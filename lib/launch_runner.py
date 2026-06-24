import os
import sys
import io
import json
import time
import argparse
import socket
from enum import IntEnum, auto
import dataclasses
from dataclasses import dataclass
import logging
import threading

import boto3
import docker
from docker.errors import APIError, DockerException, NotFound
from docker.types import DeviceRequest
import tarfile

import lang_utils as lu
from logging_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--name_prefix', type=str, default=None)
parser.add_argument('--s3_endpoint_url', type=str, default='https://s3.ru-7.storage.selcloud.ru:443')
parser.add_argument('--s3_bucket_name', type=str, default='neurolab')
parser.add_argument('--key_prefix', type=str, default='runners')
parser.add_argument('--heartbeat_interval', type=int, default=10)
parser.add_argument('--log_level', type=str, default='info')
parser.add_argument('-e', action='append', default=[]) # env vars to forward
args = parser.parse_args()
env_vars = {}

for env_var in args.e:
    pos = env_var.find('=')
    assert pos > -1, env_var
    env_key = env_var[:pos]
    env_val = env_var[pos+1:]
    assert env_key, env_var
    assert env_val, env_var
    env_vars[env_key] = env_val

LOG = Logging.get()
LOG.enable('syslog', False)
LOG.enable('stdout', False)
LOG.enable('verbose_stdout', True)
LOG.set_log_level('all', logging.getLevelName(args.log_level.upper()))

name_prefix = lu.coalesce(args.name_prefix, socket.gethostname())
runner_name = f'{name_prefix}_{time.time():.3f}'
LOG(f'Runner "{runner_name}" starting')

docker_client = docker.from_env()

def check_gpu_presence():
    test_request = docker.types.DeviceRequest(
        count=1,  # Requesting just 1 for a lightweight test
        capabilities=[['gpu']], 
        driver='nvidia'
    )
    try:
        # Launch a lightweight, instant-exit test container
        docker_client.containers.run(
            image='alpine:latest',
            command='true',
            device_requests=[test_request],
            remove=True # Auto-cleanup
        )
        return True
    except APIError as e:
        LOG.error(f'Failed to launch "alpine:latest" with GPU capabilities: {str(e)}')
        error_msg = str(e).lower()
        # Detect missing hardware or broken driver bindings
        if 'capabilities' in error_msg or 'gpu' in error_msg:
            return False
            
        raise e

is_gpu_present = check_gpu_presence()
LOG(f'{is_gpu_present=}')

s3_session = boto3.Session()
s3_credentials = s3_session.get_credentials()
env_vars['AWS_ACCESS_KEY_ID'] = s3_credentials.access_key
env_vars['AWS_SECRET_ACCESS_KEY'] = s3_credentials.secret_key
env_vars['AWS_DEFAULT_REGION'] = s3_session.region_name

s3 = s3_session.client('s3', endpoint_url=args.s3_endpoint_url)

class State(IntEnum):
    IDLE = auto()
    IMAGE_PULL = auto()
    RUNNING = auto()

state = State.IDLE
launch_id = None
launch = None
pull_result = None
pull_finished_event = None
pull_thread = None
container = None

@dataclass(slots=True)
class ResultMetadata:
    is_ok: bool = None
    error_message: str = None
    error_code: int = None
    runner_name: str = None

    def asdict(self):
        return dict(
            is_ok=str(self.is_ok), 
            error_message=lu.coalesce(self.error_message, ''), 
            error_code=lu.when(self.error_code, lambda: str(self.error_code),  ''),
            runner_name=runner_name,
        )

# To be called in a separate thread (to avoid block if main process)
def pull_image(image_name, pull_result, finish_event):
    try:
        # stream=True returns a generator yielding status updates
        output_stream = docker_client.api.pull(image_name, stream=True, decode=True)
    
        for line in output_stream:
            LOG.debug(f'Pulling: {json.dumps(line)}')
            
        pull_result.is_ok = True
    except Exception as e:
        pull_result.is_ok = False
        pull_result.error_message = f'Failed to pull launch image "{image_name}": {str(e)}'
        LOG.error(pull_result.error_message)
    finally:
        finish_event.set()

LOG(f'Runner ready')
        
while True:
    sleep_interval = args.heartbeat_interval
    heartbeat_key = f'{args.key_prefix}/heartbeats/{runner_name}/{int(time.time())}{lu.when(launch_id, lambda: '_' + launch_id, '')}'
    s3.put_object(
        Key=heartbeat_key,
        Bucket=args.s3_bucket_name,
        Body=b'',
    )
    LOG.debug(
        f'Heartbeat sent "{heartbeat_key}", ' +
        f'state={state.name}' +
        lu.when(container is not None, lambda: f', container "{container.name}" ({container.short_id})', ''),
    )

    if state == State.IDLE:
        assert launch_id is None
        assert launch is None
        assert pull_result is None
        assert pull_finished_event is None
        assert pull_thread is None
        assert container is None
        
        response = s3.list_objects_v2(Bucket=args.s3_bucket_name, Prefix=f'{args.key_prefix}/pending_launches/{runner_name}')
        
        for obj in response.get('Contents', []):
            key = obj['Key']
            launch_id = os.path.basename(key)
            LOG(f'Processing new launch "{launch_id}"')
            
            obj = s3.get_object(Bucket=args.s3_bucket_name, Key=key)
            
            with io.BytesIO(obj['Body'].read()) as b:
                launch = json.load(b)
    
            s3.delete_object(Bucket=args.s3_bucket_name, Key=key)

            pull_result = ResultMetadata(is_ok=False)
            pull_finished_event = threading.Event()
            pull_thread = threading.Thread(
                target=pull_image, 
                args=(launch['launch_image'], pull_result, pull_finished_event),
                daemon=True # Daemon ensures the thread dies if the main script kills itself
            )
            pull_thread.start()
            state = State.IMAGE_PULL
            sleep_interval = 0
            LOG(f'Started pull of launch image "{launch['launch_image']}"')
            break

    elif state == State.IMAGE_PULL:
        assert launch_id is not None
        assert launch is not None
        assert pull_result is not None
        assert pull_finished_event is not None
        assert pull_thread is not None
        assert container is None

        if pull_finished_event.is_set():
            try:
                if pull_result.is_ok == False:
                    metadata = ResultMetadata(
                        is_ok=False, 
                        error_message=f'Failed to pull launch image: {pull_result.error_message}', 
                        error_code=1,
                        runner_name=runner_name,
                    )
                    s3.put_object(
                        Key=f'{args.key_prefix}/complete_launches/{runner_name}/{launch_id}',
                        Bucket=args.s3_bucket_name, 
                        Body=b'',
                        ContentType='application/octet-stream',
                        Metadata=metadata.asdict(),
                    )
                    launch_id = None
                    launch = None
                    container = None
                    state = State.IDLE
                    sleep_interval = 0
                else:
                    try:
                        device_requests = []

                        if is_gpu_present:
                            device_requests.append(DeviceRequest(count=-1, capabilities=[["gpu"]]))
                            
                        container = docker_client.containers.run(
                            image=launch['launch_image'],
                            environment=env_vars,
                            shm_size=lu.coalesce(launch.get('shm_size'), '16G'),
                            volumes=['/dev/log:/dev/log'], # for logging
                            device_requests=device_requests,
                            detach=True,
                            remove=False,  # Keep container after exit so we can fetch its files/status
                        )
                        LOG(f'Container "{container.name}" ({container.short_id}) started for "{launch_id}"')
                        state = State.RUNNING
                        sleep_interval = 0
                    except DockerException as e:
                        error_message = f'Failed to start container for "{launch_id}": {str(e)}'
                        LOG.error(error_message)
                        metadata = ResultMetadata(
                            is_ok=False, 
                            error_message=error_message, 
                            error_code=1, 
                            runner_name=runner_name,
                        )
                        s3.put_object(
                            Key=f'{args.key_prefix}/complete_launches/{runner_name}/{launch_id}',
                            Bucket=args.s3_bucket_name, 
                            Body=b'',
                            ContentType='application/octet-stream',
                            Metadata=metadata.asdict(),
                        )
                        launch_id = None
                        launch = None
                        container = None
                        state = State.IDLE
                        sleep_interval = 0
            finally:
                pull_result = None
                pull_finished_event = None
                pull_thread = None
            
    elif state == State.RUNNING:
        assert launch_id is not None
        assert launch is not None
        assert pull_result is None
        assert pull_finished_event is None
        assert pull_thread is None
        assert container is not None
        is_container_lost = False

        try:
            container.reload()
            LOG.debug(f'{container.status=}')
        except NotFound:
            is_container_lost = True
            LOG.error(f'Container is lost')
    
        if container.status in ['exited', 'dead'] or is_container_lost:
            response = b''
            
            if is_container_lost:
                metadata = ResultMetadata(is_ok=False, error_message='Container is lost', error_code=1, runner_name=runner_name)
            else:
                exit_attrs = container.attrs["State"]
                LOG.debug(f'{exit_attrs=}')
                exit_code = exit_attrs['ExitCode']
                
                LOG(f'Container "{container.name}" ({container.short_id}) for "{launch_id}" finished: status="{container.status}", exit code={exit_code}')
    
                if exit_code != 0:
                    error_message = exit_attrs['Error']
                    error_message = lu.when(error_message is None or error_message == '', f'See logs of container "{container.name}" ({container.short_id})', error_message)
                    metadata = ResultMetadata(is_ok=False, error_message=error_message, error_code=exit_code, runner_name=runner_name)
                else:
                    metadata = ResultMetadata(is_ok=True, runner_name=runner_name)
            
                    if 'result_fname' in launch:
                        result_fname = launch['result_fname']
                        LOG.debug(f'Fetching "{result_fname}" from container')
                        try:
                            # get_archive returns a raw tar stream of the target file/folder
                            stream, stat = container.get_archive(result_fname)
                            file_data = b''
                            
                            for chunk in stream:
                                file_data += chunk
                                
                            with tarfile.open(fileobj=io.BytesIO(file_data)) as tar:
                                result_fname_f = tar.extractfile(stat['name'])
                                response = result_fname_f.read()
                
                            LOG.info(f'Fetched {len(response)} bytes of result file "{result_fname}" from container')
                        except DockerException as e:
                            metadata = ResultMetadata(
                                is_ok=False, 
                                error_message=f'Failed to fetch result file "{result_fname}" in container: {str(e)}', 
                                error_code=1,
                                runner_name=runner_name,
                            )
                            LOG.error(metadata.error_message)

            s3.put_object(
                Key=f'{args.key_prefix}/complete_launches/{runner_name}/{launch_id}',
                Bucket=args.s3_bucket_name, 
                Body=response,
                ContentType='application/octet-stream',
                Metadata=metadata.asdict(),
            )

            if metadata.is_ok and launch.get('keep_container', False) == False and not is_container_lost:
                container.remove()
                LOG(f'Container removed')
            
            LOG(f'Launch "{launch_id}" completed {lu.when(metadata.is_ok, 'succesfully', 'with FAILURE')}')
            launch_id = None
            launch = None
            container = None
            sleep_interval = 0
            state = State.IDLE
    
    time.sleep(sleep_interval)
    



