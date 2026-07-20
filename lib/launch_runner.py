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
import botocore
import botocore.config
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
parser.add_argument('--user', type=str, default=None) # user in form of user_id:group_id to use to exec container (e.g. 1000:1000)
parser.add_argument('--mps', action='store_true') # use nvidia MPS server
parser.add_argument('--gpu', type=int, default=None) # which GPU to use (None means default/first one)
parser.add_argument('--cpuset_cpus', type=str, default=None) # value of cpuset_cpus to forward to container
parser.add_argument('--max_failed_heartbeats_count', type=int, default=5) # how many heartbeats failures in a row must happen before give up
parser.add_argument('--max_failed_pending_launch_gets_count', type=int, default=10) # how many failed attempts to get a pending launch in a row must happen before give up
parser.add_argument('--max_failed_result_uploads_count', type=int, default=10) # how many failed attempts to upload result of a launch in a row must happen before give up

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
    RUN = auto()
    RESULT_UPLOAD = auto()

state = State.IDLE
launch_id = None
launch_start_time = None
launch = None
pull_result = None
pull_finished_event = None
pull_thread = None
container = None
failed_heartbeats_count = 0
failed_pending_launch_gets_count = 0
failed_result_uploads_count = 0
run_result = None

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
    my_time = time.time()
    
    try:
        heartbeat_key = f'{args.key_prefix}/heartbeats/{runner_name}/{int(my_time)}{lu.when(launch_id is not None, lambda: '_' + launch_id + '|' + str(int(my_time - launch_start_time)), '')}'
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
        failed_heartbeats_count = 0
    except botocore.exceptions.ClientError as e:
        LOG.error(f'Failed to send heartbeat: {str(e)}')
        failed_heartbeats_count += 1

    if failed_heartbeats_count >= args.max_failed_heartbeats_count:
        raise Exception(f'Threshold of failed heartbeats ' + 
                        f'({failed_heartbeats_count} vs {args.max_failed_heartbeats_count}) reached, giving up')

    if state == State.IDLE:
        assert launch_id is None
        assert launch_start_time is None
        assert launch is None
        assert pull_result is None
        assert pull_finished_event is None
        assert pull_thread is None
        assert container is None
        s3_context = None

        try:
            s3_context = 'list_objects_v2'
            response = s3.list_objects_v2(Bucket=args.s3_bucket_name, Prefix=f'{args.key_prefix}/pending_launches/{runner_name}')

            for obj in response.get('Contents', []):
                key = obj['Key']
                launch_id = os.path.basename(key)
                launch_start_time = time.time()
                LOG(f'Processing new launch "{launch_id}"')

                s3_context = 'get_object'
                obj = s3.get_object(Bucket=args.s3_bucket_name, Key=key)
                
                with io.BytesIO(obj['Body'].read()) as b:
                    launch = json.load(b)

                s3_context = 'delete_object'
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
                
            failed_pending_launch_gets_count = 0
        except botocore.exceptions.ClientError as e:
            assert s3_context is not None
            LOG.error(f'Failed to get pending launches when doing {s3_context}: {str(e)}')
            failed_pending_launch_gets_count += 1

        if failed_pending_launch_gets_count >= args.max_failed_pending_launch_gets_count:
            raise Exception(f'Threshold of failed gets of pending launch ' + 
                            f'({failed_pending_launch_gets_count} vs {args.max_failed_pending_launch_gets_count}) reached, giving up')

    elif state == State.IMAGE_PULL:
        assert launch_id is not None
        assert launch_start_time is not None
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
                    run_result = dict(
                        Key=f'{args.key_prefix}/complete_launches/{runner_name}/{launch_id}',
                        Bucket=args.s3_bucket_name, 
                        Body=b'',
                        ContentType='application/octet-stream',
                        Metadata=metadata.asdict(),
                    )
                    launch_id = None
                    launch_start_time = None
                    launch = None
                    container = None
                    state = State.RESULT_UPLOAD
                    sleep_interval = 0
                else:
                    try:
                        device_requests = []

                        if is_gpu_present:
                            device_requests.append(DeviceRequest(count=-1, capabilities=[["gpu"]]))

                        kwargs = dict(
                            image=launch['launch_image'],
                            environment=env_vars,
                            shm_size=lu.coalesce(launch.get('shm_size'), '16G'),
                            volumes=['/dev/log:/dev/log'], # for logging
                            device_requests=device_requests,
                            detach=True,
                            remove=False,  # Keep container after exit so we can fetch its files/status
                        )

                        if args.user is not None:
                            kwargs['user'] = args.user

                        if args.mps:
                            kwargs['ipc_mode'] = 'host'
                            mps_dir_name = '/tmp/nvidia-mps'
                            mps_dir_name += lu.when(args.gpu is not None, f'-gpu{args.gpu}', '')
                            kwargs['volumes'].append(f'{mps_dir_name}:{mps_dir_name}')
                            # There may be conflict with default behavior of NVIDIA Container Toolkit 
                            # which automatically mounts /tmp/nvidia-mps if it sees MPS enabled on host.
                            # But if there several GPUs on host then default mount may ruin configuration 
                            # since there may be multiple MPS servers on host with distinct pipe dirs.
                            # So we better be explicit and use named MPS pipe dir here
                            env_vars['CUDA_MPS_PIPE_DIRECTORY'] = mps_dir_name
                        else:
                            if args.gpu is not None:
                                env_vars['CUDA_DEVICE'] = f'cuda:{args.gpu}'

                        if args.cpuset_cpus is not None:
                            kwargs['cpuset_cpus'] = args.cpuset_cpus

                        LOG.debug(f'Starting container with params: {kwargs}')
                        container = docker_client.containers.run(**kwargs)
                        LOG(f'Container "{container.name}" ({container.short_id}) started for "{launch_id}"')
                        state = State.RUN
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
                        run_result = dict(
                            Key=f'{args.key_prefix}/complete_launches/{runner_name}/{launch_id}',
                            Bucket=args.s3_bucket_name, 
                            Body=b'',
                            ContentType='application/octet-stream',
                            Metadata=metadata.asdict(),
                        )
                        launch_id = None
                        launch_start_time = None
                        launch = None
                        container = None
                        state = State.RESULT_UPLOAD
                        sleep_interval = 0
            finally:
                pull_result = None
                pull_finished_event = None
                pull_thread = None
            
    elif state == State.RUN:
        assert launch_id is not None
        assert launch_start_time is not None
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

        is_run_over = container.status in ['exited', 'dead'] or is_container_lost
        is_run_abort = False

        if not is_run_over:
            try:
                response = s3.list_objects_v2(Bucket=args.s3_bucket_name, Prefix=f'{args.key_prefix}/abort_launches/{runner_name}')
    
                for obj in response.get('Contents', []):
                    key = obj['Key']
                    abort_launch_id = os.path.basename(key)
                    LOG(f'Got abort request for launch "{abort_launch_id}"')
                    s3.delete_object(Bucket=args.s3_bucket_name, Key=key)
                    is_run_abort = is_run_abort or abort_launch_id == launch_id
            except botocore.exceptions.ClientError as e:
                LOG.warn(f'Failed to get abort requests: {str(e)}')
    
        if is_run_over:
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

            run_result = dict(
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
            launch_start_time = None
            launch = None
            container = None
            sleep_interval = 0
            state = State.RESULT_UPLOAD
        elif is_run_abort:
            container.remove(force=True)
            LOG.error(f'Container "{container.name}" ({container.short_id}) removed due to abort of launch "{launch_id}"')
            launch_id = None
            launch_start_time = None
            launch = None
            container = None
            run_result = None
            sleep_interval = 0
            state = State.IDLE

    elif state == State.RESULT_UPLOAD:
        assert run_result is not None
        
        try:
            s3.put_object(**run_result)
            run_result = None
            state = State.IDLE
            failed_result_uploads_count = 0
        except botocore.exceptions.ClientError as e:
            LOG.error(f'Failed to upload run result: {str(e)}')
            failed_result_uploads_count += 1

        if failed_result_uploads_count >= args.max_failed_result_uploads_count:
            raise Exception(f'Threshold of failed result uploads ' + 
                            f'({failed_result_uploads_count} vs {args.max_failed_result_uploads_count}) reached, giving up')
        
    
    time.sleep(sleep_interval)
    



