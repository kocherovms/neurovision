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

import boto3
import docker
from docker.errors import APIError, DockerException
import tarfile

import lang_utils as lu
from logging_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--s3_endpoint_url', type=str, default='https://s3.ru-7.storage.selcloud.ru:443')
parser.add_argument('--s3_bucket_name', type=str, default='neurolab')
parser.add_argument('--key_prefix', type=str, default='runners')
parser.add_argument('--poll_interval', type=int, default=7)
parser.add_argument('--heartbeat_interval', type=int, default=10)
parser.add_argument('--log_level', type=str, default='info')
args = parser.parse_args()

LOG = Logging.get()
LOG.enable('syslog', False)
LOG.enable('stdout', False)
LOG.enable('verbose_stdout', True)
LOG.set_log_level('all', logging.getLevelName(args.log_level.upper()))

s3 = boto3.client('s3', endpoint_url=args.s3_endpoint_url)
runner_name = f'{socket.gethostname()}_{int(time.time())}'
LOG.info(f'Runner "{runner_name}" ready')

class State(IntEnum):
    IDLE = auto()
    BUSY = auto()

state = State.IDLE
launch_id = None
launch = None
container = None
docker_client = docker.from_env()

@dataclass(slots=True)
class ResultMetadata:
    is_ok: bool = None
    error_message: str = None
    error_code: int = None

    def asdict(self):
        return dict(
            is_ok=str(self.is_ok), 
            error_message=lu.coalesce(self.error_message, ''), 
            error_code=lu.when(self.error_code, lambda: str(self.error_code),  ''),
        )

while True:
    heartbeat_key = f'{args.key_prefix}/heartbeats/{runner_name}/{int(time.time())}{lu.when(launch_id, lambda: '_' + launch_id, '')}'
    s3.put_object(
        Key=heartbeat_key,
        Bucket=args.s3_bucket_name,
        Body=b'',
    )
    LOG.debug(f'Heartbeat sent "{heartbeat_key}"')

    if state == State.IDLE:
        response = s3.list_objects_v2(Bucket=args.s3_bucket_name, Prefix=f'{args.key_prefix}/pending_launches/{runner_name}')
        
        for obj in response.get('Contents', []):
            key = obj['Key']
            launch_id = os.path.basename(key)
            LOG(f'Processing new launch "{launch_id}"')
            
            obj = s3.get_object(Bucket=args.s3_bucket_name, Key=key)
            
            with io.BytesIO(obj['Body'].read()) as b:
                launch = json.load(b)
    
            s3.delete_object(Bucket=args.s3_bucket_name, Key=key)
            
            try:
                container = docker_client.containers.run(
                    image=launch['launch_image'],
                    detach=True,
                    remove=False,  # Keep container after exit so we can fetch its files/status
                )
                LOG(f'Container "{launch['launch_image']}" started for "{launch_id}"')
                state = State.BUSY
            except DockerException as e:
                LOG.error(f'Failed to start container "{launch['launch_image']}" for "{launch_id}": {e}')
                s3.put_object(
                    Key=f'{args.key_prefix}/complete_launches/{runner_name}/{launch_id}',
                    Bucket=args.s3_bucket_name, 
                    Body=b'',
                    ContentType='application/octet-stream',
                    Metadata=ResultMetadata(is_ok=False, error_message=str(e), error_code=1).asdict(),
                )
                launch_id = None
                launch = None
                container = None
                state = State.IDLE
            
            break
            
    elif state == State.BUSY:
        assert launch_id is not None
        assert launch is not None
        assert container is not None

        container.reload()
        LOG.debug(f'{container.status=}')
    
        if container.status in ['exited', 'dead']:
            exit_attrs = container.attrs["State"]
            LOG.debug(f'{exit_attrs=}')
            exit_code = exit_attrs['ExitCode']
            
            LOG(f'Container for "{launch_id}" finished (status="{container.status}"), exit code={exit_code}')
            response = b''

            if exit_code == 0:
                metadata = ResultMetadata(is_ok=True)
        
                if 'result_fname' in launch:
                    result_fname = launch['result_fname']
                    LOG.debug(f'Fetching "{result_fname}" from container')
                    # get_archive returns a raw tar stream of the target file/folder
                    stream, stat = container.get_archive(result_fname)
                    file_data = b''
                    
                    for chunk in stream:
                        file_data += chunk
                        
                    with tarfile.open(fileobj=io.BytesIO(file_data)) as tar:
                        result_fname_f = tar.extractfile(stat['name'])
                        response = result_fname_f.read()
        
                    LOG.info(f'Fetched {len(response)} bytes of result file "{result_fname}" from container')
            else:
                metadata = ResultMetadata(is_ok=False, error_message=exit_attrs['Error'], error_code=exit_code)

            print(f'kms@ {metadata.asdict()=}')
            
            s3.put_object(
                Key=f'{args.key_prefix}/complete_launches/{runner_name}/{launch_id}',
                Bucket=args.s3_bucket_name, 
                Body=response,
                ContentType='application/octet-stream',
                Metadata=metadata.asdict(),
            )
            LOG(f'Launch "{launch_id}" is complete {lu.when(metadata.is_ok, 'succesfully', 'with FAILURE')}')
            launch_id = None
            launch = None
            container = None
            state = State.IDLE
    
    time.sleep(args.heartbeat_interval)
    



