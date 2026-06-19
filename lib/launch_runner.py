import os
import sys
import io
import json
import time
import argparse
import socket
from enum import IntEnum, auto
import logging

import boto3
import docker
from docker.errors import APIError

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
LOG(f'Runner "{runner_name}" ready')

class State(IntEnum):
    IDLE = auto()
    BUSY = auto()

state = State.IDLE
launch_id = None
launch = None
docker_client = docker.from_env()

while True:
    s3.put_object(
        Key=f'{args.key_prefix}/heartbeats/{runner_name}/{int(time.time())}',
        Bucket=args.s3_bucket_name,
        Body=''.encode(),
    )
    LOG.debug('Heartbeat sent')

    if state == State.IDLE:
        response = s3.list_objects_v2(Bucket=args.s3_bucket_name, Prefix=f'{args.key_prefix}/pending_launches/{runner_name}')

        for obj in response.get('Contents', []):
            launch_id = obj['Key']
            launch = args.s3.get_object(Bucket=args.s3_bucket_name, Key=launch_id)
            
            with io.BytesIO(job['Body'].read()) as b:
                job = json.load(b)

            s3.delete_object(Bucket=args.s3_bucket_name, Key=launch_id)
            
            LOG(f'New launch "{launch_id}": {launch}')
            state = State.BUSY
            container = docker_client.containers.run(
                image=launch['image_name'],
                detach=True,
                remove=False,  # Keep container after exit so we can fetch its files/status
            )

            break
            
    elif state == State.BUSY:
        LOG(f'Launch #{launch_id} complete')
        response = ''.encode() # TBD make real
        response = s3.put_object(
            Key=f'{key_prefix}/complete_launches/{runner_name}/{launch_id}',
            Bucket=args.s3_bucket_name, 
            Body=response,
        )
        launch_id = None
        launch = None
        state = State.IDLE
    
    time.sleep(args.heartbeat_interval)
    



