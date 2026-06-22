import os
import sys
import io
import json
import time
from enum import IntEnum, auto
import dataclasses
from dataclasses import dataclass

import pika
import boto3
import botocore

import lang_utils as lu
from logging_utils import *

RMQ_DEFAULT_CONNECTION_URL = 'amqp://guest:guest@rabbitmq:5672/%2F'
RMQ_LAUNCH_REQUESTS_QUEUE_NAME = 'launch_requests'
RMQ_MAGIC_REPLY_QUEUE_NAME = 'amq.rabbitmq.reply-to'

class LaunchRequest:
    def __init__(self, rmq_connection_url):
        connection_parameters = pika.URLParameters(rmq_connection_url)
        self.connection = pika.BlockingConnection(connection_parameters)
        self.channel = self.connection.channel()
        self.channel.basic_consume(
            queue=RMQ_MAGIC_REPLY_QUEUE_NAME, 
            on_message_callback=self.on_reply, 
            auto_ack=True
        )
        self.result_headers = None
        self.result_body = None

    def __call__(self, request):
        self.result_headers = None
        self.result_body = None
        properties = pika.spec.BasicProperties(
            reply_to=RMQ_MAGIC_REPLY_QUEUE_NAME, 
            delivery_mode=pika.DeliveryMode.Persistent
        )
        self.channel.basic_publish(
            exchange='', 
            routing_key=RMQ_LAUNCH_REQUESTS_QUEUE_NAME, 
            body=json.dumps(request).encode(),
            properties=properties
        )
        self.channel.start_consuming()
        result_headers = self.result_headers
        result_body = self.result_body
        self.result_headers = None
        self.result_body = None
        return result_headers, result_body

    def on_reply(self, ch, method, properties, body):
        Logging.get().debug(f'on_reply: {method=}, {properties=}, len(body)={len(body)}')
        self.result_headers = properties.headers
        self.result_body = body
        self.channel.close()

    @staticmethod
    def run(request, rmq_connection_url=RMQ_DEFAULT_CONNECTION_URL):
        client = LaunchRequest(rmq_connection_url)
        return client(request)
    
@dataclass(slots=True)
class LaunchRunner:
    name: str = None
    launch_id: object = None # None - runner is idle, otherwise - it executes given launch
    eol_time: object = None # computed end of life time, extended by heartbeat

class LaunchStatus(IntEnum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETE = auto()

@dataclass(slots=True)
class Launch:
    id: str = None
    reply_to: str = None
    status: object = None
    request: object = None

class LaunchDispatcher:
    def __init__(self, rmq_connection_url, s3_endpoint_url, s3_bucket_name, key_prefix, launches_fname):
        self.rmq_connection_parameters = pika.URLParameters(rmq_connection_url)
        self.rmq_connection = pika.BlockingConnection(self.rmq_connection_parameters)
        self.rmq_connection.call_later(delay=1, callback=self.on_idle)
        self.rmq_channel = self.rmq_connection.channel()
        queue = self.rmq_channel.queue_declare(
            queue=RMQ_LAUNCH_REQUESTS_QUEUE_NAME,
            durable=True,
            arguments={'x-single-active-consumer': True},
        )

        self.s3 = boto3.client('s3', endpoint_url=s3_endpoint_url)
        self.s3_bucket_name = s3_bucket_name
        self.key_prefix = key_prefix

        # Runners management
        self.runners = {}
        self.eol_duration = 60
        
        self.launches = {}
        self.launches_fname = launches_fname

        if self.launches_fname is not None and os.path.exists(self.launches_fname):
            Logging.get().debug(f'Loading launches from "{self.launches_fname}"')
            
            with open(self.launches_fname, 'r') as f:
                loaded = json.load(f)
                self.launches.update(dict(map(lambda kv: (kv[0], Launch(**kv[1])), loaded.items())))

                for k, v in self.launches.items():
                    Logging.get().debug(f'Loaded launch "{k}": {v}')
                
                Logging.get().info(f'Loaded {len(self.launches)} launches')

    def run(self):
        self.rmq_channel.basic_qos(prefetch_count=1) # max 1 unacked message, i.e. serial processing
        self.rmq_channel.basic_consume(
            queue=RMQ_LAUNCH_REQUESTS_QUEUE_NAME, 
            on_message_callback=self.on_request, 
            auto_ack=False,
        )
        self.rmq_channel.start_consuming()

    def on_idle(self):
        Logging.get().debug(f'on_idle')
        my_time = time.time()
        is_launches_dirty = False

        try:
            # Update state of runners - find out which are alive/dead
            response = self.s3.list_objects_v2(Bucket=self.s3_bucket_name, Prefix=f'{self.key_prefix}/heartbeats')
    
            for obj in response.get('Contents', []):
                key = obj['Key']
                Logging.get().debug(f'Processing heartbeat "{key}"')
                key_payload = os.path.basename(key)
                sep_index = key_payload.find('_')
    
                if sep_index == -1:
                    heartbeat_time = lu.from_str(int, key_payload, 0)
                    running_launch_id = None
                else:
                    assert sep_index > 0, sep_index
                    heartbeat_time = lu.from_str(int, key_payload[:sep_index], 0)
                    running_launch_id = key_payload[sep_index+1:]
                    
                runner_name = os.path.basename(os.path.dirname(key))
                assert len(runner_name) > 0
    
                if not runner_name in self.runners:
                    # New runner
                    eol_time = heartbeat_time + self.eol_duration
    
                    if eol_time > my_time:
                        self.runners[runner_name] = LaunchRunner(
                            name=runner_name,
                            eol_time=eol_time,
                            launch_id=running_launch_id,
                        )
                        Logging.get().info(f'New runner "{runner_name}" with {lu.when(running_launch_id, f'running launch "{running_launch_id}"', 'no running launch')}')
                    else:
                        Logging.get().debug(f'Ignoring stale heartbeat for "{runner_name}": {eol_time=}, {my_time=}, delta={my_time - eol_time}')
                else:
                    # Update runner
                    self.runners[runner_name].eol_time = heartbeat_time + self.eol_duration
    
                    if self.runners[runner_name].launch_id != running_launch_id:
                        Logging.get().warn(f'Inconsistency: runner "{runner_name}": {self.runners[runner_name].launch_id=} vs {running_launch_id=}!')
    
                self.s3.delete_object(Bucket=args.s3_bucket_name, Key=key)
    
            # Recyle dead runners
            dead_runner_names = []
            
            for runner_name, runner in self.runners.items():
                eol_time = self.runners[runner_name].eol_time
                
                if eol_time < my_time:
                    Logging.get().info(f'Runner "{runner_name}" is dead: {eol_time=}, {my_time=}, delta={my_time - eol_time}')
                    dead_runner_names.append(runner_name)
    
            # Move back launches from dead runners
            for runner_name in dead_runner_names:
                runner = self.runners[runner_name]
                del self.runners[runner_name]
                assert not runner_name in self.runners
    
                if runner.launch_id is not None:
                    launch_id = runner.launch_id
                    
                    if not launch_id in self.launches:
                        Logging.get().warn(f'Dead runner "{runner_name}" refers to unknown launch "{launch_id}"')
                    else:
                        launch = self.launches[launch_id]
                        
                        if launch.status != LaunchStatus.RUNNING:
                            Logging.get().warn(f'Inconsistency: launch "{launch_id}" assigned for dead runner "{runner_name}" has status={launch.status.name} which is not RUNNING!')
    
                        launch.status = LaunchStatus.PENDING
                        is_launches_dirty = True
                        Logging.get().info(f'Launch "{launch_id}" brought back to PENDING status from dead runner "{runner_name}"')
    
            # Collect launch results
            response = self.s3.list_objects_v2(Bucket=self.s3_bucket_name, Prefix=f'{self.key_prefix}/complete_launches')
    
            for obj in response.get('Contents', []):
                key = obj['Key']
                launch_id = os.path.basename(key)
                runner_name = os.path.basename(os.path.dirname(key))
                Logging.get().info(f'Processing complete launch "{launch_id}" from runner "{runner_name}"')
                launch_result = self.s3.get_object(Bucket=self.s3_bucket_name, Key=key)
                launch_result_metadata = launch_result['Metadata']
                Logging.get().debug(f'{launch_result_metadata=}')
                launch_result_body = launch_result['Body'].read()
                Logging.get().info(f'Launch result size={len(launch_result_body)}, {type(launch_result_body)=}')
    
                if not runner_name in self.runners:
                    Logging.get().warn(f'Unknown runner "{runner_name}", do not know which runner to mark free')
                else:
                    self.runners[runner_name].launch_id = None
                    Logging.get().info(f'Runner "{runner_name}" is marked free')
    
                if not launch_id in self.launches:
                    Logging.get().warn(f'Unknown launch "{launch_id}", do not know where to return result')
                else:
                    launch = self.launches[launch_id]
    
                    if launch.status != LaunchStatus.RUNNING:
                        Logging.get().warn(f'Launch "{launch_id}" has status={launch.status.name} which is not RUNNING. Inconsistency!')
                    
                    properties = pika.spec.BasicProperties(
                        delivery_mode=pika.DeliveryMode.Persistent,
                        headers=dict(
                            is_ok=launch_result_metadata['is_ok'] == str(True),
                            error_message=launch_result_metadata.get('error_message', None),
                            error_code=lu.when(launch_result_metadata.get('error_code'), lambda: int(launch_result_metadata['error_code']), None),
                        ),
                    )
                    self.rmq_channel.basic_publish(
                        exchange='', 
                        routing_key=launch.reply_to, 
                        body=launch_result_body,
                        properties=properties,
                    )
    
                    Logging.get().info(f'Launch results sent to {launch.reply_to}')
                    del self.launches[launch_id]
                    not launch_id in self.launches
                    is_launches_dirty = True
    
                self.s3.delete_object(Bucket=args.s3_bucket_name, Key=key)
    
            # Dispatch pending launches to runners
            free_runner_names = set(map(lambda kv: kv[0], filter(lambda kv: kv[1].launch_id is None, self.runners.items())))
            pending_launches_count = list(filter(lambda kv: kv[1].status == LaunchStatus.PENDING, self.launches.items()))
            Logging.get().debug(f'Free runners count={len(free_runner_names)}, pending launches count={len(pending_launches_count)}')
    
            if free_runner_names:
                for launch_id, launch in pending_launches_count:
                    free_runner_name = free_runner_names.pop()
                    free_runner = self.runners[free_runner_name]
                    free_runner.launch_id = launch_id
                    self.s3.put_object(
                        Key=f'{args.key_prefix}/pending_launches/{free_runner_name}/{launch_id}',
                        Bucket=args.s3_bucket_name,
                        Body=json.dumps(launch.request).encode(),
                    )
    
                    launch.status = LaunchStatus.RUNNING
                    is_launches_dirty = True
                    Logging.get().info(f'Launch "{launch_id}" dispatched to runner "{free_runner_name}"')
    
                    if not free_runner_names:
                        break
    
            if is_launches_dirty:
                self.save_launches()
        except botocore.exceptions.HTTPClientError as e:
            Logging.get().error(f'Connectivity error: {str(e)}')
        except botocore.exceptions.ConnectionError as e:
            Logging.get().error(f'Connectivity error: {str(e)}')
        
        self.rmq_connection.call_later(delay=1, callback=self.on_idle)

    def on_request(self, ch, method, properties, body):
        Logging.get().debug(f'on_message: method={method}, properties={properties}, len(body)={len(body)}')
        reply_to = properties.reply_to

        request = json.loads(body.decode())
        launch = Launch(
            id=f'{os.path.basename(request['launch_image'])}_{time.time():.3f}',
            reply_to=reply_to,
            status=LaunchStatus.PENDING,
            request=request,
        )
        assert not launch.id in self.launches, launch.id
        self.launches[launch.id] = launch
        Logging.get().info(f'New launch requested: {dataclasses.asdict(launch)}')
        self.save_launches()
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def save_launches(self):
        if self.launches_fname is None:
            return
            
        with open(self.launches_fname, 'w') as f:
            json.dump(dict(map(lambda kv: (kv[0], dataclasses.asdict(kv[1])), self.launches.items())), f)
            Logging.get().info(f'Saved {len(self.launches)} launches to "{self.launches_fname}"')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--rmq_connection_url', type=str, default='amqp://guest:guest@rabbitmq:5672/%2F')
    parser.add_argument('--s3_endpoint_url', type=str, default='https://s3.ru-7.storage.selcloud.ru:443')
    parser.add_argument('--s3_bucket_name', type=str, default='neurolab')
    parser.add_argument('--key_prefix', type=str, default='runners')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--launches_fname', type=str, default=None)
    args = parser.parse_args()
    
    LOG = Logging.get()
    LOG.enable('syslog', False)
    LOG.enable('stdout', False)
    LOG.enable('verbose_stdout', True)
    LOG.set_log_level('all', logging.getLevelName(args.log_level.upper()))

    dispatcher = LaunchDispatcher(
        rmq_connection_url=args.rmq_connection_url,
        s3_endpoint_url=args.s3_endpoint_url,
        s3_bucket_name=args.s3_bucket_name,
        key_prefix=args.key_prefix,
        launches_fname=args.launches_fname,
    )

    dispatcher.run()

