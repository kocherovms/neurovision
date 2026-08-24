import os
import sys
import pickle
import io
import json
from functools import lru_cache
import time

import pika
import pika.exceptions
import boto3

import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as plt_backend_agg

import torch

import av # video support

import lang_utils as lu
from logging_utils import *
from artifact_registry import *

RMQ_EVENTS_EXCHANGE_NAME = 'events'
RMQ_EVENTS_QUEUE_NAME = 'events'
RMQ_DEFAULT_CONNECTION_URL = 'amqp://guest:guest@rabbitmq:5672/%2F'

S3_DEFAULT_ENDPOINT_URL = 'https://s3.ru-7.storage.selcloud.ru:443'
S3_DEFAULT_BUCKET_NAME = 'neurolab'
S3_DEFAULT_KEY_PREFIX = 'launches'

def _figure_to_image(figure, close):
    canvas = plt_backend_agg.FigureCanvasAgg(figure)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    image_chw = np.moveaxis(image_hwc, source=2, destination=0)
    
    if close:
        plt.close(figure)
    
    return image_chw

class RmqSummaryBase:
    def __init__(self, rmq_connection_url=RMQ_DEFAULT_CONNECTION_URL):
        self.connection_parameters = pika.URLParameters(rmq_connection_url)
        self.reconnect()
        exchange = self.channel.exchange_declare(
            exchange=RMQ_EVENTS_EXCHANGE_NAME, 
            exchange_type='topic',
            durable=True,
        )
        queue = self.channel.queue_declare(
            queue=RMQ_EVENTS_QUEUE_NAME,
            durable=True,
            arguments={'x-single-active-consumer': True},
        )
        self.channel.queue_bind(
            exchange=RMQ_EVENTS_EXCHANGE_NAME, 
            queue=RMQ_EVENTS_QUEUE_NAME,
        )

    def reconnect(self):
        self.connection = pika.BlockingConnection(self.connection_parameters)
        self.channel = self.connection.channel()
        
class RmqSummaryWriter(RmqSummaryBase):
    def __init__(self, log_dir, rmq_connection_url=RMQ_DEFAULT_CONNECTION_URL):
        super().__init__(rmq_connection_url)
        self.log_dir = log_dir
        self.scalar_batch = []

    def add_scalar(self, tag, scalar_value, global_step, is_batched=True):
        """
        is_batched=True is an optimization. All values will be staged and written only on flush(). This way
        we avoid overload of RabbitMQ with too frequent messages
        """
        if is_batched:
            self.scalar_batch.append((tag, scalar_value, global_step))
            return
            
        properties = self._create_message_properties('add_scalar')
        properties.headers['tag'] = tag
        properties.headers['global_step'] = global_step
        
        with io.BytesIO() as b:
            if isinstance(scalar_value, torch.Tensor) or isinstance(scalar_value, np.ndarray):
                scalar_value = scalar_value.item()
                
            pickle.dump(scalar_value, b)
            self._robust_publish(body=b.getvalue(), properties=properties)
    
    def add_text(self, tag, text_string, global_step):
        properties = self._create_message_properties('add_text')
        properties.headers['tag'] = tag
        properties.headers['global_step'] = global_step
        self._robust_publish(body=text_string.encode(), properties=properties)
        
    def add_figure(self, tag, figure, global_step, close):
        properties = self._create_message_properties('add_figure')
        properties.headers['tag'] = tag
        properties.headers['global_step'] = global_step

        with io.BytesIO() as b:
            image = _figure_to_image(figure, close)
            pickle.dump(image, b)
            self._robust_publish(body=b.getvalue(), properties=properties)

    # To distinguish from SummaryWriter.add_video which incepts raw video frames
    def add_video_file(self, tag, video_file, global_step):
        properties = self._create_message_properties('add_video_file')
        properties.headers['tag'] = tag
        properties.headers['global_step'] = global_step

        if isinstance(video_file, io.IOBase):
            video_file.seek(0)
            self._robust_publish(body=video_file.read(), properties=properties)
        else:
            with open(video_file, 'rb') as f:
                self._robust_publish(body=f.read(), properties=properties)

    def add_file(self, file, file_name):
        properties = self._create_message_properties('add_file')
        properties.headers['file_name'] = file_name

        if isinstance(file, io.IOBase):
            file.seek(0)
            body = file.read()
        else:
            with open(file, 'rb') as f:
                body = f.read()

        self._robust_publish(body=body, properties=properties)

    def add_hparams(self, hparam_dict, metric_dict, run_name):
        properties = self._create_message_properties('add_hparams')
        properties.headers['run_name'] = run_name

        message = dict(hparam_dict=hparam_dict, metric_dict=metric_dict)
        body = json.dumps(message)
        self._robust_publish(body=body.encode(), properties=properties)

    def flush(self):
        properties = self._create_message_properties('flush')
        
        if not self.scalar_batch:
            self._robust_publish(body='', properties=properties)
        else:
            scalar_values = []
            
            for i, (tag, scalar_value, global_step) in enumerate(self.scalar_batch):
                properties.headers[f'tag_{i}'] = tag
                properties.headers[f'global_step_{i}'] = global_step
                
                if isinstance(scalar_value, torch.Tensor) or isinstance(scalar_value, np.ndarray):
                    scalar_value = scalar_value.item()

                scalar_values.append(scalar_value)
                
            with io.BytesIO() as b:
                pickle.dump(scalar_values, b)
                self._robust_publish(body=b.getvalue(), properties=properties)

            self.scalar_batch = []

    def _robust_publish(self, body, properties):
        for attempt_no in range(2):
            try:
                self.channel.basic_publish(
                    exchange=RMQ_EVENTS_EXCHANGE_NAME, 
                    routing_key=RMQ_EVENTS_QUEUE_NAME, 
                    body=body,
                    properties=properties)
                break
            except (pika.exceptions.StreamLostError, pika.exceptions.ConnectionClosedByBroker) as e:
                if attempt_no == 0:
                    self.reconnect()
                else:
                    raise

    def _create_message_properties(self, method):
        return pika.spec.BasicProperties(
            headers={
                'log_dir': self.log_dir,
                'method': method,
            },
            delivery_mode=pika.DeliveryMode.Persistent,
        )

class S3SummaryWriter:
    def __init__(self, log_dir, s3_endpoint_url=S3_DEFAULT_ENDPOINT_URL, s3_bucket_name=S3_DEFAULT_BUCKET_NAME, key_prefix=S3_DEFAULT_KEY_PREFIX):
        self.log_dir = log_dir
        self.s3_bucket_name = s3_bucket_name
        self.s3 = boto3.client('s3', endpoint_url=s3_endpoint_url)
        self.key_prefix = key_prefix
        self.batch = []
        self.batch_counter = 0

    def add_scalar(self, tag, scalar_value, global_step):
        if isinstance(scalar_value, (np.ndarray, torch.Tensor)):
            scalar_value = scalar_value.item()
            
        batch_item = dict(
            method='add_scalar',
            tag=tag,
            scalar_value=scalar_value,
            global_step=global_step,
        )
        self.batch.append(batch_item)
    
    def add_text(self, tag, text_string, global_step):
        batch_item = dict(
            method='add_text',
            tag=tag,
            text_string=text_string,
            global_step=global_step,
        )
        self.batch.append(batch_item)
        
    def add_figure(self, tag, figure, global_step, close):
        with io.BytesIO() as b:
            image = _figure_to_image(figure, close)
            pickle.dump(image, b)
            b.seek(0)
            batch_item = dict(
                method='add_figure',
                tag=tag,
                figure=b.getvalue(),
                global_step=global_step,
            )
            
        self.batch.append(batch_item)

    # To distinguish from SummaryWriter.add_video which incepts raw video frames
    def add_video_file(self, tag, video_file, global_step):
        batch_item = dict(
            method='add_video_file',
            tag=tag,
            video_file=None,
            global_step=global_step,
        )
        
        if isinstance(video_file, io.IOBase):
            video_file.seek(0)
            batch_item['video_file'] = video_file.getvalue()
        else:
            assert isinstance(video_file, str), type(video_file)
            
            with open(video_file, 'rb') as f:
                batch_item['video_file'] = f.read()

        self.batch.append(batch_item)

    def add_file(self, file, file_name):
        batch_item = dict(
            method='add_file',
            file=None,
            file_name=file_name,
        )
        
        if isinstance(file, io.StringIO):
            file.seek(0)
            batch_item['file'] = file.getvalue().encode('utf-8') # turn to bytes
        elif isinstance(file, io.BytesIO):
            file.seek(0)
            batch_item['file'] = file.getvalue()
        else:
            assert isinstance(file, str), type(file)
            
            with open(file, 'rb') as f:
                batch_item['file'] = f.read()

        self.batch.append(batch_item)

    def add_hparams(self, hparam_dict, metric_dict, run_name):
        batch_item = dict(
            method='add_hparams',
            hparam_dict=hparam_dict, 
            metric_dict=metric_dict, 
            run_name=run_name,
        )
        self.batch.append(batch_item)
    
    def flush(self):
        if not self.batch:
            return

        key = f'{self.key_prefix}/{self.log_dir}/metrics/batch_{self.batch_counter:09d}.pkl'
        with io.BytesIO() as b:
            pickle.dump(self.batch, b)
            b.seek(0)
            self.s3.put_object(
                Bucket=self.s3_bucket_name,
                Key=key,
                Body=b,
                ContentType='application/octet-stream'
            )
        self.batch_counter += 1
        self.batch = []

class RmqSummaryCollector(RmqSummaryBase):
    def __init__(self, base_log_dir, rmq_connection_url):
        super().__init__(rmq_connection_url)
        self.base_log_dir = base_log_dir

    def run(self):
        self.channel.basic_qos(prefetch_count=1) # max 1 unacked message, i.e. serial processing
        self.channel.basic_consume(
            queue=RMQ_EVENTS_QUEUE_NAME, 
            on_message_callback=self.on_message, 
            auto_ack=False)
        self.channel.start_consuming()

    def on_message(self, ch, method, properties, body):
        Logging.get().debug(f'on_message: method={method}, properties={properties}, len(body)={len(body)}')
        logic_method = properties.headers['method']
        log_dir = properties.headers['log_dir']
        global_step = properties.headers.get('global_step', None)
        global_step = lu.when(global_step, lambda: int(global_step), global_step) # all headers are strings
        tag = properties.headers.get('tag', None)
        run_name = properties.headers.get('run_name', None)

        match logic_method:
            case 'add_scalar':
                with io.BytesIO(body) as b:
                    scalar_value = pickle.load(b)
                    self.get_summary_writer(log_dir).add_scalar(tag, scalar_value, global_step)
                    Logging.get().info(f'add_scalar, {log_dir=}, {tag=}, {scalar_value=}, {global_step=}')
            case 'add_text':
                text_string = body.decode()
                self.get_summary_writer(log_dir).add_text(tag, text_string, global_step)
                Logging.get().info(f'add_text, {log_dir=}, {tag=}, {text_string[:1000]=}, {global_step=}')
            case 'add_figure':
                with io.BytesIO(body) as b:
                    image_data = pickle.load(b)
                    self.get_summary_writer(log_dir).add_image(tag, image_data, global_step)
                    Logging.get().info(f'add_figure, {log_dir=}, {tag=}, {image_data.shape=}, {global_step=}')
            case 'add_video_file':
                video_file_len = len(body)
                # Perform transcoding and upload to tensorboard UI. Very slow. In fact produces animated GIF from video file
                with io.BytesIO(body) as b:
                    container = av.open(b)
                    fps = float(container.streams.video[0].average_rate)
                    fps = lu.when(fps > 60, 60, fps)
                    frames = []
                    
                    for frame in container.decode(video=0):
                        # Convert to RGB and then to a torch tensor
                        img = frame.to_image().convert('RGB')
                        frames.append(torch.from_numpy(np.array(img)))
    
                    video_tensor = torch.stack(frames) 
                    # 3. Permute to PyTorch format: (N, T, C, H, W)
                    # add Batch dim (N=1) and move Channels (C) to index 2
                    video_tensor = video_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)
                    self.get_summary_writer(log_dir).add_video(tag, video_tensor, global_step, fps)
                    Logging.get().info(f'add_video_file, {log_dir=}, {tag=}, {video_tensor.shape=} ({video_tensor.dtype}) ({video_file_len} bytes), {global_step=}, {fps=}')
            case 'add_file':
                file_len = len(body)
                full_log_dir = os.path.join(self.base_log_dir, log_dir.lstrip('/'))
                file_name = properties.headers['file_name']
                
                with open(os.path.join(full_log_dir, file_name), 'wb') as f:
                    f.write(body)

                Logging.get().info(f'add_file, {log_dir=}, {file_name=} ({file_len} bytes)')
            case 'add_hparams':
                with io.BytesIO(body) as b:
                    message = json.load(b)
                    hparam_dict = message['hparam_dict']
                    metric_dict = message['metric_dict']
                    self.get_summary_writer(log_dir).add_hparams(hparam_dict, metric_dict, run_name=run_name)
                    Logging.get().info(f'add_hparams, {log_dir=}, {hparam_dict=}, {metric_dict=}, {run_name=}')
            case 'flush':
                sw = self.get_summary_writer(log_dir)

                if len(body) > 0:
                    with io.BytesIO(body) as b:
                        scalar_values = pickle.load(b)

                        for i, scalar_value in enumerate(scalar_values):
                            tag = properties.headers[f'tag_{i}']
                            global_step = int(properties.headers[f'global_step_{i}'])
                            sw.add_scalar(tag, scalar_value, global_step)
                            Logging.get().info(f'add_scalar (on flush), {log_dir=}, {tag=}, {scalar_value=}, {global_step=}')
                
                sw.flush()
                Logging.get().info('flush')
            case _:
                assert False, f'Unknown method="{logic_method}"'
        
        ch.basic_ack(delivery_tag=method.delivery_tag)

    @lru_cache(maxsize=100)
    def get_summary_writer(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(self.base_log_dir, log_dir.lstrip('/'))
        Logging.get().info(f'Creating SummaryWriter for log_dir={log_dir} (base_log_dir={self.base_log_dir})')
        return SummaryWriter(log_dir=log_dir)


class S3SummaryCollector:
    def __init__(self,
                 base_log_dir, 
                 s3_endpoint_url, 
                 s3_bucket_name, 
                 key_prefix,
                 nexus_url='http://nexus:8081', 
                 nexus_auth=('bot', 'bot'), 
                 maven_repo='model-registry'):
        self.base_log_dir = base_log_dir
        self.s3_bucket_name = s3_bucket_name
        self.s3 = boto3.client('s3', endpoint_url=s3_endpoint_url)
        self.key_prefix = key_prefix
        self.nexus_url = nexus_url
        self.nexus_auth = nexus_auth
        self.maven_repo = maven_repo

    def process_new_data(self):
        response = self.fault_tolerant_s3('list_objects_v2', Bucket=self.s3_bucket_name, Prefix=self.key_prefix)
        is_any_processing = False
        
        # S3 naturally returns keys sorted alphabetically (batch_000000.json, batch_000001.json, etc.)
        for obj in response.get('Contents', []):
            key = obj['Key']
            Logging.get().debug(f'Processing {key=}')
            
            if self.key_prefix is not None and len(self.key_prefix) > 0:
                unprefixed_key = key[len(self.key_prefix) + 1:] # +1 for slash
                # Logging.get().debug(f'Unprefixed key="{key}"')
            else:
                unprefixed_key = key
            
            key_parts = unprefixed_key.split('/')
                
            if len(key_parts) < 4:
                raise ValueError(f'Key "{key}" has invalid format')

            if 'metrics' in key_parts:
                kind = 'metrics'
            elif 'assets' in key_parts:
                kind = 'assets'
            else:
                raise ValueError(f'Key "{key}" does not contain known kind markers')  
            
            kind_index = key_parts.index(kind)
            log_dir = '/'.join(key_parts[:kind_index])

            match kind:
                case 'metrics': 
                    Logging.get().info(f'New metrics batch: {key=}')
                    obj = self.fault_tolerant_s3('get_object', Bucket=self.s3_bucket_name, Key=key)
                    obj_body = self.fault_tolerant_foo(lambda: obj['Body'].read(), "metrics/obj['Body'].read()") # botocore.exceptions.ResponseStreamingError may be thrown here

                    with io.BytesIO(obj_body) as b: 
                        batch = pickle.load(b)
                        self.process_metrics_batch(log_dir, batch)
                case 'assets': 
                    Logging.get().info(f'New asset: {key=}')
                    obj = self.fault_tolerant_s3('get_object', Bucket=self.s3_bucket_name, Key=key)
                    obj_body = self.fault_tolerant_foo(lambda: obj['Body'].read(), "assets/obj['Body'].read()") # botocore.exceptions.ResponseStreamingError may be thrown here
                    self.process_asset(obj_body, obj['Metadata']) 
                case _: 
                    raise ValueError(f'Key "{key}" has unsupported {kind=}')

            self.fault_tolerant_s3('delete_object', Bucket=self.s3_bucket_name, Key=key)
            Logging.get().debug(f'Processed and deleted {key=}')
            is_any_processing = True

        return is_any_processing

    def process_metrics_batch(self, log_dir, batch):
        is_dirty = False
        
        for batch_item in batch:
            assert isinstance(batch_item, dict), type(batch_item)
            global_step = batch_item.get('global_step', None)
            global_step = lu.when(global_step, lambda: int(global_step), global_step) # all headers are strings
            tag = batch_item.get('tag', None)
            
            match batch_item['method']:
                case 'add_scalar':
                    scalar_value = batch_item['scalar_value']
                    self.get_summary_writer(log_dir).add_scalar(tag, scalar_value, global_step)
                    Logging.get().info(f'add_scalar, {log_dir=}, {tag=}, {scalar_value=}, {global_step=}')
                case 'add_text':
                    text_string = batch_item['text_string']
                    self.get_summary_writer(log_dir).add_text(tag, text_string, global_step)
                    Logging.get().info(f'add_text, {log_dir=}, {tag=}, {text_string[:1000]=}, {global_step=}')
                case 'add_figure':
                    body = batch_item['figure']
                    assert isinstance(body, bytes)
                    with io.BytesIO(body) as b:
                        image_data = pickle.load(b)
                        self.get_summary_writer(log_dir).add_image(tag, image_data, global_step)
                        Logging.get().info(f'add_figure, {log_dir=}, {tag=}, {image_data.shape=}, {global_step=}')
                case 'add_video_file':
                    body = batch_item['video_file']
                    assert isinstance(body, bytes)
                    video_file_len = len(body)
                    # Perform transcoding and upload to tensorboard UI. Very slow. In fact produces animated GIF from video file
                    with io.BytesIO(body) as b:
                        container = av.open(b)
                        fps = float(container.streams.video[0].average_rate)
                        fps = lu.when(fps > 60, 60, fps)
                        frames = []
                        
                        for frame in container.decode(video=0):
                            # Convert to RGB and then to a torch tensor
                            img = frame.to_image().convert('RGB')
                            frames.append(torch.from_numpy(np.array(img)))
        
                        video_tensor = torch.stack(frames) 
                        # 3. Permute to PyTorch format: (N, T, C, H, W)
                        # add Batch dim (N=1) and move Channels (C) to index 2
                        video_tensor = video_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)
                        self.get_summary_writer(log_dir).add_video(tag, video_tensor, global_step, fps)
                        Logging.get().info(f'add_video_file, {log_dir=}, {tag=}, {video_tensor.shape=} ({video_tensor.dtype}) ({video_file_len} bytes), {global_step=}, {fps=}')
                case 'add_file':
                    body = batch_item['file']
                    assert isinstance(body, bytes)
                    file_len = len(body)
                    full_log_dir = os.path.join(self.base_log_dir, log_dir.lstrip('/'))
                    file_name = batch_item['file_name']
                    
                    with open(os.path.join(full_log_dir, file_name), 'wb') as f:
                        f.write(body)
    
                    Logging.get().info(f'add_file, {log_dir=}, {file_name=} ({file_len} bytes)')
                case 'add_hparams':
                    hparam_dict = batch_item['hparam_dict']
                    metric_dict = batch_item['metric_dict']
                    run_name = batch_item['run_name']
                    self.get_summary_writer(log_dir).add_hparams(hparam_dict, metric_dict, run_name=run_name)
                    Logging.get().info(f'add_hparams, {log_dir=}, {hparam_dict=}, {metric_dict=}, {run_name=}')
                case _:
                    raise ValueError(f'Unknown method="{batch_item['method']}"')

            is_dirty = True

        if is_dirty:
            self.get_summary_writer(log_dir).flush()

    def process_asset(self, asset, metadata):
        artifact_registry = ArtifactRegistry(
            metadata['maven_group_id'], 
            nexus_url=self.nexus_url, 
            download_nexus_url=self.nexus_url, 
            nexus_auth=self.nexus_auth, 
            maven_repo=self.maven_repo
        )
        artifact_registry.attach_asset(
            comp_name=metadata['comp_name'],
            comp_version=metadata['comp_version'],
            asset=io.BytesIO(asset),
            asset_classifier=metadata.get('asset_classifier', None),
            asset_ext=metadata['asset_ext'],
            replace=True,
        )
    
    @lru_cache(maxsize=100)
    def get_summary_writer(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(self.base_log_dir, log_dir.lstrip('/'))
        Logging.get().info(f'Creating SummaryWriter for log_dir={log_dir} (base_log_dir={self.base_log_dir})')
        return SummaryWriter(log_dir=log_dir)

    def fault_tolerant_s3(self, method_name, *args, **kwargs):
        retries_count = 5
        method = getattr(self.s3, method_name)

        for retry_number in range(retries_count):
            try:
                return method(*args, **kwargs)
            except botocore.exceptions.BotoCoreError as e:
                retry_interval = 2 ** retry_number
                Logging.get().error(f'Failed to self.s3.{method_name}: {str(e)}. Retrying in {retry_interval} seconds')
                time.sleep(retry_interval)

        raise Exception(f'Max number {retries_count} of retries for self.s3.{method_name} reached, giving up')

    def fault_tolerant_foo(self, foo, foo_desc):
        retries_count = 5

        for retry_number in range(retries_count):
            try:
                return foo()
            except botocore.exceptions.BotoCoreError as e:
                retry_interval = 2 ** retry_number
                Logging.get().error(f'Failed to {foo_desc}: {str(e)}. Retrying in {retry_interval} seconds')
                time.sleep(retry_interval)

        raise Exception(f'Max number {retries_count} of retries for {foo_desc} reached, giving up')
        
if __name__ == "__main__":
    import argparse
    
    LOG = Logging.get()
    LOG.app_name = 'metrics_collector'
    LOG.enable('syslog', False)
    LOG.enable('stdout', False)
    LOG.enable('verbose_stdout', True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_log_dir', type=str, default='/logdir/focus')

    rabbitmq_group = parser.add_argument_group('RabbitMQ Options')
    rabbitmq_group.add_argument('--rmq_connection_url', type=str, default='')

    s3_group = parser.add_argument_group('S3 Options')
    s3_group.add_argument('--s3_endpoint_url', type=str, default='')
    s3_group.add_argument('--s3_bucket_name', type=str, default='')
    s3_group.add_argument('--key_prefix', type=str, default=S3_DEFAULT_KEY_PREFIX)
    s3_group.add_argument('--poll_interval', type=int, default=10)

    args = parser.parse_args()

    has_rmq = bool(args.rmq_connection_url)
    has_s3 = bool(args.s3_endpoint_url or args.s3_bucket_name)

    if has_rmq and has_s3:
        parser.error('Cannot specify both RabbitMQ and S3 configurations')

    if not has_rmq and not has_s3:
        parser.error('You must specify either RabbitMQ configuration or S3 configuration')

    if has_s3 and not (args.s3_endpoint_url and args.s3_bucket_name):
        parser.error('When using S3, both --s3_endpoint_url and --s3_bucket_name are required')

    if has_rmq:
        LOG.info(f'Collecting metrics from RabbitMQ: {args}')
        collector = RmqSummaryCollector(
            args.base_log_dir, 
            args.rmq_connection_url
        )
        collector.run()
    elif has_s3:
        import botocore.exceptions
        LOG.info(f'Collecting metrics and assets from S3: {args}')
        collector = S3SummaryCollector(
            args.base_log_dir, 
            s3_endpoint_url=args.s3_endpoint_url, 
            s3_bucket_name=args.s3_bucket_name, 
            key_prefix=args.key_prefix,
        )

        while True:
            if not collector.process_new_data():
                Logging.get().info(f'Did not process any data, going to sleep for {args.poll_interval} seconds')
                
            time.sleep(args.poll_interval)
