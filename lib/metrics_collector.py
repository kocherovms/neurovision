import os
import sys
import pickle
import io
import json
from functools import lru_cache

import pika
import pika.exceptions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as plt_backend_agg

import torch

import av # video support

import lang_utils as lu

RMQ_EVENTS_EXCHANGE_NAME = 'events'
RMQ_EVENTS_QUEUE_NAME = 'events'
RMQ_DEFAULT_CONNECTION_URL = 'amqp://guest:guest@rabbitmq:5672/%2F'

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

    def add_scalar(self, tag, scalar_value, global_step, is_batched=False):
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
            image = self._figure_to_image(figure, close)
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
        if not self.scalar_batch:
            properties = self._create_message_properties('flush')
            self._robust_publish(body='', properties=properties)
        else:
            scalar_values = []
            
            for i, (tag, scalar_value, global_step) in enumerate(scalar_batch):
                properties.headers[f'tag_{i}'] = tag
                properties.headers[f'global_step_{i}'] = global_step
                
                if isinstance(scalar_value, torch.Tensor) or isinstance(scalar_value, np.ndarray):
                    scalar_value = scalar_value.item()

                scalar_values.append(scalar_value)
                
            with io.BytesIO() as b:
                pickle.dump(scalar_values, b)
                self._robust_publish(body=b.getvalue(), properties=properties)

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

    def _figure_to_image(self, figure, close):
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        
        if close:
            plt.close(figure)
        
        return image_chw

    def _create_message_properties(self, method):
        return pika.spec.BasicProperties(
            headers={
                'log_dir': self.log_dir,
                'method': method,
            },
            delivery_mode=pika.DeliveryMode.Persistent,
        )

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
        # print(f'on_message: method={method}, properties={properties}, len(body)={len(body)}')
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
                    print(f'add_scalar, {log_dir=}, {tag=}, {scalar_value=}, {global_step=}')
            case 'add_text':
                text_string = body.decode()
                self.get_summary_writer(log_dir).add_text(tag, text_string, global_step)
                print(f'add_text, {log_dir=}, {tag=}, {text_string[:1000]=}, {global_step=}')
            case 'add_figure':
                with io.BytesIO(body) as b:
                    image_data = pickle.load(b)
                    self.get_summary_writer(log_dir).add_image(tag, image_data, global_step)
                    print(f'add_figure, {log_dir=}, {tag=}, {image_data.shape=}, {global_step=}')
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
                    print(f'add_video_file, {log_dir=}, {tag=}, {video_tensor.shape=} ({video_tensor.dtype}) ({video_file_len} bytes), {global_step=}, {fps=}')
            case 'add_file':
                file_len = len(body)
                full_log_dir = os.path.join(self.base_log_dir, log_dir.lstrip('/'))
                file_name = properties.headers['file_name']
                
                with open(os.path.join(full_log_dir, file_name), 'wb') as f:
                    f.write(body)

                print(f'add_file, {log_dir=}, {file_name=} ({file_len} bytes)')
            case 'add_hparams':
                with io.BytesIO(body) as b:
                    message = json.load(b)
                    hparam_dict = message['hparam_dict']
                    metric_dict = message['metric_dict']
                    self.get_summary_writer(log_dir).add_hparams(hparam_dict, metric_dict, run_name=run_name)
                    print(f'add_hparams, {log_dir=}, {hparam_dict=}, {metric_dict=}, {run_name=}')
            case 'flush':
                sw = self.get_summary_writer(log_dir)

                if len(body) > 0:
                    with io.BytesIO(body) as b:
                        scalar_values = pickle.load(b)

                        for i, scalar_value in enumerate(scalar_values):
                            tag = properties.headers[f'tag_{i}']
                            global_step = int(properties.headers[f'global_step_{i}'])
                            sw.add_scalar(tag, scalar_value, global_step)
                            print(f'add_scalar (on flush), {log_dir=}, {tag=}, {scalar_value=}, {global_step=}')
                
                sw.flush()
                print('flush')
            case _:
                assert False, f'Unknown method="{logic_method}"'
        
        ch.basic_ack(delivery_tag=method.delivery_tag)

    @lru_cache(maxsize=100)
    def get_summary_writer(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(self.base_log_dir, log_dir.lstrip('/'))
        print(f'Creating SummaryWriter for log_dir={log_dir} (base_log_dir={self.base_log_dir})')
        return SummaryWriter(log_dir=log_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_log_dir', type=str, default='/logdir')
    parser.add_argument('--rmq_connection_url', type=str, default='amqp://guest:guest@rabbitmq:5672/%2F')
    args = parser.parse_args()
    print(f'Running collector with args={args}')
    collector = RmqSummaryCollector(args.base_log_dir, args.rmq_connection_url)
    collector.run()
