import os, io
import configparser
import numpy as np
import IPython
from PIL import Image, ImageDraw
import logging
import logging.handlers

class Config:
    DEFAULTS = dict(
        hdc_n=10000,
        sample_size=128,
        cortical_columns_count=1,
        cortical_column_receptive_field_size=128, # for 'random' this is the number of sensor groups, for 'radial' - radius
        cortical_columns_layout='random',
        encoding_type='normal',
        sensor_groups_count=256,
        sensors_count=256,
        retina_layout='grid',
        dataset_source='dataset_source',
        dataset_path='dataset',
        dataset_sample_count=10000,
        dataset_train_samples_count=10000,
        dataset_test_samples_count=2000,
        dataset_metadata_file_name='_metadata.json',
        output_path='out',
        db_file_name_prefix='',
        hdv_db_file_name='hdv.db',
        train_db_file_name='train.db',
        test_db_file_name='test.db',
    )
    
    def __init__(self, section_name='DEFAULT'):
        super()
        self.section_name = section_name
        self.reload()

    def reload(self):
        config = configparser.ConfigParser(defaults=type(self).DEFAULTS)

        if os.path.exists('config.txt'):
            config.read('config.txt')

        sections = [config['DEFAULT']]

        if self.section_name != 'DEFAULT':
            sections.append(config[self.section_name])

        for section in sections:
            for k in section:
                typ = type(type(self).DEFAULTS[k]) # enforce proper type (e.g. kernel_size must by int, not string)
                setattr(self, k, typ(section[k]))

class Logging(object):
    def __init__(self):
        self.logger = logging.getLogger('kmslog')
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.hasHandlers():
            syslogHandler = logging.handlers.SysLogHandler(address='/dev/log', facility=logging.handlers.SysLogHandler.LOG_LOCAL0)
            syslogHandler.ident = 'kmstag:'
            self.logger.addHandler(syslogHandler)

        self.prefix_stanzas = dict()
        self.prefix_stanzas_order = []
        self.prefix = ''
        
    def __call__(self, s):
        self.logger.debug(f'{self.prefix} {s}')

    def push_prefix(self, stanza_name, stanza_value):
        if stanza_name in self.prefix_stanzas:
            self.prefix_stanzas[stanza_name] = stanza_value
        else:
            self.prefix_stanzas[stanza_name] = stanza_value
            self.prefix_stanzas_order.append(stanza_name)

        self.update_prefix()

    def pop_prefix(self, stanza_name):
        if stanza_name in self.prefix_stanzas:
            del self.prefix_stanzas[stanza_name]

        try:
            self.prefix_stanzas_order.remove(stanza_name)
        except ValueError:
            pass

        self.update_prefix()

    def update_prefix(self):
        self.prefix = '[' + ','.join(map(lambda s: f'{s}={self.prefix_stanzas[s]}', self.prefix_stanzas_order)) + ']'    

# from https://gist.github.com/parente/691d150c934b89ce744b5d54103d7f1e
def _html_src_from_raw_image_data(data):
    """Base64 encodes image bytes for inclusion in an HTML img element"""
    img_obj = IPython.display.Image(data=data)
    for bundle in img_obj._repr_mimebundle_():
        for mimetype, b64value in bundle.items():
            if mimetype.startswith('image/'):
                return f'data:{mimetype};base64,{b64value}'

def display_images(images, captions=None, row_height='auto'):
    figures = []
    
    for image_index, image in enumerate(images):
        if isinstance(image, bytes) or isinstance(image, Image.Image):
            if isinstance(image, bytes):
                bts = image
            else:
                b = io.BytesIO()
                image.save(b, format='PNG')
                bts = b.getvalue()
            
            src = _html_src_from_raw_image_data(bts)
        else:
            src = image
            #caption = f'<figcaption style="font-size: 0.6em">{image}</figcaption>'

        caption = ''
        
        if not captions is None:
            if isinstance(captions, dict):
                caption = captions.get(id(image), '')
            else:
                assert len(captions) == len(images)
                caption = captions[image_index]

            if caption:
                caption = f'<figcaption style="font-size: 0.6em">{caption}</figcaption>'
        
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="{src}" style="height: {row_height}">
              {caption}
            </figure>
        ''')
    return IPython.display.HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')

def display_images_grid(images, col_count, col_width=None, captions=None):
    figures = []
    
    for image_index, image in enumerate(images):
        assert isinstance(image, bytes) or isinstance(image, Image.Image)

        if isinstance(image, bytes):
            bts = image
        else:
            b = io.BytesIO()
            image.save(b, format='PNG')
            bts = b.getvalue()
        
        src = _html_src_from_raw_image_data(bts)

        caption = ''

        if not captions is None:
            if isinstance(captions, dict):
                caption = str(captions.get(id(image), ''))
            else:
                assert len(captions) == len(images), (len(captions), len(images))
                caption = str(captions[image_index])

            if caption:
                caption = f'<figcaption style="font-size: 0.6em">{caption}</figcaption>'
        
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="{src}" style="height: auto">
              {caption}
            </figure>
        ''')

    if not col_width:
        if len(images) > 0 and isinstance(images[0], Image.Image):
            col_width = images[0].width

    if not col_width: 
        col_width='auto'
    else:
        col_width = f'{col_width}px'
        
    return IPython.display.HTML(data=f'''<div style="
        display: grid; 
        grid-template-columns: repeat({col_count}, {col_width});
        column-gap: 1px;
        row-gap: 1px;">
        {''.join(figures)}
    </div>''')

def matrix_to_image(m):
    m = m.ravel()
    sz = int(np.sqrt(m.shape[0]))
    assert sz * sz == m.shape[0]
    return Image.frombytes('L', size=(sz, sz), data=m.astype('b'))

def lay_grid(image, step=16):
    draw = ImageDraw.Draw(image)

    for c in range(step - 1, image.height, step):
        draw.line([0, c, image.width, c], fill='gray')
        draw.line([c, 0, c, image.height], fill='gray')

    return image