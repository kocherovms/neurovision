import os

from logging_utils import *

class CommandListener:
    def __init__(self, fifo_fname, buffer_size=16):
        self.fifo_fname = fifo_fname
        self.buffer_size = buffer_size

    def get_command(self):
        if self.fifo_fname is None or not os.path.exists(self.fifo_fname):
            return None
            
        try:
            fd = os.open(self.fifo_fname, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as e:
            # ENXIO means no process has the FIFO open for writing yet
            if e.errno != errno.ENXIO:
                Logging.get().error(f'Failed to open "{self.fifo_fname}": {str(e)}')
                
            return None
            
        try:
            data = os.read(fd, self.buffer_size)
            return data.decode().strip()
        finally:
            os.close(fd)


