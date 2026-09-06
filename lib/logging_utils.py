import os
import sys
import time
import datetime
import logging
import logging.handlers
from dataclasses import dataclass

import lang_utils as lu

def if_verbose(verbosity, verbosity_threshold, func):
    if verbosity < verbosity_threshold:
        return

    func()
    sys.stdout.flush()

class Logging:
    @dataclass
    class Sink:
        logger: object = None
        prepare_message: object = None
        is_enabled: bool = True

    # super simple singleton
    _instance = None
    
    @staticmethod
    def get(use_raw_stdout=False, log_fname=None):
        if Logging._instance is None:
            return Logging(use_raw_stdout, log_fname)

        return Logging._instance

    @staticmethod
    def error(s, with_duration=True, when=True):
        Logging.get()(s, with_duration, when, log_level=logging.ERROR)
    
    @staticmethod
    def warn(s, with_duration=True, when=True):
        Logging.get()(s, with_duration, when, log_level=logging.WARN)
    
    @staticmethod
    def info(s, with_duration=True, when=True):
        Logging.get()(s, with_duration, when, log_level=logging.INFO)

    @staticmethod
    def debug(s, with_duration=True, when=True):
        Logging.get()(s, with_duration, when, log_level=logging.DEBUG)

    @staticmethod
    def trace(s, with_duration=True, when=True):
        Logging.get()(s, with_duration, when, log_level=logging.DEBUG - 1)

    def __init__(self, use_raw_stdout=False, log_fname=None):
        syslog_logger = logging.getLogger('kmslog_syslog')
        
        if not syslog_logger.hasHandlers():
            syslog_handler = logging.handlers.SysLogHandler(address='/dev/log', facility=logging.handlers.SysLogHandler.LOG_LOCAL0)
            syslog_handler.ident = 'kmstag:'
            syslog_logger.addHandler(syslog_handler)

        stdout = lu.when(use_raw_stdout, lambda: open('/dev/stdout', 'w', buffering=1), sys.stdout)
        stdout_logger = logging.getLogger('kmslog_stdout')
        
        if not stdout_logger.hasHandlers():
            stream_handler = logging.StreamHandler(stdout)
            stdout_logger.addHandler(stream_handler)

        verbose_stdout_logger = logging.getLogger('kmslog_verbose_stdout')
        
        if not verbose_stdout_logger.hasHandlers():
            stream_handler = logging.StreamHandler(stdout)
            verbose_stdout_logger.addHandler(stream_handler)

        self.sinks = dict(
            syslog=Logging.Sink(syslog_logger, self.prepare_syslog_message, is_enabled=True),
            stdout=Logging.Sink(stdout_logger, self.prepare_stdout_message, is_enabled=True),
            verbose_stdout=Logging.Sink(verbose_stdout_logger, self.prepare_verbose_stdout_message, is_enabled=False),
        )
        
        if log_fname is not None:
            file_logger = logging.getLogger('kmslog_file')
            
            if not file_logger.hasHandlers():
                file_handler = logging.FileHandler(log_fname, mode='a', encoding='utf-8')
                file_logger.addHandler(file_handler)

            self.sinks['file'] = Logging.Sink(file_logger, self.prepare_file_message, is_enabled=False)

        self.set_log_level('all', logging.DEBUG)

        self.app_name = 'MAIN'
        self.prefix_stanzas = dict()
        self.prefix_stanzas_order = []
        self.prefix = ''
        self.last_log_time = time.time()
        Logging._instance = self

    def __call__(self, s, with_duration=True, when=True, log_level=logging.INFO):
        if not when:
            return
            
        t = time.time()
        
        for sink in self.sinks.values():
            if sink.is_enabled:
                msg = sink.prepare_message(s, t, with_duration)
                sink.logger.log(log_level, msg)

        self.last_log_time = t

    def enable(self, sink_name, is_enabled):
        assert sink_name == 'all' or sink_name in self.sinks

        if sink_name == 'all':
            for sink in self.sinks.values():
                sink.is_enabled = is_enabled
        else:
            sink = self.sinks[sink_name]
            sink.is_enabled = is_enabled

    def set_log_level(self, sink_name, log_level):
        assert sink_name == 'all' or sink_name in self.sinks
        old_log_level = None

        if sink_name == 'all':
            for sink in self.sinks.values():
                old_log_level = sink.logger.level
                sink.logger.setLevel(log_level)
        else:
            sink = self.sinks[sink_name]
            old_log_level = sink.logger.level
            sink.logger.setLevel(log_level)

        return old_log_level
            
    def prepare_syslog_message(self, s, t, with_duration):
        msg = ' ' # without this space following 'PID:'... will be considered as a part of syslogtag by rsyslog, so separate forcibly
        msg += f'PID:{os.getpid():<10} APP:{self.app_name:<15}'
        msg += ' ' + self.prefix
        msg += ' ' if self.prefix else ''
        
        if with_duration:
            duration = max(0, t - self.last_log_time)
            msg += f'{duration:>9.3f} >> '
            
        msg += s
        return msg

    def prepare_stdout_message(self, s, t, with_duration):
        return s

    def prepare_verbose_stdout_message(self, s, t, with_duration):
        msg = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S.%f')
        msg += ' ' + self.prefix
        msg += ' ' if self.prefix else ''
        
        if with_duration:
            duration = max(0, t - self.last_log_time)
            msg += f'{duration:>9.3f} >> '
            
        msg += s
        return msg

    def prepare_file_message(self, s, t, with_duration):
        return self.prepare_verbose_stdout_message(s, t, with_duration)

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
        self.prefix = ''

        if self.prefix_stanzas_order:
            self.prefix = '[' + ','.join(map(lambda s: f'{s}={self.prefix_stanzas[s]}', self.prefix_stanzas_order)) + ']'

    def auto_prefix(self, stanza_name, stanza_value, *args):
        d = {stanza_name: stanza_value}
        
        if args:
            assert len(args) % 2 == 0, f'args count is not even: {len(args)}'
            
            for ind in range(0, len(args), 2):
                d[args[ind]] = args[ind+1] 
            
        return AutoLoggingPrefix(self, d)

    def auto_log_level(self, log_level):
        return AutoLoggingLogLevel(self, log_level)

class AutoLoggingPrefix:
    def __init__(self, logging, d):
        self.logging = logging
        self.d = d

        for k, v in d.items():
            self.logging.push_prefix(k, v)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k in self.d:
            self.logging.pop_prefix(k)
            
        return False

class AutoLoggingLogLevel:
    def __init__(self, logging, level):
        self.logging = logging
        self.old_log_level = self.logging.set_log_level('all', level) 

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logging.set_log_level('all', self.old_log_level) 
        return False
