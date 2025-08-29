import sys
import os
import re
import time
import pathlib
import inspect
import keyword
from functools import reduce
import operator
from collections import OrderedDict, Counter
from typing import Iterable
from contextlib import contextmanager
import logging
from logging.handlers import RotatingFileHandler

from colorama import init

init(autoreset=True, convert=True)        # enable Windows console colours

from domiknows.config import config

def extract_args(*args, **kwargs):
    if '_stack_back_level_' in kwargs and kwargs['_stack_back_level_']:
        level = kwargs['_stack_back_level_']
    else:
        level = 0
    frame = inspect.currentframe()
    for _ in range(level + 1):
        frame = frame.f_back
    # print(frame.f_code.co_names)
    names = frame.f_code.co_names[1:]  # skip the first one, the callee name
    names = [name for name in names if not keyword.iskeyword(
        name) and not name in dir(__builtins__)]  # skip python keywords and buildins
    # print(names)
    if not all(name in frame.f_locals for name in names):
        raise TypeError(('Please do not use any expression, but direct variable names,' +
                         ' in caller on Line {} in File {}.')
                        .format(frame.f_lineno, frame.f_code.co_filename))
    return OrderedDict((name, frame.f_locals[name]) for name in names)

def log(*args, **kwargs):
    args = extract_args(_stack_back_level_=1)
    for k, v in args.items():
        v = '{}'.format(v)
        if '\n' in v:
            print('{}:\n{}'.format(k, v))
        else:
            print('{}: {}'.format(k, v))
     
def close_file_handlers(filename):
    """
    Close all file handlers that are using the specified file.
    This prevents permission errors when trying to move/rename log files.
    """
    # Normalize the path for comparison
    target_path = os.path.abspath(filename)
    
    # Get all existing loggers
    loggers = [logging.getLogger()] + [logging.getLogger(name) 
                                      for name in logging.root.manager.loggerDict]
    
    handlers_to_close = []
    
    for logger in loggers:
        if hasattr(logger, 'handlers'):
            for handler in logger.handlers[:]:  # Create a copy of the list
                if hasattr(handler, 'baseFilename'):
                    # Check if this handler uses the target file
                    handler_path = os.path.abspath(handler.baseFilename)
                    if handler_path == target_path:
                        handlers_to_close.append((logger, handler))
    
    # Close and remove the handlers
    for logger, handler in handlers_to_close:
        try:
            handler.close()
            logger.removeHandler(handler)
        except Exception:
            # Continue even if we can't close a handler
            pass
        
def move_existing_logfile_with_timestamp(logFilename, logBackupCount):
    if os.path.exists(logFilename):
        # Check if file is empty - don't move empty files
        if os.path.getsize(logFilename) == 0:
            try:
                os.remove(logFilename)
                print(f"Removed empty log file: {logFilename}")
            except OSError as e:
                print(f"Warning: Could not remove empty log file {logFilename}: {e}")
            return
        # Close any existing handlers for this file first
        close_file_handlers(logFilename)
        
        # Add retry mechanism for file operations
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                # Read first line for timestamp
                try:
                    with open(logFilename, "r", encoding="utf-8", errors="ignore") as f:
                        first_line = f.readline().strip()
                except (IOError, OSError):
                    # If can't read file, use current timestamp
                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                else:
                    # Try to extract timestamp from first line
                    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)", first_line)
                    if match:
                        date_time = match.group(1).replace(":", "-").replace(" ", "_")
                        milliseconds = match.group(2)
                        timestamp = f"{date_time}-{milliseconds}"
                    else:
                        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                
                # Compose new filename in the "previous" subdirectory
                log_dir = os.path.dirname(logFilename) or "."
                previous_dir = os.path.join(log_dir, "previous")
                
                # Create "previous" subdirectory if it doesn't exist
                pathlib.Path(previous_dir).mkdir(parents=True, exist_ok=True)
                
                base = os.path.splitext(os.path.basename(logFilename))[0]
                ext = os.path.splitext(logFilename)[1]
                new_name = os.path.join(previous_dir, f"{base}_{timestamp}{ext}")
                
                # Handle case where target file already exists
                counter = 1
                original_new_name = new_name
                while os.path.exists(new_name):
                    name_without_ext = os.path.splitext(original_new_name)[0]
                    new_name = f"{name_without_ext}_{counter}{ext}"
                    counter += 1
                
                # Try to rename the file
                os.rename(logFilename, new_name)
                break  # Success, exit retry loop
                
            except (OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # If all retries failed, log the error but don't crash
                    print(f"Warning: Could not move log file {logFilename}: {e}")
                    return

        # Remove oldest files if exceeding backup count (now in previous subdirectory)
        try:
            all_files = os.listdir(previous_dir)
            rotated = []
            for f in all_files:
                if f.startswith(base + "_") and f.endswith(ext):
                    full_path = os.path.join(previous_dir, f)
                    if os.path.isfile(full_path):  # Ensure it's a file, not directory
                        rotated.append(f)
            
            # Sort by modification time (oldest first)
            rotated.sort(key=lambda x: os.path.getmtime(os.path.join(previous_dir, x)))
            
            # Remove oldest files if exceeding backup count
            while len(rotated) > logBackupCount:
                oldest_file = os.path.join(previous_dir, rotated.pop(0))
                try:
                    os.remove(oldest_file)
                except OSError:
                    # Continue if file can't be removed
                    pass
                    
        except OSError:
            # If directory listing fails, skip cleanup
            pass
                
# Global variable to track if error/warning logger has been initialized
_error_warning_logger_initialized = False

def setup_error_warning_logger(log_dir='logs'):
    """
    Setup a global error/warning logger that captures WARNING and ERROR level logs
    from all loggers in the application.
    
    Args:
        log_dir (str): Directory where the error/warning log file will be created
        
    Returns:
        logging.Logger: The error/warning logger instance
    """
    global _error_warning_logger_initialized
    
    if _error_warning_logger_initialized:
        return logging.getLogger('errorWarning')
    
    # Create directory
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_path = os.path.join(log_dir, 'errorWarning.log')
    
    # Move existing log file with timestamp before creating new handler
    move_existing_logfile_with_timestamp(log_path, 10)  # Keep 10 backup files
    
    # Create error/warning logger
    error_logger = logging.getLogger('errorWarning')
    error_logger.handlers.clear()
    error_logger.setLevel(logging.WARNING)  # Capture WARNING and above (ERROR, CRITICAL)
    
    # Create file handler
    handler = RotatingFileHandler(
        log_path,
        mode='a',
        maxBytes=5*1024*1024*1024,  # 5GB
        backupCount=4,
        encoding='utf-8',
        delay=False
    )
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:%(funcName)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    error_logger.addHandler(handler)
    error_logger.propagate = False
    
    # Create a custom handler that will be added to the root logger
    # to capture all WARNING and ERROR messages from any logger
    class ErrorWarningHandler(logging.Handler):
        def __init__(self, target_logger):
            super().__init__()
            self.target_logger = target_logger
            self.setLevel(logging.WARNING)
            
        def emit(self, record):
            # Forward the record to the error/warning logger
            self.target_logger.handle(record)
    
    # Add the custom handler to the root logger to capture all warnings/errors
    root_logger = logging.getLogger()
    error_warning_handler = ErrorWarningHandler(error_logger)
    root_logger.addHandler(error_warning_handler)
    
    _error_warning_logger_initialized = True
    
    print("Error/Warning log file is in: %s" % handler.baseFilename)
    
    return error_logger

def setup_logger(config=None, default_filename='app.log'):
    """
    Setup a logger with file rotation and timestamp-based backup.
    
    Args:
        config (dict): Configuration dictionary with logging parameters
        default_filename (str): Default log filename if not specified in config
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Initialize error/warning logger on first call
    if not _error_warning_logger_initialized:
        log_dir = 'logs'
        if config and isinstance(config, dict):
            log_dir = config.get('log_dir', log_dir)
        setup_error_warning_logger(log_dir)
    
    # Default values
    logName = __name__
    logLevel = logging.CRITICAL
    logFilename = default_filename
    logFilesize = 5*1024*1024*1024  # 5GB
    logBackupCount = 4
    logFileMode = 'a'
    logDir = 'logs'
    timestampBackupCount = 10
    
    # Override with config values if provided
    if config and isinstance(config, dict):
        logName = config.get('log_name', logName)
        logLevel = config.get('log_level', logLevel)
        logFilename = config.get('log_filename', logFilename)
        logFilesize = config.get('log_filesize', logFilesize)
        logBackupCount = config.get('log_backupCount', logBackupCount)
        logFileMode = config.get('log_fileMode', logFileMode)
        logDir = config.get('log_dir', logDir)
        timestampBackupCount = config.get('timestamp_backup_count', timestampBackupCount)
    
    # if logFilename is missing extension .log add it
    if not logFilename.endswith('.log'):
        logFilename += '.log'

    # Handle case where logFilename already contains a path
    if os.path.dirname(logFilename):
        # logFilename already contains path information
        log_path = logFilename
        log_dir = os.path.dirname(logFilename)
    else:
        # logFilename is just a filename, add to logDir
        log_path = os.path.join(logDir, logFilename)
        log_dir = logDir
    
    # Create directory
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Move existing log file with timestamp before creating new handler
    move_existing_logfile_with_timestamp(log_path, timestampBackupCount)
    
    # Create logger
    logger = logging.getLogger(logName)
    logger.handlers.clear()  # Clear existing handlers
    logger.setLevel(logLevel)
    
    # Create file handler
    handler = RotatingFileHandler(
        log_path, 
        mode=logFileMode, 
        maxBytes=logFilesize, 
        backupCount=logBackupCount, 
        encoding='utf-8', 
        delay=False
    )
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:%(funcName)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    logger.propagate = False
    
    print("Log file for %s is in: %s" % (logName, handler.baseFilename))
    
    return logger

noUseTimeLog = False
myLoggerTime = None
def getRegrTimer_logger(_config=None):
    global noUseTimeLog
    global myLoggerTime
    
    if myLoggerTime:
        if noUseTimeLog:
            myLoggerTime.addFilter(lambda record: False)
        return myLoggerTime
    
    myLoggerTime = setup_logger(_config, 'regrTimer.log')
    
    if noUseTimeLog:
        myLoggerTime.addFilter(lambda record: False)
    else:
        myLoggerTime.info('--- Starting new run ---')
    
    return myLoggerTime

productionMode = False
reuseModel = False
def setProductionLogMode(no_UseTimeLog = False, reuse_model=True):
    global productionMode
    global reuseModel
    global noUseTimeLog
    global myLoggerTime
    productionMode = True
    reuseModel = reuse_model
    ilpOntSolverLog = logging.getLogger("ilpOntSolver")
    ilpOntSolverLog.addFilter(lambda record: False)
    dataNodeLog = logging.getLogger("dataNode")
    dataNodeLog.addFilter(lambda record: False)
    dataNodeBuilderLog = logging.getLogger("dataNodeBuilder")
    dataNodeBuilderLog.addFilter(lambda record: False)
    
    noUseTimeLog = no_UseTimeLog
    if noUseTimeLog:
        if myLoggerTime != None:
            myLoggerTime.addFilter(lambda record: False)
    
def getProductionModeStatus():
    return productionMode

def getReuseModel():
    return reuseModel

dnSkeletonMode = False
dnSkeletonModeFull = False
def setDnSkeletonMode(dnSkeleton, full=False):
    if not isinstance(dnSkeleton, bool):
        dnSkeleton = False
        return
    
    global dnSkeletonMode
    dnSkeletonMode = dnSkeleton
    if full:
        global dnSkeletonModeFull
        dnSkeletonModeFull = True
    
def getDnSkeletonMode():
    global dnSkeletonMode
    return dnSkeletonMode

def getDnSkeletonModeFull():
    global dnSkeletonModeFull
    return dnSkeletonModeFull
    
def printablesize(ni):
    if hasattr(ni, 'shape'):
        return 'tensor' + str(tuple(ni.shape)) + ''
    elif isinstance(ni, Iterable):
        if len(ni) > 0:
            return 'iterable(' + str(len(ni)) + ')' + '[' + printablesize(ni[0]) + ']'
        else:
            return 'iterable(' + str(len(ni)) + ')[]'
    else:
        return str(type(ni))

def entuple(args):
    if isinstance(args, tuple):
        return args
    return (args,)


def detuple(*args):
    if isinstance(args, tuple) and len(args) == 1:
        return args[0]
    return args


def enum(inst, cls=None, offset=0):
    if isinstance(inst, cls):
        enum = {offset: inst}.items()
    elif isinstance(inst, OrderedDict):
        enum = inst.items()
    elif isinstance(inst, dict):
        enum = inst.items()
        # if inst:  # if dict not empty
        #     # NB: stacklevel
        #     #     1 - utils.enum()
        #     #     2 - concept.Concept.relation_type.<local>.update.<local>.create()
        #     #     3 - concept.Concept.__getattr__.<local>.handle()
        #     #     4 - __main__()
        #     warnings.warn('Please use OrderedDict rather than dict to prevent unpredictable order of arguments.' +
        #                   'For this instance, {} is used.'
        #                   .format(OrderedDict((k, v.name) for k, v in inst.items())),
        #                   stacklevel=4)
    elif isinstance(inst, Iterable):
        enum = enumerate(inst, offset)
    else:
        raise TypeError('Unsupported type of instance ({}). Use cls specified type, OrderedDict or other Iterable.'
                        .format(type(inst)))

    return enum


@contextmanager
def hide_class(inst, clsinfo, sub=True):  # clsinfo is a type of a tuple of types
    if isinstance(inst, clsinfo):
        if isinstance(clsinfo, type):
            clsinfo = (clsinfo,)

        from six.moves import builtins
        isinstance_orig = builtins.isinstance

        def _isinstance(inst_, clsinfo_):  # clsinfo_ is a type of a tuple of types
            if inst_ is inst:
                if isinstance_orig(clsinfo_, type):
                    clsinfo_ = (clsinfo_,)
                clsinfo_ = [cls_
                            for cls_ in clsinfo_
                            if not (
                                sub and issubclass(cls_, clsinfo)
                            ) and not (
                                not sub and cls_ in clsinfo
                            )]
                clsinfo_ = tuple(clsinfo_)
                # NB: isinstance(inst, ()) == False
            return isinstance_orig(inst_, clsinfo_)

        builtins.isinstance = _isinstance

        try:
            yield inst
        finally:
            builtins.isinstance = isinstance_orig
    else:
        yield inst


@contextmanager
# clsinfo is a type of a tuple of types
def hide_inheritance(cls, clsinfo, sub=True, hidesub=True):
    if issubclass(cls, clsinfo):
        if isinstance(clsinfo, type):
            clsinfo = (clsinfo,)

        from six.moves import builtins
        isinstance_orig = builtins.isinstance
        issubclass_orig = builtins.issubclass

        def _isinstance(inst, clsinfo_):
            if (hidesub and isinstance_orig(inst, cls)
                ) or (
                not hidesub and type(inst) is cls
            ):
                # not sure would this hurt somewhere?
                # the following issubclass is dynamic!
                return any(issubclass(cls_, clsinfo_) for cls_ in {type(inst), inst.__class__})
            return isinstance_orig(inst, clsinfo_)

        def _issubclass(cls_, clsinfo_):  # clsinfo_ is a type of a tuple of types
            if (hidesub and issubclass_orig(cls_, cls)
                ) or (
                not hidesub and cls_ is cls
            ):
                if isinstance_orig(clsinfo_, type):
                    clsinfo_ = (clsinfo_,)
                clsinfo_ = [cls__
                            for cls__ in clsinfo_
                            if not (
                                sub and issubclass_orig(cls__, clsinfo)
                            ) and not (
                                not sub and cls__ in clsinfo
                            )]
                clsinfo_ = tuple(clsinfo_)
            return issubclass_orig(cls_, clsinfo_)

        builtins.isinstance = _isinstance
        builtins.issubclass = _issubclass

        try:
            yield
        finally:
            builtins.isinstance = isinstance_orig
            builtins.issubclass = issubclass_orig
    else:
        yield

        from contextlib import contextmanager


def singleton(cls, getter=None, setter=None):
    if getter is None:
        def getter(*args, **kwargs):
            if hasattr(cls, '__singleton__'):
                return cls.__singleton__
            return None
    if setter is None:
        def setter(obj):
            cls.__singleton__ = obj

    __old_new__ = cls.__new__

    def __new__(cls, *args, **kwargs):
        obj = getter(*args, **kwargs)
        if obj is None:
            obj = __old_new__(cls, *args, **kwargs)
            obj.__i_am_the_new_singoton__ = True
            setter(obj)
            return obj
        else:
            return obj

    __old_init__ = cls.__init__

    def __init__(self, *args, **kwargs):
        if hasattr(self, '__i_am_the_new_singoton__') and self.__i_am_the_new_singoton__:
            del self.__i_am_the_new_singoton__
            __old_init__(self, *args, **kwargs)

    cls.__new__ = staticmethod(__new__)
    cls.__init__ = __init__
    return cls


class WrapperMetaClass(type):
    def __call__(cls, inst, *args, **kwargs):
        if not isinstance(inst, tuple(cls.mro())):
            raise TypeError(
                'Only cast from {}, while {} is given.'.format(super(cls), type(inst)))

        inst.__class__ = cls
        # no need to call cls.__new__ because we do not need new instance
        inst.__init__(*args, **kwargs)
        return inst


def optional_arg_decorator(fn, test=None):
    def wrapped_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and (test is None or test(args[0])):
            return fn(args[0])
        else:
            def real_decorator(decoratee):
                return fn(decoratee, *args, **kwargs)
            return real_decorator
    return wrapped_decorator


def optional_arg_decorator_for(test):
    return lambda fn: optional_arg_decorator(fn, test)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def guess_device(data_item):
    import torch
    poll = Counter()
    for value in data_item.values():
        if isinstance(value, dict):
            poll += guess_device(value)
        elif isinstance(value, torch.Tensor):
            poll[value.device] += 1
        else:
            poll[None] += 1
    return poll


def find_base(s, n):
    from scipy.optimize import minimize, minimize_scalar
    # NB: `n` here is the number of terms in this "geometric series", including 0 and last k.
    #     So the "n+1" in the original formula is `n` here.
    length = lambda b: (1 - b ** n) / (1 - b)
    res = minimize_scalar(lambda b : (length(b) - s) ** 2, method='bounded', bounds=(1, (s-1)**(1./n)))
    return res.x


def get_prop_result(prop, data):
    vals = []
    mask = None
    for name, sensor in prop.items():
        sensor(data)
        if hasattr(sensor, 'get_mask'):
            if mask is None:
                mask = sensor.get_mask(data)
            else:
                assert mask == sensor.get_mask(data)
        tensor = data[sensor.fullname]
        vals.append(tensor)
    label = vals[0]  # TODO: from readersensor
    pred = vals[1]  # TODO: from learner
    return label, pred, mask

def isbad(x):
    return (
        x != x or  # nan
        abs(x) == float('inf')  # inf
    )


# consume(it) https://stackoverflow.com/q/50937966
if sys.implementation.name == 'cpython':
    import collections
    def consume(it):
        collections.deque(it, maxlen=0)
else:
    def consume(it):
        for _ in it:
            pass


class Namespace(dict):
    def __init__(self, __dict=None, **kwargs):
        dict_ = __dict or {}
        dict_.update(kwargs)
        for k, v in dict_.items():
            if isinstance(v, dict):
                dict_[k] = Namespace(v)
        super().__init__(dict_)
        self.__dict__ = self

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, ','.join('\'{}\':{}'.format(k,v) for k,v in self.items()))

    def clone(self):
        from copy import copy
        return Namespace(copy(self))

    def deepclone(self):
        from copy import deepcopy
        return Namespace(deepcopy(self))

    def __getattr__(self, key):
        return self.get(key, None)


def caller_source():
    import inspect

    for frame in inspect.getouterframes(inspect.currentframe(), context=1)[2:]:
        if frame.code_context is not None:
            try:
                with open(frame.filename, 'r') as fin:
                    return fin.read()
                break
            except FileNotFoundError as ex:
                ex = type(ex)('{}\n'
                              'Please run from a file base environment, '
                              'rather than something like notebook.'.format(ex))
                raise ex
    else:
        raise RuntimeError('Who is calling?')


def dict_zip(*dicts, fillvalue=None):  # https://codereview.stackexchange.com/a/160584
    all_keys = {k for d in dicts for k in d.keys()}
    return {k: [d.get(k, fillvalue) for d in dicts] for k in all_keys}


def wrap_batch(values, fillvalue=0):
    import torch

    if isinstance(values, (list, tuple)):
        if isinstance(values[0], dict):
            values = dict_zip(*values, fillvalue=fillvalue)
            values = {k: wrap_batch(v, fillvalue=fillvalue) for k, v in values.items()}
        elif isinstance(values[0], torch.Tensor):
            values = torch.stack(values)
    elif isinstance(values, dict):
        values = {k: wrap_batch(v, fillvalue=fillvalue) for k, v in values.items()}
    return values
