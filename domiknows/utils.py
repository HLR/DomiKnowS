import shutil
import sys
import os
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

from tqdm import tqdm as tqdm_original
from tqdm.asyncio import tqdm as tqdm_asyncio_original

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
        
        # Create run timestamp for subfolder
        run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Get log directory and create previous/run structure
        log_dir = os.path.dirname(logFilename) or "."
        previous_dir = os.path.join(log_dir, "previous")
        run_dir = os.path.join(previous_dir, f"run_{run_timestamp}")
        
        # Create run subdirectory
        pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
        
        # Add retry mechanism for file operations
        max_retries = 3
        retry_delay = 0.1
        
        # Move all files from log directory to run subfolder
        try:
            all_files = os.listdir(log_dir)
            for file_item in all_files:
                source_path = os.path.join(log_dir, file_item)
                
                # Skip directories and the previous directory itself
                if os.path.isdir(source_path):
                    continue
                
                # Move each file to the run directory
                for attempt in range(max_retries):
                    try:
                        target_path = os.path.join(run_dir, file_item)
                        os.rename(source_path, target_path)
                        break
                        
                    except (OSError, PermissionError) as e:
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                        else:
                            print(f"Warning: Could not move file {source_path}: {e}")
                            break
                            
        except OSError as e:
            print(f"Warning: Could not list directory {log_dir}: {e}")
            return
        
        # Clean up old run directories - keep only 10
        try:
            if os.path.exists(previous_dir):
                all_items = os.listdir(previous_dir)
                run_dirs = []
                
                for item in all_items:
                    item_path = os.path.join(previous_dir, item)
                    if os.path.isdir(item_path) and item.startswith("run_"):
                        run_dirs.append(item)
                
                # Sort by modification time (oldest first)
                run_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(previous_dir, x)))
                
                # Remove oldest run directories if exceeding backup count
                while len(run_dirs) > logBackupCount:
                    oldest_dir = os.path.join(previous_dir, run_dirs.pop(0))
                    try:
                        shutil.rmtree(oldest_dir)
                        print(f"Removed old run directory: {oldest_dir}")
                    except OSError as e:
                        print(f"Warning: Could not remove old run directory {oldest_dir}: {e}")
                        
        except OSError:
            # If directory operations fail, skip cleanup
            pass
                
# Global variables for error/warning logger
_error_warning_logger_initialized = False
_error_warning_logger = None
_error_warning_handler_class = None
_error_warning_log_dir = None


def setup_error_warning_logger(log_dir='logs'):
    """
    Setup a global error/warning logger configuration that will create the actual
    log file only when the first warning/error is detected.
    
    Args:
        log_dir (str): Directory where the error/warning log file will be created
        
    Returns:
        logging.Logger: The error/warning logger instance (initially without file handler)
    """
    global _error_warning_logger_initialized, _error_warning_logger, _error_warning_handler_class, _error_warning_log_dir
    
    if _error_warning_logger_initialized:
        return _error_warning_logger
    
    # Store log directory for later use
    _error_warning_log_dir = log_dir
    
    # Create error/warning logger without file handler initially
    _error_warning_logger = logging.getLogger('errorWarning')
    _error_warning_logger.handlers.clear()
    _error_warning_logger.setLevel(logging.WARNING)  # Capture WARNING and above (ERROR, CRITICAL)
    _error_warning_logger.propagate = False
    
    # Create a custom handler class that will be added to individual loggers
    class ErrorWarningHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.setLevel(logging.WARNING)
            self._file_handler_created = False
            
        def emit(self, record):
            # Forward the record to the error/warning logger
            if record.levelno >= logging.WARNING:
                # Create file handler on first warning/error
                if not self._file_handler_created:
                    self._create_file_handler()
                    self._file_handler_created = True
                
                _error_warning_logger.handle(record)
        
        def _create_file_handler(self):
            """Create the actual log file and handler when first warning/error occurs"""
            global _error_warning_log_dir, _error_warning_logger
            
            # Create directory
            pathlib.Path(_error_warning_log_dir).mkdir(parents=True, exist_ok=True)
            
            log_path = os.path.join(_error_warning_log_dir, 'errorWarning.log')
            
            # Move existing log file with timestamp before creating new handler
            move_existing_logfile_with_timestamp(log_path, 10)  # Keep 10 backup files
            
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
            _error_warning_logger.addHandler(handler)
            
            print("Error/Warning log file created: %s" % handler.baseFilename)
    
    _error_warning_handler_class = ErrorWarningHandler
    _error_warning_logger_initialized = True
    
    return _error_warning_logger

def add_error_warning_handler_to_logger(logger):
    """
    Add error/warning handler to an existing logger to capture its WARNING and ERROR messages.
    
    Args:
        logger: The logger instance to add the handler to
    """
    global _error_warning_handler_class, _error_warning_logger_initialized
    
    if not _error_warning_logger_initialized or not _error_warning_handler_class:
        return
    
    # Check if this logger already has an ErrorWarningHandler
    for handler in logger.handlers:
        if isinstance(handler, _error_warning_handler_class):
            return  # Already has the handler
    
    # Add the error/warning handler
    error_warning_handler = _error_warning_handler_class()
    logger.addHandler(error_warning_handler)

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
    
    # Add error/warning handler to capture WARNING and ERROR messages
    add_error_warning_handler_to_logger(logger)
    
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


class SafeTqdm(tqdm_original):
    """Safe wrapper for tqdm that handles missing attributes and nested bars"""
    
    def __init__(self, *args, **kwargs):
        # Disable nested progress bars by default for better compatibility
        # Set position explicitly to avoid conflicts
        if 'position' not in kwargs and hasattr(tqdm_original, '_instances'):
            kwargs['position'] = len(tqdm_original._instances)
        
        # Add leave=False for inner progress bars to prevent conflicts
        if kwargs.get('position', 0) > 0 and 'leave' not in kwargs:
            kwargs['leave'] = False
            
        try:
            super().__init__(*args, **kwargs)
        except (OSError, ValueError) as e:
            # If terminal positioning fails, fall back to simpler display
            kwargs['position'] = None
            kwargs['dynamic_ncols'] = False
            super().__init__(*args, **kwargs)
    
    def __del__(self):
        try:
            # Ensure last_print_t exists before closing
            if not hasattr(self, 'last_print_t'):
                self.last_print_t = getattr(self, 'start_t', 0)
            super().__del__()
        except (AttributeError, TypeError, OSError):
            # Silently ignore any errors during cleanup
            pass

class SafeTqdmAsync(tqdm_asyncio_original):
    """Safe wrapper for tqdm_asyncio that handles missing attributes"""
    
    def __init__(self, *args, **kwargs):
        if 'position' not in kwargs and hasattr(tqdm_asyncio_original, '_instances'):
            kwargs['position'] = len(tqdm_asyncio_original._instances)
        
        if kwargs.get('position', 0) > 0 and 'leave' not in kwargs:
            kwargs['leave'] = False
            
        try:
            super().__init__(*args, **kwargs)
        except (OSError, ValueError):
            kwargs['position'] = None
            kwargs['dynamic_ncols'] = False
            super().__init__(*args, **kwargs)
    
    def __del__(self):
        try:
            if not hasattr(self, 'last_print_t'):
                self.last_print_t = getattr(self, 'start_t', 0)
            super().__del__()
        except (AttributeError, TypeError, OSError):
            pass

def safe_tqdm(*args, **kwargs):
    """
    Safe tqdm wrapper that handles:
    - Missing attributes during cleanup
    - Nested progress bar conflicts
    - Terminal positioning issues
    
    Usage:
        from utils import safe_tqdm as tqdm
        
        for item in tqdm(items):
            # Nested bars work automatically
            for subitem in tqdm(subitems):
                process(subitem)
    """
    if kwargs.get('asyncio', False):
        kwargs.pop('asyncio')
        return SafeTqdmAsync(*args, **kwargs)
    return SafeTqdm(*args, **kwargs)

# For backwards compatibility
tqdm = safe_tqdm