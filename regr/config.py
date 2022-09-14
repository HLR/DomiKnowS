import logging

config = {
    # Logging configuration for regrTimer
    'ifLog': True,
    'log_name' : 'regrTimer', 
    'log_level' : logging.INFO,
    'log_filename' : 'logs/regrTimer',
    'log_filesize' : 5*1024*1024*1024,
    'log_backupCount' : 5,
    'log_fileMode' : 'a'
}