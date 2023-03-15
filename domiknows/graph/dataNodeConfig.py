import logging

dnConfig = {
    # Logging configuration for DataNode and DataNoteBuilder
    'ifLog': True,
    'log_name' : 'dataNode', 
    'log_level' : logging.INFO,
    'log_filename' : 'logs/datanode.log',
    'log_filesize' : 5*1024*1024*1024,
    'log_backupCount' : 0,
    'log_fileMode' : 'w'
}