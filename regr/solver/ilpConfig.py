import os
import logging
REGR_SOLVER = 'REGR_SOLVER' in os.environ and os.environ['REGR_SOLVER']

ilpConfig = {
    # variable controlling what ILP solver is used  - one of "Gurobi", "GEKKO", None
    'ilpSolver' : REGR_SOLVER or 'Gurobi',
    
    # 'log_name'
    'log_level' : logging.DEBUG,
    'log_filename' : 'ilpOntSolver.log',
    'log_filesize' : 5*1024*1024*1024,
    'log_backupCount' : 5,
    'log_fileMode' : 'a'
}