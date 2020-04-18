import os
import logging
REGR_SOLVER = os.environ.get('REGR_SOLVER', 'Gurobi')

ilpConfig = {
    # variable controlling what ILP solver is used  - one of "Gurobi", "GEKKO", None
    'ilpSolver' : REGR_SOLVER,

    # Logging configuration for ilpOntSolver
    'log_name' : 'ilpOntSolver', 
    'log_level' : logging.DEBUG,
    'log_filename' : 'ilpOntSolver.log',
    'log_filesize' : 5*1024*1024*1024,
    'log_backupCount' : 5,
    'log_fileMode' : 'a'
}