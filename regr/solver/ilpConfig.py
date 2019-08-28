import os
REGR_SOLVER = 'REGR_SOLVER' in os.environ and os.environ['REGR_SOLVER']

ilpConfig = {
    # variable controlling what ILP solver is used  - one of "Gurobi", "GEKKO", None
    'ilpSolver' : REGR_SOLVER or 'Gurobi'
}
