if __package__ is None or __package__ == '':
    from domiknows.solver.ilpConfig import ilpConfig 
else:
    from .ilpConfig import ilpConfig 
    
if ilpConfig['ilpSolver'] == "Gurobi":
    from gurobipy import *
elif ilpConfig['ilpSolver'] == "GEKKO":
    from gekko import GEKKO
        
class gekkoILPBooleanProcessor:
    
    # variable controlling what ILP solver is used
    ilpSolver = "Gurobi" # "Gurobi", "GEKKO", "None"
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
        
        if _ildConfig is not None:
            self.ilpSolver = _ildConfig['ilpSolver']

    # Negation
    def notVar(m, _var):
        if ilpSolver=="Gurobi":
             _notVar=m.addVar(vtype=GRB.BINARY, name="not_%s" % (_var))
             
             m.addConstr(1 - _var, GRB.EQUAL, _notVar)
        elif ilpSolver == "GEKKO":
            _notVar=m.Var(0, lb=0, ub=1, integer=True, name="not_%s" % (_var))
            
            m.Equation(1 - _var - _notVar == 0)
    
        return _notVar
    
    # Conjunction 2 variable
    def and2Var(m, _var1, _var2):
        if ilpSolver=="Gurobi":
            _andVar=m.addVar(vtype=GRB.BINARY, name="and_%s_%s" % (_var1, _var2))
            
            m.addConstr(_andVar - _var1, GRB.LESS_EQUAL, 0)
            m.addConstr(_andVar - _var2, GRB.LESS_EQUAL, 0)
            
            m.addConstr(_var1 + _var2 - _andVar - 1, GRB.LESS_EQUAL, 0)
        elif ilpSolver == "GEKKO":
            _andVar=m.Var(0, lb=0, ub=1, integer=True, name="and_%s_%s" % (_var1, _var2))
    
            m.Equation(_andVar - _var1 <= 0)
            m.Equation(_andVar - _var2 <= 0)
            
            m.Equation(_var1 + _var2 - _andVar <= 1)
            
        return _andVar
    
    # Conjunction
    def andVar(m, *_var):
        if len(_var) < 1:
            return None
        
        if len(_var) == 1:
            for currentVar in _var:
                return currentVar
        
        _andVarName = "and"
        for currentVar in _var:
            _andVarName += "_%s" % (currentVar)
            
        if ilpSolver=="Gurobi":
            _andVar = m.addVar(vtype=GRB.BINARY, name=_andVarName)
        elif ilpSolver == "GEKKO":
            _andVar = m.Var(0, lb=0, ub=1, integer=True, name=_andVarName)
    
        for currentVar in _var:
            if ilpSolver=="Gurobi":
                m.addConstr(_andVar - currentVar, GRB.LESS_EQUAL, 0)
            elif ilpSolver == "GEKKO":
                m.Equation(_andVar - currentVar <= 0)
    
        if ilpSolver=="Gurobi":
            _varSumLinExpr = LinExpr()
            for currentVar in _var:
                _varSumLinExpr.addTerms(1.0, currentVar)
        
            m.addConstr(_varSumLinExpr, GRB.LESS_EQUAL, _andVar + len(_var) - 1)
        elif ilpSolver == "GEKKO":
            m.Equation(m.sum(list(_var)) - _andVar - len(_var) <= -1)
    
        return _andVar
    
    # Disjunction 2 variables
    def or2Var(m, _var1, _var2):
        if ilpSolver=="Gurobi":
            _orVar=m.addVar(vtype=GRB.BINARY, name="or_%s_%s" % (_var1, _var2))
            
            m.addConstr(_orVar, GRB.LESS_EQUAL, _var1)
            m.addConstr(_orVar, GRB.LESS_EQUAL, _var2)
            
            m.addConstr(_var1 + _var2, GRB.GREATER_EQUAL, _orVar)
        elif ilpSolver == "GEKKO":
            _orVar=m.Var(0, lb=0, ub=1, integer=True, name="or_%s_%s" % (_var1, _var2))
     
            m.Equation(_orVar - _var1 <= 0)
            m.Equation(_orVar - _var2 <= 0)
            
            m.Equation(_var1 + _var2 - _orVar >= 0)
    
        return _orVar
    
    # Disjunction
    def orVar(m, *_var):
        if len(_var) < 1:
            return None
        
        if len(_var) == 1:
            for currentVar in _var:
                return currentVar
        
        _orVarName = "or"
        for currentVar in _var:
            _orVarName += "_%s" % (currentVar)
        
        if ilpSolver=="Gurobi":
            _orVar = m.addVar(vtype=GRB.BINARY, name=_orVarName)
        elif ilpSolver == "GEKKO":
            _orVar = m.Var(0, lb=0, ub=1, integer=True, name=_orVarName)
    
        for currentVar in _var:
            if ilpSolver=="Gurobi":
                m.addConstr(_orVar, GRB.LESS_EQUAL, currentVar)
            elif ilpSolver == "GEKKO":
                m.Equation(_orVar - currentVar <= 0)
    
        if ilpSolver=="Gurobi":
            _varSumLinExpr = LinExpr()
            for currentVar in _var:
                _varSumLinExpr.addTerms(1.0, currentVar)
        
            m.addConstr(_varSumLinExpr, GRB.GREATER_EQUAL, _orVar)
        elif ilpSolver == "GEKKO":
             m.Equation(m.sum(list(_var)) - _orVar >= 0)
             
        return _orVar
    
    # Nand (Alternative denial) 2 variables
    def nand2Var(m, _var1, _var2):
        if ilpSolver=="Gurobi":
            _nandVar = m.addVar(vtype=GRB.BINARY, name="nand_%s_%s" % (_var1, _var2))
            
            m.addConstr(1 - _nandVar, GRB.LESS_EQUAL, _var1)
            m.addConstr(1 - _nandVar, GRB.LESS_EQUAL, _var2)
        
            m.addConstr(_var1 + _var2, GRB.LESS_EQUAL, 2 - _nandVar)
        elif ilpSolver == "GEKKO":
            _nandVar = m.Var(0, lb=0, ub=1, integer=True, name="nand_%s_%s" % (_var1, _var2))
    
            m.Equation(1 - _nandVar - _var1 <= 0)
            m.Equation(1 - _nandVar - _var2 <= 0)
        
            m.Equation(_var1 + _var2 + _nandVar <= 2)
        
        return _nandVar
    
    # Nand (Alternative denial)
    def nandVar(m, *_var):
        if len(_var) < 1:
            return None
        
        if len(_var) == 1:
            for currentVar in _var:
                return currentVar
        
        _nandVarName = "nand"
        for currentVar in _var:
            _nandVarName += "_%s" % (currentVar)
            
        if ilpSolver=="Gurobi":
            _nandVar = m.addVar(vtype=GRB.BINARY, name=_nandVarName)
        elif ilpSolver == "GEKKO":
            _nandVar = m.Var(0, lb=0, ub=1, integer=True, name=_nandVarName)
    
        for currentVar in _var:
            if ilpSolver=="Gurobi":
                m.addConstr(1 - _nandVar, GRB.LESS_EQUAL, currentVar)
            elif ilpSolver == "GEKKO":
                m.Equation(currentVar + _nandVar >= 1)
        
        if ilpSolver=="Gurobi":
            _varSumLinExpr = LinExpr()
            for currentVar in _var:
                _varSumLinExpr.addTerms(1.0, currentVar)
    
            m.addConstr(_varSumLinExpr, GRB.LESS_EQUAL, len(_var) - _nandVar)
        elif ilpSolver == "GEKKO":
            m.Equation(m.sum(list(_var)) + _nandVar <= len(_var))
    
        return _nandVar
    
    # Nor (Joint Denial) i2 variables
    def nor2Var(m, _var1, _var2):
        if ilpSolver=="Gurobi":
            _norVar = m.addVar(vtype=GRB.BINARY, name="nor_%s_%s"%(_var1, _var2))
            
            m.addConstr(_var1, GRB.LESS_EQUAL, 1 - _norVar)
            m.addConstr(_var2, GRB.LESS_EQUAL, 1 - _norVar)
            
            m.addConstr(_var1 + _var2, GRB.GREATER_EQUAL, 1 - _norVar)
        elif ilpSolver == "GEKKO":
            _norVar = m.Var(0, lb=0, ub=1, integer=True, name="nor_%s_%s"%(_var1, _var2))
            
            m.Equation(_norVar + _var1 <= 1)
            m.Equation(_norVar + _var2 <= 1)
            
            m.Equation(_var1 + _var2 + _norVar >= 1)
            
        return _norVar
    
    # Nor (Joint Denial)
    def norVar(m, *_var):
        if len(_var) < 1:
            return None
        
        if len(_var) == 1:
            for currentVar in _var:
                return currentVar
        
        _norVarName = "nor"
        for currentVar in _var:
            _norVarName += "_%s"%(currentVar)
           
        if ilpSolver=="Gurobi":
            _norVar = m.addVar(vtype=GRB.BINARY, name=_norVarName)
        elif ilpSolver == "GEKKO":
            _norVar = m.Var(0, lb=0, ub=1, integer=True, name=_norVarName)
    
        for currentVar in _var:
            if ilpSolver=="Gurobi":
                m.addConstr(currentVar, GRB.LESS_EQUAL, 1 - _norVar)
            elif ilpSolver == "GEKKO":
                m.Equation(currentVar + _norVar  <= 1)
        
        if ilpSolver=="Gurobi":
            _varSumLinExpr = LinExpr()
            for currentVar in _var:
                _varSumLinExpr.addTerms(1.0, currentVar)
            
            m.addConstr(_varSumLinExpr, GRB.GREATER_EQUAL, 1 - _norVar)
        elif ilpSolver == "GEKKO":
            m.Equation(m.sum(list(_var)) + _nandVar >= 1)
    
        return _norVar
    
    # Exclusive Disjunction
    def xorVar(m, _var1, _var2):
        if ilpSolver=="Gurobi":
            _xorVar = m.addVar(vtype=GRB.BINARY, name="xor_%s_%s"%(_var1, _var2))
            
            m.addConstr(_var1 + _var2 + _xorVar, GRB.LESS_EQUAL, 2)
            m.addConstr(-_var1 - _var2 + _xorVar, GRB.LESS_EQUAL, 0)
            m.addConstr(_var1 - _var2 + _xorVar, GRB.GREATER_EQUAL, 0)
            m.addConstr(-_var1 + _var2 + _xorVar, GRB.GREATER_EQUAL, 0)
        elif ilpSolver == "GEKKO":
            _xorVar = m.Var(0, lb=0, ub=1, integer=True, name="xor_%s_%s"%(_var1, _var2))
            
            m.Equation(_var1 + _var2 + _xorVar <= 2)
            m.Equation(-_var1 - _var2 + _xorVar <= 0)
            m.Equation(_var1 - _var2 + _xorVar >= 0)
            m.Equation(-_var1 + _var2 + _xorVar >= 0)
            
        return _xorVar
    
    # Implication
    def ifVar(m, _var1, _var2):
        if ilpSolver=="Gurobi":
            _ifVar = m.addVar(vtype=GRB.BINARY, name="if_%s_%s"%(_var1, _var2))
            
            m.addConstr(1 - _var1 , GRB.LESS_EQUAL, _ifVar)
            m.addConstr(_var2 , GRB.LESS_EQUAL, _ifVar)
            m.addConstr(1 - _var1 + _var2, GRB.GREATER_EQUAL, _ifVar)
        elif ilpSolver == "GEKKO":
            _ifVar = m.Var(0, lb=0, ub=1, integer=True, name="if_%s_%s"%(_var1, _var2))
            
            m.Equation(_var1 + _ifVar >= 1)
            m.Equation(_var2 - _ifVar <= 0)
            m.Equation(_ifVar + _var1 - _var2 <= 1)
            
        return _ifVar
           
    # Equivalence 
    def epqVar(m, _var1, _var2):
        if ilpSolver=="Gurobi":
            _epqVar = m.addVar(vtype=GRB.BINARY, name="epq_%s_%s"%(_var1, _var2))
            
            m.addConstr(_var1 + _var2 - _epqVar, GRB.LESS_EQUAL, 1)
            m.addConstr(_var1 + _var2 + _epqVar , GRB.GREATER_EQUAL, 1)
            m.addConstr(-_var1 + _var2 + _epqVar, GRB.LESS_EQUAL, 1)
            m.addConstr(_var1 - _var2 + _epqVar, GRB.LESS_EQUAL, 1)
        elif ilpSolver == "GEKKO":
            _epqVar = m.Var(0, lb=0, ub=1, integer=True, name="epq_%s_%s"%(_var1, _var2))
            
            m.Equation(_var1 + _var2 - _epqVar <= 1)
            m.Equation(_var1 + _var2 + _epqVar >= 1)
            m.Equation(-_var1 + _var2 + _epqVar <= 1)
            m.Equation(_var1 - _var2 + _epqVar <= 1)
     
        return _epqVar
    
    def main() :
        # Create a new Gurobi model
        m = Model("andVarTest")
        m.Params.InfUnbdInfo = 1
        m.params.outputflag = 1
        
        _var1 = m.addVar(vtype=GRB.BINARY, name="_var1")
        _var2 = m.addVar(vtype=GRB.BINARY, name="_var2")
    
        m.addConstr(_var1, GRB.EQUAL, 1)
        m.addConstr(_var2, GRB.EQUAL, 1)
    
        _andVar = andVar(m, _var1, _var2)
        m.update()
    
        Q = LinExpr()
        Q += _andVar
        m.setObjective(Q, GRB.MAXIMIZE)
    
        m.update()
        
        print(m.getObjective()) 
        for const in m.getConstrs():
            print(m.getConstrByName(const.ConstrName))
            
        m.optimize()
           
        print(m.printStats())
    
        if m.status == GRB.Status.OPTIMAL:
            print('Optimal solution was found')
        elif m.status == GRB.Status.INFEASIBLE:
            print('Model was proven to be infeasible.')
            exit()
        elif m.status == GRB.Status.INF_OR_UNBD:
            print('Model was proven to be infeasible or unbound.')
            exit()
        elif m.status == GRB.Status.UNBOUNDED:
            print('Model was proven to be unbound.')
            exit()
        else:
            print('Optimal solution not was found - error code %i'%(m.status))
            exit()
    
        m.update()
        m.printAttr('x') 
        
        #print(_andVar.getAttr(GRB.Attr.X))
    
    if __name__ == '__main__' :
        main()