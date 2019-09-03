if __package__ is None or __package__ == '':
    from regr.solver.ilpConfig import ilpConfig 
    from regr.solver.ilpBooleanMethods import ilpBooleanProcessor 
else:
    from .ilpConfig import ilpConfig 
    from .ilpBooleanMethods import ilpBooleanProcessor 

class gurobiILPBooleanProcessor(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()

    # Negation
    def notVar(m, _var):
        _notVar=m.addVar(vtype=GRB.BINARY, name="not_%s" % (_var))
        
        m.addConstr(1 - _var, GRB.EQUAL, _notVar)
    
        return _notVar
    
    # Conjunction 2 variable
    def and2Var(m, _var1, _var2):
        _andVar=m.addVar(vtype=GRB.BINARY, name="and_%s_%s" % (_var1, _var2))
            
        m.addConstr(_andVar - _var1, GRB.LESS_EQUAL, 0)
        m.addConstr(_andVar - _var2, GRB.LESS_EQUAL, 0)
            
        m.addConstr(_var1 + _var2 - _andVar - 1, GRB.LESS_EQUAL, 0)
            
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
            
        _andVar = m.addVar(vtype=GRB.BINARY, name=_andVarName)
        for currentVar in _var:
            m.addConstr(_andVar - currentVar, GRB.LESS_EQUAL, 0)

        _varSumLinExpr = LinExpr()
        for currentVar in _var:
            _varSumLinExpr.addTerms(1.0, currentVar)
        
        m.addConstr(_varSumLinExpr, GRB.LESS_EQUAL, _andVar + len(_var) - 1)
    
        return _andVar
    
    # Disjunction 2 variables
    def or2Var(m, _var1, _var2):
        _orVar=m.addVar(vtype=GRB.BINARY, name="or_%s_%s" % (_var1, _var2))
            
        m.addConstr(_orVar, GRB.LESS_EQUAL, _var1)
        m.addConstr(_orVar, GRB.LESS_EQUAL, _var2)
            
        m.addConstr(_var1 + _var2, GRB.GREATER_EQUAL, _orVar)
    
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
        
        _orVar = m.addVar(vtype=GRB.BINARY, name=_orVarName)

        for currentVar in _var:
            m.addConstr(_orVar, GRB.LESS_EQUAL, currentVar)

        _varSumLinExpr = LinExpr()
        for currentVar in _var:
            _varSumLinExpr.addTerms(1.0, currentVar)
        
        m.addConstr(_varSumLinExpr, GRB.GREATER_EQUAL, _orVar)
             
        return _orVar
    
    # Nand (Alternative denial) 2 variables
    def nand2Var(m, _var1, _var2):
        _nandVar = m.addVar(vtype=GRB.BINARY, name="nand_%s_%s" % (_var1, _var2))
            
        m.addConstr(1 - _nandVar, GRB.LESS_EQUAL, _var1)
        m.addConstr(1 - _nandVar, GRB.LESS_EQUAL, _var2)
        
        m.addConstr(_var1 + _var2, GRB.LESS_EQUAL, 2 - _nandVar)
        
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
            
        _nandVar = m.addVar(vtype=GRB.BINARY, name=_nandVarName)
        for currentVar in _var:
            m.addConstr(1 - _nandVar, GRB.LESS_EQUAL, currentVar)

        _varSumLinExpr = LinExpr()
        for currentVar in _var:
            _varSumLinExpr.addTerms(1.0, currentVar)
    
        m.addConstr(_varSumLinExpr, GRB.LESS_EQUAL, len(_var) - _nandVar)
    
        return _nandVar
    
    # Nor (Joint Denial) i2 variables
    def nor2Var(m, _var1, _var2):
        _norVar = m.addVar(vtype=GRB.BINARY, name="nor_%s_%s"%(_var1, _var2))
            
        m.addConstr(_var1, GRB.LESS_EQUAL, 1 - _norVar)
        m.addConstr(_var2, GRB.LESS_EQUAL, 1 - _norVar)
            
        m.addConstr(_var1 + _var2, GRB.GREATER_EQUAL, 1 - _norVar)
        
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
           
        _norVar = m.addVar(vtype=GRB.BINARY, name=_norVarName)
        for currentVar in _var:
            m.addConstr(currentVar, GRB.LESS_EQUAL, 1 - _norVar)
        
        _varSumLinExpr = LinExpr()
        for currentVar in _var:
            _varSumLinExpr.addTerms(1.0, currentVar)
            
            m.addConstr(_varSumLinExpr, GRB.GREATER_EQUAL, 1 - _norVar)
    
        return _norVar
    
    # Exclusive Disjunction
    def xorVar(m, _var1, _var2):
        _xorVar = m.addVar(vtype=GRB.BINARY, name="xor_%s_%s"%(_var1, _var2))
            
        m.addConstr(_var1 + _var2 + _xorVar, GRB.LESS_EQUAL, 2)
        m.addConstr(-_var1 - _var2 + _xorVar, GRB.LESS_EQUAL, 0)
        m.addConstr(_var1 - _var2 + _xorVar, GRB.GREATER_EQUAL, 0)
        m.addConstr(-_var1 + _var2 + _xorVar, GRB.GREATER_EQUAL, 0)
            
        return _xorVar
    
    # Implication
    def ifVar(m, _var1, _var2):
        _ifVar = m.addVar(vtype=GRB.BINARY, name="if_%s_%s"%(_var1, _var2))
            
        m.addConstr(1 - _var1 , GRB.LESS_EQUAL, _ifVar)
        m.addConstr(_var2 , GRB.LESS_EQUAL, _ifVar)
        m.addConstr(1 - _var1 + _var2, GRB.GREATER_EQUAL, _ifVar)
            
        return _ifVar
           
    # Equivalence 
    def epqVar(m, _var1, _var2):
        _epqVar = m.addVar(vtype=GRB.BINARY, name="epq_%s_%s"%(_var1, _var2))
            
        m.addConstr(_var1 + _var2 - _epqVar, GRB.LESS_EQUAL, 1)
        m.addConstr(_var1 + _var2 + _epqVar , GRB.GREATER_EQUAL, 1)
        m.addConstr(-_var1 + _var2 + _epqVar, GRB.LESS_EQUAL, 1)
        m.addConstr(_var1 - _var2 + _epqVar, GRB.LESS_EQUAL, 1)
        
        return _epqVar
    