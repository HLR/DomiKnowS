# Gurobi
from gurobipy import *

def notVar(m, _var):
    _notVar = m.addVar(vtype=GRB.BINARY, name="not_%s" % (_var))
    m.addConstr(1 - _var, GRB.EQUAL, _notVar)
            
    return _notVar

def andVar(m, _var1, _var2):
    _andVar = m.addVar(vtype=GRB.BINARY, name="and_%s_%s" % (_var1, _var2))
    
    m.addConstr(_andVar, GRB.LESS_EQUAL, _var1)
    m.addConstr(_andVar, GRB.LESS_EQUAL, _var2)
    
    m.addConstr(_var1 + _var2, GRB.LESS_EQUAL, _andVar + 1)

    return _andVar

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
        m.addConstr(_andVar, GRB.LESS_EQUAL, currentVar)
    
    _varSumLinExpr = LinExpr()
    for currentVar in _var:
        _varSumLinExpr += currentVar

    m.addConstr(_varSumLinExpr, GRB.LESS_EQUAL, _andVar + len(_var) - 1)

    return _andVar

def orVar(m, _var1, _var2):
    _orVar = m.addVar(vtype=GRB.BINARY, name="or_%s_%s" % (_var1, _var2))
    
    m.addConstr(_orVar, GRB.LESS_EQUAL, _var1)
    m.addConstr(_orVar, GRB.LESS_EQUAL, _var2)
    
    m.addConstr(_var1 + _var2, GRB.GREATER_EQUAL, _orVar)

    return _orVar

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
        _varSumLinExpr += currentVar

    m.addConstr(_varSumLinExpr, GRB.GREATER_EQUAL, _orVar)

    return _orVar

def nandVar(m, _var1, _var2):
    _nandVar = m.addVar(vtype=GRB.BINARY, name="nand_%s_%s" % (_var1, _var2))
    
    m.addConstr(1 - _nandVar, GRB.LESS_EQUAL, _var1)
    m.addConstr(1 - _nandVar, GRB.LESS_EQUAL, _var2)
    
    m.addConstr(_var1 + _var2, GRB.LESS_EQUAL, 2 - _nandVar)

    return _nandVar

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
        m.addConstr(_nandVar, GRB.LESS_EQUAL, currentVar)
    
    _varSumLinExpr = LinExpr()
    for currentVar in _var:
        _varSumLinExpr += currentVar

    m.addConstr(_varSumLinExpr, GRB.LESS_EQUAL, _nandVar + len(_var) - 1)

    return _nandVar

def norVar(m, _var1, _var2):
    _norVar = m.addVar(vtype=GRB.BINARY, name="nor_%s_%s"%(_var1, _var2))
    
    m.addConstr(1 - _norVar, GRB.LESS_EQUAL, _var1)
    m.addConstr(1 - _norVar, GRB.LESS_EQUAL, _var2)
    
    m.addConstr(_var1 + _var2, GRB.GREATER_EQUAL, 1 - _norVar)

    return _norVar

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
        m.addConstr(_norVar, GRB.LESS_EQUAL, currentVar)
    
    _varSumLinExpr = LinExpr()
    for currentVar in _var:
        _varSumLinExpr += currentVar

    m.addConstr(_varSumLinExpr, GRB.GREATER_EQUAL, 1 - _norVar)

    return _norVar

def xorVar(m, _var1, _var2):
    _xorVar = m.addVar(vtype=GRB.BINARY, name="xor_%s_%s"%(_var1, _var2))
    
    m.addConstr(_var1 + _var2 + _xorVar, GRB.LESS_EQUAL, 2)
    m.addConstr(-_var1 - _var2 + _xorVar, GRB.LESS_EQUAL, 0)
    m.addConstr(_var1 - _var2 + _xorVar, GRB.GREATER_EQUAL, 0)
    m.addConstr(-_var1 + _var2 + _xorVar, GRB.GREATER_EQUAL, 0)

    return _xorVar

def ifVar(m, _var1, _var2):
    _ifVar = m.addVar(vtype=GRB.BINARY, name="if_%s_%s"%(_var1, _var2))
    
    m.addConstr(1 - _var1 , GRB.LESS_EQUAL, _ifVar)
    m.addConstr(_var2 , GRB.LESS_EQUAL, _ifVar)
    m.addConstr(1 - _var1 + _var2, GRB.GREATER_EQUAL, _ifVar)

    return _ifVar

def epqVar(m, _var1, _var2):
    _epqVar = m.addVar(vtype=GRB.BINARY, name="epq_%s_%s"%(_var1, _var2))
    
    m.addConstr(_var1 + _var2 - _epqVar, GRB.LESS_EQUAL, 1)
    m.addConstr(_var1 + _var2 + _epqVar , GRB.GREATER_EQUAL, 1)
    m.addConstr(-_var1 + _var2 + _epqVar, GRB.LESS_EQUAL, 1)
    m.addConstr(_var1 - _var2 + _epqVar, GRB.LESS_EQUAL, 1)

    return _epqVar

def main() :
    # Create a new Gurobi model
    m = Model("andVarTest")
    m.params.outputflag = 0
    
    _var1 = m.addVar(vtype=GRB.BINARY, name="_var1")
    _var2 = m.addVar(vtype=GRB.BINARY, name="_var2")

    m.addConstr(_var1, GRB.EQUAL, 1)
    m.addConstr(_var2, GRB.EQUAL, 2)

    _andVar = andVar(m, _var1, _var2)

    Q = _andVar
    m.setObjective(Q, GRB.MAXIMIZE)

    m.update()
    m.optimize()
    m.update()

    print(m.printStats())
    
    #print(_andVar.getAttr(GRB.Attr.X))

if __name__ == '__main__' :
    main()