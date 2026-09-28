"""Bounded performance probes; timings are evidence, not universal thresholds."""
import importlib
import logging
import time
from collections import OrderedDict
from types import SimpleNamespace
import pytest
from gurobipy import GRB
from domiknows.graph.logicalConstrain import LogicalConstrain


@pytest.mark.parametrize('rows', [256,1024,4096])
def test_D10_singleton_assembly_scaling(rows, record_property):
    lc=object.__new__(LogicalConstrain)
    operands=OrderedDict((str(k),[[k] for _ in range(rows)]) for k in range(16))
    start=time.perf_counter()
    result=lc._collectVariableSetups('0',list(operands)[1:],operands)
    record_property('assembly_seconds',time.perf_counter()-start)
    assert len(result)==rows
    assert all(row==[list(range(16))] for row in result)


@pytest.mark.parametrize('count', [256,1024,4096])
def test_D10_name_construction_scaling_without_solver(count, monkeypatch, record_property):
    """Real preprocessing function with lightweight external-expression doubles.

    Measures Python name construction, NOT Gurobi construction/optimization.
    """
    module=importlib.import_module('domiknows.solver.gurobiILPBooleanMethods')
    class Var:
        def __init__(self,name): self.VarName=name
    class Expr:
        def __init__(self): self.terms=[]
        def addTerms(self,c,v): self.terms.append((c,v))
        def size(self): return len(self.terms)
        def __str__(self): return '<recording expression>'
    monkeypatch.setattr(module,'Var',Var)
    monkeypatch.setattr(module,'LinExpr',Expr)
    processor=module.gurobiILPBooleanProcessor()
    variables=[Var('long_variable_name_'+str(i)) for i in range(count)]
    start=time.perf_counter()
    result=processor.preprocessLogicalMethodVar(variables,'COUNT','count')
    record_property('python_preprocess_seconds',time.perf_counter()-start)
    expected=''.join('count_'+v.VarName+'_' for v in variables)[:-1][:200]
    assert result['varName']==expected
    assert result['No_of_ilp']==count
    assert result['varSumLinExpr'].terms==[(1.0,v) for v in variables]


@pytest.mark.parametrize('status',[GRB.TIME_LIMIT,GRB.INFEASIBLE])
def test_D10_IIS_dispatch_after_nonoptimal_status(status,tmp_path,monkeypatch,record_property):
    """Real post-solve dispatch with a status-recording external model double.

    No optimizer is run. INFEASIBLE documents default diagnostic dispatch;
    TIME_LIMIT must NOT be treated as proof of infeasibility.
    """
    module=importlib.import_module('domiknows.solver.gurobiILPOntSolver')
    monkeypatch.setattr(module,'_default_log_dir',lambda:str(tmp_path))
    calls=[]
    model=SimpleNamespace(status=status,NumVars=1,NumConstrs=1,
        optimize=lambda:None,update=lambda:None,computeIIS=lambda:calls.append('IIS'),
        write=lambda path:calls.append('write'))
    solver=object.__new__(module.gurobiILPOntSolver)
    solver.myLogger=solver.myLoggerTime=logging.getLogger('repro.status')
    solver.reuse_model=False
    solver.addLogicalConstrains=lambda *args,**kwargs:None
    runs={}
    solver.processILPModelForP(0,{0:[]},model,{},None,False,False,1,False,runs,cacheModel=False)
    record_property('iis_calls',calls.count('IIS'))
    assert not runs[0]['solved']
    if status==GRB.TIME_LIMIT:
        assert 'IIS' not in calls, 'TIME_LIMIT is not proven infeasibility, but computeIIS was called'
    else:
        assert calls.count('IIS')==1  # Observed policy, not a speed certification.
