"""Bounded performance probes; timings are evidence, not universal thresholds."""
import importlib
import logging
import os
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


def run_status_dispatch(status, tmp_path, monkeypatch):
    module=importlib.import_module('domiknows.solver.gurobiILPOntSolver')
    monkeypatch.setattr(module,'_default_log_dir',lambda:str(tmp_path))
    calls=[]
    model=SimpleNamespace(status=status,NumVars=1,NumConstrs=1,
        optimize=lambda:None,update=lambda:None,computeIIS=lambda:calls.append('IIS'),
        write=lambda path:calls.append(('write',os.path.basename(path))))
    solver=object.__new__(module.gurobiILPOntSolver)
    solver.myLogger=solver.myLoggerTime=logging.getLogger('repro.status')
    solver.reuse_model=False
    solver.addLogicalConstrains=lambda *args,**kwargs:None
    runs={}
    solver.processILPModelForP(0,{0:[]},model,{},None,False,False,1,False,runs,cacheModel=False)
    assert not runs[0]['solved']
    return calls


@pytest.mark.parametrize('status',[GRB.INFEASIBLE,GRB.INF_OR_UNBD])
def test_D10_iis_can_be_disabled_for_proven_infeasible_models(status,tmp_path,monkeypatch):
    from domiknows.utils import getComputeIIS, setComputeIIS
    assert getComputeIIS() is True, 'default keeps the existing diagnostic'
    enabled=run_status_dispatch(status,tmp_path,monkeypatch)
    assert enabled.count('IIS')==1 and ('write','GurobiInfeasible.ilp') in enabled
    setComputeIIS(False)
    try:
        disabled=run_status_dispatch(status,tmp_path,monkeypatch)
    finally:
        setComputeIIS(True)
    assert 'IIS' not in disabled
    assert ('write','GurobiInfeasible.ilp') not in disabled


def test_D10_iis_flag_never_enables_iis_after_time_limit(tmp_path,monkeypatch):
    from domiknows.utils import setComputeIIS
    setComputeIIS(True)
    assert 'IIS' not in run_status_dispatch(GRB.TIME_LIMIT,tmp_path,monkeypatch)


def test_D10_iis_flag_validation_and_environment(monkeypatch):
    import domiknows
    from domiknows.utils import _computeIISFromEnvironment, setComputeIIS
    with pytest.raises(TypeError):
        setComputeIIS('no')
    assert domiknows.getComputeIIS is not None and domiknows.setComputeIIS is setComputeIIS
    monkeypatch.delenv('DOMIKNOWS_COMPUTE_IIS',raising=False)
    assert _computeIISFromEnvironment() is True
    for value in ('0','false','No','OFF'):
        monkeypatch.setenv('DOMIKNOWS_COMPUTE_IIS',value)
        assert _computeIISFromEnvironment() is False
    monkeypatch.setenv('DOMIKNOWS_COMPUTE_IIS','1')
    assert _computeIISFromEnvironment() is True


@pytest.mark.parametrize('flag,per_call,expected',[
    (True,None,1),(False,None,0),       # None follows the process-wide flag
    (True,False,0),(False,True,1),      # an explicit per-call choice wins
])
def test_D10_iis_per_call_override(flag,per_call,expected,tmp_path,monkeypatch):
    from domiknows.utils import setComputeIIS
    module=importlib.import_module('domiknows.solver.gurobiILPOntSolver')
    monkeypatch.setattr(module,'_default_log_dir',lambda:str(tmp_path))
    calls=[]
    model=SimpleNamespace(status=GRB.INFEASIBLE,NumVars=1,NumConstrs=1,
        optimize=lambda:None,update=lambda:None,computeIIS=lambda:calls.append('IIS'),
        write=lambda path:None)
    solver=object.__new__(module.gurobiILPOntSolver)
    solver.myLogger=solver.myLoggerTime=logging.getLogger('repro.status')
    solver.reuse_model=False
    solver.addLogicalConstrains=lambda *args,**kwargs:None
    setComputeIIS(flag)
    try:
        solver.processILPModelForP(0,{0:[]},model,{},None,False,False,1,False,{},cacheModel=False,
                                   computeIIS=per_call)
    finally:
        setComputeIIS(True)
    assert calls.count('IIS')==expected
