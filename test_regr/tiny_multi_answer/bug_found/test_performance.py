"""Bounded performance probes; timings are evidence, not universal thresholds."""
import gc
import importlib
import inspect
import itertools
import logging
import os
import time
from collections import OrderedDict
from types import SimpleNamespace
import pytest
import torch
from gurobipy import GRB
from domiknows.graph import Graph, Concept
from domiknows.graph.dataNode import DataNode
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


def pair_graph(n):
    """Root -> n items and n*n pair nodes linked to their two items."""
    with Graph('perf_graph') as graph:
        scene=Concept(name='scene'); item=Concept(name='item'); scene.contains(item)
        pair=Concept(name='pair'); arg1,arg2=pair.has_a(arg1=item,arg2=item)
    root=DataNode(instanceID=0,ontologyNode=scene)
    items=[]
    for i in range(n):
        node=DataNode(instanceID=i,ontologyNode=item); root.addChildDataNode(node); items.append(node)
    for i,j in itertools.product(range(n),repeat=2):
        node=DataNode(instanceID=i*n+j,ontologyNode=pair)
        node.addRelationLink(arg1.name,items[i]); node.addRelationLink(arg2.name,items[j])
        root.addChildDataNode(node)
    return root,item,pair


def best_of(repeats,fn):
    """Minimum wall time; GC is paused because its cost grows with the object
    count and would blur the algorithmic shape being checked."""
    times=[]
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            start=time.perf_counter(); result=fn(); times.append(time.perf_counter()-start)
    finally:
        gc.enable()
    return min(times),result


def test_D10_datanode_construction_scales_roughly_linearly(record_property):
    """Quadratic list-membership tests made 4x the nodes cost ~16x (12 s at 14.5k nodes).

    Compares 4x growth in node count; the threshold is far above linear (4x)
    and far below quadratic (16x), so it is a shape check, not a speed limit.
    """
    small,_=best_of(3,lambda:pair_graph(60))
    large,_=best_of(3,lambda:pair_graph(120))
    record_property('construction_seconds_small_large',(small,large))
    assert large/small<10, f'4x nodes took {large/small:.1f}x (small={small:.3f}s, large={large:.3f}s)'


def test_D10_find_datanodes_scales_roughly_linearly(record_property):
    small_root,small_item,small_pair=pair_graph(40)
    large_root,large_item,large_pair=pair_graph(80)
    small,found_small=best_of(3,lambda:small_root.findDatanodes(select=small_pair))
    large,found_large=best_of(3,lambda:large_root.findDatanodes(select=large_pair))
    record_property('find_seconds_small_large',(small,large))
    assert len(found_small)==40*40 and len(found_large)==80*80
    assert [d.id for d in found_large]==sorted({d.id for d in found_large})  # no duplicates, graph order kept
    assert large/small<10, f'4x nodes took {large/small:.1f}x (small={small:.4f}s, large={large:.4f}s)'


def test_D10_relation_links_ignore_duplicates_and_survive_removal():
    """The link-membership cache must keep exact list semantics."""
    with Graph('links') as graph:
        scene=Concept(name='scene'); item=Concept(name='item'); scene.contains(item)
    root=DataNode(instanceID=0,ontologyNode=scene)
    nodes=[DataNode(instanceID=i,ontologyNode=item) for i in range(4)]
    for node in nodes: root.addChildDataNode(node)
    for node in nodes: root.addChildDataNode(node)           # duplicates ignored
    assert root.relationLinks['contains']==nodes
    assert all(node.impactLinks['contains']==[root] for node in nodes)
    root.removeChildDataNode(nodes[1])
    assert root.relationLinks['contains']==[nodes[0],nodes[2],nodes[3]]
    root.addChildDataNode(nodes[1])                           # removed node can be re-added
    assert root.relationLinks['contains']==[nodes[0],nodes[2],nodes[3],nodes[1]]
    root.resetChildDataNode()                                 # list replaced behind the cache
    root.addChildDataNode(nodes[0])
    assert root.relationLinks['contains']==[nodes[0]]


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


# --- D10 performance: ILP build ------------------------------------------------

def _scalar_epsilon_reference(stored, epsilon):
    """The previous getProbability clamp: scalar max/min on a view of `stored`."""
    value = stored.squeeze(0)
    if not torch.isnan(value[0]).item() and epsilon is not None:
        value[0] = max(epsilon, min(1 - epsilon, value[0]))
        value[1] = max(epsilon, min(1 - epsilon, value[1]))
    return value


SPECIAL_PROBABILITIES = [0.0, 1.0, 0.5, 1e-9, 1 - 1e-9, 1e-5, 1 - 1e-5, 0.99999,
                         float('nan'), float('inf'), -float('inf'), -0.3, 1.7]


def test_D10_probability_clamp_matches_the_scalar_reference_including_side_effect():
    """The vector clamp must return the same values, NaN handling included, and
    clamp the stored softmax in place exactly as the scalar code did."""
    module = importlib.import_module('domiknows.solver.gurobiILPOntSolver')
    solver = object.__new__(module.gurobiILPOntSolver)
    solver.constraintConstructor = SimpleNamespace(conceptIsMultiClass=lambda concept: False)
    concept = ('flag', 'flag', 0, 1)
    for first, second in itertools.product(SPECIAL_PROBABILITIES, repeat=2):
        stored_new = torch.tensor([[first, second]], dtype=torch.float32)
        stored_old = stored_new.clone()
        got = solver.getProbability(SimpleNamespace(getAttribute=lambda *a: stored_new), concept,
                                    key=('local', 'softmax'), fun=None, epsilon=1e-5)
        want = _scalar_epsilon_reference(stored_old, 1e-5)
        nan_to_marker = lambda t: torch.nan_to_num(t, nan=-7.0)
        assert torch.equal(nan_to_marker(got), nan_to_marker(want)), (first, second, got, want)
        assert torch.equal(nan_to_marker(stored_new), nan_to_marker(stored_old)),             f'stored tensor differs for {(first, second)}: {stored_new} vs {stored_old}'


def test_D10_logical_constraint_signature_is_inspected_once_per_call(monkeypatch):
    """createLogicalConstrains used to run inspect.signature for every grounded row."""
    calls = []
    real = inspect.signature
    monkeypatch.setattr(inspect, 'signature', lambda fn, *a, **k: calls.append(fn) or real(fn, *a, **k))
    lc = object.__new__(LogicalConstrain)
    def builder(model, *args, onlyConstrains=False):
        return 1
    rows = 500
    lc.createLogicalConstrains('AND', builder, object(),
        OrderedDict(a=[[i] for i in range(rows)], b=[[i] for i in range(rows)]), True)
    assert len(calls) == 1, f'inspect.signature called {len(calls)} times for {rows} rows'


def test_D10_find_datanodes_cache_scope_semantics():
    root, item, pair = pair_graph(6)
    outside = root.findDatanodes(select=item)
    with DataNode.findDatanodesCache():
        first = root.findDatanodes(select=item)
        second = root.findDatanodes(select=item)
        assert [d.id for d in first] == [d.id for d in outside]
        assert first is not second, 'every caller gets its own list'
        second.clear()
        assert [d.id for d in root.findDatanodes(select=item)] == [d.id for d in outside],             'mutating a returned list must not corrupt the cache'
        # queries the cache does not cover are answered normally
        assert len(root.findDatanodes(select=pair)) == 36
        # a link change inside the scope invalidates it
        extra = DataNode(instanceID=999, ontologyNode=item)
        root.addChildDataNode(extra)
        assert 999 in [d.getInstanceID() for d in root.findDatanodes(select=item)]
    assert DataNode._findCache is None, 'the cache must be gone after the scope'
    assert 999 in [d.getInstanceID() for d in root.findDatanodes(select=item)]


def test_D10_find_datanodes_cache_is_not_stale_across_scopes():
    root, item, pair = pair_graph(4)
    with DataNode.findDatanodesCache():
        assert len(root.findDatanodes(select=item)) == 4
    root.addChildDataNode(DataNode(instanceID=500, ontologyNode=item))
    with DataNode.findDatanodesCache():
        assert len(root.findDatanodes(select=item)) == 5
        with DataNode.findDatanodesCache():          # nested scopes share the outer cache
            assert len(root.findDatanodes(select=item)) == 5
        assert DataNode._findCache is not None, 'inner scope must not drop the outer cache'
    assert DataNode._findCache is None
