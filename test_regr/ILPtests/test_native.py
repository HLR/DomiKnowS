"""Native-solver reproducers. Skipped explicitly without a valid license.

These tests are prepared, NOT certified as reproduced when skipped.
"""
import itertools
import math
import time
import pytest
import torch
from domiknows.graph import Graph, Concept
from domiknows.graph.dataNode import DataNode
from domiknows.graph.logicalConstrain import execute, existsL, andL, notL, miotaL, ifL
from domiknows.solver.answerModule import AnswerSolver


def activate(root, graph):
    node = DataNode(instanceID=0, ontologyNode=graph.get_constraint_concept())
    node.attributes['ELC0/label'] = torch.tensor(1.)
    root.addChildDataNode(node)
    root.inferLocal(keys=('softmax',))
    root.setActiveExecutableLCs()


def solve(root, graph):
    solver, concepts = root.getILPSolver(root.collectConceptsAndRelations())
    return AnswerSolver(graph, solver=solver, compiled=True).solve_active_constraints(
        root, ['ELC0'], concepts, key=('local','softmax'), fun=torch.log,
        populate=True, raise_on_infeasible=True)


def test_D01_native_selector_obeys_forced_false(native_license):
    with Graph('selector') as graph:
        scene = Concept(name='scene'); item = Concept(name='item'); scene.contains(item)
        flag = item(name='flag')
        notL(flag('x'))
        execute(miotaL(flag('x'), threshold=.5, hard=False))
    root = DataNode(instanceID=0, ontologyNode=scene)
    child = DataNode(instanceID=0, ontologyNode=item)
    child.attributes['<flag>'] = torch.tensor([math.log(.01), math.log(.99)])
    root.addChildDataNode(child); activate(root, graph)
    result = solve(root, graph)
    assert child.attributes['<flag>/ILP'].item() == 0
    # miotaL answers are candidate-aligned 0/1 lists (README, ExecutableInference):
    # the one candidate is forced false, so it is not selected.
    assert result['hypotheses']['ELC0'] == [0]


def test_D03_native_inverse_argument_order(native_license):
    with Graph('inverse') as graph:
        scene = Concept(name='scene'); item = Concept(name='item'); scene.contains(item)
        pair = Concept(name='pair'); arg1,arg2 = pair.has_a(arg1=item,arg2=item)
        left = pair(name='left'); right = pair(name='right')
        ifL(left('x','y'),right('y','x'))
        ifL(right('y','x'),left('x','y'))
    root = DataNode(instanceID=0, ontologyNode=scene)
    objects=[]; pairs={}
    for i in range(2):
        node=DataNode(instanceID=i, ontologyNode=item); root.addChildDataNode(node); objects.append(node)
    for i,j in itertools.product(range(2),repeat=2):
        node=DataNode(instanceID=i*2+j,ontologyNode=pair)
        node.addRelationLink(arg1.name,objects[i]); node.addRelationLink(arg2.name,objects[j])
        for concept,truth in [(left,(i,j)==(0,1)),(right,(i,j)==(1,0))]:
            p=.99 if truth else .01
            node.attributes[f'<{concept.name}>']=torch.tensor([math.log(1-p),math.log(p)])
        root.addChildDataNode(node); pairs[i,j]=node
    root.inferLocal(keys=('softmax',))
    solver,concepts=root.getILPSolver(root.collectConceptsAndRelations())
    result=solver._calculateILPSelection(root,*concepts,key=('local','softmax'),fun=torch.log,
        populate=True,forceFreshModel=True,raiseOnInfeasible=True,compiled=True)
    assert result is not None
    for i,j in pairs:
        l=pairs[i,j].attributes['<left>/ILP'].item()
        r=pairs[j,i].attributes['<right>/ILP'].item()
        assert l==r, 'Inverse constraint violated under explicit endpoint identities'
        assert l==float((i,j)==(0,1)), 'Already-consistent unconstrained MAP world was changed'


@pytest.mark.parametrize('case', range(6))
def test_D04_native_unequal_scope_join_matches_exhaustive_worlds(native_license, case):
    with Graph('joint') as graph:
        scene=Concept(name='scene'); item=Concept(name='item'); scene.contains(item)
        a=item(name='a'); b=item(name='b'); pair=Concept(name='pair')
        arg1,arg2=pair.has_a(arg1=item,arg2=item); r=pair(name='r')
        execute(existsL(andL(a('x'),r('x','y'),r('y','z'),b('z'))))
    root=DataNode(instanceID=0,ontologyNode=scene); nodes=[]
    probs=[.13,.82,.77,.21,.12,.89,.23,.71]
    if case%2: probs[:2]=[.12,.19]
    if case>=2: probs[4:]=[.83,.11,.76,.14]
    if case>=4: probs[2:4]=[.17,.24]
    for i in range(2):
        node=DataNode(instanceID=i,ontologyNode=item); root.addChildDataNode(node); nodes.append(node)
        for c,p in [(a,probs[i]),(b,probs[2+i])]:
            node.attributes[f'<{c.name}>']=torch.tensor([math.log(1-p),math.log(p)],dtype=torch.float64)
    for i,j in itertools.product(range(2),repeat=2):
        node=DataNode(instanceID=i*2+j,ontologyNode=pair)
        node.addRelationLink(arg1.name,nodes[i]); node.addRelationLink(arg2.name,nodes[j]); root.addChildDataNode(node)
        p=probs[4+i*2+j]; node.attributes['<r>']=torch.tensor([math.log(1-p),math.log(p)],dtype=torch.float64)
    activate(root,graph); result=solve(root,graph)
    worlds=[]
    for z in itertools.product([0,1],repeat=8):
        objective=sum(math.log(p if v else 1-p) for p,v in zip(probs,z))
        answer=any(z[i] and z[4+i*2+j] and z[4+j*2+k] and z[2+k]
                   for i,j,k in itertools.product(range(2),repeat=3))
        worlds.append((objective,answer))
    objective,answer=max(worlds)
    assert result['hypotheses']['ELC0']==answer
    assert abs(result['objective']-objective)<1e-5


def test_D08_native_repeat_fresh_models(native_license):
    with Graph('repeat') as graph:
        scene=Concept(name='scene'); item=Concept(name='item'); scene.contains(item)
        flag=item(name='flag'); execute(existsL(flag('x')))
    root=DataNode(instanceID=0,ontologyNode=scene)
    node=DataNode(instanceID=0,ontologyNode=item);root.addChildDataNode(node)
    for expected in (True,False,True):
        p=.9 if expected else .1
        node.attributes['<flag>']=torch.tensor([math.log(1-p),math.log(p)])
        node.attributes.pop('<flag>/local/softmax',None)
        if not root.getChildDataNodes() or len(root.getChildDataNodes())==1:
            activate(root,graph)
        else: root.inferLocal(keys=('softmax',))
        result=solve(root,graph)
        assert result['hypotheses']['ELC0']==expected


def test_D10_native_constraint_name_scaling(native_license, record_property):
    import gurobipy as gp
    from domiknows.solver.gurobiILPBooleanMethods import gurobiILPBooleanProcessor
    processor=gurobiILPBooleanProcessor()
    with gp.Model() as model:
        model.Params.OutputFlag=0
        variables=[model.addVar(vtype=gp.GRB.BINARY,name=f'v{i}') for i in range(1024)]
        model.update()
        for n in (64,256,1024):
            start=time.perf_counter()
            result=processor.preprocessLogicalMethodVar(variables[:n],'COUNT','count')
            elapsed=time.perf_counter()-start
            record_property(f'name_seconds_{n}',elapsed)
            assert result['No_of_ilp']==n
            assert len(result['varName'])<=254
    # Timing is diagnostic, not a machine-dependent hard failure threshold.


def test_D10_native_repeated_and_construction(native_license, record_property):
    """Measure synchronization and duplicate auxiliary creation, not correctness loss."""
    import gurobipy as gp
    from domiknows.solver.gurobiILPBooleanMethods import gurobiILPBooleanProcessor
    processor=gurobiILPBooleanProcessor()
    with gp.Model() as model:
        model.Params.OutputFlag=0
        a=model.addVar(vtype=gp.GRB.BINARY,name='a')
        b=model.addVar(vtype=gp.GRB.BINARY,name='b')
        model.update()
        class Counter:
            updates=0
            def __getattr__(self,name): return getattr(model,name)
            def update(self): self.updates+=1; model.update()
        proxy=Counter()
        start=time.perf_counter()
        outputs=[processor.andVar(proxy,a,b) for _ in range(25)]
        model.update()
        record_property('elapsed_seconds',time.perf_counter()-start)
        record_property('update_calls',proxy.updates)
        record_property('auxiliary_variables',model.NumVars-2)
        record_property('constraints',model.NumConstrs)
        assert len(outputs)==25


def test_miota_answer_is_a_zero_one_list_that_follows_the_solved_world(native_license):
    """Three candidate datanodes: the answer is the documented candidate-aligned 0/1
    list, and a hard constraint changes it (no constraint: thresholded scores;
    notL(flag): nothing can be selected)."""
    def answer(forbid):
        from domiknows.graph import Graph as G, Concept as C, Relation as R
        from domiknows.solver import ilpOntSolverFactory
        G.clear(); C.clear(); R.clear(); ilpOntSolverFactory.clear()
        DataNode.collectedConceptsAndRelations = None
        with Graph('miota_format') as graph:
            scene = Concept(name='scene'); item = Concept(name='item'); scene.contains(item)
            flag = item(name='flag')
            if forbid:
                notL(flag('x'))
            execute(miotaL(flag('x'), threshold=.5, hard=False))
        root = DataNode(instanceID=0, ontologyNode=scene)
        for index, probability in enumerate((.9, .2, .7)):
            child = DataNode(instanceID=index, ontologyNode=item)
            child.attributes['<flag>'] = torch.log(torch.tensor([1 - probability, probability]))
            root.addChildDataNode(child)
        activate(root, graph)
        return solve(root, graph)['hypotheses']['ELC0']
    assert answer(forbid=False) == [1, 0, 1]
    assert answer(forbid=True) == [0, 0, 0]
