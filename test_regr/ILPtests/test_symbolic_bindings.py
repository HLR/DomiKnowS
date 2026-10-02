"""Exercise real compiled ILP-mode binding with a Boolean recording backend.

The backend evaluates fixed truth assignments, not Gurobi optimization. No
library code is replaced; flags match ILP (loss/sample/verify/circuit=False).
Native Gurobi counterparts are in test_native.py and remain license-gated.
"""
import itertools
import logging
from types import SimpleNamespace
import torch
import pytest
from domiknows.graph import Concept, Graph
from domiknows.graph.dataNode import DataNode
from domiknows.graph.logicalConstrain import andL, ifL, existsL
from domiknows.solver.compiled.formula import CompiledModeExecutor


class BooleanRecorder:
    grad = False
    def __init__(self): self.calls=[]
    def ifVar(self, model, a, b, onlyConstrains=False):
        self.calls.append(('if',a,b))
        return float(not bool(a) or bool(b))
    def andVar(self, model, *values, onlyConstrains=False):
        self.calls.append(('and',*values))
        return float(all(values))
    def countVar(self, model, *values, onlyConstrains=False, limitOp='>=', limit=1, **kwargs):
        self.calls.append(('count',*values))
        total=sum(values)
        return float({'>=':total>=limit,'<=':total<=limit,'==':total==limit}[limitOp])


def populate_pair_scene(graph,scene,item,pair,arg1,arg2,unary,predicates):
    root=DataNode(instanceID=0,ontologyNode=scene);root.current_device='cpu'
    nodes=[]
    for i in range(2):
        dn=DataNode(instanceID=i,ontologyNode=item);root.addChildDataNode(dn);nodes.append(dn)
        for concept,values in unary:
            put(dn,concept,values[i])
    for i,j in itertools.product(range(2),repeat=2):
        dn=DataNode(instanceID=2*i+j,ontologyNode=pair)
        dn.addRelationLink(arg1.name,nodes[i]);dn.addRelationLink(arg2.name,nodes[j]);root.addChildDataNode(dn)
        for concept,values in predicates:
            put(dn,concept,values[2*i+j])
    return root


def put(dn,concept,truth):
    dn.attributes[f'<{concept.name}>']=torch.tensor([1.-truth,truth])
    dn.attributes[f'<{concept.name}>/local/softmax']=torch.tensor([1.-truth,truth])
    dn.attributes[f'<{concept.name}>/xP']={0:{0:float(truth)}}


def compiled(lc,graph,root):
    solver=SimpleNamespace(myGraph={graph},myLogger=logging.getLogger('repro'))
    executor=CompiledModeExecutor(solver)
    recorder=BooleanRecorder()
    output=executor.construct(lc,recorder,root,key='/xP',headLC=False,
                              model=SimpleNamespace(update=lambda:None),p=0,
                              loss=False,sample=False,verify=False,circuit=False)
    return output,recorder.calls


@pytest.mark.parametrize('reverse',[False,True])
def test_D03_compiled_inverse_bindings_fixed_truth(reverse):
    with Graph('inverse_bindings') as graph:
        scene=Concept(name='scene');item=Concept(name='item');scene.contains(item)
        pair=Concept(name='pair');arg1,arg2=pair.has_a(arg1=item,arg2=item)
        left=pair(name='left');right=pair(name='right')
        rule=ifL(left('x','y'),right('y','x') if reverse else right('x','y'))
    root=populate_pair_scene(graph,scene,item,pair,arg1,arg2,[],
                             [(left,[0,1,0,0]),(right,[0,0,1,0] if reverse else [0,1,0,0])])
    output,calls=compiled(rule,graph,root)
    implication_calls=[row for row in calls if row[0]=='if']
    assert len(implication_calls)==4, (output,calls)
    assert all((not bool(a)) or bool(b) for _,a,b in implication_calls), calls


@pytest.mark.parametrize('hops',[1,2,3])
def test_D04_compiled_join_retains_known_existential_witness(hops):
    with Graph('join_bindings') as graph:
        scene=Concept(name='scene');item=Concept(name='item');scene.contains(item)
        a=item(name='a');b=item(name='b');pair=Concept(name='pair')
        arg1,arg2=pair.has_a(arg1=item,arg2=item);r=pair(name='r')
        names=list('xyzw')[:hops+1]
        rule=existsL(andL(a(names[0]),*[r(x,y) for x,y in zip(names,names[1:])],b(names[-1])))
    root=populate_pair_scene(graph,scene,item,pair,arg1,arg2,
                            [(a,[1,0]),(b,[0,1] if hops%2 else [1,0])],[(r,[0,1,1,0])])
    output,calls=compiled(rule,graph,root)
    def leaves(value):
        if isinstance(value,(tuple,list)):
            return [x for item in value for x in leaves(item)]
        return [value]
    # x=0,y=1 is an explicit satisfying witness.
    assert 1.0 in leaves(output[0]), (output,calls)
