from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from domiknows.program.gradient_manager import GradientConflictManager
from .batch_gradients import BatchGradientManager
from .source_transform import transform
from .source_views import allocate, observed_views, native_reset_configuration
from .protocol import sha


def test_provenance_hash_supports_scalar_buffers_and_detects_changes():
    from .train import tensor_digest
    model=torch.nn.Module();model.register_buffer('counter',torch.tensor(0))
    before=tensor_digest(model)
    assert tensor_digest(model)==before
    model.counter.add_(1)
    assert tensor_digest(model)!=before


def test_accumulated_pcgrad_matches_full_domain_minibatch_not_last_example():
    model=torch.nn.Linear(2,1,bias=False)
    first=torch.optim.SGD(model.parameters(),lr=.1)
    expected=GradientConflictManager(model,first); actual=BatchGradientManager(model,first)
    samples={'eai':[torch.tensor([[3.,-1.]]),torch.tensor([[-1.,4.]])],
             'vlabench':[torch.tensor([[-4.,2.]]),torch.tensor([[1.,-3.]])]}
    expected.begin_step()
    for domain,values in samples.items(): expected.capture(domain,sum(model(v).sum() for v in values)/2)
    wanted=[g.clone() for g in expected.resolve_gradients()];expected.abort_step()
    actual.begin_step()
    for domain,values in samples.items():
        for value in values:actual.capture(domain,model(value).sum()/2)
    result=actual.resolve_gradients()
    assert all(torch.allclose(a,b) for a,b in zip(result,wanted))
    actual.abort_step()


@dataclass(frozen=True)
class Example:
    episode_id:str='case'
    entities:tuple=('apple','bowl','table')
    operation_sequence:tuple=({'name':'pick','params':{'target_entity_name':0,'duration':2}},
                              {'name':'place','params':{'target_container_name':'bowl'}})
    image_paths:tuple=()
    segmented_image_paths:tuple=()
    task_target_container:object=1
    task_target_entity:object='apple'
    visible_entities:tuple=(0,'bowl')
    object_candidates:tuple=(('pick','subject',(0,)),)


def test_source_three_cycle_preserves_named_facts_and_only_remaps_pointers(tmp_path):
    scene=tmp_path/'scene';scene.write_bytes(b'private simulator configuration')
    image=tmp_path/'image';image.write_bytes(b'new marker image')
    record={'episode_id':'case','canonical_entities':['apple','bowl','table'],
        'scene_path':str(scene),'scene_sha256':sha(scene),'canonical_image':str(image),
        'files':{str(image):sha(image)},'augmentations':[{'new_to_old':[2,0,1],'image':str(image)}]}
    original=Example()
    value=transform(original,record,epoch=0,augmented=True,pointer_keys={'target_entity_name','target_container_name'})
    assert value.entities==('table','apple','bowl')
    assert value.operation_sequence[0]['params']=={'target_entity_name':1,'duration':2}
    assert value.operation_sequence[1]['params']['target_container_name']=='bowl'
    assert value.task_target_container==2 and value.task_target_entity=='apple'
    assert value.visible_entities==(1,'bowl') and value.object_candidates[0][2]==(1,)
    assert original.operation_sequence[0]['params']['target_entity_name']==0
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='marker image changed'):
        transform(original,record,epoch=0,augmented=False,pointer_keys=set())


def test_source_allocation_covers_frozen_training_validation_and_excludes_test():
    groups={'train':[f'train{i}' for i in range(3600)],'validation':[f'valid{i}' for i in range(450)],
            'test':[f'test{i}' for i in range(450)]}
    rows=allocate({'seed':42,'splits':groups})
    assert len(rows)==4050 and not any(r['episode_id'].startswith('test') for r in rows)
    groups['validation'][0]=groups['train'][0]
    with pytest.raises(ValueError,match='overlap'):allocate({'seed':42,'splits':groups})


def test_saved_named_background_is_restored_without_rewriting_source_or_goals():
    original={'task':{'scene':{'name':'living_room_0','position':[-2,-2.5,0]},
                      'conditions':{'contain':{'entities':['apple']}}}}
    restored,overrides=native_reset_configuration(original)
    assert restored['task']['scene']['use_default_scene'] is False
    assert overrides=={'scene.use_default_scene':False}
    assert 'use_default_scene' not in original['task']['scene']
    assert restored['task']['conditions']==original['task']['conditions']
    restored,_=native_reset_configuration({'task':{'scene':{'name':'default'}}})
    assert restored['task']['scene']['use_default_scene'] is True
    with pytest.raises(ValueError,match='contradicts'):
        native_reset_configuration({'task':{'scene':{'name':'living_room_0','use_default_scene':True}}})


def test_native_render_uses_canonical_inventory_order_and_ignores_hidden_fields():
    a,b=object(),object()
    env=SimpleNamespace(task=SimpleNamespace(entities={'alpha':SimpleNamespace(geoms=[a]),
                                                     'zebra':SimpleNamespace(geoms=[b])}),
                        physics=SimpleNamespace(bind=lambda g:SimpleNamespace(element_id=2 if g is a else 7)))
    rgb=np.full((4,480,480,3),100,dtype=np.uint8)
    seg=np.full((4,480,480,2),-1,dtype=int)
    seg[:,10:14,20:24,0]=7;seg[:,30:34,40:44,0]=2
    raw,centers=observed_views(env,('zebra','alpha'),{'rgb':rgb,'segmentation':seg,
                                                    'hidden_goal':'alpha','reference_plan':['alpha']})
    assert raw.size==(960,960) and centers[:2]==[(0,21,11),(1,41,31)]
    with pytest.raises(ValueError,match='complete canonical'):
        observed_views(env,('alpha',),{'rgb':rgb,'segmentation':seg})
    # The canonical source table also includes the declared background scene.
    # It is stored separately from task.entities, so preserve its pointer and
    # bind its actual scene geometry rather than deleting the canonical entry.
    env.task.scene=SimpleNamespace(name='living_room_0',mjcf_model=SimpleNamespace(find_all=lambda kind:[a]))
    _,centers=observed_views(env,('zebra','alpha','living_room_0'),{'rgb':rgb,'segmentation':seg})
    assert centers[:3]==[(0,21,11),(1,41,31),(2,41,31)]
