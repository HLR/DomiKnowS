import copy
import json
from pathlib import Path
import pytest

from ..generalization.rlbench_protocol import planning_context,plan_validity,validate_calibration
from ..generalization.libero_protocol import sha


def public():
    return {'instruction':'put the item in the cupboard','entities':['entity_000','entity_001'],
            'movable':['entity_000'],'positions':{'entity_000':[0,0,0],'entity_001':[1,0,0]},
            'images':['/private-storage/task_variation_1/raw.png']}


def test_context_excludes_storage_paths_task_ids_and_scoring_fields_from_prompt():
    first=public();first.update(task='oracle',reference=[1],success_sensor=[2])
    second=copy.deepcopy(first);second.update(task='different oracle',reference=[9],success_sensor=[10])
    assert planning_context(first)==planning_context(second)
    context=planning_context(first)
    assert 'private-storage' not in context['instruction'] and 'oracle' not in context['instruction']
    assert context['entity_table']==('entity_000','entity_001')
    assert context['generation_static_fixture_ids']==[1]
    assert 'Declared dynamic entities: ["entity_000"]' in context['instruction']


def test_supported_plan_validity_uses_observations_only_and_preserves_pointers():
    pick=lambda i:{'name':'pick','params':{'target_entity_name':i}}
    place=lambda i:{'name':'place','params':{'target_container_name':i}}
    assert plan_validity([pick(0),place(1)],public())['valid_plan']
    for plan in [[pick(1),place(0)],[pick(0),place(0)],[place(1)],[pick(True)],[pick(9)],
                 [{'name':'open','params':{'target_entity_name':1}}]]:
        assert not plan_validity(plan,public())['valid_plan']
    for plan in [None,{},'pick',[{'name':'pick','params':None}]]:
        assert not plan_validity(plan,public())['valid_plan']


def test_planning_requires_matching_full_calibration_before_loading_a_model(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    for name in ['rlbench_bridge.py','rlbench_executor.py','rlbench_campaign.py']:(tmp_path/name).write_text(name)
    gate={'mode':'calibrate','episodes':100,'success_rate':.89,'manifest_sha256':sha(manifest),
          'bridge_sha256':sha(tmp_path/'rlbench_bridge.py'),'executor_sha256':sha(tmp_path/'rlbench_executor.py'),
          'runner_sha256':sha(tmp_path/'rlbench_campaign.py')}
    report=tmp_path/'calibration.json';report.write_text(json.dumps(gate))
    with pytest.raises(ValueError,match='90/100'):validate_calibration(manifest,report,tmp_path)
    gate['success_rate']=.9;report.write_text(json.dumps(gate))
    assert validate_calibration(manifest,report,tmp_path)==gate
    manifest.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='manifest'):validate_calibration(manifest,report,tmp_path)
    manifest.write_text('{}');(tmp_path/'rlbench_bridge.py').write_text('changed')
    with pytest.raises(ValueError,match='source'):validate_calibration(manifest,report,tmp_path)
