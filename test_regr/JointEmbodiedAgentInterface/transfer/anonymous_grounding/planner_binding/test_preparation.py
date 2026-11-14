import pytest
from ...protocol import freeze,sha
from .freeze_external import carry_calibration


def test_libero_only_observations_keep_legacy_extractor_keys_without_source_test_data():
    from .prepare_inputs import observation_manifest
    rows=[{'episode_id':'libero_object/task/state40'}];manifest=observation_manifest(rows)
    assert manifest['vlabench']==[] and manifest['libero']==rows and manifest['references_read'] is False
    assert [(dataset,r) for dataset in ('vlabench','libero') for r in manifest[dataset]]==[('libero',rows[0])]


def test_scene_fingerprint_covers_arena_table_and_ignores_robot_state():
    from types import SimpleNamespace as S
    from xml.etree.ElementTree import Element
    import numpy as np
    from .freeze_external import scene
    data=S(body_xpos=np.zeros((3,3)),body_xquat=np.zeros((3,4)))
    env=S(env=S(objects_dict={'package':S(root_body='object_body')},fixtures_dict={},
        mujoco_arena=S(table_body=Element('body',{'name':'table'}))),
        sim=S(data=data,model=S(body_names=['object_body','table','robot'],geom_names=[],
                               body_name2id={'object_body':0,'table':1,'robot':2}.__getitem__)))
    spec={'entities':['main_table','package'],'entity_types':{'main_table':'table','package':'box'}}
    before=scene(env,spec);data.body_xpos[2,0]=3;assert scene(env,spec)==before
    data.body_xpos[1,2]=.8;assert scene(env,spec)!=before


def test_calibration_reuse_requires_identical_records_and_native_code(tmp_path):
    bridge=tmp_path/'libero_campaign.py';executor=tmp_path/'fixed_executor.py'
    bridge.write_text('frozen bridge');executor.write_text('frozen executor')
    records=[{'episode_id':str(i),'split':'calibration','state_sha256':str(i)} for i in range(100)]
    before={'episodes':records+[{'episode_id':'old','split':'evaluation'}]}
    after={'episodes':[{'episode_id':'fresh','split':'evaluation'}]+records}
    old=tmp_path/'old.json';new=tmp_path/'new.json';freeze(old,before);freeze(new,after)
    gate={'manifest_sha256':sha(old),'success_rate':.96,'executor_source_sha256':sha(executor),'bridge_source_sha256':sha(bridge)}
    reused=carry_calibration(before,after,gate,old,new,bridge)
    assert reused['calibration_reused'] and reused['success_rate']==.96 and reused['original_manifest_sha256']==sha(old)
    executor.write_text('changed')
    with pytest.raises(ValueError,match='changed'):carry_calibration(before,after,gate,old,new,bridge)
    executor.write_text('frozen executor');after['episodes'][-1]={**after['episodes'][-1],'state_sha256':'changed'}
    with pytest.raises(ValueError,match='calibration states'):carry_calibration(before,after,gate,old,new,bridge)


def test_settled_audit_rejects_physical_overlap_despite_different_robot_states():
    from .settled_audit import verify
    rows=[{'episode_id':str(i),'scene_sha256':str(i),'input_sha256':str(i),'cohort':'fresh' if i>=800 else 'reserved'} for i in range(900)]
    assert verify(rows)['passed']
    rows[-1]['scene_sha256']=rows[0]['scene_sha256']
    with pytest.raises(ValueError,match='settled overlap'):verify(rows)
