import json
import pytest
from ...protocol import freeze,sha,read
from .report import report


def fixture(root):
    variants=['baseline','raw','contract','raw-contract','aligned','aligned-contract'];config={'conditions':variants}
    for key in ('observations','features','libero_manifest','archive_predictions'):
        path=root/key;path.write_text('{}');config[key]=str(path);config[key+'_sha256']=sha(path)
    config['archive_execution']=str(root/'archive')
    ids=[f'{suite}/task{task}/state{state}' for suite in ('libero_object','libero_spatial') for task in range(10) for state in range(5)]
    plan=[{'name':'pick','params':{'target_entity_name':0}},{'name':'place','params':{'target_container_name':1}}]
    for variant in variants:
        folder=root/'planning'/variant;folder.mkdir(parents=True)
        rows=[{'episode_id':i,'suite':i.split('/')[0],'prediction':plan,'input_hash':'matched','observation_hash':'same'} for i in ids]
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        freeze(folder/'summary.json',{'overall':{'valid_plans':100},'suites':{
            suite:{'grounder_source_correct':20,'planner_source_correct':10} for suite in ('libero_object','libero_spatial')}})
        execution=root/('archive' if variant=='baseline' else 'execution/'+variant);execution.mkdir(parents=True)
        native=[{'episode_id':i,'prediction':plan,'success':variant!='baseline' and j==0,
                 'termination_reason':'completed'} for j,i in enumerate(ids)]
        (execution/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in native))
    freeze(root/'planning/baseline/baseline_gate.json',{'passed':True,'episodes':100,'mismatches':[]})
    return config


def test_report_keeps_native_abstract_scope_and_pairing_separate(tmp_path):
    config=fixture(tmp_path);report(tmp_path,config);result=read(tmp_path/'results.json')
    raw=result['conditions']['raw']['cohorts']
    assert raw['all']['native_successes']==1 and raw['libero_object']['native_successes']==1
    assert raw['libero_spatial']['native_successes']==0
    assert raw['all']['paired_vs_baseline']['wins']==1 and raw['all']['paired_vs_baseline']['task_clusters']==20
    assert result['conditions']['baseline']['execution_reused'] is True
    assert result['conditions']['raw']['execution_reused'] is False


@pytest.mark.parametrize('which',['physical_plan','rendered_input'])
def test_report_rejects_unmatched_physical_plan_or_constraint_prompt(tmp_path,which):
    config=fixture(tmp_path)
    path=tmp_path/('execution/raw/episodes.jsonl' if which=='physical_plan' else 'planning/contract/predictions.jsonl')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if which=='physical_plan':rows[0]['prediction']=[]
    else:rows[0]['input_hash']='changed'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='different predictions|changed actual model inputs'):report(tmp_path,config)
    assert not (tmp_path/'results.json').exists()
