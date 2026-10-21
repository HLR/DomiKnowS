from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from .. import eval_planning


@pytest.mark.parametrize('mode', ['visual','metadata-assisted'])
@pytest.mark.parametrize('updated', [False, True])
def test_joint_cli_propagates_observation_mode_and_records_it(tmp_path, monkeypatch, mode, updated):
    checkpoint = tmp_path/'checkpoint.pt'
    checkpoint.write_bytes(b'fixture')
    output = tmp_path/'metrics.json'
    args = eval_planning.build_eval_parser().parse_args([
        '--checkpoint',str(checkpoint),'--device','cpu','--split','all',
        '--output',str(output),'--observation-mode',mode,
    ])
    args.updated_data = updated
    from test_regr.VLABenchAgentInterface import repair_density_labels
    calls = []
    def update(selected):
        calls.append(selected)
        return selected
    monkeypatch.setattr(repair_density_labels, 'updated_density_examples', update)
    assert eval_planning.build_eval_parser().parse_args(['--checkpoint',str(checkpoint)]).observation_mode == 'visual'
    examples = [SimpleNamespace(task='reflection_qa',dimension='PhysLaw')]
    seen = []
    def load(root, *, observation_mode):
        seen.append(observation_mode)
        return examples
    monkeypatch.setattr(eval_planning, 'load_planning_examples', load)
    runtime = SimpleNamespace(domain_scope=lambda domain:nullcontext(), eai_vocabulary=object(), vlabench_vocabulary=object())
    monkeypatch.setattr(eval_planning,'_prepare_eval',lambda args,device:(runtime,{}))
    planner = SimpleNamespace(eval=lambda:None, for_domain=lambda domain:object())
    monkeypatch.setattr(eval_planning,'JointQwenVLPlanner',SimpleNamespace(from_pretrained=lambda **kwargs:planner))
    monkeypatch.setattr(eval_planning,'_controller',lambda *args:object())
    monkeypatch.setattr(eval_planning,'load_joint_checkpoint',lambda *args,**kwargs:{'stage':'stage2','epoch':4})
    monkeypatch.setattr(eval_planning,'_runtime_adapter',lambda *args:object())
    def evaluate(planner, selected, adapter):
        assert selected is examples
        return {'planning_metrics':{'overall':{},'dimensions':{}}}
    monkeypatch.setattr(eval_planning,'evaluate_planner',evaluate)
    report = eval_planning.run_planning_evaluation(args)
    assert seen == [mode]
    assert report['examples_evaluated'] == 1
    assert report['observation_mode'] == json.loads(output.read_text())['observation_mode'] == mode
    assert report['updated_data'] is updated
    assert report['reference_labels'] == ('density-material-button-v1' if updated else 'as-stored')
    assert len(calls) == int(updated)


@pytest.mark.parametrize('mode', ['visual','metadata-assisted'])
@pytest.mark.parametrize('updated', [False, True])
def test_joint_test_split_preparation_uses_selected_observation_mode(monkeypatch, mode, updated):
    from test_regr.EmbodiedAgentInterface import dataset as eai_dataset
    from .. import main, world_graph
    args = eval_planning.build_eval_parser().parse_args(['--checkpoint','unused','--observation-mode',mode])
    args.vlabench_planning_dir = 'unused'
    args.updated_data = updated
    from test_regr.VLABenchAgentInterface import repair_density_labels
    calls = []
    def update(selected):
        calls.append(selected)
        return selected
    monkeypatch.setattr(repair_density_labels, 'updated_density_examples', update)
    monkeypatch.setattr(eai_dataset,'load_eai_dataset',lambda **kwargs:[{'generation_vocab':('<eos>','action')}])
    monkeypatch.setattr(world_graph,'build_joint_world_graph',lambda name:object())
    monkeypatch.setattr(world_graph,'build_joint_runtime',lambda *args,**kwargs:object())
    monkeypatch.setattr(main,'_ordered_union',lambda *args:())
    monkeypatch.setattr(main,'_action_object_constraints',lambda *args:())
    seen = []
    def load(root, *, observation_mode):
        seen.append(observation_mode)
        return [object() for _ in range(20)]
    monkeypatch.setattr(eval_planning,'load_planning_examples',load)
    _, splits = eval_planning._prepare_eval(args, 'cpu')
    assert seen == [mode]
    assert sum(map(len,splits.values())) == 20 and splits['test']
    assert len(calls) == int(updated)


def test_updated_label_cli_is_opt_in_and_has_explicit_alias():
    parser = eval_planning.build_eval_parser()
    assert not parser.parse_args(['--checkpoint','unused']).updated_data
    assert parser.parse_args(['--checkpoint','unused','--updated-data']).updated_data
    assert parser.parse_args(['--checkpoint','unused','--updated-labels']).updated_data
