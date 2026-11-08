from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ..generalization.libero_protocol import specification, generation_input, reference_plan, abstract_score
from ..generalization.shared_constraints import bind_shared, rule_trace, SHARED_RULES
from ..generalization.fixed_executor import bounded_osc_action, FixedExecutor
from ..utils.generalization_protocol import freeze_json
from .test_joint_pipeline import joint_fixture as base_joint_fixture, make_planner, TinyController


@pytest.fixture
def joint_fixture():
    return base_joint_fixture.__wrapped__()


def test_bddl_goal_and_interest_cannot_change_generation_inputs(tmp_path):
    text = '''(define (problem Example) (:domain robosuite)
      (:language Pick the bowl and put it on the plate)
      (:objects bowl_1 bowl_2 - bowl plate_1 - plate) (:fixtures table_1 - table)
      (:obj_of_interest bowl_1 plate_1) (:init (On bowl_1 table_1) (On bowl_2 table_1))
      (:goal (And (On bowl_1 plate_1))))'''
    path = tmp_path/'task.bddl'; path.write_text(text)
    original = specification(path)
    path.write_text(text.replace('(:goal (And (On bowl_1', '(:goal (And (On bowl_2')
                        .replace('(:obj_of_interest bowl_1', '(:obj_of_interest bowl_2'))
    changed = specification(path)
    assert generation_input(original, ['raw.png', 'ids.png']) == generation_input(changed, ['raw.png', 'ids.png'])
    assert original['entities'] == sorted(original['entities'])
    assert reference_plan(original) != reference_plan(changed)
    assert abstract_score(reference_plan(original), original)['goal_success']
    assert not abstract_score(reference_plan(original), changed)['goal_success']


def test_unsupported_external_goal_is_not_translated_to_another_skill(tmp_path):
    p = tmp_path/'task.bddl'
    p.write_text('(define (problem X) (:language Open drawer) (:objects drawer_1 - drawer) (:goal (Open drawer_1)))')
    with pytest.raises(ValueError, match='unsupported external goal'):
        specification(p)


def test_bddl_goal_region_names_cannot_leak_target_role(tmp_path):
    p = tmp_path/'object.bddl'
    p.write_text('''(define (problem X) (:language Put soup in basket)
      (:regions (target_object_region (:target floor) (:ranges ((0 0 1 1))))
                (contain_region (:target basket_1)))
      (:objects soup_1 - soup basket_1 - basket) (:fixtures floor - floor)
      (:init (On soup_1 floor_target_object_region))
      (:goal (And (In soup_1 basket_1_contain_region))))''')
    spec = specification(p)
    public = generation_input(spec, [])
    assert 'target_object_region' not in json.dumps(public)
    assert public['initial_relations'] == [['On', 'entity_002', 'entity_001']]
    assert public['entities'] == ['entity_000', 'entity_001', 'entity_002']
    assert 'soup_1' not in json.dumps(public)
    assert reference_plan(spec)[1]['params']['target_container_name'] == spec['entities'].index('basket_1')
    assert abstract_score(reference_plan(spec), spec)['goal_success']


def test_shared_rules_keep_legitimate_rearrangement_and_reject_self_placement(joint_fixture):
    _, runtime = joint_fixture
    from test_regr.VLABenchAgentInterface.plan_codec import plan_to_tokens
    from test_regr.VLABenchAgentInterface.world_graph import build_vlabench_world_graph
    from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary
    vocab = PlanVocabulary.from_world(build_vlabench_world_graph(), max_entities=4)
    context = {'entity_table': ('apple', 'plate', 'basket')}
    dfa = bind_shared(vocab, 6, context)
    assert {r.name for r in dfa.program.rules} == SHARED_RULES
    def labels(ops):
        return [vocab.label_for_token(t) for t in plan_to_tokens(ops, context['entity_table'], world=runtime.world.vlabench)]
    pick = lambda i: {'name': 'pick', 'params': {'target_entity_name': i}}
    place = lambda i: {'name': 'place', 'params': {'target_container_name': i}}
    # Earlier destinations may be moved, and earlier objects may be picked again.
    assert dfa.accepts(labels([pick(0), place(1), pick(1), place(2)]))
    assert dfa.accepts(labels([pick(0), place(1), pick(0), place(2)]))
    assert not dfa.accepts(labels([pick(0), place(0)]))
    trace = rule_trace(dfa, labels([pick(0), place(1)]))
    assert trace['triggered_rejections'].get('anti_self_placement', 0) > 0


def test_overlap_inventory_uses_actual_runtime_manifest(joint_fixture):
    from ..generalization.overlap_report import overlap_report
    _, runtime = joint_fixture
    report = overlap_report(runtime.knowledge_manifest, {'vlabench:none':{'examples':450},
        'vlabench:full-metadata':{'examples':450}, 'libero:shared-verified':{'examples':100}})
    assert len(report['families']) == 8
    assert report['verified_transfer']['whitelisted_world_rule_count'] == 5
    assert report['inventories']['eai_world']
    assert report['measured']['vlabench:none']['groundable'] == 'no generation constraints'
    assert 'privileged' in report['measured']['vlabench:full-metadata']['groundable']


def test_osc_translation_rotation_and_gripper_bounds():
    action = bounded_osc_action([.2, -.2, .01], [0, 0, 0], [.2, -.2, 0], [.05]*3+[.1]*3, 1)
    assert np.allclose(action, [1, -1, .2, 1, -1, 0, 1])
    assert bounded_osc_action([0]*3, [0]*3, [0]*3, [.1]*6, -1)[-1] == -1
    with pytest.raises(ValueError):
        bounded_osc_action([np.nan, 0, 0], [0]*3, [0]*3, [.1]*6, 1)


def test_fixed_executor_never_reads_native_goal_and_rejects_invalid_ids():
    class Env:
        def check_success(self):
            pytest.fail('executor must not query the native goal')
    executor = FixedExecutor(Env(), ['bowl', 'plate'])
    result = executor.execute([{'name':'pick','params':{'target_entity_name':2}}], {})
    assert result['termination_reason'] == 'invalid entity pointer'
    assert result['steps'] == 0


def test_fixed_executor_uses_predicted_object_and_never_substitutes_after_failure(monkeypatch):
    executor = FixedExecutor(object(), ['wanted', 'distractor'])
    attempted = []
    def failed_pick(name):
        attempted.append(name)
        raise RuntimeError('grasp failed')
    monkeypatch.setattr(executor, 'pick', failed_pick)
    result = executor.execute([{'name':'pick', 'params':{'target_entity_name':1}}], {})
    assert attempted == ['distractor']
    assert result['termination_reason'] == 'grasp failed'
    result = executor.execute([{'name':'open', 'params':{'target_entity_name':0}}], {})
    assert result['termination_reason'] == 'unsupported skill; no replacement or assistance'
    executor.horizon = 0
    import sys
    monkeypatch.setitem(sys.modules, 'robosuite.utils', SimpleNamespace(transform_utils=None))
    with pytest.raises(TimeoutError, match='horizon'):
        executor.tick([0., 0., 0.], 1)


def test_quantized_bias_cast_requires_exact_ownership_match():
    from ..checkpoint import _restore_quantized_bias_representation
    Linear4bit = type('Linear4bit', (torch.nn.Module,), {})
    class Planner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Linear4bit()
            self.layer.bias = torch.nn.Parameter(torch.ones(2), requires_grad=False)
            self.layer.compute_dtype = torch.bfloat16
        def compute_parameter_ownership_checksum(self):
            return str(self.layer.bias.dtype)
    planner = Planner()
    _restore_quantized_bias_representation(planner, {'compatibility': {'parameter_ownership_checksum':'other-drift'}})
    assert planner.layer.bias.dtype == torch.float32
    initial = {'compatibility': {'parameter_ownership_checksum':'torch.float32'}}
    _restore_quantized_bias_representation(planner, initial)
    assert planner.layer.bias.dtype == torch.float32 and 'representation_restoration' not in initial
    trained = {'compatibility': {'parameter_ownership_checksum':'torch.bfloat16'}}
    _restore_quantized_bias_representation(planner, trained)
    assert planner.layer.bias.dtype == torch.bfloat16
    assert trained['representation_restoration'] == ['layer.bias']


def test_frozen_reset_restores_osc_joint_reference(tmp_path):
    from ..generalization.libero_campaign import reset_episode
    state = tmp_path/'state.npy'; np.save(state, [9., 8., 7.])
    references = []; seeds = []
    controller = SimpleNamespace(update_initial_joints=lambda q: references.append(q.copy()))
    robot = SimpleNamespace(controller=controller, _ref_joint_pos_indexes=[0, 2])
    class Env:
        env = SimpleNamespace(robots=[robot])
        sim = SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(3)))
        def seed(self, value):seeds.append(value)
        def reset(self):self.sim.data.qpos[:] = np.random.normal(size=3)
        def set_init_state(self, value):
            self.sim.data.qpos[:] = value
            return {'observation':'frozen'}
    record = {'episode_id':'fixed-state05', 'state':str(state)}
    assert reset_episode(Env(),record) == reset_episode(Env(),record)
    assert seeds[0] == seeds[1]
    assert all(np.array_equal(q, [9.,7.]) for q in references)


def test_frozen_manifest_rejects_changed_checkpoint_or_probe_settings(tmp_path):
    p = tmp_path/'settings.json'
    freeze_json(p, {'checkpoint':'a', 'policy':'none'})
    freeze_json(p, {'checkpoint':'a', 'policy':'none'})
    with pytest.raises(ValueError, match='differs'):
        freeze_json(p, {'checkpoint':'b', 'policy':'none'})
    with pytest.raises(ValueError, match='differs'):
        freeze_json(p, {'checkpoint':'a', 'policy':'shared-verified'})


def test_internal_option_does_not_certify_clean_data_or_accept_renamed_copy(tmp_path, monkeypatch):
    from test_regr.VLABenchAgentInterface import dataset_protocol as protocol
    from test_regr.VLABenchAgentInterface.test_dataset_protocol import episode
    root = tmp_path/'official'; episode(root, 'ep')
    (root/'dataset_role.json').write_text(json.dumps({'role':'evaluation','provenance':{'repo_id':'VLABench/vlm_evaluation_v1.0'}}))
    monkeypatch.setattr(protocol, 'OFFICIAL_EVALUATION', root)
    with pytest.raises(ValueError, match='Official VLABench'):
        protocol.guard_training_dataset(root)
    result = protocol.guard_training_dataset(root, tmp_path/'internal', allow_internal_evaluation_training=True)
    assert result['official_evaluation_training_seen'] is True
    assert result['role'] == 'internal_evaluation_training'
    import shutil
    shutil.copytree(root, tmp_path/'renamed')
    with pytest.raises(ValueError, match='canonical'):
        protocol.guard_training_dataset(tmp_path/'renamed', allow_internal_evaluation_training=True)


def test_metadata_observation_mode_reaches_training_loader(joint_fixture, monkeypatch):
    from .. import main
    examples, runtime = joint_fixture
    monkeypatch.setattr(main, 'load_eai_dataset', lambda **kwargs: examples)
    monkeypatch.setattr(main, 'build_joint_world_graph', lambda *a: runtime.world)
    def load(root, **kwargs):
        assert kwargs['observation_mode'] == 'metadata-assisted'
        raise RuntimeError('metadata loader inspected')
    monkeypatch.setattr(main, 'load_planning_examples', load)
    args = main.build_parser().parse_args(['train-agent', '--ce-dfa-mass-only', '--knowledge', 'metadata', '--observation-mode', 'metadata-assisted'])
    with pytest.raises(RuntimeError, match='loader inspected'):
        main._prepare(args, torch.device('cpu'))


def test_initialization_snapshot_precedes_eai_warmup(joint_fixture, monkeypatch, tmp_path):
    from .. import main
    from ..utils import generalization_protocol
    from test_regr.VLABenchAgentInterface import dataset_protocol
    examples, runtime = joint_fixture
    planner = make_planner(runtime); events = []
    monkeypatch.setattr(dataset_protocol, 'guard_training_dataset', lambda *a, **k: None)
    monkeypatch.setattr(main, '_prepare', lambda *a: (runtime, examples, examples,
        {'train':[], 'validation':[], 'test':[]}, {'train':[], 'validation':[], 'test':[]}))
    monkeypatch.setattr(main.JointQwenVLPlanner, 'from_pretrained', lambda **k: planner)
    monkeypatch.setattr(main, '_controller', lambda *a: TinyController())
    monkeypatch.setattr(main, '_control_loaders', lambda *a: {'train':[]})
    monkeypatch.setattr(main, 'load_control_task_instructions', lambda *a: {})
    monkeypatch.setattr(generalization_protocol, 'freeze_inputs', lambda *a: 'frozen-hash')
    monkeypatch.setattr(generalization_protocol, 'freeze_model_source', lambda *a: 'model-hash')
    def save(path, **kwargs):
        assert kwargs['stage'] == 'initialization' and kwargs['epoch'] == -1
        assert kwargs['metrics']['training_updates'] == 0
        events.append('saved'); return path
    monkeypatch.setattr(main, 'save_joint_checkpoint', save)
    def warmup(self, *a, **k):
        assert events == ['saved']
        raise RuntimeError('warmup boundary')
    monkeypatch.setattr(main.JointSolverPOIProgram, 'train_eai_warmup', warmup)
    args = main.build_parser().parse_args(['train-agent','--ce-only','--stage1-epochs','0',
        '--save-initialization','--stage1-eai-warmup-steps','1','--device','cpu','--output',str(tmp_path)])
    with pytest.raises(RuntimeError, match='warmup boundary'):
        main.command_train_agent(args)
