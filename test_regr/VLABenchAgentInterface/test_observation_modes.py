"""Visual benchmark scoring is distinct from metadata-assisted verification."""
from dataclasses import replace
import json

import pytest
import torch

from .dataset import load_planning_examples, load_scene_planning_examples
from .test_reflection_observations import components, supplied, write_episode
from .test_qa_and_pick_history import accepted, runtime
from .training import evaluate_planner


def test_visual_default_ignores_answer_metadata_and_sidecars_but_keeps_all_buttons(tmp_path, runtime):
    episode = write_episode(tmp_path, components([.9,.1,.5], ['button0','button1','button2']))
    config_path = episode/'env_config/env_config.json'
    config = json.loads(config_path.read_text())
    config.update(task_target_entity='button2', task_skill_sequence=['pick'], planning_facts={
        'goals': {'positive': [['pressed','button2']]},
        'selections': [{'action':'press','candidates':['button2'],'criterion':'min','scores':{'button2':.5}}],
    })
    config_path.write_text(json.dumps(config))
    # A property sidecar is not even parsed in visual mode.
    (episode/'input/property_observations.json').write_text('not valid JSON')
    visual = load_planning_examples(tmp_path)[0]
    assert visual.planning_observation_profile == 'visual_qa_v1'
    assert visual.task_target_entity is None
    assert visual.task_skill_sequence == ('press',)
    assert 'goals' not in visual.planning_facts
    assert set(visual.planning_facts['selections'][0]['scores'].values()) == {None}
    policy = runtime.dfa_for('vlabench', visual.as_reward_item())
    for i in range(3):
        assert accepted(runtime, policy, [{'name':'press','params':{'target_entity_name':i}}], visual.entities)
    safe = load_scene_planning_examples(tmp_path)[0]
    assert visual.entities == safe.entities
    assert visual.planning_facts == safe.planning_facts
    # Neither hidden target annotations nor novel reference arguments alter inputs.
    config['task']['conditions']['press_button']['target_button'] = 'button1'
    config['task_target_entity'] = 'button1'
    config_path.write_text(json.dumps(config))
    (episode/'output/operation_sequence.json').write_text(json.dumps({'skill_sequence':[
        {'name':'press','params':{'target_entity_name':'reference_only'}}]}))
    changed = load_planning_examples(tmp_path)[0]
    assert changed.entities == visual.entities
    assert changed.planning_facts == visual.planning_facts
    assert 'reference_only' not in changed.entities


def test_metadata_assisted_is_explicit_and_changes_only_property_selection(tmp_path, runtime):
    episode = write_episode(tmp_path, components())
    (episode/'input/property_observations.json').write_text(json.dumps(supplied({'button0':.9,'button1':.1,'button2':.5})))
    visual = load_planning_examples(tmp_path)[0]
    assisted = load_planning_examples(tmp_path, observation_mode='metadata-assisted')[0]
    assert assisted.planning_observation_profile == 'reflection_observations_v1'
    for example, expected in ((visual, {0,1,2}), (assisted, {2})):
        policy = runtime.dfa_for('vlabench', example.as_reward_item())
        assert {i for i in range(3) if accepted(runtime, policy,
            [{'name':'press','params':{'target_entity_name':i}}],example.entities)} == expected
    with pytest.raises(ValueError, match='observation_mode'):
        load_planning_examples(tmp_path, observation_mode='oracle')


def test_all_fourteen_ungrounded_reflection_predictions_are_scored_against_labels(tmp_path, runtime):
    write_episode(tmp_path, components())
    original = load_planning_examples(tmp_path)[0]
    examples = [replace(original, episode_id=f'PhysicsLaw/reflection_qa/example{i}',
        operation_sequence=({'name':'press','params':{'target_entity_name':i % 3}},)) for i in range(14)]
    class VisualPlanner(torch.nn.Module):
        def generate_plan(self, **kwargs):
            # Fixed prediction, independent of each episode's answer label.
            return [{'name':'press','params':{'target_entity_name':0}}]
    result = evaluate_planner(VisualPlanner(), examples, runtime)
    metrics = result['planning_metrics']
    assert result['examples'] == metrics['dimensions']['PhysLaw']['examples'] == 14
    assert result['generation_failures'] == 0
    assert result['valid'] == 1
    assert metrics['overall']['SR'] == 100
    assert metrics['overall']['PR'] == pytest.approx(100 * 5 / 14)
    assert metrics['overall']['score'] == pytest.approx(40 + 60 * 5 / 14)
    assert result['observation_profiles'] == {'visual_qa_v1':14}
    assert result['grounding_coverage']['selection'] == {'unavailable':14}
    assert len(result['observation_reports']) == 14
