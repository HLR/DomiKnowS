import ast
import itertools
import json
from types import SimpleNamespace

import pytest

from domiknows.generation.grounded import Operation, PlanningFacts
from ..dataset import load_planning_examples, load_scene_planning_examples
from ..physics_grounding import physics_planning_facts
from ..utils.preserve_reflection_observations import (
    ENTITY, GENERATOR, patch_entity, patch_generator, preserve,
)
from .test_qa_and_pick_history import accepted, runtime
from ..training import evaluate_planner


def components(values=None, buttons=None):
    mirror = {"class": "Mirrors", "name": "mirrors"}
    if values is not None:
        mirror["reflectance"] = values
    if buttons is not None:
        mirror["option_buttons"] = buttons
    return [{"class": "Button", "name": f"button{i}"} for i in range(3)] + [mirror]


def supplied(values):
    return {"version": 1, "source": "reflectometer", "observations": [
        {"entity": name, "property": "reflectance", "unit": "fraction", "value": value}
        for name, value in values.items()
    ]}


def requirement(scene, question="medium reflection ratio", observation=None):
    facts, profile = physics_planning_facts("reflection_qa", question, scene, observations=observation)
    from ..knowledge.physics_lcs import physics_requirement
    return physics_requirement(facts, "reflection_qa"), facts, profile


@pytest.mark.parametrize("values", list(itertools.permutations((.2, .6, .9))))
@pytest.mark.parametrize("question, expected_value", [
    ("smallest reflection ratio", .2), ("medium reflection ratio", .6), ("largest reflection ratio", .9),
])
def test_explicit_mirror_bindings_follow_values_not_button_numbers(values, question, expected_value):
    # Deliberately different orders for bindings, component names and values.
    buttons = ["button2", "button0", "button1"]
    req, facts, profile = requirement(list(reversed(components(list(values), buttons))), question)
    assert profile == "reflection_observations_v1"
    assert facts['observation_report']['status'] == 'observed'
    assert {b for b in buttons if req.check(Operation('press', b)).satisfied} == {buttons[values.index(expected_value)]}


def test_unbound_reflectance_and_missing_values_are_not_recovered_from_order():
    req, facts, profile = requirement(components([.2, .6, .9]))
    assert profile == 'reflection_unavailable_v1'
    assert facts['observation_report']['status'] == 'unavailable'
    assert 'binding' in facts['observation_report']['reasons'][0]
    assert all(req.check(Operation('press', b)).satisfied is None for b in ('button0','button1','button2'))
    _, facts, _ = requirement(components())
    assert len(facts['observation_report']['reasons']) == 2


def test_partial_observation_preserves_known_losers_and_discloses_unknown_winners():
    req, facts, _ = requirement(components(), 'smallest reflection ratio', supplied({'button0': .9, 'button1': .3}))
    assert req.check(Operation('press', 'button0')).satisfied is False
    assert req.check(Operation('press', 'button1')).satisfied is None
    assert facts['observation_report']['status'] == 'partial'
    assert facts['observation_report']['missing_candidates'] == ['button2']


@pytest.mark.parametrize('observation', [
    supplied({'button0': -1}), supplied({'button0': 1.01}), supplied({'button0': True}),
    supplied({'button0': float('nan')}), supplied({'unknown': .2}),
    {**supplied({'button0': .2}), 'version': 2},
])
def test_invalid_explicit_reflectance_fails_closed(observation):
    with pytest.raises(ValueError):
        requirement(components(), observation=observation)


def test_metadata_and_supplied_observation_conflicts_are_rejected():
    with pytest.raises(ValueError, match='conflicting'):
        requirement(components([.2,.6,.9], ['button0','button1','button2']), observation=supplied({'button0': .7}))
    with pytest.raises(ValueError, match='parallel'):
        requirement(components([.2], ['button0','button1']))
    with pytest.raises(ValueError, match='duplicate'):
        requirement(components([.2,.6], ['button0','button0']))
    wrong_units = supplied({'button0': .2})
    wrong_units['observations'][0]['unit'] = 'percent'
    with pytest.raises(ValueError, match='unit'):
        requirement(components(), observation=wrong_units)
    with pytest.raises(ValueError, match='universe'):
        physics_planning_facts('reflection_qa', 'medium reflection ratio', [],
                               observations=supplied({'button0': .2}))


def write_episode(root, scene):
    episode = root / 'PhysicsLaw/reflection_qa/example0'
    for name in ('input','output','env_config'):
        (episode/name).mkdir(parents=True)
    (episode/'input/instruction.txt').write_text('Choose the mirror with the medium reflection ratio.')
    (episode/'env_config/env_config.json').write_text(json.dumps({'task': {'components': scene,
        'conditions': {'press_button': {'target_button': 'button0'}, 'reflectance': [1,0,1]}}}))
    (episode/'output/operation_sequence.json').write_text(json.dumps({'skill_sequence': [
        {'name': 'press', 'params': {'target_entity_name': 'button0'}}]}))
    return episode


def test_sidecar_loading_label_isolation_compiled_masks_and_missing_report(tmp_path, runtime):
    episode = write_episode(tmp_path, components())
    unavailable = load_planning_examples(tmp_path, observation_mode="metadata-assisted")[0]
    assert unavailable.planning_observation_profile == 'reflection_unavailable_v1'
    assert unavailable.planning_facts['observation_report']['status'] == 'unavailable'
    (episode/'input/property_observations.json').write_text(json.dumps(supplied({'button0': .9,'button1': .1,'button2': .5})))
    example = load_planning_examples(tmp_path, observation_mode="metadata-assisted")[0]
    safe = load_scene_planning_examples(tmp_path, observation_mode="metadata-assisted")[0]
    assert example.planning_facts == safe.planning_facts
    policy = runtime.dfa_for('vlabench', example.as_reward_item())
    for button in range(3):
        plan = [{'name': 'press', 'params': {'target_entity_name': button}}]
        assert accepted(runtime, policy, plan, example.entities) == (button == 2)
    state = policy.start_state
    for token in ('skill:press', 'arg:target_entity_name'):
        state = policy.step(state, runtime.vocabulary.label_for_token(token))
    assert policy.allowed_tokens(state, 1) == {runtime.vocabulary.label_for_token('obj:2')}
    import torch
    class Planner(torch.nn.Module):
        def generate_plan(self, **kwargs):
            return [{'name': 'press', 'params': {'target_entity_name': 2}}]
    metrics = evaluate_planner(Planner(), [unavailable, example], runtime)
    assert metrics['observation_profiles'] == {'reflection_unavailable_v1': 1, 'reflection_observations_v1': 1}
    assert [r['status'] for r in metrics['observation_reports']] == ['unavailable','observed']
    assert metrics['grounding_coverage']['selection'] == {'unavailable': 1, 'checked': 1}


ENTITY_SOURCE = '''class Mirrors(FlatContainer):
    def _build(self,
               reflectance:list,
               **kwargs):
        super()._build(**kwargs)
        for i, ratio in enumerate(reflectance):
            self.mjcf_model.asset.add("material", name=f"reflectance_{i}", reflectance=ratio)
'''
GENERATOR_SOURCE = '''def load_objects():
    mirror_config = self.get_entity_config("mirrors", reflectance=candidate_reflections)
'''


def test_source_patch_round_trip_preserves_bindings_and_is_idempotent(tmp_path):
    materials = {}
    class Base:
        def _build(self, **kwargs):
            self.mjcf_model = SimpleNamespace(
                asset=SimpleNamespace(add=lambda kind, name, **values: materials.update({name: SimpleNamespace(**values)})),
                find=lambda kind, name: materials[name],
            )
        def save(self, physics):
            return {'class': 'Mirrors', 'name': 'mirrors'}
    patched = patch_entity(ENTITY_SOURCE)
    namespace = {'FlatContainer': Base}
    exec(compile(ast.parse(patched), '<patched Mirrors>', 'exec'), namespace)
    instance = namespace['Mirrors']()
    instance._build(reflectance=[.9,.2,.5], option_buttons=['button1','button2','button0'])
    saved = instance.save(None)
    req, _, _ = requirement(components()[:3]+[saved])
    assert req.check(Operation('press','button0')).satisfied is True
    assert saved['reflectance'] == [.9,.2,.5]
    assert saved['option_buttons'] == ['button1','button2','button0']
    assert patch_entity(patched) == patched
    assert patch_generator(patch_generator(GENERATOR_SOURCE)) == patch_generator(GENERATOR_SOURCE)
    upstream = tmp_path/'upstream'
    for name, source in ((ENTITY, ENTITY_SOURCE), (GENERATOR, GENERATOR_SOURCE)):
        path = upstream/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    report = preserve(upstream, apply=True, backup_dir=tmp_path/'backup')
    assert report['applied'] and len(report['files']) == 2
    assert not preserve(upstream)['files']
    assert (tmp_path/'backup/00.original').read_text() == GENERATOR_SOURCE
