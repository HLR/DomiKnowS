from copy import deepcopy
import json
from pathlib import Path
import pytest
from .protocol import (adapt, compare_baseline, context, digest, episodes, freeze,
                       indexed, public_input, require_predecessor, sha)


def test_semantic_adapter_does_not_invent_target_bindings_or_change_pointer_table():
    base = {'instruction': 'put mustard in cupboard', 'entity_table': ('entity_000', 'entity_001'),
            'images': ('front.png',), 'generation_static_fixture_ids': [1]}
    before = deepcopy(base)
    adapted = adapt(base, 'semantic-prompt', base['instruction'])
    assert base == before and adapted['entity_table'] == base['entity_table']
    assert adapted['images'] == base['images']
    assert 'object relocation' in adapted['instruction']
    assert 'mustard' not in adapted['instruction'][len(base['instruction']):]
    assert 'entity_001' not in adapted['instruction']
    assert adapt(base, 'baseline', base['instruction']) == base


def test_prompt_is_independent_of_episode_reference_and_task_id(tmp_path):
    image = tmp_path/'front.png'; image.write_bytes(b'pixels')
    public = tmp_path/'public.json'
    value = {'instruction': 'put rubbish in bin', 'entities': ['entity_000'],
             'images': [str(image)], 'movable': ['entity_000'],
             'positions': {'entity_000': [0, 0, 0]}, 'reference': 'SECRET', 'task_id': 'SECRET'}
    public.write_text(json.dumps(value))
    episode = {'generation_input': str(public), 'scoring_private': '/must/not/open',
               'files': {str(public): sha(public), str(image): sha(image)}}
    result = public_input(episode)
    first = adapt(context(result, 'rlbench'), 'semantic-prompt', result['instruction'])
    value['reference'] = 'CHANGED'; value['task_id'] = 'CHANGED'
    public.write_text(json.dumps(value)); episode['files'][str(public)] = sha(public)
    second_public = public_input(episode)
    second = adapt(context(second_public, 'rlbench'), 'semantic-prompt', second_public['instruction'])
    assert first == second and 'SECRET' not in json.dumps(first)


@pytest.mark.parametrize('field', ['checkpoint', 'manifest', 'probe', 'source'])
def test_resume_rejects_changed_fingerprints(tmp_path, field):
    path = tmp_path/'settings.json'; freeze(path, {field: 'a'}); freeze(path, {field: 'a'})
    with pytest.raises(ValueError, match='fingerprint'):
        freeze(path, {field: 'b'})


def test_resume_normalizes_json_tuples_without_accepting_changed_binding_values(tmp_path):
    path = tmp_path/'bindings.json'
    freeze(path, {'centers': [(0, 10, 20)]})
    freeze(path, {'centers': [(0, 10, 20)]})
    freeze(path, {'centers': [[0, 10, 20]]})
    with pytest.raises(ValueError, match='fingerprint'):
        freeze(path, {'centers': [(0, 11, 20)]})


def test_baseline_comparison_requires_exact_predictions_inputs_validity_and_errors():
    row = {'episode_id': 'a', 'prediction': [], 'generation_error': None, 'valid_plan': False, 'input_hash': 'hash'}
    assert compare_baseline([row], [row], ['a'])['passed']
    for field, wrong in [('prediction', [1]), ('input_hash', 'other'), ('valid_plan', True), ('generation_error', 'error')]:
        assert not compare_baseline([{**row, field: wrong}], [row], ['a'])['passed']
    with pytest.raises(ValueError, match='coverage'):
        indexed([row, row], ['a'])


def test_cannot_start_later_method_without_completed_predecessor(tmp_path):
    require_predecessor('semantic-prompt', tmp_path)
    with pytest.raises(ValueError, match='previous method'):
        require_predecessor('id-augmentation', tmp_path)
    freeze(tmp_path/'semantic-prompt/qualification.json', {'technical_validation_passed': True})
    require_predecessor('id-augmentation', tmp_path)


def test_evaluation_split_coverage_and_overlap():
    manifest = {'episodes': [{'episode_id': str(i), 'split': 'evaluation'} for i in range(100)]}
    assert len(episodes(manifest)) == 100
    manifest['episodes'].append({'episode_id': '0', 'split': 'training'})
    with pytest.raises(ValueError, match='disjoint'):
        episodes(manifest)


def test_tampered_public_image_rejected(tmp_path):
    image = tmp_path/'image'; image.write_bytes(b'original')
    public = tmp_path/'public'; public.write_text(json.dumps({'images': [str(image)]}))
    episode = {'generation_input': str(public), 'files': {str(public): sha(public), str(image): sha(image)}}
    image.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='image hash'):
        public_input(episode)


def test_target_allocations_preserve_test_and_calibration_indices():
    from .data import allocation
    for dataset, count, low in [('libero', 20, 10), ('rlbench', 10, 20)]:
        rows = allocation(dataset, [f'case{i}' for i in range(count)])
        assert rows == allocation(dataset, [f'case{i}' for i in range(count)])
        assert len(rows) == count*30
        assert len({r['episode_id'] for r in rows}) == len(rows)
        assert len({r['seed'] for r in rows}) == len(rows)
        assert min(r['index'] for r in rows) == low
        assert sum(r['split'] == 'validation' for r in rows) == count*10


@pytest.mark.parametrize('field', ['episode_id', 'scene_fingerprint', 'input_fingerprint'])
def test_actual_scene_overlaps_fail_instead_of_filtering(field):
    from .data import audit_disjoint
    train = {'episode_id': 'train', 'scene_fingerprint': 'scene-a', 'input_fingerprint': 'image-a', 'split': 'training'}
    test = {'episode_id': 'test', 'scene_fingerprint': 'scene-b', 'input_fingerprint': 'image-b', 'split': 'evaluation'}
    assert audit_disjoint([train], [test])['passed']
    with pytest.raises(ValueError, match='overlap'):
        audit_disjoint([train], [{**test, field: train[field]}])


def test_blueprints_cannot_pass_frozen_scene_integrity_gate():
    from .data import allocation, audit_disjoint
    with pytest.raises(ValueError, match='actual frozen'):
        audit_disjoint(allocation('libero', ['case']), [])


def test_evaluator_cli_has_no_inherited_option_conflict():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-m',
        'test_regr.JointEmbodiedAgentInterface.transfer.evaluate', '--help'],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--method' in result.stdout and '--manifest' in result.stdout


def test_native_arrays_are_frozen_as_json_without_exposing_private_fields():
    import numpy as np
    from .prepare_data import native_json
    value = {'positions': {'entity_000': np.array([1., 2., 3.])}, 'count': np.int64(3)}
    assert native_json(value) == {'positions': {'entity_000': [1., 2., 3.]}, 'count': 3}


def test_completed_planning_resume_does_not_require_a_free_gpu(tmp_path, monkeypatch):
    from . import campaign
    checkpoint = tmp_path/'checkpoint'; checkpoint.write_bytes(b'weights')
    freeze(tmp_path/'jobs/libero-baseline-planning.json', {'exit_code': 0})
    freeze(tmp_path/'libero/baseline/settings.json', {'checkpoint_sha256': sha(checkpoint)})
    freeze(tmp_path/'libero/baseline/summary.json', {'complete': True})
    monkeypatch.setattr(campaign, 'idle_gpu', lambda *a: pytest.fail('completed job requested GPU'))
    campaign.launch_planning(tmp_path, {'checkpoint': str(checkpoint), 'source': str(tmp_path)}, 'libero', 'baseline')


def test_preparation_shards_cover_each_case_exactly_once():
    from .data import shard_cases
    cases = [f'case{i}' for i in range(20)]
    shards = [shard_cases(cases, i, 4) for i in range(4)]
    assert all(len(s) == 5 for s in shards)
    assert sorted(x for s in shards for x in s) == sorted(cases)
    with pytest.raises(ValueError, match='shard'):
        shard_cases(cases, 4, 4)


def test_grounding_diagnostic_preserves_canonical_ids_and_rejects_boolean_pointers():
    from .report import roles, reference_roles
    reference = {'entities': ['mustard', 'cupboard'], 'goals': [['in', 'mustard', 'cupboard']]}
    expected = {'source': [0], 'destination': [1]}
    assert reference_roles(reference, 'libero') == expected
    plan = [{'name': 'pick', 'params': {'target_entity_name': 0}},
            {'name': 'place', 'params': {'target_container_name': 1}}]
    assert roles(plan) == expected
    assert reference_roles({'reference': plan}, 'rlbench') == expected
    plan[1]['params']['target_container_name'] = True
    assert roles(plan)['destination'] == [None]


def test_idle_capacity_allocation_requires_known_service_and_empty_task_lists(monkeypatch):
    from . import campaign
    from types import SimpleNamespace
    worker = {'Id': 'abcdefghijkl1234', 'State': {'Running': True, 'Pid': 123}}
    gpu_pids = ['GPU-test, 123\n']; tasks = [[]]
    monkeypatch.setattr(campaign.subprocess, 'check_output', lambda cmd, **kw:
        gpu_pids[0] if cmd[0] == 'nvidia-smi' else json.dumps([worker]))
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=json.dumps({'celery@abcdefghijkl': tasks[0]})))
    assert campaign.idle_service_available(5, 'GPU-test')
    tasks[0] = [{'task': 'queued'}]
    assert not campaign.idle_service_available(5, 'GPU-test')
    tasks[0] = []; gpu_pids[0] += 'GPU-test, 456\n'
    assert not campaign.idle_service_available(5, 'GPU-test')


def test_native_resume_adopts_original_container_without_relaunch(tmp_path, monkeypatch):
    from . import campaign
    from types import SimpleNamespace
    cmd = ['docker', 'run', '-d', '--name', 'owned', 'pinned-image', 'evaluate']
    job = tmp_path/'jobs/native.json'
    freeze(job, {'command': cmd, 'container_id': 'full-container-id', 'pid': 1234,
                 'log': 'original.log', 'started_utc': 'original-time'})
    worker = {'Id': 'full-container-id', 'Name': '/owned', 'Image': 'pinned-image', 'State': {'Pid': 1234}}
    monkeypatch.setattr(campaign.subprocess, 'check_output', lambda cmd, **kw:
        json.dumps([worker]) if cmd[1] == 'inspect' else '0\n')
    monkeypatch.setattr(campaign.subprocess, 'Popen', lambda *a, **kw: SimpleNamespace(wait=lambda: 0))
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **kw: pytest.fail('restarted live simulation'))
    campaign.run(tmp_path, 'native', cmd, {})
    result = json.loads(job.read_text())
    assert result['container_id'] == 'full-container-id' and result['pid'] == 1234
    assert result['exit_code'] == 0 and result['started_utc'] == 'original-time'
    assert result['previous_logs'] == ['original.log']
