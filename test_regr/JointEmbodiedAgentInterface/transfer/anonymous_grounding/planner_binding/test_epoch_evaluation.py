import json
import pytest
from .epoch_evaluation import episode_order, verify_inputs, POLICIES
from ...protocol import sha, freeze


def test_frozen_ids_preserve_pointer_record_and_reject_missing_or_duplicate():
    rows = [{'episode_id': 'b', 'canonical_entities': ['cup', 'plate']},
            {'episode_id': 'a', 'canonical_entities': ['plate', 'cup']}]
    result = episode_order(rows, ['a', 'b'], 2)
    assert result[0] is rows[1] and result[0]['canonical_entities'] == ['plate', 'cup']
    with pytest.raises(ValueError): episode_order(rows[:1], ['a', 'b'], 2)
    with pytest.raises(ValueError): episode_order([rows[0], rows[0]], ['a', 'b'], 2)


def test_checkpoint_manifest_and_source_changes_reject_resume(tmp_path):
    paths = [tmp_path/name for name in ('checkpoint.pt', 'manifest.json', 'source.py')]
    for p in paths: p.write_text(p.name)
    config = {'input_hashes': {str(p): sha(p) for p in paths}}
    verify_inputs(config)
    for p in paths:
        before = p.read_text(); p.write_text('changed')
        with pytest.raises(ValueError, match='input changed'): verify_inputs(config)
        p.write_text(before)


def test_policy_settings_cannot_be_reused_with_different_checkpoint(tmp_path):
    path = tmp_path/'settings.json'
    original = {'checkpoint_sha256': 'first', 'policy': 'none'}
    freeze(path, original); freeze(path, original)
    for changed in ({**original, 'checkpoint_sha256': 'second'}, {**original, 'policy': 'shared-verified'}):
        with pytest.raises(ValueError, match='fingerprint'): freeze(path, changed)
    assert json.loads(path.read_text()) == original
    assert POLICIES == ('none', 'shared-verified', 'full-metadata', 'executor')


def test_one_planner_failure_does_not_skip_other_native_evaluations():
    from concurrent.futures import Future, ThreadPoolExecutor
    from .epoch_campaign import finish_evaluations
    futures = []
    for result in ((1, 'none'), (3, 'shared-verified'), (4, 'full-metadata')):
        f = Future(); f.set_result(result); futures.append(f)
    failed = Future(); failed.set_exception(RuntimeError('bounded EAI search failed')); futures.append(failed)
    completed = []
    with ThreadPoolExecutor(max_workers=2) as native:
        with pytest.raises(RuntimeError, match='successful conditions preserved'):
            finish_evaluations(futures, native, lambda epoch, policy: completed.append((epoch, policy)))
    assert sorted(completed) == [(1, 'none'), (3, 'shared-verified')]
