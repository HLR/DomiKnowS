"""Integration checks for public-only generation and archived baseline gates."""
import json
from types import SimpleNamespace
import pytest

from .evaluate import evaluate
from .protocol import context, digest, freeze, sha


def fixture(tmp_path, monkeypatch):
    from .. import eval_generalization
    image = tmp_path/'image.png'; image.write_bytes(b'frozen pixels')
    public = {'instruction': 'put the object in the container',
        'entities': ['entity_000', 'entity_001'], 'movable': ['entity_000'],
        'positions': {'entity_000': [0, 0, 0], 'entity_001': [1, 0, 0]},
        'images': [str(image)], 'reference': 'MUST NOT ENTER PROMPT', 'task_id': 'SECRET'}
    public_file = tmp_path/'public.json'; freeze(public_file, public)
    records = [{'episode_id': f'episode{i}', 'split': 'evaluation',
        'generation_input': str(public_file), 'scoring_private': str(tmp_path/'DOES_NOT_EXIST'),
        'files': {str(public_file): sha(public_file), str(image): sha(image)}} for i in range(100)]
    manifest = tmp_path/'manifest.json'; freeze(manifest, {'episodes': records})
    planner = SimpleNamespace(_prompt_base=lambda domain, bound: bound['instruction']+str(bound['entity_table']),
                              _chat=lambda prompt, count: prompt+str(count))
    prediction = [{'name': 'pick', 'params': {'target_entity_name': 0}},
                  {'name': 'place', 'params': {'target_container_name': 1}}]
    calls = []
    def generate(planner, runtime, bound, policy):
        assert 'SECRET' not in json.dumps(bound) and 'MUST NOT' not in json.dumps(bound)
        assert tuple(bound['entity_table']) == ('entity_000', 'entity_001')
        calls.append((policy, bound))
        return prediction, None, {'triggered_rejections': {}, 'applicable_events': {}}, None
    monkeypatch.setattr(eval_generalization, 'generate', generate)
    clean_public = {k: v for k, v in public.items() if k not in ('reference', 'task_id')}
    input_hash = digest(eval_generalization.input_record(planner, context(clean_public, 'rlbench')))
    archive = tmp_path/'archive'
    for policy in ('none', 'shared-verified'):
        path = archive/policy; path.mkdir(parents=True)
        (path/'predictions.jsonl').write_text(''.join(json.dumps({'episode_id': record['episode_id'],
            'prediction': prediction, 'generation_error': None, 'valid_plan': True,
            'input_hash': input_hash})+'\n' for record in records))
    return manifest, archive, planner, calls


def test_exact_baseline_then_semantic_prompts_keep_references_private(tmp_path, monkeypatch):
    manifest, archive, planner, calls = fixture(tmp_path, monkeypatch)
    baseline = evaluate(planner, None, dataset='rlbench', manifest_path=manifest, method='baseline',
                        output=tmp_path/'baseline', settings={}, archive=archive)
    semantic = evaluate(planner, None, dataset='rlbench', manifest_path=manifest, method='semantic-prompt',
                        output=tmp_path/'semantic-prompt', settings={}, archive=archive)
    assert len(calls) == 400
    assert baseline['none']['valid_plan_rate'] == semantic['none']['valid_plan_rate'] == 1
    assert semantic['none']['input_hashes'] == semantic['shared-verified']['input_hashes']
    assert semantic['none']['input_hashes'] != baseline['none']['input_hashes']


def test_baseline_mismatch_is_archived_and_stops_the_run(tmp_path, monkeypatch):
    manifest, archive, planner, calls = fixture(tmp_path, monkeypatch)
    path = archive/'none/predictions.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['prediction'] = []
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='reproduction failed'):
        evaluate(planner, None, dataset='rlbench', manifest_path=manifest, method='baseline',
                 output=tmp_path/'baseline', settings={}, archive=archive)
    assert len(calls) == 100
    assert not (tmp_path/'baseline/none/report.json').exists()
    proof = json.loads((tmp_path/'baseline/none/baseline_reproduction.json').read_text())
    assert not proof['passed'] and proof['mismatches'][0]['episode_id'] == 'episode0'


def test_permutation_predictions_are_inverse_mapped_before_native_plan_scoring(tmp_path, monkeypatch):
    from . import evaluate as module
    from .permutation import EntityPermutation
    manifest, archive, planner, calls = fixture(tmp_path, monkeypatch)
    data = json.loads(manifest.read_text())
    for record in data['episodes']:
        record.update(canonical_generation_input=record['generation_input'], trial=0,
                      original_reference_sha256='opaque-reference-hash')
    manifest.write_text(json.dumps(data))
    # Transformation integrity is exercised with real files in the preparation
    # tests; isolate generation/scoring direction here with a semantic mismatch.
    monkeypatch.setattr(module, 'verified_episode_permutation', lambda episode, public: EntityPermutation((1, 0)))
    reports = evaluate(planner, None, dataset='rlbench', manifest_path=manifest,
        method='id-sensitivity', output=tmp_path/'sensitivity', settings={}, archive=archive)
    assert reports['none']['valid_plan_rate'] == 0
    row = json.loads((tmp_path/'sensitivity/none/predictions.jsonl').read_text().splitlines()[0])
    assert row['permuted_prediction'][0]['params']['target_entity_name'] == 0
    assert row['prediction'][0]['params']['target_entity_name'] == 1
