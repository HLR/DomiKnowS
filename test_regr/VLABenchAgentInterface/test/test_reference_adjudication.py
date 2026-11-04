import json
from dataclasses import replace

import pytest

from ..official import official_protocol as protocol
from ..official.reference_adjudication import (
    VERSION, audit_dataset, digest, input_hashes, load_adjudications, score_adjudicated,
)
from .test_official_protocol import make_episode


def plan(pointer):
    return [{'name': 'pick', 'params': {'target_entity_name': pointer}}, {'name': 'lift', 'params': {}}]


def fixture(tmp_path):
    data = tmp_path/'data'
    make_episode(data, 'select_poker', 1, 'Pick the queen of hearts', plan(1))
    examples = protocol.load_official_examples(data)
    e = examples[0]
    manifest = {'version': VERSION, 'dataset_fingerprint': protocol.dataset_fingerprint(examples), 'entries': [{
        'episode_id': e.episode_id, 'original_reference_sha256': e.reference_sha256,
        'input_hashes': input_hashes(e), 'status': 'approved', 'reference': plan(2),
        'review': {'reviewer': 'review-test', 'reason': 'verified card/tag mismatch',
                   'evidence': ['numbered image: queen is tag 2', 'instruction: queen of hearts']}}]}
    path = tmp_path/'review.json'
    path.write_text(json.dumps(manifest))
    return data, examples, manifest, path


def test_same_saved_replies_get_separate_scores_without_changing_official_outputs(tmp_path):
    data, examples, manifest, path = fixture(tmp_path)
    out = tmp_path/'run'
    protocol.run_official_protocol(examples, lambda _: '```json\n'+json.dumps(plan(2))+'\n```',
                                   output_dir=out, model_name='frozen', progress=lambda _: None)
    protected = [out/'summary.json', out/'final_scores.jsonl', out/'outputs.jsonl',
                 data/examples[0].episode_id/'output/operation_sequence.json']
    hashes = {p: digest(p) for p in protected}
    summary = score_adjudicated(examples, out/'outputs.jsonl', path, out)
    assert not summary['standard'] and summary['scoring_only'] and summary['changed_references'] == 1
    original = summary['original_reference_scores']['official_mapped_overall']['legacy_score']
    corrected = summary['adjudicated_reference_scores']['official_mapped_overall']['legacy_score']
    assert corrected > original
    assert {p: digest(p) for p in protected} == hashes
    assert (out/'adjudicated_summary.json').exists()


def test_reference_changes_do_not_change_zero_shot_generation_inputs(tmp_path):
    _, examples, _, _ = fixture(tmp_path)
    e = examples[0]
    changed = replace(e, reference=tuple(plan(3)), reference_payload={'skill_sequence': plan(3)}, reference_sha256='changed')
    assert protocol.build_ti_list(protocol.load_official_prompt(), e) == protocol.build_ti_list(protocol.load_official_prompt(), changed)


@pytest.mark.parametrize('mutation', ['reference_hash', 'input_hash', 'duplicate', 'unknown_episode',
                                     'reviewer', 'reason', 'evidence', 'string_id', 'unknown_skill', 'wrong_argument', 'pending'])
def test_unreviewed_or_mismatched_overrides_are_rejected(tmp_path, mutation):
    _, examples, payload, path = fixture(tmp_path)
    entry = payload['entries'][0]
    if mutation == 'reference_hash':entry['original_reference_sha256'] = 'wrong'
    elif mutation == 'input_hash':entry['input_hashes']['image'] = 'wrong'
    elif mutation == 'duplicate':payload['entries'].append(entry.copy())
    elif mutation == 'unknown_episode':entry['episode_id'] = '../elsewhere'
    elif mutation in ('reviewer', 'reason', 'evidence'):entry['review'][mutation] = ''
    elif mutation == 'string_id':entry['reference'][0]['params']['target_entity_name'] = '2'
    elif mutation == 'unknown_skill':entry['reference'][0]['name'] = 'teleport'
    elif mutation == 'wrong_argument':entry['reference'][0]['params'] = {'target_container_name': 2}
    elif mutation == 'pending':entry['status'] = 'needs_review'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):load_adjudications(path, examples)


def test_image_edit_invalidates_review_even_with_unchanged_reference(tmp_path):
    _, examples, _, path = fixture(tmp_path)
    examples[0].mask_path.write_bytes(b'changed tags')
    with pytest.raises(ValueError, match='inputs changed'):load_adjudications(path, examples)


def test_changed_review_requires_new_supplementary_directory(tmp_path):
    _, examples, payload, path = fixture(tmp_path)
    outputs = tmp_path/'outputs.jsonl'
    outputs.write_text(json.dumps({'episode_id': examples[0].episode_id, 'task': examples[0].task, 'skill_sequence': plan(2)})+'\n')
    out = tmp_path/'scores'
    score_adjudicated(examples, outputs, path, out)
    payload['entries'][0]['review']['reason'] = 'a later revision of the review'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='different inputs or review settings'):
        score_adjudicated(examples, outputs, path, out)


def test_prediction_source_cannot_be_overwritten_by_supplementary_scoring(tmp_path):
    _, examples, _, path = fixture(tmp_path)
    outputs = tmp_path/'adjudicated_scores.jsonl'
    outputs.write_text(json.dumps({'episode_id': examples[0].episode_id, 'task': examples[0].task, 'skill_sequence': plan(2)})+'\n')
    before = digest(outputs)
    with pytest.raises(ValueError, match='cannot overwrite'):
        score_adjudicated(examples, outputs, path, tmp_path)
    assert digest(outputs) == before


def test_offline_audit_flags_scene_conflicts_but_never_approves_or_edits_labels(tmp_path):
    data = tmp_path/'data'
    wrong = [{'name': 'pick', 'params': {'target_entity_name': 2}}, {'name': 'pull', 'params': {}}]
    make_episode(data, 'select_drink_semantic', 1, 'Pick the juice', wrong, 'Semantic')
    folder = data/'Semantic/select_drink_semantic/example1'
    (folder/'env_config').mkdir()
    (folder/'env_config/env_config.json').write_text(json.dumps({'task': {
        'components': [{'name': 'fridge_open'}, {'name': 'juice'}, {'name': 'redbull'}],
        'conditions': {'not_contain': {'container': 'fridge_open', 'entities': ['juice']}}}}))
    files = {p: digest(p) for p in data.rglob('*') if p.is_file()}
    output = tmp_path/'review.json'
    result = audit_dataset(data, output)
    assert len(result['entries']) == 1
    entry = result['entries'][0]
    assert entry['status'] == 'needs_review' and entry['reference'] is None
    assert entry['flags'][0]['scene_targets'] == ['juice']
    assert {p: digest(p) for p in files} == files
    with pytest.raises(ValueError, match='overwrite'):audit_dataset(data, output)
