"""Offline reference review and supplementary scoring; never a generation input.

Original official labels/scores stay untouched. Scene metadata is read only by
the audit command, and cannot enter the evaluator's prompt or grounding path.
"""
from pathlib import Path
import argparse
import hashlib
import json

VERSION = 'vlabench-reference-adjudication-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_hashes(example):
    return {'instruction': hashlib.sha256(example.instruction.encode()).hexdigest(),
            'image': digest(example.image_path), 'numbered_image': digest(example.mask_path)}


def validate_reference(plan):
    from ..world_domain import SKILL_ARGUMENTS, OPTIONAL_SKILL_ARGUMENTS
    if not isinstance(plan, list) or not plan:
        raise ValueError('an adjudicated reference must be a nonempty operation list')
    for op in plan:
        if not isinstance(op, dict) or set(op) != {'name', 'params'}:
            raise ValueError('reference operations need exactly name and params')
        name, params = op['name'], op['params']
        if not isinstance(name, str) or name not in SKILL_ARGUMENTS or not isinstance(params, dict):
            raise ValueError('unknown reference skill or invalid params')
        required = set(SKILL_ARGUMENTS[name])
        optional = set(OPTIONAL_SKILL_ARGUMENTS.get(name, ()))
        if not required <= set(params) or not set(params) <= required | optional:
            raise ValueError('reference arguments differ from the declared skill schema')
        if any(type(v) is not int or not 0 <= v < 512 for v in params.values()):
            raise ValueError('adjudicated references require integer image-tag IDs in [0, 512)')
    return plan


def load_adjudications(path, examples):
    from .official_protocol import dataset_fingerprint
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('version') != VERSION or payload.get('dataset_fingerprint') != dataset_fingerprint(examples):
        raise ValueError('adjudication manifest version or original dataset fingerprint differs')
    by_id = {e.episode_id: e for e in examples}
    seen, approved = set(), {}
    if not isinstance(payload.get('entries'), list):
        raise ValueError('adjudication entries must be a list')
    for entry in payload['entries']:
        eid = entry.get('episode_id')
        if eid not in by_id or eid in seen:
            raise ValueError('unknown or duplicate adjudicated episode')
        seen.add(eid)
        example = by_id[eid]
        if entry.get('original_reference_sha256') != example.reference_sha256 or entry.get('input_hashes') != input_hashes(example):
            raise ValueError('adjudication source reference or episode inputs changed')
        status = entry.get('status')
        if status not in ('needs_review', 'approved', 'rejected'):
            raise ValueError('unknown adjudication review status')
        if status != 'approved':
            continue
        review = entry.get('review', {})
        if (not isinstance(review, dict) or any(not isinstance(review.get(k), str) or not review[k].strip()
                                              for k in ('reviewer', 'reason'))
                or not isinstance(review.get('evidence'), list) or not review['evidence']
                or any(not isinstance(x, str) or not x.strip() for x in review['evidence'])):
            raise ValueError('approved adjudications require reviewer, reason and evidence')
        approved[eid] = validate_reference(entry.get('reference'))
    if not approved:
        raise ValueError('manifest contains no approved adjudications; review entries before scoring')
    return approved


def score_adjudicated(examples, outputs_path, manifest_path, output_dir):
    """Rescore the same saved replies into separate, explicitly non-standard files."""
    from .official_protocol import official_report, score_official_answer, dataset_fingerprint
    approved = load_adjudications(manifest_path, examples)
    manifest_hash, outputs_hash = digest(manifest_path), digest(outputs_path)
    answers = {}
    for line in Path(outputs_path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('episode_id') in answers:
            raise ValueError('duplicate saved prediction')
        answers[row['episode_id']] = row
    if set(answers) != {e.episode_id for e in examples}:
        raise ValueError('saved prediction coverage differs from the adjudication dataset')
    original, adjudicated = [], []
    for e in examples:
        answer = answers[e.episode_id]
        if answer.get('task') != e.task:
            raise ValueError('saved prediction task differs from the original episode')
        ok = 'skill_sequence' in answer
        sequence = answer.get('skill_sequence')
        common = {'episode_id': e.episode_id, 'task': e.task, 'format_error': not ok}
        original.append({**common, 'scores': score_official_answer(e.task, e.reference, ok, sequence)})
        adjudicated.append({**common, 'reference_changed': e.episode_id in approved and approved[e.episode_id] != list(e.reference),
                            'scores': score_official_answer(e.task, approved.get(e.episode_id, e.reference), ok, sequence)})
    summary = {'protocol': VERSION, 'standard': False, 'scoring_only': True,
               'dataset_fingerprint': dataset_fingerprint(examples),
               'manifest_sha256': manifest_hash, 'predictions_sha256': outputs_hash,
               'examples': len(examples), 'approved_entries': len(approved),
               'changed_references': sum(x['reference_changed'] for x in adjudicated),
               'original_reference_scores': official_report(original),
               'adjudicated_reference_scores': official_report(adjudicated)}
    if digest(manifest_path) != manifest_hash or digest(outputs_path) != outputs_hash:
        raise ValueError('adjudication manifest or predictions changed during scoring')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protected = {Path(outputs_path).resolve(), Path(manifest_path).resolve()}
    if any((output_dir/name).resolve() in protected for name in ('adjudicated_scores.jsonl', 'adjudicated_summary.json')):
        raise ValueError('supplementary outputs cannot overwrite prediction or review inputs')
    prior = output_dir/'adjudicated_summary.json'
    if prior.exists():
        saved = json.loads(prior.read_text(encoding='utf-8'))
        if any(saved.get(k) != summary[k] for k in ('manifest_sha256', 'predictions_sha256', 'dataset_fingerprint')):
            raise ValueError('adjudicated output records different inputs or review settings; use a new directory')
    # No write to summary.json, final_scores.jsonl, outputs.jsonl or original data.
    (output_dir/'adjudicated_scores.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in adjudicated), encoding='utf-8')
    (output_dir/'adjudicated_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    return summary


def audit_dataset(root, output):
    """Flag simple scene/reference conflicts for human review, with no auto-repair."""
    from .official_protocol import load_official_examples, dataset_fingerprint
    from ..dataset import _extract_entities
    from ..repair_density_labels import scene_answer
    if Path(output).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('review manifests must be outside the benchmark dataset')
    examples = load_official_examples(root)
    entries = []
    for e in examples:
        config_path = Path(root)/e.episode_id/'env_config/env_config.json'
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding='utf-8'))
        entities = _extract_entities(config)
        conditions = config.get('task', {}).get('conditions') or {}
        def targets(skill):
            return {entities[p] for op in e.reference if op['name'] == skill
                    for p in [op['params'].get('target_entity_name')]
                    if type(p) is int and 0 <= p < len(entities)}
        flags = []
        condition, required, skill = None, (), 'pick'
        if isinstance(conditions.get('press_button'), dict):
            condition, required, skill = 'press_button', (conditions['press_button'].get('target_button'),), 'press'
        elif e.task.startswith('select_'):
            for kind in ('not_contain', 'is_grasped', 'contain'):
                if isinstance(conditions.get(kind), dict) and conditions[kind].get('entities'):
                    condition, required = kind, conditions[kind]['entities']
                    break
        elif e.task.startswith('add_condiment') and isinstance(conditions.get('pour'), dict):
            condition, required = 'pour', (conditions['pour'].get('target_entity'),)
        if condition and all(isinstance(x, str) for x in required) and not set(required) <= targets(skill):
            flags.append({'kind': 'reference_scene_target_conflict', 'condition': condition,
                          'scene_targets': list(required), 'reference_targets': sorted(targets(skill))})
        if e.task == 'density_qa':
            try:
                button, _, _ = scene_answer(config, e.instruction)
                if button not in targets('press'):
                    flags.append({'kind': 'density_material_reference_conflict', 'scene_derived_button': button})
            except (ValueError, KeyError, TypeError) as error:
                flags.append({'kind': 'density_unresolved', 'reason': str(error)})
        if flags:
            entries.append({'episode_id': e.episode_id, 'original_reference_sha256': e.reference_sha256,
                            'input_hashes': input_hashes(e), 'status': 'needs_review',
                            'reference': None, 'flags': flags,
                            'review': {'reviewer': '', 'reason': '', 'evidence': []}})
    payload = {'version': VERSION, 'dataset_fingerprint': dataset_fingerprint(examples),
               'scope': 'Offline review only; scene conditions may also be wrong. No decoder inputs or labels are changed.',
               'examples': len(examples), 'entries': entries}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('refusing to overwrite an existing review manifest')
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--planning-dir', type=Path, required=True)
    audit.add_argument('--output', type=Path, required=True)
    score = sub.add_parser('score')
    for name in ('planning-dir', 'outputs', 'manifest', 'output-dir'):
        score.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'audit':
        result = audit_dataset(args.planning_dir, args.output)
        print(json.dumps({'examples': result['examples'], 'needs_review': len(result['entries'])}))
    else:
        from .official_protocol import load_official_examples
        result = score_adjudicated(load_official_examples(args.planning_dir), args.outputs, args.manifest, args.output_dir)
        print(json.dumps({'examples': result['examples'], 'changed_references': result['changed_references'], 'standard': False}))


if __name__ == '__main__':
    main()
