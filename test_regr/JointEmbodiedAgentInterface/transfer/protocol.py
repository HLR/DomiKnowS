"""CPU-only transfer contracts. No scoring-private inputs enter prompt adapters."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

METHODS = ('semantic-prompt', 'id-augmentation', 'grounding', 'shared-format',
           'target-adaptation', 'replanning')
POLICIES = ('none', 'shared-verified')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def freeze(path, value):
    path = Path(path)
    serialized = json.dumps(value, indent=2, sort_keys=True) + '\n'
    canonical = json.loads(serialized)
    if path.exists():
        if read(path) != canonical:
            raise ValueError('resume fingerprint differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(serialized, encoding='utf-8')
    temporary.replace(path)


def episodes(manifest):
    selected = [e for e in manifest['episodes'] if e['split'] == 'evaluation']
    ids = [e['episode_id'] for e in selected]
    others = {e['episode_id'] for e in manifest['episodes'] if e['split'] != 'evaluation'}
    if len(ids) != 100 or len(set(ids)) != 100 or set(ids) & others:
        raise ValueError('require 100 unique evaluation identities disjoint from other splits')
    return selected


def public_input(episode):
    """Read public fields only; even unknown public keys are removed by allowlist."""
    path = episode['generation_input']
    if sha(path) != episode['files'][path]:
        raise ValueError('public input hash differs')
    raw = read(path)
    keys = ('instruction', 'entities', 'images', 'movable', 'positions',
            'initial_relations', 'entity_types', 'visible_entity_ids')
    public = {key: raw[key] for key in keys if key in raw}
    for image in public['images']:
        if sha(image) != episode['files'][image]:
            raise ValueError('image hash differs')
    return public


def context(public, dataset):
    if dataset == 'rlbench':
        from ..generalization.rlbench_protocol import planning_context
        return {**planning_context(public), 'grounding_instruction': public['instruction']}
    if dataset != 'libero':
        raise ValueError('unsupported external dataset')
    return {'grounding_instruction': public['instruction'],
            'instruction': public['instruction'] + '\nInitial scene relations: '
            + json.dumps(public['initial_relations']),
            'entity_table': public['entities'], 'images': tuple(public['images']),
            'generation_entity_types': public['entity_types'],
            'generation_static_fixture_ids': [i for i, name in enumerate(public['entities'])
                                              if name not in public['movable']]}


SEMANTIC_VERSION = 1
OPERATION_GUIDANCE = (
    '\nGeneric operation definitions: pick(entity_id) acquires the observed object; '
    'place(destination_entity_id) releases the held object at the observed destination. '
    'Resolve requested object and destination from the instruction and observations. '
    'Entity numbers are identifiers, not object categories. Do not assume a fixed destination number.'
)


def adapt(base_context, method, instruction):
    value = deepcopy(base_context)
    if method == 'baseline':
        return value
    if method != 'semantic-prompt':
        raise NotImplementedError('method is not yet validated for rollout: ' + method)
    # Language-only classification: no benchmark task IDs, object lookup, or gold pointers.
    family = ('object relocation to an instructed destination' if re.search(
        r'\b(put|place|move|relocate)\b', instruction, re.I) else
        'follow the instruction using supported object manipulation operations')
    value['instruction'] += '\nTask semantics: ' + family + '.' + OPERATION_GUIDANCE
    return value


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def indexed(rows, expected):
    ids = [row['episode_id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('episode coverage differs')
    return {row['episode_id']: row for row in rows}


def compare_baseline(actual, archived, expected):
    actual, archived = indexed(actual, expected), indexed(archived, expected)
    mismatches = []
    for key in sorted(actual):
        for field in ('prediction', 'generation_error', 'input_hash', 'valid_plan'):
            if actual[key].get(field) != archived[key].get(field):
                mismatches.append({'episode_id': key, 'field': field})
    return {'passed': not mismatches, 'episodes': len(actual), 'mismatches': mismatches}


def require_predecessor(method, root):
    if method not in METHODS:
        raise ValueError('unknown method')
    position = METHODS.index(method)
    if position:
        prior = Path(root) / METHODS[position - 1] / 'qualification.json'
        if not prior.exists() or read(prior).get('technical_validation_passed') is not True:
            raise ValueError('previous method has not completed technical validation and evaluation')
