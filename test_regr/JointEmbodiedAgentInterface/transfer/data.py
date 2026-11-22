"""Disjoint target-training allocations and scene-integrity gates."""
from collections import Counter
from .protocol import digest


def shard_cases(cases, index, count):
    if count < 1 or not 0 <= index < count or count > len(cases):
        raise ValueError('invalid task-case shard')
    return sorted(cases)[index::count]


def allocation(dataset, cases, seed=42, *, exclude_libero_confirmation=False):
    """Blueprints only: native preparation must freeze actual states and images."""
    if dataset not in ('libero', 'rlbench'):
        raise ValueError('unsupported dataset')
    if len(set(cases)) != len(cases):
        raise ValueError('duplicate task cases')
    ranges = ({'training': range(10, 30), 'validation': range(30, 35 if exclude_libero_confirmation else 40)} if dataset == 'libero'
              else {'training': range(20, 40), 'validation': range(40, 50)})
    rows = []
    for case in sorted(cases):
        for split, indices in ranges.items():
            for index in indices:
                suffix = f'state{index:02d}' if dataset == 'libero' else f'reset{index:02d}'
                episode_id = f'{case}/{suffix}'
                rows.append({'episode_id': episode_id, 'split': split, 'index': index,
                             'seed': int(digest([seed, dataset, episode_id])[:8], 16)})
    return rows


def audit_disjoint(prepared, reserved):
    """Fail on overlap rather than replacing hard scenes or filtering examples."""
    from .repair.reservations import guard
    guard({'episodes':prepared})
    required = ('episode_id', 'scene_fingerprint', 'input_fingerprint', 'split')
    for row in [*prepared, *reserved]:
        if any(key not in row for key in required):
            raise ValueError('actual frozen scenes and inputs required; blueprints cannot pass')
    by_identity = {}; by_scene = {}; by_input = {}
    for row in [*prepared, *reserved]:
        for field, seen in [('episode_id', by_identity), ('scene_fingerprint', by_scene),
                            ('input_fingerprint', by_input)]:
            value = row[field]
            if value in seen and seen[value] != row['split']:
                raise ValueError(f'cross-split {field} overlap')
            seen[value] = row['split']
    ids = [r['episode_id'] for r in prepared]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate prepared identities')
    return {'passed': True, 'counts': dict(Counter(r['split'] for r in prepared)),
            'prepared_episodes': len(prepared), 'reserved_episodes': len(reserved)}
