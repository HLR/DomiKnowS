"""Contracts for the preregistered matched-transfer evaluation."""
from contextlib import contextmanager
from collections import defaultdict
import random

POLICIES = ('none', 'shared-verified', 'full-metadata', 'executor')


def snapshot_key(epoch=None, snapshot=None):
    if (epoch is None) == (snapshot is None):
        raise ValueError('provide exactly one of --epoch or --snapshot')
    if epoch is not None:
        if epoch not in range(1, 5): raise ValueError('epoch must be 1--4')
        return str(epoch), f'epoch-{epoch}'
    if snapshot == 'initialization': return snapshot, snapshot
    if snapshot in tuple(f'epoch-{n}' for n in range(1, 5)): return snapshot[-1], snapshot
    raise ValueError('unknown snapshot')


def selected_datasets(policy, requested=None):
    datasets = tuple(requested or (('libero',) if policy == 'executor' else ('vlabench', 'libero')))
    if len(set(datasets)) != len(datasets) or not set(datasets) <= {'vlabench', 'libero'}:
        raise ValueError('invalid dataset selection')
    if policy == 'executor' and 'vlabench' in datasets:
        raise ValueError('executor policy is LIBERO-only')
    return tuple(d for d in datasets if not (policy == 'full-metadata' and d == 'libero'))


@contextmanager
def exact_policy_search(policy, records, maximum=1000000):
    """Increase computation, never approximate an unknown reachability answer."""
    from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
    seen = set(); pending = [policy]; searches = []
    while pending:
        node = pending.pop()
        if id(node) in seen: continue
        seen.add(id(node))
        search = getattr(node, '_reachability', None)
        if search is not None: searches.append((search, search.max_state_expansions))
        for name in ('base_dfa', 'dfa'):
            child = getattr(node, name, None)
            if child is not None: pending.append(child)
    original = policy.allowed_tokens
    def allowed(state, remaining_steps=None):
        while True:
            try: return original(state, remaining_steps=remaining_steps)
            except SemanticSearchLimitExceeded:
                candidates = [s for s, _ in searches if s.max_state_expansions < maximum]
                if not candidates: raise
                before = [s.max_state_expansions for s in candidates]
                for s in candidates: s.max_state_expansions = min(maximum, s.max_state_expansions*10)
                records.append({'old_caps': before, 'new_caps': [s.max_state_expansions for s in candidates],
                                'remaining_steps': remaining_steps, 'hard_constraints_relaxed': False})
    policy.allowed_tokens = allowed
    try: yield
    finally:
        policy.allowed_tokens = original
        for search, cap in searches: search.max_state_expansions = cap


def enrich_decision(record):
    result = dict(record); values = record.get('unmasked_logits', {})
    ranked = sorted(values.items(), key=lambda x: (-x[1], x[0]))
    result['raw_top_logit_margin'] = ranked[0][1]-ranked[1][1] if len(ranked) >= 2 else None
    allowed = set(record.get('allowed', ()))
    rejected = [v for k, v in ranked if k not in allowed]
    retained = [v for k, v in ranked if k in allowed]
    result['best_allowed_minus_best_rejected'] = max(retained)-max(rejected) if retained and rejected else None
    return result


def paired_interval(before, after, ids, *, confidence=.95, samples=10000, seed=42):
    if set(before) != set(ids) or set(after) != set(ids) or len(set(ids)) != len(ids):
        raise ValueError('paired outcome identities differ')
    groups = defaultdict(list); changes = []
    for identity in ids:
        change = int(after[identity]['success'])-int(before[identity]['success'])
        changes.append(change); groups[identity.rsplit('/', 1)[0]].append(change)
    clusters = list(groups.values()); rng = random.Random(seed); boot = []
    for _ in range(samples):
        values = [v for group in rng.choices(clusters, k=len(clusters)) for v in group]
        boot.append(sum(values)/len(values))
    boot.sort(); tail = (1-confidence)/2
    return {'delta': sum(changes)/len(changes), 'wins': changes.count(1), 'losses': changes.count(-1),
            'unchanged': changes.count(0), 'confidence': confidence,
            'task_cluster_interval': [boot[int(tail*samples)], boot[min(samples-1, int((1-tail)*samples))]],
            'task_clusters': len(clusters), 'samples': samples, 'seed': seed}
