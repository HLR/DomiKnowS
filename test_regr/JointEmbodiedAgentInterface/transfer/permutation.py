"""Observation-consistent ID changes, with explicit old/new pointer directions."""
from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations
import math
import random
import re
from .protocol import digest


@dataclass(frozen=True)
class EntityPermutation:
    # order[new_id] = old_id. This is also the inverse map for predictions.
    order: tuple

    def __post_init__(self):
        object.__setattr__(self, 'order', tuple(self.order))
        if (not self.order or any(type(i) is not int for i in self.order)
                or set(self.order) != set(range(len(self.order)))):
            raise ValueError('entity order must be an integer bijection')

    @property
    def old_to_new(self):
        inverse = [0]*len(self.order)
        for new, old in enumerate(self.order): inverse[old] = new
        return tuple(inverse)

    def encode_pointer(self, old):
        if type(old) is not int or not 0 <= old < len(self.order):
            raise ValueError('invalid original entity pointer')
        return self.old_to_new[old]

    def decode_pointer(self, new):
        if type(new) is not int or not 0 <= new < len(self.order):
            raise ValueError('invalid permuted entity pointer')
        return self.order[new]

    def plan(self, value, *, inverse=False, pointer_keys=('target_entity_name', 'target_container_name'), strict=True):
        result = deepcopy(value)
        convert = self.decode_pointer if inverse else self.encode_pointer
        for op in result if isinstance(result, list) else []:
            params = op.get('params', {}) if isinstance(op, dict) else {}
            if not isinstance(params, dict): continue
            for key in set(params) & set(pointer_keys):
                try: params[key] = convert(params[key])
                except ValueError:
                    if strict: raise
                    # Invalid model pointers stay invalid; never repair them with a gold ID.
        return result

    def public(self, original):
        allowed = {'instruction', 'entities', 'images', 'movable', 'positions',
                   'entity_types', 'initial_relations', 'visible_entity_ids'}
        if set(original)-allowed:
            raise ValueError('unsupported public fields; scoring data must remain separate')
        public = deepcopy(original); n = len(self.order)
        names = [f'entity_{i:03d}' for i in range(n)]
        if public['entities'] != names:
            raise ValueError('anonymous canonical public entity ordering required')
        aliases = {name: names[self.encode_pointer(i)] for i, name in enumerate(names)}
        public['entities'] = names
        public['movable'] = sorted(aliases[name] for name in public['movable'])
        for key in ('positions', 'entity_types'):
            if key in public: public[key] = {aliases[name]: value for name, value in public[key].items()}
        if 'initial_relations' in public:
            public['initial_relations'] = sorted([predicate, aliases[left], aliases[right]]
                                                for predicate, left, right in public['initial_relations'])
        if 'visible_entity_ids' in public:
            public['visible_entity_ids'] = sorted(self.encode_pointer(i) for i in public['visible_entity_ids'])
        def replace(match):
            name = match.group(0)
            if name not in aliases: raise ValueError('instruction mentions an unknown entity ID')
            return aliases[name]
        public['instruction'] = re.sub(r'\bentity_\d{3}\b', replace, public['instruction'])
        return public


def deterministic_permutations(episode_id, count, *, seed=42, trials=5):
    if count < 2 or trials < 1:
        raise ValueError('counterfactual pick/place needs at least two entities and one trial')
    identity = tuple(range(count)); unique = min(trials, math.factorial(count)-1)
    rng = random.Random(int(digest([seed, episode_id, 'entity-permutations-v1']), 16))
    orders = []
    if count <= 3:
        choices = [p for p in permutations(identity) if p != identity]
        rng.shuffle(choices); orders = choices[:unique]
    else:
        while len(orders) < unique:
            proposal = list(identity); rng.shuffle(proposal); proposal = tuple(proposal)
            if proposal != identity and proposal not in orders: orders.append(proposal)
    return [EntityPermutation(orders[i % unique]) for i in range(trials)]


def verified_episode_permutation(episode, public):
    """Validate the observed transformation without opening a scoring label."""
    from .protocol import public_input, sha
    if any(sha(path) != expected for path, expected in episode['files'].items()):
        raise ValueError('frozen permutation input or reference differs')
    if sha(episode['scoring_private']) != episode['original_reference_sha256']:
        raise ValueError('permutation reference changed')
    permutation = EntityPermutation(episode['new_to_old'])
    if list(permutation.old_to_new) != episode['old_to_new']:
        raise ValueError('inverse pointer map differs')
    canonical_record = {**episode, 'generation_input': episode['canonical_generation_input']}
    canonical = public_input(canonical_record)
    expected = permutation.public(canonical)
    if public['images'][0] != canonical['images'][0] or len(public['images']) != 2:
        raise ValueError('permutation must preserve the original raw view')
    proof = episode['annotation_proof']
    if (sha(public['images'][1]) != proof['counterfactual_sha256']
            or sha(canonical['images'][1]) != proof['numbered_sha256']
            or list(permutation.order) != proof['new_to_old']):
        raise ValueError('permuted marker image or binding differs')
    expected['images'] = public['images']
    if expected != public:
        raise ValueError('entity tables, instruction or scene bindings were not permuted consistently')
    return permutation
