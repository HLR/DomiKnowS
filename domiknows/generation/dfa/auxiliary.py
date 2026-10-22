"""Existential projection of local, one-hot auxiliary token-state formulas.

Recognizes an andL family containing exactL over every value of an auxiliary
EnumConcept on token positions, unary token/state formulas, a sequence_start
implication, and immediate-successor implications over that same enum. Names
and enforcement metadata have no semantic role. All shared auxiliary values
are solved together before projection; intersecting their projections would
incorrectly allow incompatible witnesses for different constraints.
"""
from collections import deque

from .core import DFA


class AuxiliaryFormulaError(ValueError):
    """A formula uses auxiliary states outside the supported local fragment."""


def _expression(lc):
    """Read actual LC operands, including the flattened concept/V pairs."""
    operands = []
    elements = list(lc.e)
    index = 0
    while index < len(elements):
        item = elements[index]
        if hasattr(item, 'e'):
            operands.append(_expression(item))
        elif isinstance(item, tuple) and len(item) == 4:
            if index + 1 >= len(elements) or not hasattr(elements[index + 1], 'v'):
                raise AuxiliaryFormulaError('auxiliary atoms require explicit position variables')
            variable = elements[index + 1]
            if variable.relVarInfo is not None:
                raise AuxiliaryFormulaError('auxiliary atoms do not support relation-variable bindings')
            operands.append(('atom', item[0], item[2], variable.name, variable.v))
            index += 1
        elif isinstance(item, int):
            operands.append(('number', item))
        else:
            raise AuxiliaryFormulaError(f'unsupported auxiliary operand: {item!r}')
        index += 1
    return (type(lc).__name__, *operands)


def _atoms(expr):
    if expr[0] == 'atom':
        yield expr
    else:
        for child in expr[1:]:
            if isinstance(child, tuple):
                yield from _atoms(child)


def _truth(expr, atom, universe):
    op, *children = expr
    if op == 'atom':
        return atom(expr)
    if op not in {'andL', 'orL', 'notL', 'ifL'}:
        raise AuxiliaryFormulaError(f'unsupported auxiliary Boolean operator: {op}')
    values = [_truth(child, atom, universe) for child in children]
    if op == 'andL':
        result = universe
        for value in values:
            result &= value
        return result
    if op == 'orL':
        result = 0
        for value in values:
            result |= value
        return result
    if op == 'notL' and len(values) == 1:
        return universe ^ values[0]
    if op == 'ifL' and len(values) == 2:
        return (universe ^ values[0]) | values[1]
    raise AuxiliaryFormulaError(f'invalid auxiliary Boolean arity: {op}')


def _family_domain(lc, bundle):
    if type(lc).__name__ != 'andL':
        return None
    for child in lc.e:
        if type(child).__name__ != 'exactL':
            continue
        atoms = [item for item in child.e if isinstance(item, tuple) and len(item) == 4]
        if not atoms:
            continue
        auxiliary = atoms[0][0]
        from domiknows.graph import EnumConcept
        if (auxiliary is not bundle.generated_token and isinstance(auxiliary, EnumConcept)
                and any(rel.dst is bundle.token for rel in auxiliary._out.get('is_a', ()))):
            return auxiliary
    return None


def compile_auxiliary_state_families(graph, bundle):
    """Jointly project all head formulas sharing an auxiliary enum domain."""
    from .graph_discovery import _is_latent_marked, _walk_lc
    heads = {
        name: lc for name, lc in graph.logicalConstrains.items()
        if getattr(lc, 'headLC', True)
        and not getattr(lc, '_generation_contextual_dfa_specs', ())
        and not (_is_latent_marked(lc) and not hasattr(lc, '_generation_dfa_constraint'))
    }
    domains = {}
    for name, lc in heads.items():
        auxiliary = _family_domain(lc, bundle)
        if auxiliary is not None:
            domains.setdefault(auxiliary, name)
    compiled = {}
    for auxiliary, primary in domains.items():
        related = {name: lc for name, lc in heads.items() if any(
            isinstance(item, tuple) and len(item) == 4 and item[0] is auxiliary for item in _walk_lc(lc)
        )}
        if set(related) & set(compiled):
            raise AuxiliaryFormulaError('a head formula cannot mix auxiliary enum domains')
        dfa = compile_auxiliary_state_family(heads[primary], bundle, companions=[
            lc for name, lc in related.items() if name != primary
        ])
        compiled.update({name: (() if name != primary else (dfa,)) for name in related})
    return compiled


def compile_auxiliary_state_family(lc, bundle, *, companions=()):
    """Return a projected DFA, or None when this is not an auxiliary family.

    The exactL declaration explicitly establishes the one-hot domain. Every
    other formula must be unary, start-bound, or pair-bound over this domain;
    malformed or mixed families fail instead of silently weakening constraints.
    """
    if type(lc).__name__ != 'andL':
        return None
    domains = [child for child in lc.e if type(child).__name__ == 'exactL']
    if not domains:
        return None
    domain = _expression(domains[0])
    atoms = list(_atoms(domain))
    if not atoms or atoms[0][1] is bundle.generated_token:
        return None
    auxiliary = atoms[0][1]
    from domiknows.graph import EnumConcept
    if not isinstance(auxiliary, EnumConcept):
        return None
    if not any(rel.dst is bundle.token for rel in auxiliary._out.get('is_a', ())):
        return None
    size = len(auxiliary.enum)
    if (len(domains) != 1 or domain[-1] != ('number', 1)
            or len(atoms) != size or {a[2] for a in atoms} != set(range(size))
            or any(a[1] is not auxiliary or a[4] is not None or a[3] != atoms[0][3] for a in atoms)):
        raise AuxiliaryFormulaError('auxiliary exactL must cover every enum value once at one position')

    state_mask = (1 << size) - 1
    alphabet = frozenset(range(bundle.vocabulary.label_count))
    label_mask = (1 << len(alphabet)) - 1
    unary, starts, pairs = [], [], []
    children = [child for child in lc.e if child is not domains[0]]
    for companion in companions:
        if _family_domain(companion, bundle) is auxiliary:
            for child in companion.e:
                if type(child).__name__ == 'exactL':
                    if _expression(child) != domain:
                        raise AuxiliaryFormulaError('shared auxiliary domains require identical exactL declarations')
                else:
                    children.append(child)
        else:
            children.append(companion)
    for child in children:
        expr = _expression(child)
        if expr[0] == 'ifL' and expr[1][0] == 'atom' and expr[1][1] in (bundle.sequence_start, bundle.is_before_rel):
            relation = expr[1]
            if len(expr) != 3 or relation[2] is not None or relation[4] is not None:
                raise AuxiliaryFormulaError('invalid auxiliary relation guard')
            roles = ((bundle.sequence_start_token,) if relation[1] is bundle.sequence_start
                     else (bundle.first_token, bundle.second_token))
            for a in _atoms(expr[2]):
                if a[1] is not auxiliary or a[2] not in range(size) or a[4] not in tuple((relation[3], role) for role in roles):
                    raise AuxiliaryFormulaError('auxiliary relation operands must use the guarded endpoint paths')
            (starts if relation[1] is bundle.sequence_start else pairs).append(expr[2])
        else:
            variables = set()
            for a in _atoms(expr):
                if a[1] not in (auxiliary, bundle.generated_token) or a[4] is not None or a[2] is None:
                    raise AuxiliaryFormulaError('unary auxiliary formulas require pathless enum predicates')
                variables.add(a[3])
            if len(variables) != 1:
                raise AuxiliaryFormulaError('unary auxiliary operands must share a position variable')
            unary.append(expr)

    # Unary formulas provide the labels compatible with each possible witness.
    allowed_labels = []
    for state in range(size):
        def unary_atom(a):
            return (label_mask if a[2] == state else 0) if a[1] is auxiliary else 1 << a[2]
        allowed = label_mask
        for expr in unary:
            allowed &= _truth(expr, unary_atom, label_mask)
        allowed_labels.append(allowed)

    initial = state_mask
    for expr in starts:
        initial &= _truth(expr, lambda a: 1 << a[2], state_mask)
    successors = []
    for state in range(size):
        def pair_atom(a):
            return (state_mask if state == a[2] else 0) if a[4][1] is bundle.first_token else 1 << a[2]
        allowed = state_mask
        for expr in pairs:
            allowed &= _truth(expr, pair_atom, state_mask)
        successors.append(allowed)

    compatible = {label: sum(1 << state for state, labels in enumerate(allowed_labels) if labels & (1 << label))
                  for label in alphabet}
    # Subset construction existentially eliminates the auxiliary assignments.
    start = -1
    states, transitions = {start}, {}
    queue = deque([start])
    while queue:
        source = queue.popleft()
        possible = initial if source == start else 0
        if source != start:
            remaining = source
            while remaining:
                bit = remaining & -remaining
                possible |= successors[bit.bit_length() - 1]
                remaining ^= bit
        for label in alphabet:
            target = possible & compatible[label]
            if not target:
                continue
            transitions[source, label] = target
            if target not in states:
                states.add(target)
                queue.append(target)
    # Local universal formulas accept all realizable prefixes. Termination is
    # an ordinary EOS existence LC, compiled alongside this projected family.
    return DFA(frozenset(states), alphabet, transitions, start, frozenset(states))
