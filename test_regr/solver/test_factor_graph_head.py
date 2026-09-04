"""R3 — factor-graph heads: exact constrained inference as the forward pass.

The failure mode is a head that trains well while the constraints do nothing
structural, so these checks are **exactness-based**, not metric-based. Two
findings from building this are encoded as tests, because both contradict a
plausible-sounding assumption:

1. **The arithmetic-circuit derivative identity does not hold here.**
   ``P(l|phi) = dZ/dw * w / Z`` needs a *smooth* circuit; these diagrams are
   reduced, and binary leaves tie ``p`` to ``1-p``. Marginals are therefore
   computed by conditioning, which is verified exact against brute force.

2. **Argmax of exact marginals can still violate the constraint.** MPM is not
   MAP. "Satisfied by construction" is a property of MAP decoding and of the
   distribution — not of a per-group argmax.

Run with ``-s`` to see the measured MPM violation rates.
"""

import itertools
import math

import pytest
import torch

from domiknows.solver.circuitBooleanMethods import CircuitLeaf, circuitBooleanMethods
from domiknows.program.model.factorGraphHead import (
    FactorGraphHead, VariableGroup, FactorGraphReport, semantic_loss,
)

BACKENDS = ('bdd', 'pysdd')


def _binary_leaf(index, probability):
    key = (f'v{index}', index, 0)
    return CircuitLeaf(key, probability, ('binary', key), 1,
                       (1.0 - probability, probability))


def _brute(probabilities, predicate, query=None):
    """Exact ``Z`` and ``P(var[query] | phi)`` by enumeration — ground truth."""
    total = held = 0.0
    for assignment in itertools.product((False, True), repeat=len(probabilities)):
        weight = math.prod(p if v else 1.0 - p
                           for p, v in zip(probabilities, assignment))
        if predicate(*assignment):
            total += weight
            if query is not None and assignment[query]:
                held += weight
    return total, (held / total if total > 0 else float('nan'))


def _brute_map(probabilities, predicate):
    best = None
    for assignment in itertools.product((False, True), repeat=len(probabilities)):
        if not predicate(*assignment):
            continue
        weight = math.prod(p if v else 1.0 - p
                           for p, v in zip(probabilities, assignment))
        if best is None or weight > best[0]:
            best = (weight, assignment)
    return best


#: (name, probabilities, build(processor, leaves), python predicate)
FORMULAS = [
    ('A and B', [0.7, 0.3],
     lambda c, L: c.andVar(None, L[0], L[1]), lambda a, b: a and b),
    ('A or B', [0.7, 0.3],
     lambda c, L: c.orVar(None, L[0], L[1]), lambda a, b: a or b),
    ('A -> B', [0.8, 0.2],
     lambda c, L: c.ifVar(None, L[0], L[1]), lambda a, b: (not a) or b),
    ('A -> (B and C)', [0.8, 0.2, 0.6],
     lambda c, L: c.ifVar(None, L[0], c.andVar(None, L[1], L[2])),
     lambda a, b, d: (not a) or (b and d)),
    ('(A&B)|(A&C)', [0.5, 0.5, 0.5],
     lambda c, L: c.orVar(None, c.andVar(None, L[0], L[1]),
                          c.andVar(None, L[0], L[2])),
     lambda a, b, d: (a and b) or (a and d)),
    ('nand(A,B)', [0.6, 0.6],
     lambda c, L: c.nandVar(None, L[0], L[1]), lambda a, b: not (a and b)),
    ('xor(A,B)', [0.3, 0.8],
     lambda c, L: c.xorVar(None, L[0], L[1]), lambda a, b: a != b),
    ('A&B | ~A&C', [0.4, 0.6, 0.9],
     lambda c, L: c.orVar(None, c.andVar(None, L[0], L[1]),
                          c.andVar(None, c.notVar(None, L[0]), L[2])),
     lambda a, b, d: (a and b) or ((not a) and d)),
    # A is the only real variable: B is simplified out of the diagram entirely,
    # which is what breaks the derivative identity.
    ('A (B free)', [0.7, 0.4],
     lambda c, L: c.andVar(None, L[0], c.orVar(None, L[1], c.notVar(None, L[1]))),
     lambda a, b: a),
]


# --------------------------------------------------------------------------- #
# Verification 1 — the marginal identity (this gates everything else)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name,probabilities,build,predicate', FORMULAS,
                         ids=[f[0] for f in FORMULAS])
def test_marginals_match_brute_force(backend, name, probabilities, build, predicate):
    """Constrained marginals equal brute-force ``P(literal | phi)``, exactly."""
    processor = circuitBooleanMethods(backend=backend)
    processor.begin_evaluation()
    tensors = [torch.tensor(p, dtype=torch.float64, requires_grad=True)
               for p in probabilities]
    leaves = [_binary_leaf(i, t) for i, t in enumerate(tensors)]
    node = build(processor, leaves)

    expected_z, _ = _brute(probabilities, predicate)
    assert processor.wmc(node).item() == pytest.approx(expected_z, abs=1e-12)

    marginals = processor.marginals(node, leaves)
    for i, marginal in enumerate(marginals):
        _, expected = _brute(probabilities, predicate, query=i)
        assert marginal.item() == pytest.approx(expected, abs=1e-10), (
            f'{name}: marginal for v{i}')


@pytest.mark.parametrize('backend', BACKENDS)
def test_categorical_marginals_match_brute_force(backend):
    """Multi-valued (EnumConcept) groups: exact, and normalised per variable."""
    dist_x = [0.5, 0.3, 0.2]
    dist_y = [0.2, 0.3, 0.5]
    processor = circuitBooleanMethods(backend=backend)
    processor.begin_evaluation()

    def group(name, instance, distribution):
        tensors = [torch.tensor(p, dtype=torch.float64) for p in distribution]
        return [CircuitLeaf((name, instance, k), tensors[k],
                            ('categorical', name, instance), k, tuple(tensors),
                            categorical=True)
                for k in range(len(distribution))]

    X = group('X', 0, dist_x)
    Y = group('Y', 1, dist_y)
    node = processor.ifVar(None, X[0], Y[2])          # X=0 -> Y=2

    def brute(query_var=None, query_val=None):
        total = held = 0.0
        for x, y in itertools.product(range(3), range(3)):
            if not ((x != 0) or (y == 2)):
                continue
            w = dist_x[x] * dist_y[y]
            total += w
            if query_var is not None and (x, y)[query_var] == query_val:
                held += w
        return total, (held / total if total else float('nan'))

    assert processor.wmc(node).item() == pytest.approx(brute()[0], abs=1e-12)

    for var_index, leaves in enumerate((X, Y)):
        marginals = processor.marginals(node, leaves)
        assert sum(m.item() for m in marginals) == pytest.approx(1.0, abs=1e-9)
        for k, marginal in enumerate(marginals):
            assert marginal.item() == pytest.approx(
                brute(var_index, k)[1], abs=1e-10)


@pytest.mark.parametrize('backend', BACKENDS)
def test_naive_derivative_of_wmc_is_unsound_which_is_why_smoothing_exists(backend):
    """The defect the smoothing fix addresses, pinned so it cannot regress.

    Differentiating the *plain* ``wmc`` w.r.t. the caller's source probability
    is unsound twice over: on ``A and (B or not B)`` the reduction removes ``B``
    so the gradient is ``None``, and a binary leaf ties ``p`` to ``1-p`` so the
    ratio mixes both literals. ``marginals`` avoids both — it smooths the
    evaluation and differentiates w.r.t. the registered branch weights.
    """
    processor = circuitBooleanMethods(backend=backend)
    processor.begin_evaluation()
    tensors = [torch.tensor(p, dtype=torch.float64, requires_grad=True)
               for p in (0.7, 0.4)]
    leaves = [_binary_leaf(i, t) for i, t in enumerate(tensors)]
    node = processor.andVar(None, leaves[0],
                            processor.orVar(None, leaves[1],
                                            processor.notVar(None, leaves[1])))
    grads = torch.autograd.grad(processor.wmc(node), tensors, allow_unused=True)
    assert grads[1] is None, 'B vanished from the diagram — the naive route fails'

    # Both supported methods still recover B's true (unconstrained) marginal.
    for method in ('auto', 'conditioning'):
        marginals = processor.marginals(node, leaves, method=method)
        assert marginals[1].item() == pytest.approx(0.4, abs=1e-10), method
        assert marginals[0].item() == pytest.approx(1.0, abs=1e-10), method


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name,probabilities,build,predicate', FORMULAS,
                         ids=[f[0] for f in FORMULAS])
def test_gradient_and_conditioning_marginals_agree(backend, name, probabilities,
                                                   build, predicate):
    """The fast identity and the assumption-free reference must not diverge."""
    processor = circuitBooleanMethods(backend=backend)
    processor.begin_evaluation()
    tensors = [torch.tensor(p, dtype=torch.float64, requires_grad=True)
               for p in probabilities]
    leaves = [_binary_leaf(i, t) for i, t in enumerate(tensors)]
    node = build(processor, leaves)

    fast = processor.marginals(node, leaves, method='auto')
    reference = processor.marginals(node, leaves, method='conditioning')
    for a, b in zip(fast, reference):
        assert a.item() == pytest.approx(b.item(), abs=1e-10), name


def test_gradient_marginals_stay_differentiable_for_training():
    """The identity must be usable *inside* a forward pass, not just for readout."""
    processor = circuitBooleanMethods(backend='bdd')
    processor.begin_evaluation()
    tensors = [torch.tensor(p, dtype=torch.float64, requires_grad=True)
               for p in (0.8, 0.2)]
    leaves = [_binary_leaf(i, t) for i, t in enumerate(tensors)]
    node = processor.ifVar(None, leaves[0], leaves[1])

    marginal = processor.marginals(node, leaves, method='gradient')[1]
    (-torch.log(marginal)).backward()
    for tensor in tensors:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


# --------------------------------------------------------------------------- #
# Phase C — MAP
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name,probabilities,build,predicate', FORMULAS,
                         ids=[f[0] for f in FORMULAS])
def test_map_matches_brute_force_argmax(backend, name, probabilities, build, predicate):
    """Max-product MAP equals brute-force argmax, in both value and assignment."""
    processor = circuitBooleanMethods(backend=backend)
    processor.begin_evaluation()
    tensors = [torch.tensor(p, dtype=torch.float64) for p in probabilities]
    leaves = [_binary_leaf(i, t) for i, t in enumerate(tensors)]
    node = build(processor, leaves)

    value, assignment = processor.map_assignment(node, leaves=leaves)
    expected_value, _ = _brute_map(probabilities, predicate)

    decoded = tuple(bool(assignment.get(('binary', (f'v{i}', i, 0)), 0))
                    for i in range(len(probabilities)))
    # Optima can tie (``nand`` with equal probabilities has two), so require an
    # optimum — matching value and a satisfying assignment — not a specific one.
    assert float(value) == pytest.approx(expected_value, abs=1e-12), name
    assert predicate(*decoded), f'{name}: MAP assignment violates the constraint'
    achieved = math.prod(p if v else 1.0 - p
                         for p, v in zip(probabilities, decoded))
    assert achieved == pytest.approx(expected_value, abs=1e-12), name


def test_map_charges_variables_the_reduction_removed():
    """A skipped variable must still be scored — max has no 'sums to one' identity.

    ``A -> B`` with ``p_A=0.8, p_B=0.2``: scoring ``A=False`` without charging
    ``B`` gives 0.2 and wins, but the true optimum is ``A=B=True`` at 0.16.
    """
    processor = circuitBooleanMethods(backend='bdd')
    processor.begin_evaluation()
    tensors = [torch.tensor(p, dtype=torch.float64) for p in (0.8, 0.2)]
    leaves = [_binary_leaf(i, t) for i, t in enumerate(tensors)]
    node = processor.ifVar(None, leaves[0], leaves[1])

    value, assignment = processor.map_assignment(node, leaves=leaves)
    assert float(value) == pytest.approx(0.16, abs=1e-12)
    assert assignment[('binary', ('v0', 0, 0))] == 1
    assert assignment[('binary', ('v1', 1, 0))] == 1


def test_unsatisfiable_constraint_reports_no_assignment():
    processor = circuitBooleanMethods(backend='bdd')
    processor.begin_evaluation()
    t = torch.tensor(0.5, dtype=torch.float64)
    leaf = _binary_leaf(0, t)
    node = processor.andVar(None, leaf, processor.notVar(None, leaf))
    value, assignment = processor.map_assignment(node, leaves=[leaf])
    assert float(value) == 0.0
    assert assignment == {}


@pytest.mark.parametrize('backend', BACKENDS)
def test_map_is_available_on_every_shipped_backend(backend):
    """Both managers smooth explicitly, so both support exact MAP."""
    processor = circuitBooleanMethods(backend=backend)
    assert processor.supports_map is True
    assert backend in circuitBooleanMethods.MAP_BACKENDS


@pytest.mark.parametrize('backend', BACKENDS)
def test_map_on_categorical_groups_matches_brute_force(backend):
    """Multi-valued groups: exactly-one holds and the argmax is the true optimum."""
    dist_x = [0.5, 0.3, 0.2]
    dist_y = [0.2, 0.3, 0.5]
    processor = circuitBooleanMethods(backend=backend)
    processor.begin_evaluation()

    def group(name, instance, distribution):
        tensors = [torch.tensor(p, dtype=torch.float64) for p in distribution]
        return [CircuitLeaf((name, instance, k), tensors[k],
                            ('categorical', name, instance), k, tuple(tensors),
                            categorical=True)
                for k in range(len(distribution))]

    X = group('X', 0, dist_x)
    Y = group('Y', 1, dist_y)
    node = processor.ifVar(None, X[0], Y[2])          # X=0 -> Y=2

    best = max(((dist_x[x] * dist_y[y], (x, y))
                for x, y in itertools.product(range(3), range(3))
                if (x != 0) or (y == 2)), key=lambda item: item[0])

    value, assignment = processor.map_assignment(node, leaves=X + Y)
    decoded = (assignment[('categorical', 'X', 0)],
               assignment[('categorical', 'Y', 1)])
    assert decoded == best[1]
    assert float(value) == pytest.approx(best[0], abs=1e-12)


# --------------------------------------------------------------------------- #
# The head
# --------------------------------------------------------------------------- #

E = 5  # conll04-shaped entity_label / pair_label width


def _typing_head(backend='bdd', **kwargs):
    """`work_for(pair) -> people(e0) and organization(e1)`, as a factor graph."""
    groups = [VariableGroup('e0', E), VariableGroup('e1', E), VariableGroup('p', E)]

    def build(processor, leaves):
        return processor.ifVar(
            None, leaves['p'][0],
            processor.andVar(None, leaves['e0'][0], leaves['e1'][1]))

    return FactorGraphHead(groups, build, backend=backend,
                           name='work_for_typing', **kwargs)


def _random_beliefs(head, sharpness=1.0, requires_grad=False):
    out = {}
    for group in head.groups:
        t = torch.softmax(torch.randn(group.size) * sharpness, dim=-1)
        out[group.name] = t.requires_grad_(True) if requires_grad else t
    return out


def test_head_returns_normalised_marginals_and_reports_exact():
    torch.manual_seed(0)
    head = _typing_head()
    out = head(_random_beliefs(head))
    for group in head.groups:
        assert out[group.name].shape == (group.size,)
        assert out[group.name].sum().item() == pytest.approx(1.0, abs=1e-6)
    assert head.report.exact == ['work_for_typing']
    assert head.report.fallback == []
    assert head.report.exact_fraction == 1.0


def test_map_decode_never_violates_the_constraint():
    """Verification 2, in its sound form: MAP is constraint-respecting, always."""
    torch.manual_seed(1)
    head = _typing_head()
    for _ in range(200):
        decode = head.map_predict(_random_beliefs(head, sharpness=3.0))
        if decode['p'] == 0:
            assert decode['e0'] == 0 and decode['e1'] == 1


def test_argmax_of_marginals_can_violate_even_though_marginals_are_exact(capsys):
    """MPM is not MAP — the plan's 'argmax must never violate' is false as stated.

    The distribution *is* conditioned (marginals verified exact above), but a
    per-group argmax is a factorised readout of a non-factorised conditional and
    can land on an assignment of zero posterior probability. This test pins the
    real behaviour so nobody later 'fixes' the decoder by taking argmax.
    """
    torch.manual_seed(0)
    groups = [VariableGroup(f'g{i}', 2) for i in range(3)]

    def build(processor, leaves):
        return processor.xorVar(None, leaves['g0'][1],
                                leaves['g1'][1], leaves['g2'][1])

    head = FactorGraphHead(groups, build, backend='bdd', name='exactly_one')

    mpm_violations = map_violations = 0
    trials = 400
    for _ in range(trials):
        beliefs = _random_beliefs(head, sharpness=3.0)
        marginal_decode = {k: int(v.argmax()) for k, v in head(beliefs).items()}
        if sum(marginal_decode.values()) != 1:
            mpm_violations += 1
        if sum(head.map_predict(beliefs).values()) != 1:
            map_violations += 1

    print(f'\nexactly-one over {trials} random inputs: '
          f'MPM violations={mpm_violations}, MAP violations={map_violations}')

    assert map_violations == 0, 'MAP must never violate'
    assert mpm_violations > 0, (
        'expected argmax-of-marginals to violate sometimes; if this ever stops '
        'happening the claim in FactorGraphHead\'s warning needs revisiting')


def test_head_conditioning_moves_beliefs_toward_satisfaction():
    """Conditioning reduces the constraint's violation relative to the input."""
    torch.manual_seed(3)
    head = _typing_head()
    reduced = 0
    for _ in range(50):
        beliefs = _random_beliefs(head, sharpness=2.0)
        before = float(semantic_loss(head, beliefs))
        after = float(semantic_loss(head, head(beliefs)))
        if after <= before + 1e-9:
            reduced += 1
    assert reduced == 50, 'conditioning should never increase -log P(phi)'


def test_semantic_loss_is_much_smaller_on_the_conditioned_beliefs():
    """R2's term shrinks sharply under an R3 head — but is not identically zero.

    The plan expected ~0. That holds for the *joint*; a factorised readout of a
    non-factorised conditional cannot represent it exactly, so the honest claim
    is a large reduction, not zero. (Exact satisfaction is MAP's property.)
    """
    torch.manual_seed(0)
    head = _typing_head()
    befores, afters = [], []
    for _ in range(30):
        beliefs = _random_beliefs(head, sharpness=2.0)
        befores.append(float(semantic_loss(head, beliefs)))
        afters.append(float(semantic_loss(head, head(beliefs))))
    assert sum(afters) < 0.5 * sum(befores)


def test_batched_marginals_equal_per_row_marginals():
    """One circuit with ``[R, K]`` weights must equal R separate scalar passes.

    The sum-product recursion is pure broadcasting arithmetic, and row ``r``'s
    partition depends only on row ``r``'s weights, so ``grad(Z.sum(), w)`` is the
    per-row derivative. This is the identity the batching rests on — if it ever
    breaks, exact per-grounding inference silently becomes approximate.
    """
    torch.manual_seed(0)
    K, R = 3, 40
    groups = [VariableGroup('v0', K), VariableGroup('v1', K), VariableGroup('v2', K)]

    def build(processor, leaves):
        return processor.ifVar(None, leaves['v0'][0],
                               processor.andVar(None, leaves['v1'][0],
                                                leaves['v2'][1]))

    beliefs = {g.name: torch.softmax(torch.randn(R, K), dim=-1) for g in groups}

    per_row = {g.name: [] for g in groups}
    head = FactorGraphHead(groups, build, backend='bdd')
    for r in range(R):
        out = head({g.name: beliefs[g.name][r] for g in groups})
        for g in groups:
            per_row[g.name].append(out[g.name])

    batched = FactorGraphHead(groups, build, backend='bdd')(beliefs)
    for g in groups:
        reference = torch.stack(per_row[g.name])
        assert batched[g.name].shape == (R, K)
        assert torch.allclose(batched[g.name], reference, atol=1e-6)
        assert torch.allclose(batched[g.name].sum(-1), torch.ones(R), atol=1e-5)


def test_batched_marginals_stay_differentiable():
    """Batching must not sever the path back to the heads."""
    torch.manual_seed(0)
    K, R = 2, 16
    groups = [VariableGroup('v0', K), VariableGroup('v1', K)]

    def build(processor, leaves):
        return processor.ifVar(None, leaves['v0'][1], leaves['v1'][1])

    beliefs = {g.name: torch.softmax(torch.randn(R, K), dim=-1).requires_grad_(True)
               for g in groups}
    out = FactorGraphHead(groups, build, backend='bdd')(beliefs)
    (-torch.log(out['v0'][:, 1].clamp_min(1e-30))).sum().backward()
    for name, tensor in beliefs.items():
        assert tensor.grad is not None and tensor.grad.abs().sum() > 0, name


def test_gradients_reach_every_participating_concept():
    """The credit-assignment claim: each concept gets a constrained-joint gradient."""
    torch.manual_seed(0)
    head = _typing_head()
    beliefs = _random_beliefs(head, requires_grad=True)
    out = head(beliefs)
    (-torch.log(out['e0'][0].clamp_min(1e-30))).backward()
    for name, tensor in beliefs.items():
        assert tensor.grad is not None, f'{name} received no gradient'
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0, f'{name} received a zero gradient'


def test_circuit_size_limit_falls_back_and_reports_instead_of_approximating():
    """Verification: never silently approximate — the fallback is reported."""
    torch.manual_seed(0)
    head = _typing_head(max_nodes=1)
    beliefs = _random_beliefs(head)
    out = head(beliefs)
    for name, tensor in beliefs.items():
        assert torch.allclose(out[name], tensor)      # untouched pass-through
    assert head.report.exact == []
    assert len(head.report.fallback) == 1
    assert 'circuit-size-limit' in head.report.fallback[0]
    assert head.report.exact_fraction == 0.0


def test_unsatisfiable_head_falls_back_rather_than_dividing_by_zero():
    groups = [VariableGroup('g', 2)]

    def build(processor, leaves):
        return processor.andVar(None, leaves['g'][0],
                                processor.notVar(None, leaves['g'][0]))

    head = FactorGraphHead(groups, build, backend='bdd', name='contradiction')
    beliefs = {'g': torch.tensor([0.5, 0.5])}
    out = head(beliefs)
    assert torch.allclose(out['g'], beliefs['g'])
    assert 'unsatisfiable' in head.report.fallback[0]


def test_report_renders_the_enforced_subset():
    report = FactorGraphReport(exact=['a', 'b'], fallback=['c'])
    assert report.exact_fraction == pytest.approx(2 / 3)
    text = report.render()
    assert '2 constraint(s) enforced structurally' in text
    assert 'c' in text


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
