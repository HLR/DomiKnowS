"""Credit-assignment evidence: exact semantic loss (R2) vs t-norm relaxations.

This is the micro-benchmark behind the whole "R" line. The original concern was
that when a single constraint couples several concepts, the t-norm constraint
loss cannot give each concept's classifier a *correct* correction, because the
per-concept split is an artifact of the fuzzy semantics rather than a
probabilistic quantity. Two things are demonstrated here:

1. **Product t-norm is not weighted model counting.** On a formula that is not
   decomposable/deterministic (a variable repeated across non-disjoint
   disjuncts) the product t-norm disagrees with the exact model count, so it is
   not computing "the probability the constraint holds" at all.

2. **Gödel implication gives the antecedent exactly zero gradient.** On a
   violated implication ``A -> C`` the Gödel loss cannot push the antecedent
   down — half of the available correction is structurally discarded. Exact
   semantic loss gives every involved concept a non-zero, sign-correct gradient.

Run with ``-s`` to print the gradient-attribution table.
"""

import itertools
import math

import pytest
import torch

from domiknows.solver.circuitBooleanMethods import CircuitLeaf, circuitBooleanMethods
from domiknows.solver.lcLossBooleanMethods import lcLossBooleanMethods

TNORMS = ('G', 'L', 'P')


def _binary(name, probability, instance=0):
    probability = torch.as_tensor(probability)
    key = (name, instance, 0)
    return CircuitLeaf(key, probability, ("binary", key), 1,
                       (1.0 - probability, probability))


def _brute_force(probabilities, predicate):
    """Exact WMC by enumeration — the ground truth both methods are judged against."""
    total = 0.0
    for assignment in itertools.product((False, True), repeat=len(probabilities)):
        weight = math.prod(p if v else 1.0 - p
                           for p, v in zip(probabilities, assignment))
        if predicate(*assignment):
            total += weight
    return total


def _tnorm(tnorm):
    processor = lcLossBooleanMethods()
    processor.current_device = torch.device('cpu')
    processor.setTNorm(tnorm)
    return processor


# ---------------------------------------------------------------------------
# 1. Product t-norm != exact WMC on a non-decomposable formula
# ---------------------------------------------------------------------------

def test_product_tnorm_diverges_from_exact_wmc_when_not_decomposable():
    """phi = (A and B) or (A and C).

    'A' appears in both disjuncts and the disjuncts are not mutually exclusive,
    so the formula is neither decomposable nor deterministic. The product t-norm
    treats the two conjunctions as independent events; exact WMC does not.
    """
    pA, pB, pC = 0.5, 0.5, 0.5

    exact_expected = _brute_force([pA, pB, pC],
                                  lambda a, b, c: (a and b) or (a and c))

    # Exact circuit WMC.
    circuit = circuitBooleanMethods(backend='bdd')
    circuit.begin_evaluation()
    A = _binary('A', pA); B = _binary('B', pB, 1); C = _binary('C', pC, 2)
    node = circuit.orVar(None,
                         circuit.andVar(None, A, B),
                         circuit.andVar(None, A, C))
    exact = circuit.wmc(node).item()

    # Product t-norm over the same expression tree.
    p = _tnorm('P')
    t = lambda x: torch.tensor([x], dtype=torch.float32)
    tnorm_value = p.orVar(None,
                          p.andVar(None, t(pA), t(pB)),
                          p.andVar(None, t(pA), t(pC)),
                          onlyConstrains=False).reshape(()).item()

    # The circuit is exact...
    assert exact == pytest.approx(exact_expected, abs=1e-7)
    # ...and analytically P(A and (B or C)) = 0.5 * 0.75
    assert exact == pytest.approx(0.375, abs=1e-7)
    # ...while the product t-norm is measurably wrong.
    assert tnorm_value == pytest.approx(0.4375, abs=1e-7)
    assert abs(tnorm_value - exact) > 0.05

    print(f'\n[non-decomposable phi=(A&B)|(A&C), p=.5]  '
          f'exact WMC={exact:.4f}  product t-norm={tnorm_value:.4f}  '
          f'error={tnorm_value - exact:+.4f}')


def test_repeated_variable_disjunction_is_exact():
    """phi = A or A must be exactly P(A); the product t-norm double counts."""
    pA = 0.5
    circuit = circuitBooleanMethods(backend='bdd')
    circuit.begin_evaluation()
    A = _binary('A', pA)
    exact = circuit.wmc(circuit.orVar(None, A, A)).item()

    p = _tnorm('P')
    t = torch.tensor([pA], dtype=torch.float32)
    tnorm_value = p.orVar(None, t, t, onlyConstrains=False).reshape(()).item()

    assert exact == pytest.approx(pA, abs=1e-7)          # exact: P(A or A) = P(A)
    assert tnorm_value == pytest.approx(0.75, abs=1e-7)  # 2p - p^2
    assert tnorm_value > exact


# ---------------------------------------------------------------------------
# 2. Per-concept gradient attribution on a violated implication
# ---------------------------------------------------------------------------

def _implication_gradients(logit_a, logit_c, method):
    """d(loss)/d(logit) for the antecedent and consequent of ``A -> C``.

    Mirrors the conll04 typing rule shape ``ifL(relation, entity)``: the
    relation is the antecedent, the entity the consequent.
    """
    la = torch.tensor([logit_a], dtype=torch.float32, requires_grad=True)
    lc = torch.tensor([logit_c], dtype=torch.float32, requires_grad=True)
    pa, pc = la.sigmoid(), lc.sigmoid()

    if method == 'semantic':
        circuit = circuitBooleanMethods(backend='bdd')
        circuit.begin_evaluation()
        A = _binary('A', pa.reshape(()))
        C = _binary('C', pc.reshape(()), 1)
        probability = circuit.wmc(circuit.ifVar(None, A, C))
        loss = -torch.log(probability)
    else:
        loss = _tnorm(method).ifVar(None, pa, pc, onlyConstrains=True).reshape(())

    loss.backward()
    # A ``None`` grad means the concept never entered the autograd graph at all
    # — an even stronger form of the defect than a zero gradient. Report it as
    # 0.0 so the table is comparable across methods.
    grad = lambda t: 0.0 if t.grad is None else float(t.grad)
    return loss.item(), grad(la), grad(lc)


def test_godel_gives_antecedent_zero_gradient_semantic_loss_does_not():
    """The central claim, as an assertion.

    Antecedent probability high, consequent low → the implication is violated,
    so a correct constraint loss should push the antecedent DOWN and the
    consequent UP. Gödel can only do the latter.
    """
    logit_a, logit_c = 2.0, -1.5  # p_a ~ 0.88, p_c ~ 0.18  → violated

    _, g_ante_godel, g_cons_godel = _implication_gradients(logit_a, logit_c, 'G')
    _, g_ante_sem, g_cons_sem = _implication_gradients(logit_a, logit_c, 'semantic')

    # Gödel: the antecedent enters only through a hard comparison.
    assert g_ante_godel == pytest.approx(0.0, abs=1e-9)
    assert g_cons_godel != pytest.approx(0.0, abs=1e-9)

    # Semantic loss: both concepts get a non-zero, sign-correct correction.
    assert g_ante_sem > 0      # descending the loss lowers the antecedent
    assert g_cons_sem < 0      # descending the loss raises the consequent


def test_all_tnorms_vs_semantic_gradient_table(capsys=None):
    """Print the full attribution table across t-norms and semantic loss."""
    logit_a, logit_c = 2.0, -1.5

    rows = []
    for method in TNORMS + ('semantic',):
        loss, g_a, g_c = _implication_gradients(logit_a, logit_c, method)
        rows.append((method, loss, g_a, g_c))

    header = f'\n{"method":10s} {"loss":>9s} {"d/d logit_antecedent":>22s} {"d/d logit_consequent":>22s}'
    print(header)
    print('-' * len(header))
    for method, loss, g_a, g_c in rows:
        label = {'G': 'Godel', 'L': 'Lukasiewicz', 'P': 'Product'}.get(method, 'SEMANTIC')
        print(f'{label:10s} {loss:9.4f} {g_a:22.6f} {g_c:22.6f}')
    print('\nviolated implication A->C with p_a~0.88, p_c~0.18;'
          '\npositive antecedent gradient = "lower the antecedent", '
          'negative consequent gradient = "raise the consequent".')

    by_method = {m: (g_a, g_c) for m, _, g_a, g_c in rows}
    # Only the exact loss gives a usable two-sided correction here.
    assert by_method['G'][0] == pytest.approx(0.0, abs=1e-9)
    assert by_method['semantic'][0] > 0 and by_method['semantic'][1] < 0
    for method in TNORMS + ('semantic',):
        assert all(math.isfinite(g) for g in by_method[method])


def test_semantic_gradient_matches_closed_form():
    """Sanity-check the exact gradient against the analytic derivative.

    P(A->C) = 1 - p_a(1 - p_c), loss = -log P, so
        dloss/dp_a = (1 - p_c)/P    and    dloss/dp_c = -p_a/P.
    """
    logit_a, logit_c = 2.0, -1.5
    _, g_a, g_c = _implication_gradients(logit_a, logit_c, 'semantic')

    pa = torch.tensor(logit_a).sigmoid().item()
    pc = torch.tensor(logit_c).sigmoid().item()
    P = 1.0 - pa * (1.0 - pc)
    # chain through the sigmoid: dp/dlogit = p(1-p)
    expected_a = ((1.0 - pc) / P) * pa * (1.0 - pa)
    expected_c = (-pa / P) * pc * (1.0 - pc)

    assert g_a == pytest.approx(expected_a, rel=1e-5)
    assert g_c == pytest.approx(expected_c, rel=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
