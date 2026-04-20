import torch

from domiknows.graph import iffL as iffL_export
from domiknows.graph import equivalenceL
from domiknows.graph.logicalConstrain import iffL
from domiknows.solver.booleanMethodsCalculator import booleanMethodsCalculator
from domiknows.solver.lcLossBooleanMethods import lcLossBooleanMethods


def _to_scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].item())
    return float(value)


def test_iff_export_and_alias():
    assert iffL_export is iffL
    assert issubclass(iffL, equivalenceL)


def test_iff_truth_table_boolean_backend():
    calc = booleanMethodsCalculator()
    calc.current_device = torch.device("cpu")

    # (A, B, expected A<->B)
    cases = [
        (1, 1, 1),
        (0, 0, 1),
        (1, 0, 0),
        (0, 1, 0),
    ]

    for a, b, expected in cases:
        assert calc.equivalenceVar(None, a, b) == expected


def test_iff_truth_table_loss_backend():
    calc = lcLossBooleanMethods()
    calc.current_device = torch.device("cpu")
    calc.setTNorm("P")

    # Hard truth assignments should produce hard biconditional outputs in [0, 1].
    cases = [
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]

    for a, b, expected_success in cases:
        a_t = torch.tensor([a], dtype=torch.float32, device=calc.current_device)
        b_t = torch.tensor([b], dtype=torch.float32, device=calc.current_device)

        success = calc.equivalenceVar(None, a_t, b_t, onlyConstrains=False)
        loss = calc.equivalenceVar(None, a_t, b_t, onlyConstrains=True)

        assert _to_scalar(success) == expected_success
        assert _to_scalar(loss) == (1.0 - expected_success)


def test_iff_edge_cases_boolean_backend():
    calc = booleanMethodsCalculator()
    calc.current_device = torch.device("cpu")

    # Vacuous edge cases
    assert calc.equivalenceVar(None) == 1
    assert calc.equivalenceVar(None, 1) == 1

    # None is treated as False in equivalenceVar.
    assert calc.equivalenceVar(None, None, 0) == 1
    assert calc.equivalenceVar(None, None, 1) == 0

    # N-ary behavior: true iff all values are identical.
    assert calc.equivalenceVar(None, 0, 0, 0) == 1
    assert calc.equivalenceVar(None, 1, 1, 1) == 1
    assert calc.equivalenceVar(None, 1, 1, 0) == 0
