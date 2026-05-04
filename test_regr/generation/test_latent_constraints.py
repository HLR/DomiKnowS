import pytest
import torch

from domiknows.generation import (
    chain_exists_loss,
    implication_loss,
    soft_and,
    soft_exists,
    soft_or,
    window_all_loss,
    window_any_loss,
    window_formula_loss,
)


PER, ORG, LOC, DATE, O = range(5)


def empty_probs(seq_len=5):
    probs = torch.zeros((seq_len, 5), dtype=torch.float32)
    probs[:, O] = 1.0
    return probs


def test_soft_logic_primitives_use_product_semantics():
    a = torch.tensor(0.2)
    b = torch.tensor(0.5)
    assert soft_and(a, b).item() == pytest.approx(0.1)
    assert soft_or(a, b).item() == pytest.approx(0.6)
    assert implication_loss(torch.tensor(0.7), torch.tensor(0.8)).item() == pytest.approx(0.14)

    probs = torch.tensor(
        [
            [0.0, 0.2],
            [0.0, 0.5],
            [0.0, 0.1],
        ]
    )
    assert soft_exists(probs, label=1, start=0, end=3).item() == pytest.approx(0.64)


def test_chain_exists_loss_is_low_when_label_appears_in_window():
    satisfied = empty_probs()
    satisfied[0, PER] = 0.9
    satisfied[2, ORG] = 0.9

    violated = empty_probs()
    violated[0, PER] = 0.9

    satisfied_loss = chain_exists_loss(satisfied, PER, ORG, window=3)
    violated_loss = chain_exists_loss(violated, PER, ORG, window=3)

    assert satisfied_loss.item() == pytest.approx(0.018, abs=1e-6)
    assert violated_loss.item() == pytest.approx(0.18, abs=1e-6)
    assert satisfied_loss.item() < violated_loss.item()


def test_window_all_loss_requires_each_label_in_the_window():
    satisfied = empty_probs()
    satisfied[0, ORG] = 0.8
    satisfied[1, LOC] = 0.9
    satisfied[3, DATE] = 0.5

    missing_date = empty_probs()
    missing_date[0, ORG] = 0.8
    missing_date[1, LOC] = 0.9

    satisfied_loss = window_all_loss(satisfied, ORG, [LOC, DATE], window=4)
    missing_loss = window_all_loss(missing_date, ORG, [LOC, DATE], window=4)

    assert satisfied_loss.item() == pytest.approx(0.088, abs=1e-6)
    assert missing_loss.item() == pytest.approx(0.16, abs=1e-6)
    assert satisfied_loss.item() < missing_loss.item()


def test_window_any_loss_accepts_either_candidate_label():
    satisfied = empty_probs()
    satisfied[0, PER] = 0.8
    satisfied[2, ORG] = 0.6

    violated = empty_probs()
    violated[0, PER] = 0.8

    satisfied_loss = window_any_loss(satisfied, PER, [ORG, LOC], window=3)
    violated_loss = window_any_loss(violated, PER, [ORG, LOC], window=3)

    assert satisfied_loss.item() == pytest.approx(0.064, abs=1e-6)
    assert violated_loss.item() == pytest.approx(0.16, abs=1e-6)
    assert satisfied_loss.item() < violated_loss.item()


def test_window_formula_loss_supports_nested_and_or_formulas():
    probs = empty_probs()
    probs[0, PER] = 0.8
    probs[1, ORG] = 0.5
    probs[2, DATE] = 0.4
    probs[3, LOC] = 0.2

    loss = window_formula_loss(
        probs,
        if_label=PER,
        formula=("or", ("and", ORG, DATE), ("and", LOC, DATE)),
        window=4,
    )

    expected_rhs = 1.0 - (1.0 - 0.5 * 0.4) * (1.0 - 0.2 * 0.4)
    expected_loss = 0.8 * (1.0 - expected_rhs) / 5.0
    assert loss.item() == pytest.approx(expected_loss, abs=1e-6)


def test_batched_latent_losses_support_none_and_mean_reductions():
    satisfied = empty_probs()
    satisfied[0, PER] = 0.9
    satisfied[2, ORG] = 0.9

    violated = empty_probs()
    violated[0, PER] = 0.9

    probs = torch.stack([satisfied, violated], dim=0)
    per_batch = chain_exists_loss(probs, PER, ORG, window=3, reduction="none")
    mean_loss = chain_exists_loss(probs, PER, ORG, window=3, reduction="mean")

    assert per_batch.shape == (2,)
    assert per_batch[0].item() == pytest.approx(0.018, abs=1e-6)
    assert per_batch[1].item() == pytest.approx(0.18, abs=1e-6)
    assert mean_loss.item() == pytest.approx(per_batch.mean().item(), abs=1e-6)
