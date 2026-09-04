"""Grounding-aligned feature extraction for the amortized DualCritic."""

import pytest
import torch

from domiknows.solver.compiled.formula import CompiledConstraintEvaluator


def _matrix(layout, groundings):
    return CompiledConstraintEvaluator._layout_feature_matrix(
        layout, groundings)


def test_ragged_candidate_groups_compact_missing_rows():
    features = _matrix([
        [torch.tensor(0.2)],
        [None],
        [torch.tensor(0.8)],
    ], 2)
    assert torch.equal(features, torch.tensor([[0.2], [0.8]]))


def test_path_expansion_repeats_earlier_grounding_features():
    features = _matrix([[torch.tensor([0.2, 0.8])]], 6)
    assert torch.equal(
        features[:, 0], torch.tensor([0.2, 0.2, 0.2, 0.8, 0.8, 0.8]))


def test_reduction_retains_every_collapsed_literal_as_a_feature():
    features = _matrix([[torch.arange(6, dtype=torch.float32)]], 2)
    assert torch.equal(features, torch.tensor([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
    ]))


def test_enum_columns_remain_parallel_features():
    features = _matrix([[
        torch.tensor([0.1, 0.7]),
        torch.tensor([0.9, 0.3]),
    ]], 2)
    assert torch.equal(features, torch.tensor([
        [0.1, 0.9],
        [0.7, 0.3],
    ]))


def test_counting_groups_keep_ragged_literal_rows():
    features = _matrix([
        [torch.tensor(0.1), torch.tensor(0.2)],
        [torch.tensor(0.7)],
    ], 2)
    assert torch.equal(features[0], torch.tensor([0.1, 0.2]))
    assert features[1, 0] == pytest.approx(0.7)
    assert torch.isnan(features[1, 1])


def test_non_integral_feature_layout_fails_loudly():
    with pytest.raises(ValueError, match="cannot align"):
        _matrix([[torch.arange(5, dtype=torch.float32)]], 3)
