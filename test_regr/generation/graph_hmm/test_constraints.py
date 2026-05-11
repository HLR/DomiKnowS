"""
Tests for constraint utility functions in graph_hmm.

This module tests the core utility functions used to apply constraints in HMM models:
- Distribution projection: Ensures probability distributions respect allowed states
- Matrix operations: Projects transition/emission matrices to valid states
- Mask validation: Validates constraint masks
- Mask combination: Combines multiple constraint masks

These utilities are fundamental to enforcing DomiKnows constraints during inference.
"""
import torch
import pytest

from domiknows.generation.graph_hmm import combine_masks, project_distribution, project_matrix, project_matrix_rows, validate_mask


def test_project_distribution_preserves_forbidden_entries():
    """Test that project_distribution respects forbidden entries (where mask=0)."""
    projected = project_distribution(
        torch.tensor([0.2, 0.8], dtype=torch.float64),
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(projected, torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_project_distribution_falls_back_on_zero_allowed_mass():
    """Test that project_distribution handles edge case of zero mass in allowed states."""
    projected = project_distribution(
        torch.tensor([0.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    assert projected[0].item() == 0.0
    assert projected[1].item() == 1.0


def test_project_distribution_preserves_all_zero_mask_row():
    """Test that fully forbidden rows remain zero instead of being re-enabled."""
    projected = project_distribution(
        torch.tensor([0.2, 0.8], dtype=torch.float64),
        torch.tensor([0.0, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(projected, torch.tensor([0.0, 0.0], dtype=torch.float64))


def test_project_matrix_rows_projects_each_row():
    """Test that project_matrix_rows applies projection independently to each row."""
    projected = project_matrix_rows(
        torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
    )
    assert torch.allclose(projected[0], torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert torch.allclose(projected[1], torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_project_matrix_alias_projects_rows():
    """Test that project_matrix is an alias for project_matrix_rows."""
    projected = project_matrix(
        torch.tensor([[1.0, 9.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
    )
    assert torch.allclose(projected, torch.tensor([[1.0, 0.0]], dtype=torch.float64))


def test_validate_mask_rejects_bad_values():
    """Test that validate_mask rejects invalid mask values (negative numbers)."""
    with pytest.raises(ValueError, match="non-negative"):
        validate_mask(torch.tensor([-1.0]), (1,))


def test_combine_masks_multiplies_masks():
    """Test that combine_masks element-wise multiplies constraint masks."""
    combined = combine_masks(
        (torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 1.0]])),
        (1, 2),
    )
    assert torch.allclose(combined, torch.tensor([[1.0, 0.0]], dtype=torch.float64))
