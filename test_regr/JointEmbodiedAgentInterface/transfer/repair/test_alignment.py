import pytest
import torch
from .alignment import distribution_divergence,assert_future_independent,NATIVE_KL_LIMIT


def test_distribution_agreement_rejects_shifted_decisions():
    a=torch.tensor([8.,0.,-5.]);b=torch.tensor([0.,8.,-5.])
    assert distribution_divergence(a,a)==pytest.approx(0.,abs=1e-7)
    assert distribution_divergence(a,b)>NATIVE_KL_LIMIT
    with pytest.raises(ValueError,match='nonfinite'):distribution_divergence(a,torch.tensor([float('nan'),0.,0.]))


def test_future_exclusion_checks_prior_queries_not_future_ones():
    original=torch.zeros(5,4);changed=original.clone();changed[3:]=100.
    assert_future_independent(original,changed,3)
    changed[2,0]=1.
    with pytest.raises(ValueError,match='future source labels'):assert_future_independent(original,changed,3)
