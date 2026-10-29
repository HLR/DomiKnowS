"""Both joint domain losses must forward the runtime decoding budget."""

import pytest
import torch

from .test_stage1_dfa_loss import joint_fixture, program_with, vlabench_item
from .test_joint_pipeline import make_planner


@pytest.mark.parametrize("domain", ["eai", "vlabench"])
def test_joint_masked_loss_uses_runtime_budget_and_learns_eos(joint_fixture, monkeypatch, domain):
    examples, runtime = joint_fixture
    planner = make_planner(runtime)
    vocabulary = planner.vocabulary(domain)
    eos = int(vocabulary.eos_label)
    item = dict(examples[0]) if domain == "eai" else vlabench_item(runtime)
    key = "target_action_labels" if domain == "eai" else "target_plan_labels"
    if domain == "eai":
        action = next(label for label in range(vocabulary.label_count) if label != eos)
        item[key] = torch.tensor([action, eos, eos, eos])
    labels = item[key]
    kept = int(((labels == eos).cumsum(0) <= 1).sum())
    budgets = []

    class BudgetAwareDFA:
        start_state = 0

        def allowed_tokens(self, state, remaining_steps=None):
            budgets.append(remaining_steps)
            return {eos} if remaining_steps == 0 else set(range(vocabulary.label_count))

        def step(self, state, label):
            return state + 1

    monkeypatch.setattr(runtime, "dfa_for", lambda *args: BudgetAwareDFA())
    logits = torch.zeros(1, len(labels), vocabulary.label_count, requires_grad=True)
    monkeypatch.setattr(planner, "sequence_logits", lambda *args: logits)
    program = program_with(runtime, planner, domains=(domain,), masked_ce_domains=(domain,),
                           lookahead_domains=(domain,), pmd_domains=(), mass_weight=0.0, pmd_weight=0.0)
    with runtime.domain_scope(domain, parameter_policy="freeze_inactive"):
        loss, terms = program._domain_loss(domain, item)
    loss.backward()

    budget = runtime.max_eai_steps if domain == "eai" else runtime.max_vlabench_operations * 5 + 1
    assert budgets == [budget - step - 1 for step in range(kept)]
    assert budgets[-1] > 0
    assert logits.grad[0, kept - 1, eos] < 0
    assert logits.grad[0, kept - 1].sum().abs() < 1e-6
    assert terms["masked_ce_misses"] == 0.0
