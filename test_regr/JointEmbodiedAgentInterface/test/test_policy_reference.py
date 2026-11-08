from types import SimpleNamespace

import pytest
import torch

from ..checkpoint import load_joint_checkpoint, save_joint_checkpoint
from ..models import FreeDecoding
from ..program import JointReinforcementProgram, _context
from ..utils.policy_reference import (
    FrozenPolicyReference, PolicyReplay, resolve_regularization,
    balanced_reference_example, deterministic_policy,
)
from test_regr.VLABenchAgentInterface.simulation.program import JointEpisode, PlannerReplayDecision
from .test_joint_pipeline import joint_fixture as base_joint_fixture, make_planner, TinyController


@pytest.fixture
def joint_fixture():
    return base_joint_fixture.__wrapped__()


def make_program(runtime, planner=None, reference=None, **kwargs):
    planner = planner or make_planner(runtime)
    return JointReinforcementProgram(
        runtime, planner, TinyController(),
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.1),
        controller_optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01),
        env_factory=lambda **kwargs: None, parameter_policy="freeze_inactive", dfa_enabled=False,
        eai_supervised_weight=0, supervised_weight=0, controller_bc_weight=0,
        planner_kl_weight=1.0, planner_reward_policy="success-gated",
        planner_reference_state=reference, eai_num_samples=2,
        **kwargs,
    )


def test_exact_token_kl_detects_drift_in_unchosen_actions():
    # Chosen-token probability is unchanged. A sampled log-ratio penalty would
    # miss this redistribution of probability between the other two actions.
    logits = torch.tensor([0.5, 0.4, 0.1]).log().requires_grad_()
    current = logits.log_softmax(0)
    frozen = torch.tensor([0.5, 0.25, 0.25]).log()
    kl = PolicyReplay(current[0], (current,)).kl_from(PolicyReplay(frozen[0], (frozen,)))
    assert kl.item() == pytest.approx(0.1115718)
    kl.backward()
    assert logits.grad[1] > 0  # gradient descent restores probability to action 2
    assert logits.grad[2] < 0


def test_reference_is_immutable_and_restores_storage_modes_and_gradients():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    optimizer = torch.optim.Adam(model.parameters())
    reference = FrozenPolicyReference(model, optimizer)
    state = {name: p.clone() for name, p in reference.parameters.items()}
    with torch.no_grad():
        model[0].weight.add_(1)
    current = {name: p.detach().clone() for name, p in model.named_parameters()}
    pointers = {name: p.data_ptr() for name, p in model.named_parameters()}
    with pytest.raises(RuntimeError, match="reference failure"):
        with reference.activate(model):
            assert not model.training
            assert not torch.is_grad_enabled()
            assert torch.equal(model[0].weight, state["0.weight"])
            raise RuntimeError("reference failure")
    assert model.training and model[1].training
    for name, p in model.named_parameters():
        assert torch.equal(p, current[name]) and p.data_ptr() == pointers[name]
        assert torch.equal(reference.parameters[name], state[name])
    model(torch.ones(1, 2)).sum().backward()
    assert model[0].weight.grad is not None


def test_frozen_reference_rejects_tampered_tensor_content():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference = FrozenPolicyReference(model, optimizer)
    state = reference.state_dict()
    state["parameters"]["weight"] = state["parameters"]["weight"] + 1
    with pytest.raises(ValueError, match="fingerprint differs"):
        FrozenPolicyReference(model, optimizer, state=state)


@pytest.mark.parametrize("success,expected", [(False, [0, -0.2]), (True, [0.8, -0.2])])
def test_success_gate_only_changes_planner_credit(success, expected):
    program = object.__new__(JointReinforcementProgram)
    program.planner_reward_policy = "success-gated"
    episode = JointEpisode([None, None], [], 0.6, success, True, 10, [0.8, -0.2])
    assert program._planner_returns_for_update(episode) == expected
    assert episode.total_return == 0.6
    assert episode.planner_returns == [0.8, -0.2]  # controller/rollout evidence stays intact


def test_equal_success_rewards_still_restore_drift_without_gold_ce(joint_fixture, monkeypatch):
    examples, runtime = joint_fixture
    torch.manual_seed(8)
    program = make_program(runtime)
    planner = program.joint_planner
    eos = runtime.eai_vocabulary.eos_label
    labels = [eos]
    monkeypatch.setattr(program.eai_planner, "sample_labels_from_context",
                        lambda *a, **k: (labels, torch.zeros(())))
    monkeypatch.setattr(planner, "supervised_loss", lambda *a, **k: pytest.fail("gold CE must be disabled"))
    from test_regr.EmbodiedAgentInterface import reward
    monkeypatch.setattr(reward, "evaluate_goal_satisfaction",
                        lambda *a, **k: {"recall": 1., "is_success": True})
    with torch.no_grad():
        before = planner.label_heads["eai"].bias[eos].item()
        planner.label_heads["eai"].bias[eos].add_(1.5)
    item = {**examples[0], "reward_function": lambda *a, **k: 1.0}
    context = program.eai_planner.prepare_replay_context(_context(item, "eai"))
    dfa = program._policy_dfa("eai", item)
    def divergence():
        with torch.no_grad():
            return float(program._regularized_replay_loss("eai", context, labels, dfa,
                torch.zeros(()), max_steps=runtime.max_eai_steps))
    before_kl = divergence()
    metrics = program.train_eai_update(item)
    assert metrics["reward"] == 1
    assert metrics["planner_regularization"]["reference_kl_trajectories"] == 2
    assert metrics["planner_regularization"]["reference_kl_sum"] > 0
    assert metrics["planner_update"]["shared_parameters_changed"]
    assert planner.label_heads["eai"].bias[eos].item() < before + 1.5
    assert divergence() < before_kl
    # Both replay passes restore normal training mode.
    assert planner.training


@pytest.mark.parametrize("success", [False, True])
def test_only_successful_positive_rollout_reinforces_plan(joint_fixture, success):
    _examples, runtime = joint_fixture
    program = make_program(runtime)
    planner = program.joint_planner
    vocab = runtime.vlabench_vocabulary
    labels = tuple(vocab.label_for_token(t) for t in
                   ["skill:pick", "arg:target_entity_name", "obj:0", "<eos>"])
    decision = PlannerReplayDecision({"instruction": "pick the object", "images": (),
        "entity_table": ("object",)}, labels, FreeDecoding(vocab.label_count, vocab.eos_label), max_steps=10)
    before = {n: p.detach().clone() for n, p in planner.named_parameters()}
    with torch.no_grad():
        logprob_before = float(decision.logprob(program.vlabench_planner))
    with runtime.domain_scope("vlabench", parameter_policy="freeze_inactive"):
        program.planner_head = program.vlabench_planner
        loss = program._update_planner([JointEpisode([decision], [], 1., success, True, 1, [1.])])
    if success:
        with torch.no_grad():
            assert float(decision.logprob(program.vlabench_planner)) > logprob_before
    else:
        assert abs(loss) < 1e-6
        assert all(torch.allclose(p, before[n], atol=1e-7, rtol=0) for n, p in planner.named_parameters())
    assert len(program._reference_kls["vlabench"]) == 1  # KL still executes with zero credit


def test_success_signal_cannot_reinforce_an_invalid_plan():
    program = object.__new__(JointReinforcementProgram)
    program.planner_reward_policy = "success-gated"
    assert program._planner_returns_for_update(JointEpisode([None], [], 1., True, False, 1)) == [0.]


def test_checkpoint_resume_preserves_original_reference_not_current_rl_weights(joint_fixture, tmp_path):
    examples, runtime = joint_fixture
    observation = {"task": "toy", "instruction": "pick", "images": (), "entities": ("object",)}
    program = make_program(runtime, reference_replay_per_update=2,
        eai_supervised_examples=examples, vlabench_supervised_examples=[observation],
        reference_replay_cursors={"eai": 7, "vlabench": 4})
    planner = program.joint_planner
    fingerprint = program.policy_reference.fingerprint
    original = program.policy_reference.parameters["label_heads.eai.weight"].clone()
    with torch.no_grad():
        planner.label_heads["eai"].weight.add_(0.3)
    path = save_joint_checkpoint(tmp_path / "regularized.pt", runtime=runtime, planner=planner,
        controller=program.controller, planner_optimizer=program.planner_optimizer,
        controller_optimizer=program.controller_optimizer, stage="stage2", epoch=0, round_robin_cursor=3)
    restored = make_planner(runtime)
    payload = load_joint_checkpoint(path, runtime=runtime, planner=restored, controller=TinyController())
    weight, policy, replay, reference = resolve_regularization(SimpleNamespace(), payload)
    assert weight == 1.0 and policy == "success-gated"
    assert replay == 2
    assert restored._stage2_regularization_state["reference"]["fingerprint"] == fingerprint
    resumed = make_program(runtime, restored, reference, reference_replay_per_update=replay,
        eai_supervised_examples=examples, vlabench_supervised_examples=[observation],
        reference_replay_cursors=restored._stage2_regularization_state["replay_cursors"])
    assert resumed.policy_reference.fingerprint == fingerprint
    assert torch.equal(resumed.policy_reference.parameters["label_heads.eai.weight"], original)
    assert not torch.equal(restored.label_heads["eai"].weight, original)
    assert resumed.reference_replay_cursors == {"eai": 7, "vlabench": 4}


def test_fresh_defaults_and_resume_configuration_guards():
    assert resolve_regularization(SimpleNamespace()) == (0.05, "success-gated", 2, None)
    legacy = {"stage": "stage2"}
    assert resolve_regularization(SimpleNamespace(), legacy) == (0., "dense", 0, None)
    with pytest.raises(ValueError, match="start from the CE checkpoint"):
        resolve_regularization(SimpleNamespace(stage2_planner_kl_weight=.05), legacy)
    saved = {"stage": "stage2", "stage2_planner_regularization": {
        "version": 1, "kl_weight": .05, "reward_policy": "success-gated", "reference": None}}
    with pytest.raises(ValueError, match="missing its frozen CE reference"):
        resolve_regularization(SimpleNamespace(), saved)
    with pytest.raises(ValueError, match="start from the CE checkpoint"):
        resolve_regularization(SimpleNamespace(stage2_reference_replay_per_update=3), saved)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        resolve_regularization(SimpleNamespace(stage2_planner_kl_weight=float("nan")))


def test_balanced_reference_replay_visits_tasks_before_reusing_a_group():
    examples = [{"task": "a", "id": 0}, {"task": "a", "id": 1},
                {"task": "a", "id": 2}, {"task": "b", "id": 3}]
    assert [balanced_reference_example(examples, i)["id"] for i in range(6)] == [0, 3, 1, 3, 2, 3]


def test_dropout_disabled_in_checkpointed_forward_and_backward_without_disabling_training():
    from torch.utils.checkpoint import checkpoint
    model = torch.nn.Sequential(torch.nn.Dropout(0.9), torch.nn.Linear(2, 1, bias=False))
    x = torch.ones(1, 2, requires_grad=True)
    with deterministic_policy(model):
        assert model.training  # retain activation checkpointing in the real VLM
        loss = checkpoint(model, x, use_reentrant=False).sum()
    with deterministic_policy(model):
        loss.backward()
    assert torch.equal(model[1].weight.grad, torch.ones(1, 2))
    assert model[0].training


@pytest.mark.parametrize("domain", ["eai", "vlabench"])
def test_reference_replay_uses_frozen_predictions_and_never_reads_gold(joint_fixture, domain, monkeypatch):
    examples, runtime = joint_fixture
    class Observation(dict):
        def get(self, key, *args):
            if key in ("operation_sequence", "target_plan_labels", "target_action_labels"):
                pytest.fail("reference replay accessed a gold plan")
            return super().get(key, *args)
    observation = Observation({"instruction": "pick the object", "entities": ("object",),
                               "images": (), "task": "train-task"})
    program = make_program(runtime, reference_replay_per_update=1,
                           eai_supervised_examples=[observation], vlabench_supervised_examples=[observation])
    planner = program.joint_planner
    view = planner.for_domain(domain)
    sampled = []
    original = planner.sample_labels_from_context
    def sample(*args, **kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs["deterministic"]
        sampled.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(planner, "sample_labels_from_context", sample)
    # First create cross-update drift, then run only reference retention.
    with torch.no_grad():
        planner.label_heads[domain].bias[0].add_(0.5)
    program.planner_optimizer.zero_grad(set_to_none=True)
    with runtime.domain_scope(domain, parameter_policy="freeze_inactive"):
        loss = program._reference_retention_update(domain)
    assert loss > 0 and sampled
    assert program.reference_replay_cursors[domain] == 1
    assert planner._stage2_regularization_state["replay_cursors"][domain] == 1
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in planner.parameters())
