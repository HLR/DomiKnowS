"""Exercise real decoder masks and replay without downloading a VLM."""
from types import SimpleNamespace

import pytest
import torch

from domiknows.generation import MissingSemanticBinding, ProductiveDFA, find_predicate_monitor
from ..graph import bind_vlabench_contextual_dfa
from ..models import QwenVLPlanner
from ..program import PlannerReplayDecision, VLABenchHierarchicalReinforcementProgram, _task_pattern_dfa
from ..training import build_constraint_runtime


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(hidden_size=4)
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        hidden = self.marker.expand(*input_ids.shape, 4)
        return SimpleNamespace(hidden_states=(hidden,))


class TinyProcessor:
    def apply_chat_template(self, *args, **kwargs):
        return "scene"

    def __call__(self, **kwargs):
        return {"input_ids": torch.ones(1, 2, dtype=torch.long)}


@pytest.fixture
def setup():
    runtime = build_constraint_runtime(max_entities=4, max_operations=2, name_prefix="integration")
    planner = QwenVLPlanner(TinyBackbone(), TinyProcessor(), runtime.vocabulary)
    # The highest-scoring pointer is nonexistent; the next-highest pointer
    # becomes illegal as a placement destination after picking that object.
    with torch.no_grad():
        planner.output.weight.zero_()
        planner.output.bias.fill_(-20.0)
        for token, score in (("obj:3", 100.0), ("obj:0", 20.0), ("obj:1", 10.0),
                             ("<eos>", 50.0), ("skill:pick", 5.0), ("skill:place", 4.0)):
            planner.output.bias[runtime.vocabulary.label_for_token(token)] = score
    return runtime, planner


def test_standalone_generation_binds_world_rules_before_sampling(setup):
    runtime, planner = setup
    result = planner.generate_plan(instruction="Put the apple in the bowl", images=(),
                                   entity_table=("apple", "bowl"), dfa=runtime.dfa,
                                   world=runtime.world_bundle, max_steps=runtime.max_tokens)
    assert result == [{"name": "pick", "params": {"target_entity_name": 0}},
                      {"name": "place", "params": {"target_container_name": 1}}]


@pytest.mark.parametrize("planning_facts", [None, {
    "initial_state": {"held": {"right": None}}, "capacities": {"right": 1},
    "entity_properties": {"apple": {"graspable": True}},
    "destinations": {"apple": "bowl"},
    "compatibility": [{"subject": "apple", "destination": "bowl", "allowed": True}],
    "goals": {"positive": [["at", "apple", "bowl"]]},
}])
def test_encoded_collection_and_replay_share_exact_policy_and_logprob(setup, monkeypatch, planning_facts):
    runtime, planner = setup
    context = {"instruction": "Put the apple in the bowl", "images": (), "entity_table": ("apple", "bowl"), "planning_facts": planning_facts}
    bound = runtime.dfa_for("vlabench", context)
    policy = ProductiveDFA(_task_pattern_dfa(bound, runtime.vocabulary, ("pick", "place")))
    assert bind_vlabench_contextual_dfa(policy, runtime.vocabulary, context) is policy
    used = []
    original = planner.sample_labels_from_context
    def capture(encoded, dfa, **kwargs):
        used.append(dfa)
        return original(encoded, dfa, **kwargs)
    monkeypatch.setattr(planner, "sample_labels_from_context", capture)
    prepared = planner.prepare_replay_context(context)
    with torch.no_grad():
        encoded = planner.encode_replay_context(prepared)
        _, collected, labels = planner.sample_with_logprob(
            **context, dfa=policy, world=runtime.world_bundle, max_steps=runtime.max_tokens,
            encoded_context=encoded, return_labels=True,
        )
    assert used == [policy]
    assert not collected.requires_grad
    assert planner.model.calls == 1
    evidence = PlannerReplayDecision(prepared, tuple(labels), policy, runtime.max_tokens)
    replayed = evidence.logprob(planner)
    torch.testing.assert_close(replayed.detach(), collected)
    assert replayed.requires_grad and planner.model.calls == 2
    (-replayed).backward()
    assert planner.output.bias.grad.abs().sum() > 0
    assert planner.output.bias.grad[runtime.vocabulary.label_for_token("obj:3")] == 0
    # Replacing the valid destination by the held apple must also fail replay.
    invalid = list(labels)
    invalid[-2] = runtime.vocabulary.label_for_token("obj:0")
    with pytest.raises(RuntimeError, match="rejected DFA transition"):
        planner.replay_labels_logprob(prepared, invalid, policy, max_steps=runtime.max_tokens)


def test_context_free_decoding_and_replay_reject_unbound_program(setup):
    runtime, planner = setup
    with pytest.raises(MissingSemanticBinding, match="request-bound"):
        planner.sample_labels_from_context(torch.zeros(1, 4), runtime.dfa, max_steps=runtime.max_tokens)
    with pytest.raises(MissingSemanticBinding, match="request-bound"):
        planner.replay_labels_logprob({}, (), runtime.dfa, max_steps=runtime.max_tokens)
    assert planner.model.calls == 0


def test_optional_push_goal_collection_and_replay_agree(setup):
    runtime, planner = setup
    context = {"instruction": "Push the book to the shelf", "images": (), "entity_table": ("book", "shelf"),
               "task_skill_sequence": ("pick", "push"), "planning_facts": {
                   "initial_state": {"negative": [["pushed_to", "book", "shelf"]]},
                   "goals": {"positive": [["pushed_to", "book", "shelf"]]}, "destinations": {"book": "shelf"}}}
    policy = runtime.dfa_for("vlabench", context)
    prepared = planner.prepare_replay_context(context)
    with torch.no_grad():
        encoded = planner.encode_replay_context(prepared)
        plan, collected, labels = planner.sample_with_logprob(**context, dfa=policy, world=runtime.world_bundle,
            max_steps=runtime.max_tokens, encoded_context=encoded, return_labels=True)
    assert plan[-1] == {"name": "push", "params": {"target_container_name": 1}}
    replayed = PlannerReplayDecision(prepared, tuple(labels), policy, runtime.max_tokens).logprob(planner)
    torch.testing.assert_close(replayed.detach(), collected)


def test_rank_constrained_collection_and_replay_share_policy_and_logprob(setup):
    runtime, planner = setup
    from ..physics_grounding import physics_planning_facts
    components = [{"name": f"button{i}", "class": "Button", "subentities": [
        {"name": f"{material}_sphere", "class": "RandomGeom", "material": material}
    ]} for i, material in enumerate(("wood", "stone", "metal"))]
    facts, _ = physics_planning_facts("density_qa", "medium density", components)
    context = {"instruction": "medium density", "images": (), "entity_table": ("button0", "button1", "button2"),
               "task": "density_qa", "planning_facts": facts}
    policy = runtime.dfa_for("vlabench", context)
    prepared = planner.prepare_replay_context(context)
    with torch.no_grad():
        encoded = planner.encode_replay_context(prepared)
        _, collected, labels = planner.sample_with_logprob(
            **context, dfa=policy, world=runtime.world_bundle, max_steps=runtime.max_tokens,
            encoded_context=encoded, return_labels=True,
        )
    assert labels[-2] == runtime.vocabulary.label_for_token("obj:1")
    assert policy.accepts(labels)
    replayed = PlannerReplayDecision(prepared, tuple(labels), policy, runtime.max_tokens).logprob(planner)
    torch.testing.assert_close(replayed.detach(), collected)
    invalid = list(labels)
    invalid[-2] = runtime.vocabulary.label_for_token("obj:0")
    with pytest.raises(RuntimeError, match="rejected DFA transition"):
        planner.replay_labels_logprob(prepared, invalid, policy, max_steps=runtime.max_tokens)


def test_online_policy_binds_identities_even_for_equal_entity_counts(setup):
    runtime, _ = setup
    program = SimpleNamespace(runtime=runtime)
    bind = VLABenchHierarchicalReinforcementProgram._dfa_for_entities
    apple = bind(program, ("apple", "bowl"))
    floor = bind(program, ("floor", "bowl"))
    assert find_predicate_monitor(apple) is not find_predicate_monitor(floor)
    def pick_targets(policy):
        state = policy.start_state
        for token in ("skill:pick", "arg:target_entity_name"):
            state = policy.step(state, runtime.vocabulary.label_for_token(token))
        return policy.allowed_tokens(state)
    obj0 = runtime.vocabulary.label_for_token("obj:0")
    assert obj0 in pick_targets(apple)
    assert obj0 not in pick_targets(floor)


def test_standalone_runtime_passes_task_metadata_to_completion_monitor(setup):
    runtime, _ = setup
    policy = runtime.dfa_for("vlabench", {"entities": ("apple", "bowl"), "task": "cook_dishes"})
    sequence = ("skill:pick", "arg:target_entity_name", "obj:0", "skill:place", "arg:target_container_name", "obj:1", "<eos>")
    assert not policy.accepts([runtime.vocabulary.label_for_token(token) for token in sequence])


def test_live_agent_uses_the_bound_policy_for_generation_and_validation(setup):
    from ..agent import HierarchicalVLABenchAgent

    runtime, planner = setup
    agent = HierarchicalVLABenchAgent(planner, torch.nn.Identity(), runtime)
    decision = agent.plan("Put the apple in the bowl", (), ("apple", "bowl"))
    assert decision.valid and decision.dfa_valid
    assert decision.constraint_score == 1.0
    assert decision.plan[-1]["params"] == {"target_container_name": 1}


def test_standalone_evaluation_supplies_a_bound_policy(setup, monkeypatch):
    from ..training import evaluate_planner

    runtime, planner = setup
    original = planner.generate_plan
    def checked(**kwargs):
        assert find_predicate_monitor(kwargs["dfa"]) is not None
        return original(**kwargs)
    monkeypatch.setattr(planner, "generate_plan", checked)
    example = {"instruction": "Put the apple in the bowl", "entities": ("apple", "bowl"),
               "task": "select_fruit", "episode_id": "select_fruit/example0",
               "operation_sequence": [{"name": "pick", "params": {"target_entity_name": "apple"}},
                                      {"name": "place", "params": {"target_container_name": "bowl"}}]}
    result = evaluate_planner(planner, [example], runtime)
    assert result["valid"] == 1.0
    assert result["planning_metrics"]["overall"]["score"] == 100.0


def test_composed_task_policy_rejects_dead_prefix_before_emitting_a_skill(setup):
    from domiknows.generation import ProductiveDFA

    runtime, _ = setup
    # With only one entity, every pick/place trajectory places the object into
    # itself. The inner grammar also offers pick/lift, but that alternative is
    # excluded by the outer task-pattern policy.
    monitor = runtime.dfa_for("vlabench", {"entities": ("apple",)})
    task = _task_pattern_dfa(monitor, runtime.vocabulary, ("pick", "place"))
    policy = ProductiveDFA(task)
    assert policy.allowed_tokens(policy.start_state, remaining_steps=runtime.max_tokens - 1) == set()
    assert not policy.can_reach_accepting(policy.start_state, max_steps=runtime.max_tokens)


def test_infeasible_composite_cannot_start_or_relax_eos_to_escape(setup):
    runtime, _ = setup
    policy = runtime.dfa_for("vlabench", {"entities": ("apple", "bowl"), "task": "cook_dishes"})
    assert policy.allowed_tokens(policy.start_state, remaining_steps=runtime.max_tokens - 1) == set()



def test_explicit_stop_policy_matches_legacy_sampling_and_replay(setup):
    from domiknows.generation.dfa.stop_policy import StopPolicy
    runtime, planner = setup
    context = {"instruction": "Put the apple in the bowl", "images": (), "entity_table": ("apple", "bowl")}
    dfa = runtime.dfa_for("vlabench", context)
    encoded = planner.encode_replay_context(planner.prepare_replay_context(context))
    policy = StopPolicy(max_steps=7)
    labels, collected = planner.sample_labels_from_context(encoded, dfa, stop_policy=policy, deterministic=True)
    legacy, legacy_logprob = planner.sample_labels_from_context(encoded, dfa, max_steps=7, deterministic=True)
    assert labels == legacy and len(labels) == 7
    torch.testing.assert_close(collected, legacy_logprob)
    replayed = planner.labels_logprob_from_context(encoded, labels, dfa, stop_policy=policy)
    torch.testing.assert_close(collected, replayed)
    evidence = PlannerReplayDecision(planner.prepare_replay_context(context), tuple(labels), dfa, stop_policy=policy)
    torch.testing.assert_close(collected, evidence.logprob(planner))
    with pytest.raises(ValueError, match="not both"):
        planner.sample_labels_from_context(encoded, dfa, max_steps=7, stop_policy=policy)


@pytest.mark.parametrize("criterion", ["timeout", "callback"])
def test_stop_policy_cancellation_rejects_incomplete_plan_without_decoding(setup, criterion):
    from domiknows.generation.dfa.stop_policy import StopPolicy
    runtime, planner = setup
    dfa = runtime.dfa_for("vlabench", {"entities": ("apple", "bowl")})
    snapshots = []
    def cancel(progress):
        snapshots.append(progress)
        return True
    policy = (StopPolicy(timeout_seconds=0) if criterion == "timeout"
              else StopPolicy(external_stop_fn=cancel))
    with pytest.raises(RuntimeError, match="accepting DFA state"):
        planner.sample_labels_from_context(torch.zeros(1, 4), dfa, stop_policy=policy)
    assert planner.model.calls == 0
    if criterion == "callback":
        assert len(snapshots) == 1 and snapshots[0].step_index == 0
        assert snapshots[0].generated_labels == ()


def test_replay_does_not_rerun_collection_cancellation_callback(setup):
    from domiknows.generation.dfa.stop_policy import StopPolicy
    runtime, planner = setup
    dfa = runtime.dfa_for("vlabench", {"entities": ("apple", "bowl")})
    encoded = torch.zeros(1, 4)
    labels, sampled = planner.sample_labels_from_context(encoded, dfa, max_steps=7, deterministic=True)
    def collection_only(progress):
        raise AssertionError("collection cancellation must not run during replay")
    replayed = planner.labels_logprob_from_context(
        encoded, labels, dfa, stop_policy=StopPolicy(max_steps=7, timeout_seconds=0, external_stop_fn=collection_only),
    )
    torch.testing.assert_close(sampled, replayed)



def test_sampling_and_replay_share_eos_and_stagnation_policy(setup):
    from domiknows.generation import DFA
    from domiknows.generation.dfa.stop_policy import StopPolicy
    runtime, planner = setup
    eos = runtime.vocabulary.eos_label
    dfa = DFA(frozenset({0, 1}), frozenset({eos}), {(0, eos): 1, (1, eos): 1}, 0, frozenset({1}))
    encoded = torch.zeros(1, 4)
    policy = StopPolicy(max_steps=3, stop_on_eos_if_accepting=False)
    labels, sampled = planner.sample_labels_from_context(encoded, dfa, stop_policy=policy)
    assert labels == [eos, eos, eos]
    torch.testing.assert_close(sampled, planner.labels_logprob_from_context(encoded, labels, dfa, stop_policy=policy))
    with pytest.raises(RuntimeError, match="continues after stop-policy termination"):
        planner.labels_logprob_from_context(encoded, labels, dfa, max_steps=3)
    stagnant = StopPolicy(max_steps=3, stop_on_eos_if_accepting=False, max_stagnant_steps=1)
    stopped, _ = planner.sample_labels_from_context(encoded, dfa, stop_policy=stagnant)
    assert stopped == [eos, eos]
    with pytest.raises(RuntimeError, match="stop-policy budget"):
        planner.labels_logprob_from_context(encoded, labels, dfa, stop_policy=stagnant)



def test_live_agent_passes_explicit_task_facts_to_generation_and_verification(setup):
    from ..agent import HierarchicalVLABenchAgent
    runtime, planner = setup
    agent = HierarchicalVLABenchAgent(planner, torch.nn.Identity(), runtime)
    decision = agent.plan("Move the selected object", (), ("apple", "bowl"), task_context={
        "task_skill_sequence": ("pick", "place"), "task_target_entity": "bowl",
    })
    assert decision.valid and decision.constraint_score == 1.0
    assert decision.plan[0]["params"]["target_entity_name"] == 1
    assert decision.plan[1]["params"]["target_container_name"] == 0


def test_high_level_continuation_preserves_explicit_facts(setup):
    runtime, planner = setup
    facts = {"initial_state": {"held": {"right": "apple"}},
             "goals": {"positive": [["at", "apple", "bowl"]]}}
    result = planner.generate_plan(instruction="Place it", images=(), entity_table=("apple", "bowl"),
        dfa=runtime.dfa, world=runtime.world_bundle, max_steps=runtime.max_tokens,
        planning_facts=facts, task_skill_sequence=("place",))
    assert result == [{"name": "place", "params": {"target_container_name": 1}}]


def test_grounded_noop_stops_agent_without_controller_or_simulator_success(setup, monkeypatch):
    from .. import agent as agent_module
    runtime, planner = setup
    agent = agent_module.HierarchicalVLABenchAgent(planner, torch.nn.Identity(), runtime)
    monkeypatch.setattr(agent_module, "numbered_views_from_observation", lambda *a, **k: ((), ("apple", "bowl")))
    monkeypatch.setattr(agent_module, "_task_success", lambda env: False)
    monkeypatch.setattr(agent, "action_chunk", lambda *a, **k: pytest.fail("no-op reached controller"))
    env = SimpleNamespace(reset=lambda: None, get_observation=lambda **k: {})
    result = agent.rollout(env, "Already done", task_context={"planning_facts": {
        "initial_state": {"positive": [["at", "apple", "bowl"]]},
        "goals": {"positive": [["at", "apple", "bowl"]]},
    }})
    assert result.plans[0].valid and result.plans[0].plan == ()
    assert result.actions == ()
    assert result.reward.success == 0.0
