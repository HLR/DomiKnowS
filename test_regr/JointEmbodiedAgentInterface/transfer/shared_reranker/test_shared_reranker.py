from types import SimpleNamespace

import pytest
import torch

from domiknows.generation.grounded_facts import FactState, GroundedState
from test_regr.EmbodiedAgentInterface.observed_reward import prepare_observed_eai_goal, score_observed_eai_goal
from test_regr.EmbodiedAgentInterface.test.test_grounded_constraints import encode, policy_for
from test_regr.EmbodiedAgentInterface.world_graph import build_eai_world_graph
from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary

from ...generalization.executor_policy import DESTINATION, EXECUTOR, SUBJECT, bind_plan_policy, pick_place_profile
from ..repair.input_protocol import constraint_context
from ..repair.test_executor_policy import operation, plan_labels
from .features import OperationTrace, PublicGoal, anonymous_features, candidate_features, eai_public_goal
from .head import ReplayPool, SharedPlanReranker, encode_candidates, fit_shared_reranker, goal_transition_potential, rerank_candidates


@pytest.fixture(scope="module")
def eai_world():
    return build_eai_world_graph("shared_reranker_test_eai", include_default_constraints=True)


def _eai_case(world, text="inside(apple.1, fridge)"):
    sample = {
        "tl_goal": text,
        "transition_model": """(define (problem shared) (:domain virtualhome)
            (:objects character - character apple_1 apple_2 fridge - object)
            (:init (open fridge) (containers fridge) (not (inside apple_1 fridge)))
            (:goal (inside apple_2 fridge)))""",
        "target_action_tokens": ["grab", "apple_2", "putback", "fridge"],
    }
    goal = prepare_observed_eai_goal(sample, world_bundle=world)
    context = goal.policy_context()
    facts = context.pop("planning_facts")
    policy, vocabulary = policy_for(world, facts, **context)
    return sample, goal, policy, vocabulary


def test_public_goal_ignores_private_reference_and_pddl_goal(eai_world):
    sample, goal, _, _ = _eai_case(eai_world)
    changed = {**sample, "target_action_tokens": ["sleep"],
               "transition_model": sample["transition_model"].replace(
                   ":goal (inside apple_2 fridge)", ":goal (open fridge)")}
    changed_goal = prepare_observed_eai_goal(changed, world_bundle=eai_world)
    assert eai_public_goal(goal, eai_world) == eai_public_goal(changed_goal, eai_world)
    assert eai_public_goal(goal, eai_world).literals == ((("inside", "apple_1", "fridge"), True),)
    _, temporal, _, _ = _eai_case(eai_world, "open(fridge) then inside(apple.1, fridge)")
    assert not eai_public_goal(temporal, eai_world).literals


def test_open_world_transition_compiles_soft_regression_only():
    compiled = goal_transition_potential()
    assert len(compiled) == 9
    assert compiled[7] == pytest.approx(-0.5)
    assert all(value == 0 for index, value in enumerate(compiled) if index != 7)
    goal = PublicGoal(((("open", "drawer"), True),))
    yes = GroundedState(FactState(frozenset({("open", "drawer")})))
    no = GroundedState(FactState(negative=frozenset({("open", "drawer")})))
    unknown = GroundedState()
    confirmed = anonymous_features(yes, [OperationTrace("close", "drawer", None, yes, no)], goal)[1]
    unavailable = anonymous_features(yes, [OperationTrace("close", "drawer", None, yes, unknown)], goal)[1]
    assert (confirmed.progress, confirmed.after) == ("regress", (0., 1., 0.))
    assert (unavailable.progress, unavailable.after) == ("steady", (1., 0., 0.))


def test_eai_pairwise_fit_uses_independent_replay_and_loads(tmp_path, eai_world):
    _, observed, policy, _ = _eai_case(eai_world)
    tokens = ("grab", "putback", "apple_1", "apple_2", "fridge", "<eos>")
    from domiknows.generation.dfa.vocabulary import TokenVocabulary
    from test_regr.EmbodiedAgentInterface.utils.constraint_compiler import CompiledEAIPlannerDFA, bind_eai_policy
    from domiknows.generation.dfa.core import DFA

    vocabulary = TokenVocabulary(tokens, eos_token="<eos>")
    limit = frozenset(range(len(tokens)))
    grammar = DFA(frozenset((0, 1)), limit,
                  {(0, i): int(i == vocabulary.eos_label) for i in limit}, 0, frozenset((1,)))
    template = CompiledEAIPlannerDFA.from_dfa(grammar, policy.program, world_constraint_bundle=eai_world,
                                               generation_graph=eai_world.graph)
    policy = bind_eai_policy(template, vocabulary, observed.policy_context())
    candidates = [
        SimpleNamespace(labels=tuple(encode(vocabulary, ("grab", apple, "putback", "fridge"))),
                        log_probability=score)
        for apple, score in (("apple_2", -.1), ("apple_1", -.2))
    ]
    goal = eai_public_goal(observed, eai_world)
    head = SharedPlanReranker()
    features = encode_candidates(candidates, policy, vocabulary, goal=goal)
    assert rerank_candidates(head, candidates, policy, vocabulary, goal=goal)[0].index == 0
    assert features[1].steps[-1].progress == "advance"
    outcomes = tuple(score_observed_eai_goal(candidate.labels, observed, vocabulary,
                                            world_bundle=eai_world)["is_success"] for candidate in candidates)
    assert outcomes == (False, True)
    pool = ReplayPool("source-train", features, outcomes)
    with pytest.raises(ValueError, match="non-training"):
        fit_shared_reranker(head, [pool], source_train_ids={"other"})
    fit = fit_shared_reranker(head, [pool], source_train_ids={"source-train"}, epochs=15, lr=.025)
    assert fit["pairs"] == 1 and fit["loss"][-1] < fit["loss"][0]
    path = tmp_path / "shared.pt"
    head.save(path)
    loaded = SharedPlanReranker.load(path)
    assert rerank_candidates(loaded, candidates, policy, vocabulary, goal=goal)[0].index == 1
    torch.testing.assert_close(head(features), loaded(features))


def test_graph_adapter_matches_eai_and_keeps_pointer_permutations_legal(eai_world):
    facts = {"initial_state": {"positive": [["open", "bowl"]], "held": {"right": None}}}
    eai, eai_vocabulary = policy_for(eai_world, facts)
    eai_steps = candidate_features(encode(eai_vocabulary, ("right_grasp", "apple",
                                                           "right_place_inside", "bowl")), eai, eai_vocabulary)
    graph_vocabulary = PlanVocabulary.default(max_entities=4)
    head = SharedPlanReranker()
    for entities, source, destination in ((["a", "b", "c"], 0, 2), (["c", "b", "a"], 2, 0)):
        public = {"entities": entities, "movable": ["a", "b"], "hidden_goal": {"target": "c"}}
        graph = bind_plan_policy(EXECUTOR, graph_vocabulary, 4, constraint_context(public), pick_place_profile())
        valid = plan_labels(graph_vocabulary, [operation("pick", SUBJECT, source),
                                               operation("place", DESTINATION, destination)])
        alternative = plan_labels(graph_vocabulary, [operation("pick", SUBJECT, 1),
                                                     operation("place", DESTINATION, destination)])
        assert candidate_features(valid, graph, graph_vocabulary) == eai_steps
        assert candidate_features(alternative, graph, graph_vocabulary) == eai_steps
        candidates = [SimpleNamespace(labels=labels, log_probability=-1.) for labels in (valid, alternative)]
        assert [entry.index for entry in rerank_candidates(head, candidates, graph, graph_vocabulary)] == [0, 1]
        invalid = plan_labels(graph_vocabulary, [operation("place", DESTINATION, destination)])
        candidates.append(SimpleNamespace(labels=invalid, log_probability=0.))
        with pytest.raises(ValueError, match="accepted"):
            rerank_candidates(head, candidates, graph, graph_vocabulary)
        with pytest.raises(ValueError, match="terminal EOS"):
            candidate_features(valid[:-1], graph, graph_vocabulary)
