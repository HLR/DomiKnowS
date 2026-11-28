"""Public EAI planner inputs and reference-independent Stage 2 rewards."""

from dataclasses import replace

from domiknows.generation.dfa.predicate_monitor import find_predicate_monitor
from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
from test_regr.EmbodiedAgentInterface.observed_reward import (
    prepare_observed_eai_goal, score_observed_eai_goal,
)

from .planner_context import planner_context


OBSERVED = "observed-pddl-simpletl-v1"
LEGACY = "legacy-reference"


def input_and_goal(item, world, *, archived_prompt=False):
    goal = prepare_observed_eai_goal(
        {key: item[key] for key in ("tl_goal", "transition_model")}, world_bundle=world
    )
    actor = goal.aliases.get("character", "character")
    if actor != "character":
        if not {"agent", "character"} & goal.types.get(actor, set()):
            raise ValueError("indexed EAI agent lacks an observed actor type")
        aliases = {key: "character" if value == actor else value
                   for key, value in goal.aliases.items()}
        aliases[actor] = "character"
        types = dict(goal.types)
        types["character"] = types["character"] | types.pop(actor)

        def normalize(facts):
            return frozenset((fact[0], *("character" if name == actor else name for name in fact[1:]))
                             for fact in facts)

        goal = replace(goal, aliases=aliases, types=types, positive=normalize(goal.positive),
                       negative=normalize(goal.negative))
    if archived_prompt:
        prompt = planner_context("eai", {
            key: item[key] for key in ("causal_prompt_text", "instruction_text", "instruction", "goal", "tl_goal")
            if key in item
        })
    else:
        prompt = {"instruction": item.get("natural_language_description") or item.get("instruction") or "",
                  "goal": item["tl_goal"]}
    return {**prompt, **goal.policy_context()}, goal


def score(labels, goal, runtime):
    return score_observed_eai_goal(labels, goal, runtime.eai_vocabulary, world_bundle=runtime.world.eai)


def sample(view, context, policy, goal, runtime, *, deterministic):
    """Retain unsuccessful draws and distinguish search limits from infeasibility."""
    try:
        labels, _ = view.sample_labels(
            context, policy, max_steps=runtime.max_eai_steps, deterministic=deterministic
        )
        reason = None if labels and labels[-1] == runtime.eai_vocabulary.eos_label else "missing_eos"
    except SemanticSearchLimitExceeded:
        labels, reason = [], "semantic_search_limit"
    except RuntimeError as exc:
        if not str(exc).startswith(("eai planner DFA has no productive", "eai planner did not terminate")):
            raise
        labels, reason = [], "no_accepting_plan"
    if reason:
        return labels, {"is_success": False, "recall": 0.0, "unresolved": goal.unresolved,
                        "generation_error": reason}
    execution_policy = policy if find_predicate_monitor(policy) is not None else runtime.dfa_for("eai", context)
    if not execution_policy.accepts(labels):
        return labels, {"is_success": False, "recall": 0.0, "unresolved": goal.unresolved,
                        "execution_error": "declared_precondition_or_schema_violation"}
    return labels, score(labels, goal, runtime)


def reward_protocol(requested, payload):
    if payload and payload.get("stage") == "stage2":
        saved = payload.get("metrics", {}).get("stage2_eai_reward_protocol", LEGACY)
        if requested not in ("auto", saved):
            raise ValueError("Stage 2 EAI reward protocol differs from the checkpoint")
        return saved
    return OBSERVED if requested == "auto" else requested
