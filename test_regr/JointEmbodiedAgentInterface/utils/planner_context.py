"""One context-binding contract for row, nested-context and keyword APIs.

These are observation fields consumed by the existing domain graphs, not new
rules. Explicit context values (including empty sets and None) take precedence.
Targets/demonstration trajectories are not copied into policy context.
"""

from collections.abc import Mapping

KNOWLEDGE_CONTEXT_FIELDS = (
    "planning_facts",
    "task_target_container",
    "task_target_entity",
    "task",
    "task_name",
    "task_skill_sequence",
    "predicted_task",
    "planning_interface",
    "episode_id",
    "generation_entity_types",
    "semantic_action_permissions",
    "entities",
    "entity_table",
    "observation_mode",
    "ground_complex",
)


def planner_context(domain, source=None, *, context=None):
    """Merge observations without deriving facts from target plans or labels."""
    if domain not in ("eai", "vlabench"):
        raise ValueError(f"Unknown planner domain: {domain!r}")
    get = (
        source.get
        if isinstance(source, Mapping)
        else lambda key, default=None: getattr(source, key, default)
    )
    missing = object()
    result = {}
    for key in KNOWLEDGE_CONTEXT_FIELDS:
        value = get(key, missing)
        if value is not missing:
            result[key] = value
    if domain == "eai":
        result["instruction"] = (
            get("causal_prompt_text")
            or get("instruction_text")
            or get("instruction", get("text", ""))
        )
        result["goal"] = get("goal") or get("tl_goal", "")
    else:
        result["instruction"] = get("instruction", get("text", ""))
        result["images"] = get("images", get("segmented_image_paths", ()))
        result["entity_table"] = get("entity_table", get("entities", ()))
    nested = get("planner_context")
    if isinstance(nested, Mapping):
        result.update(nested)
    if context is not None:
        if not isinstance(context, Mapping):
            raise TypeError("planner context must be a mapping")
        result.update(context)
    return result
