"""The ``metadata`` prompt level in plain English: every rule family of ``--knowledge metadata`` and the episode's facts.

``--prompt-constraints metadata`` tells a planner what ``--knowledge metadata`` (the task level with the oracle task
identity and metadata-assisted observations) would enforce, and enforces nothing. The prompt has two parts:

* the rules (``METADATA_SECTIONS``): the grammar and causal rules of ``prompt_constraints.py``, then the generic,
  scene and task groups of the world graph and the physics, complex, poker and articulation knowledge they relay. The
  ``official`` image rules are not part of this level;
* the facts those rules read for one episode (``episode_facts``): the task name, its declared skill sequence or
  obligations, which objects are buttons, static fixtures or enclosed containers, and the simulator metadata of the
  answer options and scene roles. They come from ``task_policy.task_requirements``, the grounding the DFA binds, so the
  prompt states what the DFA would know and nothing else (no reference plan, goal condition or derived answer).

Each rule names the logical constraints it words as ``fnmatch`` patterns over their names (one rule covers a family
such as ``task__*__skill_sequence``). ``test_prompt_metadata.py`` fails when a declared LC or relayed family matches no
rule and is not listed in ``UNWORDED``.
"""
from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any, Mapping

try:
    from .prompt_constraints import PromptRule
except ImportError:  # pragma: no cover - flat-module execution
    from official.prompt_constraints import PromptRule

GENERIC_RULES = (
    PromptRule("Every operation is exactly one skill, with only the parameters that skill takes.",
               ("operation_exactly_one_skill", "operation_schema_is_valid", "skill_argument__*")),
    PromptRule("Every subtask starts with the first skill of a pattern and follows it step by step, and the plan ends "
               "only after a completed subtask.",
               ("operation_starts_subtask_pattern", "transition_matches_subtask_pattern",
                "plan_completes_subtask_pattern")),
    PromptRule("Never pick an object a second time, unless the facts below list it as reusable.",
               ("no_redundant_repeated_picks",)),
)
SCENE_RULES = (
    PromptRule("Use only the IDs of the listed objects.", ("operation_pointer_is_valid",)),
    PromptRule("Use press only on a button, and use a button only with press.",
               ("press_requires_button_target", "button_entity_requires_press_skill")),
    PromptRule("Never pick a static fixture, such as a table, a counter, a wall, the floor or the room itself. The "
               "facts below list them.",
               ("pick_excludes_static_fixtures",)),
    PromptRule("When the facts below name the destination of the task, every destination is that object.",
               ("container_destination_consistency",)),
    PromptRule("Before you place, pour or insert something into an enclosed container (such as a fridge, drawer, "
               "cabinet, microwave, oven, cooler or blender), open it with open_door, unless the facts say it is "
               "already open.",
               ("open_before_enclosed_placement",)),
)
TASK_RULES = (
    PromptRule("When the facts give the skill sequence of the task, the plan uses exactly those skills in that order "
               "and ends after the last one. When they give a repeating skill cycle, the plan repeats that cycle and "
               "ends after a complete cycle.",
               ("task_skill_sequence", "task_sequence_completion", "task__*__skill_sequence*",
                "task__*__completion*")),
    PromptRule("When the facts name the object of the first operation, the first operation acts on it.",
               ("task_initial_target",)),
    PromptRule("When the facts list required skills and a minimum number of operations, the plan uses every required "
               "skill and has at least that many operations.",
               ("composite_subgoal_completion", "task__*__obligations")),
    PromptRule("Pick or press only an object that meets the requirements of the task, and send each object to the "
               "destination the task requires for it.",
               ("grounded__affordance", "grounded__destination", "grounded__compatibility")),
    PromptRule("Hold an object, by picking it, before you place, pour, insert, push, pull or lift it, and hold one "
               "object at a time.",
               ("grounded__holding", "grounded__capacity")),
    PromptRule("Do the steps of the task in their required order, finish only when every goal of the task is "
               "reached, and leave alone the objects the task must not change.",
               ("grounded__milestone_order", "grounded__goal_completion", "grounded__preservation")),
    PromptRule("For a question answered by pressing a button, press exactly one of the answer buttons listed in the "
               "facts.",
               ("physics__press_option_membership",)),
    PromptRule("Answer a question about a physical property from the declared order of that property and each "
               "option's observed attribute, both given in the facts. Options in the same tier tie. For largest, "
               "most or fastest, no option may be above your choice; for smallest, least or slowest, no option may "
               "be below it. For second fastest, exactly one option is above it; for medium or moderately, as many "
               "options are above it as below it. To float in water, its material is below water in the density "
               "order. For magnetic, choose a magnetic option; for non-magnetic, a non-magnetic one. For a sliding "
               "question the order is the cost of sliding, so the fastest option has the lowest cost.",
               ("physics__*__max", "physics__*__min", "physics__*__n[0-9]*", "physics__density_qa__float__below")),
    PromptRule("To cook the recipe named in the instruction, pick only its ingredients and place each one on the "
               "plate. The plan ends when every ingredient in the scene is on the plate.",
               ("complex__cook__*",)),
    PromptRule("To rearrange books, pick each book and push it to the shelf before you pick the next one, in the "
               "order of their years given in the facts (the reverse order when the instruction asks for it). Every "
               "book ends on the shelf.",
               ("complex__book__*",)),
    PromptRule("For a chemistry experiment, pick only the reagents named in the instruction, pour each one into the "
               "flask, then place it back on the table. Every named reagent is poured and returned.",
               ("complex__chemistry__*",)),
    PromptRule("To hang a picture, in this order: pick the hammer, push it to the nail, place it on the counter, pick "
               "the painting, and push the painting to the nail.",
               ("complex__hammer__*",)),
    PromptRule("For a poker hand, find the best hand among the candidate cards: royal flush, then straight flush, "
               "four of a kind, full house, flush, straight, three of a kind, two pairs, one pair and high card. When "
               "two hands tie, the first one wins. Move the cards that make that hand to the destination (for a "
               "high card, the highest card).",
               ("poker__observations", "poker__category_*__required_cards")),
    PromptRule("When the task puts a drink in the fridge, the drink ends inside and the fridge ends closed. When it "
               "takes a book out, the book ends outside. A door, drawer or lid the task sets ends in its required "
               "position.",
               ("articulation__*",)),
)
# The world-graph sections of the level, in prompt order, after the grammar and causal rules.
METADATA_SECTIONS = {"generic": ("General rules", GENERIC_RULES), "scene": ("Scene rules", SCENE_RULES),
                     "task": ("Task rules", TASK_RULES)}
# Declared LCs left out, with the reason; each is a pattern like the rules' names.
UNWORDED = {
    # It restricts only with explicit initial-visibility evidence, which no planning episode supplies.
    "observation_supported_references": "needs explicit visibility evidence",
    # The official group is not part of the metadata level (it is the image-perception side level).
    "official__*": "outside the metadata level",
}


def rule_patterns(rules) -> tuple[str, ...]:
    return tuple(pattern for rule in rules for pattern in rule.lcs)


def worded_by(name: str, rules) -> tuple[PromptRule, ...]:
    """The rules whose patterns match the LC ``name`` (a world ``causal__`` prefix matches the causal rule)."""
    bare = name.removeprefix("causal__")
    return tuple(rule for rule in rules if any(fnmatchcase(bare, pattern) for pattern in rule.lcs))


def unworded(name: str) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in UNWORDED)


# ----------------------------------------------------------------------------------------------------------------
# The episode's facts.

def _ids(table: Mapping[int, str]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for index, name in table.items():
        ids.setdefault(name, index)
    return ids


def _named(names, ids) -> str:
    return ", ".join(f"{ids[name]} ({name})" if name in ids else str(name) for name in names)


def _scene_lines(table, ids, planning, requirements):
    try:
        from ..semantics import is_enclosed_container, is_static_fixture
    except ImportError:  # pragma: no cover
        from semantics import is_enclosed_container, is_static_fixture
    names = list(dict.fromkeys(table.values()))
    lines = []
    for label, test in (("Buttons", lambda n: "button" in n.lower()), ("Static fixtures", is_static_fixture),
                        ("Enclosed containers", is_enclosed_container)):
        found = [name for name in names if test(name)]
        if found:
            lines.append(f"{label}: {_named(found, ids)}.")
    if planning.reusable_entities:
        lines.append(f"Objects that may be picked again: {_named(sorted(planning.reusable_entities), ids)}.")
    held = [name for _, name in planning.initial.holdings if name is not None]
    if held:
        lines.append(f"Already held: {_named(held, ids)}.")
    opened = sorted(fact[1] for fact in planning.initial.facts.positive
                    if isinstance(fact, tuple) and len(fact) == 2 and fact[0] == "open")
    if opened:
        lines.append(f"Already open: {_named(opened, ids)}.")
    return lines


def _task_lines(requirements, ids):
    lines = []
    if requirements.task:
        lines.append(f"Task: {requirements.task}.")
    if requirements.expected_skills:
        lines.append(f"Skill sequence of the task: {', '.join(requirements.expected_skills)}.")
    if requirements.skill_cycle:
        lines.append(f"Repeating skill cycle of the task: {', '.join(requirements.skill_cycle)}.")
    schema = requirements.schema
    if schema is not None and (schema.required_skills or schema.min_operations):
        skills = ", ".join(schema.required_skills) or "none"
        lines.append(f"Required skills: {skills}; at least {schema.min_operations} operations.")
    if requirements.initial_target is not None:
        lines.append(f"Object of the first operation: {_named([requirements.initial_target], ids)}.")
    destination = requirements.planning.default_destination
    if destination is not None:
        lines.append(f"Destination of the task: {_named([destination], ids)}.")
    return lines


def _physics_lines(block, task, ids):
    try:
        from ..knowledge.tables import PHYSICS_PROPERTIES, PHYSICS_REVERSED
    except ImportError:  # pragma: no cover
        from knowledge.tables import PHYSICS_PROPERTIES, PHYSICS_REVERSED
    lines = []
    candidates = list(block.get("candidates") or ())
    if candidates:
        lines.append(f"Answer buttons: {_named(candidates, ids)}.")
    for candidate in candidates:
        measured = (block.get("measurements") or {}).get(candidate)
        observed = {**{key: value for key, value in ((block.get("attributes") or {}).get(candidate) or {}).items()
                       if value is not None},
                    **({"measured value": measured} if measured is not None else {})}
        if observed:
            values = ", ".join(f"{key} {value}" for key, value in sorted(observed.items()))
            lines.append(f"The object of button {_named([candidate], ids)}: {values}.")
    try:
        from ..knowledge.physics_lcs import criterion_chain
        from ..knowledge.tables import PHYSICS_PHRASES
    except ImportError:  # pragma: no cover
        from knowledge.physics_lcs import criterion_chain
        from knowledge.tables import PHYSICS_PHRASES
    observed_phrases = set(block.get("phrases") or ())
    # The physics LCs read the instruction's phrases in this order; the first one present decides the criterion.
    applied = next(((label, criterion) for label, criterion in criterion_chain(task) if label in observed_phrases), None)
    if applied is not None:
        label, criterion = applied
        meaning = {"max": "the highest in the declared order", "min": "the lowest in the declared order",
                   "rank:second": "second from the " + ("lowest" if task in PHYSICS_REVERSED else "highest")
                                  + " in the declared order",
                   "rank:middle": "the middle of the options in the declared order"}.get(
            criterion, f"below {criterion[6:]} in the declared order" if criterion.startswith("below:") else criterion)
        words = " or ".join(f'"{phrase}"' for phrase in PHYSICS_PHRASES.get(label, (label,)))
        lines.append(f"The instruction says {words}, which asks for the option {meaning}.")
    if task in PHYSICS_PROPERTIES:
        attribute, tiers = PHYSICS_PROPERTIES[task]
        if tiers:
            order = " < ".join("/".join(tier) for tier in tiers)
            what = "the cost of sliding" if task in PHYSICS_REVERSED else task.removesuffix("_qa").replace("_", " ")
            lines.append(f"Declared order of {what} by {attribute}, from lowest to highest: {order}.")
    return lines


def _complex_lines(block, ids):
    try:
        from ..knowledge.tables import RECIPES
    except ImportError:  # pragma: no cover
        from knowledge.tables import RECIPES
    lines = []
    roles = {role: names for role, names in (block.get("roles") or {}).items() if names}
    if roles:
        lines.append("Scene roles: " + "; ".join(f"{role}: {_named(names, ids)}" for role, names in sorted(roles.items()))
                     + ".")
    for recipe in block.get("recipes") or ():
        ingredients = RECIPES.get(recipe)
        lines.append(f"Recipe named by the instruction: {recipe}"
                     + (f", whose ingredients are {', '.join(ingredients)}." if ingredients else "."))
    years = block.get("years")
    if years:
        lines.append("Years of the books: " + "; ".join(f"{_named([name], ids)}: {year}" for name, year in years.items())
                     + ".")
    if block.get("asks_reverse"):
        lines.append("The instruction asks for the reverse order.")
    return lines


def _report_lines(raw):
    lines = []
    for key in ("observation_report", "complex_grounding_report"):
        report = raw.get(key) if isinstance(raw, Mapping) else None
        if isinstance(report, Mapping) and report.get("status") == "unavailable":
            reasons = "; ".join(report.get("reasons") or ()) or "not observed"
            lines.append(f"Not observed: {reasons}.")
    return lines


def episode_facts(context: Any, world) -> tuple[str, ...]:
    """The facts the task-level rules read for this episode, one sentence each, from ``task_requirements``."""
    try:
        from ..semantics import entity_mapping
        from ..task_policy import task_requirements
    except ImportError:  # pragma: no cover
        from semantics import entity_mapping
        from task_policy import task_requirements
    get = context.get if isinstance(context, Mapping) else lambda key, default=None: getattr(context, key, default)
    entities = get("entity_table") or get("entities") or ()
    requirements = task_requirements(context, world, entities)
    table = entity_mapping(entities)
    ids = _ids(table)
    planning = requirements.planning
    lines = [*_task_lines(requirements, ids), *_scene_lines(table, ids, planning, requirements)]
    raw = get("planning_facts")
    raw = raw if isinstance(raw, Mapping) else {}
    answers = (raw.get("physics_selection") or {}).get("candidates") if isinstance(raw.get("physics_selection"), Mapping) else None
    if answers:  # the answer buttons are listed with their objects below
        lines = [line for line in lines if not line.startswith("Buttons: ")]
    if isinstance(raw.get("physics_selection"), Mapping):
        lines += _physics_lines(raw["physics_selection"], requirements.task, ids)
    if isinstance(raw.get("complex_request"), Mapping):
        lines += _complex_lines(raw["complex_request"], ids)
    lines += _report_lines(raw)
    return tuple(lines)


def episode_facts_text(context: Any, world) -> str:
    lines = episode_facts(context, world)
    body = "\n".join(f"- {line}" for line in lines) if lines else "- No facts are known about this episode."
    return "### Facts about this episode:\n\n" + body


_EPISODE_WORLD = []


def episode_world():
    """The task-level world graph with the oracle task identity: the facts are grounded as ``--knowledge metadata``
    grounds them, whatever decodes the plan."""
    try:
        from ..world_graph import build_vlabench_world_graph
    except ImportError:  # pragma: no cover - flat-module execution
        from world_graph import build_vlabench_world_graph
    if not _EPISODE_WORLD:
        _EPISODE_WORLD.append(build_vlabench_world_graph(graph_name="vlabench_prompt_episode_facts",
                                                         knowledge_level="task", task_identity="oracle"))
    return _EPISODE_WORLD[0]


def episode_facts_for(example) -> str:
    """The facts section of a ``PlanningExample``."""
    from test_regr.JointEmbodiedAgentInterface.utils.planner_context import planner_context
    return episode_facts_text(planner_context('vlabench', example), episode_world())


def episode_prompt_text(rules: str, example) -> str:
    """``rules`` followed by the facts of ``example``."""
    return f"{rules}\n\n{episode_facts_for(example)}"
