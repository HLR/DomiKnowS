"""VLABench concepts and human-readable logical rules, grouped by knowledge tier.

Generic: schemas and domain invariants. Scene: facts about visible entities.
Task: requested sequences, obligations and grounded requirements. The selected
knowledge level activates groups; it never changes their logical meaning.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import count
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping
try:
    from .world_domain import (SKILL_ARGUMENTS, OPTIONAL_SKILL_ARGUMENTS, SUBTASK_PATTERNS,
        PRIMITIVE_TASK_PATTERNS, VLABENCH_DOMAIN_CHECKSUM)
    from .knowledge import KnowledgeState, active_task_knowledge, check_level, default_level, group_active
    from .knowledge.task_lcs import declare_task_concepts
except ImportError:
    from world_domain import (SKILL_ARGUMENTS, OPTIONAL_SKILL_ARGUMENTS, SUBTASK_PATTERNS,
        PRIMITIVE_TASK_PATTERNS, VLABENCH_DOMAIN_CHECKSUM)
    from knowledge import KnowledgeState, active_task_knowledge, check_level, default_level, group_active
    from knowledge.task_lcs import declare_task_concepts
_BUILD_IDS = count()
@dataclass(frozen=True)
class VLABenchWorldGraphBundle:
    graph: Any
    plan: Any
    operation: Any
    entity: Any
    transition: Any
    contains_operation: Any
    contains_entity: Any
    contains_transition: Any
    operation_roles: Mapping[str, Any]
    transition_roles: Mapping[str, Any]
    skills: Mapping[str, Any]
    skill_arguments: Mapping[str, tuple[str, ...]]
    subtask_patterns: tuple[tuple[str, ...], ...]
    primitive_task_patterns: Mapping[str, tuple[str, ...]]
    domain_checksum: str
    has_target_entity: Any
    has_target_container: Any
    valid_skill_type: Any
    valid_pointer: Any
    valid_operation: Any
    allowed_transition: Any
    targets_button: Any = None
    is_self_placement: Any = None
    targets_static_fixture: Any = None
    repeated_pick_entity: Any = None
    inconsistent_container: Any = None
    unopened_container_placement: Any = None
    premature_composite_eos: Any = None
    task_skill_allowed: Any = None
    task_target_allowed: Any = None
    task_sequence_complete: Any = None
    grounded_constraints: Any = None
    task_concepts: Any = None
    task_knowledge: Any = None
    knowledge_state: Any = None
    knowledge_groups: Any = None
    # Relayed knowledge declared in standalone graphs (knowledge/physics_lcs.py), active with the task group.
    physics_knowledge: Any = None
    complex_knowledge: Any = None
    facts: Any = None
    pattern_first_operation: Any = None
    pattern_start_allowed: Any = None
    pattern_complete: Any = None

    @property
    def optional_skill_arguments(self):
        return OPTIONAL_SKILL_ARGUMENTS

    @property
    def has_constraints(self) -> bool:
        return any(
            getattr(constraint, "headLC", True) and getattr(constraint, "active", True)
            for constraint in getattr(self.graph, "logicalConstrains", {}).values()
        )


# Primitive facts are observations, not precomputed policy verdicts. The same
# bindings feed ordinary DataNode verification and incremental DFA monitoring.
WORLD_FACT_DOMAINS = {
    **dict.fromkeys(("entity_value_usable", "container_value_usable", "entity_resolves", "container_resolves",
                     "entity_is_button", "container_is_button", "subject_equals_destination",
                     "entity_is_static_fixture", "destination_used_in_subtask", "target_was_destination"), "operation"),
    **dict.fromkeys(("next_is_pick", "next_is_placement", "next_has_entity", "next_has_destination",
                     "target_previously_picked", "target_reusable", "expected_destination_known",
                     "destination_matches_expected", "destination_is_enclosed", "destination_was_opened",
                     "planning_facts_supplied", "open_observed_false", "closed_observed_true"), "transition"),
}


def _record_rule(world, group, antecedent, consequent, name, level):
    from domiknows.graph.logicalConstrain import ifL
    rule = ifL(antecedent, consequent, name=name, active=group_active(level, group))
    world.knowledge_groups[group].append(rule)
    return rule


def _any(calls):
    from domiknows.graph.logicalConstrain import orL
    calls = tuple(calls)
    return calls[0] if len(calls) == 1 else orL(*calls)


def _skill_schema(world, skill):
    """A skill has its required roles, no forbidden roles, and usable values."""
    from domiknows.graph.logicalConstrain import andL, ifL, notL
    terms = [world.skills[skill]("op")]
    required = set(world.skill_arguments[skill])
    optional = set(world.optional_skill_arguments.get(skill, ()))
    for role, present, usable in (
        ("target_entity_name", world.has_target_entity, world.facts["entity_value_usable"]),
        ("target_container_name", world.has_target_container, world.facts["container_value_usable"]),
    ):
        if role in required:
            terms.append(present("op"))
        elif role not in optional:
            terms.append(notL(present("op")))
        terms.append(ifL(present("op"), usable("op")))
    return andL(*terms)


def _declare_generic_rules(world, level):
    """Schema and causal invariants that do not consult task identity."""
    from domiknows.graph.logicalConstrain import andL, exactL, notL, orL
    rule = lambda a, b, name: _record_rule(world, "generic", a, b, name, level)
    op, edge, skills, facts = world.operation, world.transition, world.skills, world.facts

    # An operation denotes exactly one skill, with that skill's argument schema.
    rule(op("op"), exactL(*(skill("op") for skill in skills.values()), 1), "operation_exactly_one_skill")
    rule(op("op"), _any(_skill_schema(world, skill) for skill in skills), "operation_schema_is_valid")
    # Progress is projected from graph.py's formulas, separately from argument
    # validity. A transition extends the current prefix, not just a skill pair.
    rule(edge("edge"), world.allowed_transition("edge"), "transition_matches_subtask_pattern")
    rule(andL(op("op"), world.pattern_first_operation("op")),
         world.pattern_start_allowed("op"), "operation_starts_subtask_pattern")
    rule(world.plan("plan"), world.pattern_complete("plan"), "plan_completes_subtask_pattern")

    # A manipulated subject must differ from its destination. The applicable
    # actions and role conditions are visible here, rather than in a violation flag.
    sensitive = orL(*(skills[s]("op") for s in ("place", "pour", "insert", "push")), world.has_target_entity("op"))
    rule(op("op"), notL(andL(world.has_target_container("op"),
                            facts["subject_equals_destination"]("op"), sensitive)), "anti_self_placement")
    # A later pick may revisit an object only when reuse is explicitly permitted.
    rule(andL(edge("edge"), facts["next_is_pick"]("edge"), facts["next_has_entity"]("edge")),
         orL(notL(facts["target_previously_picked"]("edge")), facts["target_reusable"]("edge")),
         "no_redundant_repeated_picks")

    # Preserve the individually named role requirements as readable diagnostics.
    for skill, required in world.skill_arguments.items():
        for role, present in (("target_entity_name", world.has_target_entity),
                              ("target_container_name", world.has_target_container)):
            if role in world.optional_skill_arguments.get(skill, ()):
                continue
            rule(skills[skill]("op"), present("op") if role in required else notL(present("op")),
                 f"skill_argument__{skill}__{role}")


def _declare_causal_rules(world, level):
    """Id-level causal rules, active from ``causal`` upwards (the lowest world group).

    They read only prefix comparisons of the plan's own targets: the subject picked in the current
    subtask, its earlier destinations, and every earlier destination. A pick or press starts a subtask.
    """
    from domiknows.graph.logicalConstrain import andL, notL, orL
    rule = lambda a, b, name: _record_rule(world, "causal", a, b, name, level)
    op, skills, facts = world.operation, world.skills, world.facts
    destination = lambda: andL(world.has_target_container("op"),
                               orL(*(skills[s]("op") for s in ("place", "insert", "pour", "push"))))
    rule(op("op"), notL(andL(destination(), facts["subject_equals_destination"]("op"))), "causal__anti_self_placement")
    rule(op("op"), notL(andL(destination(), facts["destination_used_in_subtask"]("op"))),
         "causal__distinct_subtask_destinations")
    rule(op("op"), notL(andL(skills["pick"]("op"), world.has_target_entity("op"), facts["target_was_destination"]("op"))),
         "causal__destinations_stay_put")


def _declare_scene_rules(world, level):
    """Scene observations constrain references, targets and container access."""
    from domiknows.graph.logicalConstrain import andL, ifL, notL, orL
    rule = lambda a, b, name: _record_rule(world, "scene", a, b, name, level)
    op, edge, facts = world.operation, world.transition, world.facts
    rule(op("op"), andL(
        ifL(world.has_target_entity("op"), facts["entity_resolves"]("op")),
        ifL(world.has_target_container("op"), facts["container_resolves"]("op")),
    ), "operation_pointer_is_valid")
    button_target = lambda: orL(facts["entity_is_button"]("op"), facts["container_is_button"]("op"))
    rule(button_target(), world.skills["press"]("op"), "button_entity_requires_press_skill")
    rule(world.skills["press"]("op"), button_target(), "press_requires_button_target")
    rule(world.skills["pick"]("op"), notL(facts["entity_is_static_fixture"]("op")), "pick_excludes_static_fixtures")
    rule(andL(edge("edge"), facts["next_has_destination"]("edge"), facts["expected_destination_known"]("edge")),
         facts["destination_matches_expected"]("edge"), "container_destination_consistency")

    # Missing open/closed observations remain unknown when explicit planning
    # facts are supplied. Without those facts, retain the legacy closed-world rule.
    access_is_known = orL(notL(facts["planning_facts_supplied"]("edge")),
                         facts["open_observed_false"]("edge"), facts["closed_observed_true"]("edge"))
    rule(andL(edge("edge"), facts["next_is_placement"]("edge"), facts["next_has_destination"]("edge"),
              facts["destination_is_enclosed"]("edge"), access_is_known),
         facts["destination_was_opened"]("edge"), "open_before_enclosed_placement")


def build_vlabench_world_graph(
    graph_name: str = "vlabench_plan_world",
    constraint_builders: Iterable[Callable[[VLABenchWorldGraphBundle], None]] = (),
    *,
    include_default_constraints: bool = True,
    semantic_parents: Mapping[str, Any] | None = None,
    knowledge_level: str | None = None,
    task_knowledge: Any = None,
    task_identity: str | None = None,
) -> VLABenchWorldGraphBundle:
    """Define the DomiKnowS plan ontology and its hard logical constraints.

    Every rule is a concept plus logical constraints, in four groups: ``causal`` (id-level causal rules),
    ``generic`` (schema, patterns, causal rules), ``scene`` (rules that need an entity table)
    and ``task`` (task facts and the ``task__<name>``
    rules of ``knowledge.task_lcs``). ``knowledge_level`` (default: ``knowledge.default_level()``) decides which
    groups are active; :func:`set_knowledge_level` changes it later.
    """
    from domiknows.graph import Concept, Graph

    state = KnowledgeState(knowledge_level or default_level(), task_identity)
    level = state.level
    task_knowledge = task_knowledge or active_task_knowledge()

    build_id = next(_BUILD_IDS)
    role_prefix = f"__vlabench_plan_{build_id}__"

    def role(name: str) -> str:
        return f"{role_prefix}{name}"

    semantic_parents = dict(semantic_parents or {})
    with Graph(graph_name) as graph:
        episode_parent = semantic_parents.get("episode")
        entity_parent = semantic_parents.get("entity")
        operation_parent = semantic_parents.get("operation")
        plan = (
            episode_parent(name="vlabench_plan")
            if episode_parent is not None else Concept(name="vlabench_plan")
        )
        operation = (
            operation_parent(name="vlabench_operation")
            if operation_parent is not None else Concept(name="vlabench_operation")
        )
        entity = (
            entity_parent(name="vlabench_entity")
            if entity_parent is not None else Concept(name="vlabench_entity")
        )
        transition = Concept(name="vlabench_operation_transition")

        contains_operation = plan.contains(operation)[0]
        contains_entity = plan.contains(entity)[0]
        contains_transition = plan.contains(transition)[0]

        operation_plan, target_entity, target_container = operation.has_a(**{
            role("operation_plan"): plan,
            role("target_entity"): entity,
            role("target_container"): entity,
        })
        current_operation, following_operation = transition.has_a(**{
            role("current_operation"): operation,
            role("following_operation"): operation,
        })

        skills = {
            name: operation(name=f"skill__{name}")
            for name in sorted(SKILL_ARGUMENTS)
        }
        has_target_entity = operation(name="operation__has_target_entity")
        has_target_container = operation(name="operation__has_target_container")
        targets_button = operation(name="operation__targets_button")
        is_self_placement = operation(name="operation__is_self_placement")
        targets_static_fixture = operation(name="operation__targets_static_fixture")
        valid_skill_type = operation(name="operation__valid_skill_type")
        valid_pointer = operation(name="operation__valid_pointer")
        valid_operation = operation(name="operation__valid_schema")
        allowed_transition = transition(name="transition__allowed_pattern")
        pattern_first_operation = operation(name="pattern__first_operation")
        pattern_start_allowed = operation(name="pattern__start_allowed")
        pattern_complete = plan(name="pattern__complete")
        repeated_pick_entity = transition(name="transition__repeated_pick_entity")
        inconsistent_container = transition(name="transition__inconsistent_container")
        unopened_container_placement = transition(name="transition__unopened_container_placement")
        premature_composite_eos = operation(name="operation__premature_composite_eos")
        task_skill_allowed = operation(name="operation__task_skill_allowed")
        task_target_allowed = operation(name="operation__task_target_allowed")
        task_sequence_complete = operation(name="operation__task_sequence_complete")

        from domiknows.generation.grounded import declare_predicates
        grounded_constraints = declare_predicates(
            operation, plan, active=group_active(level, "task"),
            constraints=include_default_constraints, detailed=True)
        task_concepts = declare_task_concepts(
            operation, skills, task_knowledge, active=group_active(level, "task"),
            constraints=include_default_constraints)
        groups = {"causal": [], "generic": [], "scene": [], "task": list(task_concepts.constraints)}
        groups["task"].extend(grounded_constraints.constraints)
        facts = {
            name: (operation if domain == "operation" else transition)(name=f"fact__{name}")
            for name, domain in WORLD_FACT_DOMAINS.items()
        }

        bundle = VLABenchWorldGraphBundle(
            grounded_constraints=grounded_constraints,
            facts=MappingProxyType(facts),
            graph=graph,
            plan=plan,
            operation=operation,
            entity=entity,
            transition=transition,
            contains_operation=contains_operation,
            contains_entity=contains_entity,
            contains_transition=contains_transition,
            operation_roles=MappingProxyType({
                "plan": operation_plan,
                "target_entity": target_entity,
                "target_container": target_container,
            }),
            transition_roles=MappingProxyType({
                "current": current_operation,
                "following": following_operation,
            }),
            skills=MappingProxyType(skills),
            skill_arguments=SKILL_ARGUMENTS,
            subtask_patterns=SUBTASK_PATTERNS,
            primitive_task_patterns=PRIMITIVE_TASK_PATTERNS,
            domain_checksum=VLABENCH_DOMAIN_CHECKSUM,
            has_target_entity=has_target_entity,
            has_target_container=has_target_container,
            valid_skill_type=valid_skill_type,
            valid_pointer=valid_pointer,
            valid_operation=valid_operation,
            allowed_transition=allowed_transition,
            pattern_first_operation=pattern_first_operation,
            pattern_start_allowed=pattern_start_allowed,
            pattern_complete=pattern_complete,
            targets_button=targets_button,
            is_self_placement=is_self_placement,
            targets_static_fixture=targets_static_fixture,
            repeated_pick_entity=repeated_pick_entity,
            inconsistent_container=inconsistent_container,
            unopened_container_placement=unopened_container_placement,
            premature_composite_eos=premature_composite_eos,
            task_skill_allowed=task_skill_allowed,
            task_target_allowed=task_target_allowed,
            task_sequence_complete=task_sequence_complete,
            task_concepts=task_concepts,
            task_knowledge=task_knowledge,
            knowledge_state=state,
            knowledge_groups=groups,
            physics_knowledge=_physics_knowledge(level) if include_default_constraints else None,
            complex_knowledge=_complex_knowledge(level) if include_default_constraints else None,
        )

        if include_default_constraints:
            _declare_causal_rules(bundle, level)
            _declare_generic_rules(bundle, level)
            _declare_scene_rules(bundle, level)

        for builder in tuple(constraint_builders):
            builder(bundle)
    return bundle


def _physics_knowledge(level):
    from domiknows.generation.lc_relay import LazyKnowledge
    try:
        from .knowledge.physics_lcs import build_physics_knowledge
    except ImportError:
        from knowledge.physics_lcs import build_physics_knowledge
    return LazyKnowledge(build_physics_knowledge, active=group_active(level, "task"))


def _complex_knowledge(level):
    from domiknows.generation.lc_relay import LazyKnowledge
    try:
        from .knowledge.complex_lcs import build_complex_knowledge
    except ImportError:
        from knowledge.complex_lcs import build_complex_knowledge
    return LazyKnowledge(build_complex_knowledge, active=group_active(level, "task"))


def set_knowledge_level(bundle: VLABenchWorldGraphBundle, level: str, task_identity: str | None = None) -> None:
    """Activate the constraint groups of ``level`` in place. Recompile any DFA built from the graph afterwards.

    ``task_identity`` (or the identity a legacy alias implies) replaces the bundle's task identity.
    """
    from .knowledge import split_level
    state = bundle.knowledge_state
    state.level, identity = split_level(level, task_identity)
    if identity is not None:
        state.task_identity = identity
    for group, constraints in (bundle.knowledge_groups or {}).items():
        for constraint in constraints:
            constraint.active = group_active(state.level, group)
    for knowledge in (bundle.physics_knowledge, bundle.complex_knowledge):
        if knowledge is not None:
            knowledge.set_active(group_active(state.level, "task"))




def __getattr__(name):
    """Keep existing world_graph imports while separating declarations and I/O."""
    import importlib
    for module in ("world_domain", "world_runtime"):
        source = importlib.import_module(f"{__package__}.{module}" if __package__ else module)
        if hasattr(source, name):
            return getattr(source, name)
    raise AttributeError(name)
