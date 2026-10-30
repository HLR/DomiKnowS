"""Declarative VLABench task facts and their executable predicate meanings.

Task metadata supplies requirements; scene composition and answer keywords do
not invent task goals. The generic compiler enforces the world LCs over these
predicates, just as it does over physical world predicates.
"""
from dataclasses import dataclass, replace
from domiknows.generation.grounded import PlanningFacts
from typing import Any, Mapping

try:
    from .visibility import visible_entity_ids
    from .semantics import entity_mapping, resolve_pointer
    from .knowledge import (
        COMPLEX_TASK_SCHEMAS, DEFAULT_TASK_KNOWLEDGE, ComplexTaskSchema, default_level, default_task_identity, task_facts_available,
    )
except ImportError:
    from visibility import visible_entity_ids
    from semantics import entity_mapping, resolve_pointer
    from knowledge import (
        COMPLEX_TASK_SCHEMAS, DEFAULT_TASK_KNOWLEDGE, ComplexTaskSchema, default_level, default_task_identity, task_facts_available,
    )


@dataclass(frozen=True)
class TaskRequirements:
    expected_skills: tuple[str, ...] | None = None
    initial_target: str | None = None
    schema: ComplexTaskSchema | None = None
    planning: PlanningFacts = PlanningFacts()
    allowed_sequences: frozenset | None = None
    skill_cycle: tuple = ()
    # Task identity as an observation for the ``task__<name>`` concepts, the plan interface that picks the declared
    # sequence, and whether ``expected_skills`` was supplied explicitly (it then overrides the declared sequence).
    task: str | None = None
    interface: str = "full"
    explicit_sequence: bool = False
    visible_entities: frozenset[int] | None = None

    def skill_allowed(self, index, skill):
        # A declared (graph) sequence is enforced by the task LCs; only explicit facts are checked here.
        expected = self.expected_skills if self.explicit_sequence else None
        return (not self.skill_cycle or skill == self.skill_cycle[index % len(self.skill_cycle)]) and (expected is None or
                (index < len(expected) and expected[index] == skill))

    def target_allowed(self, index, params, entities):
        if index != 0 or self.initial_target is None:
            return True
        if "target_entity_name" not in params:
            held = dict(self.planning.initial.holdings).get("right")
            if held is not None:
                return held == self.initial_target
        value = params.get("target_entity_name")
        name, valid = resolve_pointer(value, entity_mapping(entities))
        return valid and name == self.initial_target

    def sequence_complete(self, count):
        expected = self.expected_skills if self.explicit_sequence else None
        return (not self.skill_cycle or count % len(self.skill_cycle) == 0) and (expected is None or count == len(expected))


def knowledge_monitors(world, raw, resolve, task):
    """Requirement monitors for the declared knowledge whose observations the planning facts carry."""
    monitors = []
    block = raw.get("physics_selection") if isinstance(raw, Mapping) else None
    physics = getattr(world, "physics_knowledge", None)
    if block and physics is not None and task:
        try:
            from .knowledge.physics_lcs import PhysicsMonitor
        except ImportError:
            from knowledge.physics_lcs import PhysicsMonitor
        names = {candidate: resolve(candidate) for candidate in block["candidates"]}
        block = {**block, "candidates": [names[c] for c in block["candidates"]],
                 "attributes": {names[c]: value for c, value in (block.get("attributes") or {}).items()},
                 "measurements": {names[c]: value for c, value in (block.get("measurements") or {}).items()}}
        monitors.append(PhysicsMonitor(physics.get(), task, block))
    block = raw.get("complex_request") if isinstance(raw, Mapping) else None
    complex_knowledge = getattr(world, "complex_knowledge", None)
    if block and complex_knowledge is not None and task:
        try:
            from .knowledge.complex_lcs import ComplexMonitor
        except ImportError:
            from knowledge.complex_lcs import ComplexMonitor
        resolved = lambda names: [resolve(name) for name in names]
        block = {**block, "entities": resolved(block["entities"]),
                 "roles": {role: resolved(names) for role, names in block["roles"].items()},
                 "years": {resolve(name): year for name, year in block.get("years", {}).items()}}
        monitors.append(ComplexMonitor(complex_knowledge.get(), task, block))
    return tuple(monitors)


def task_requirements(context, world, entities):
    """Resolve explicit task data; unknown tasks add no implicit goal rules."""
    context = context or {}
    from .research.config import observation_context
    context = observation_context(context,getattr(world,"research_config",None))
    get = context.get if isinstance(context, Mapping) else lambda k, default=None: getattr(context, k, default)
    visible = visible_entity_ids(get("visible_entities"), entities)
    knowledge = getattr(world, "task_knowledge", None) or DEFAULT_TASK_KNOWLEDGE
    state = getattr(world, "knowledge_state", None)
    level = state.level if state is not None else default_level()
    predicted = (state.task_identity if state is not None else default_task_identity()) == "predicted"
    if not task_facts_available(level):
        return TaskRequirements(visible_entities=visible)  # Visibility is independent of task identity.
    normalize = lambda value: str(value or "").strip().lower().replace("-", "_")
    task = normalize(get("task") or get("task_name"))
    if predicted:
        task = normalize(get("predicted_task"))  # the episode's own task name is not used
    elif not task and get("episode_id"):
        try:
            from .planning_metrics import planning_metadata
        except ImportError:
            from planning_metrics import planning_metadata
        task = planning_metadata(str(get("episode_id")))[0] or ""
    expected = None if predicted else get("task_skill_sequence")
    if expected is None and isinstance(get("planning_facts"), Mapping):
        expected = get("planning_facts").get("required_skill_sequence")
    if expected is None and isinstance(get("planning_facts"), PlanningFacts):
        expected = get("planning_facts").required_skill_sequence or None
    explicit_sequence = expected is not None
    interface = "compact" if get("planning_interface") == "compact" else "full"
    if expected is None:
        # The declared sequence is knowledge: the task LCs enforce it; it is read here only to bound the grammar roots.
        expected = knowledge.expected(task, interface)
    if expected is not None:
        if isinstance(expected, str):
            raise ValueError("task_skill_sequence must be a sequence of skill names")
        expected = tuple(expected)
        if not expected or any(skill not in world.skill_arguments for skill in expected):
            raise ValueError("task_skill_sequence must contain known skills")
    target = get("task_target_entity")
    if target is not None:
        target, valid = resolve_pointer(target, entity_mapping(entities))
        if not valid:
            raise ValueError("task target is absent from the current scene")
    # These tasks bind their subject in final-state LCs. Opening an access
    # door before picking that subject must not fail the first-operation rule.
    if task in ("cool_drink", "set_study_table", "manage_study_table") and task in knowledge.tasks():
        target = None
    table = entity_mapping(entities)
    def resolve(value):
        name, valid = resolve_pointer(value, table)
        if not valid:
            raise ValueError(f"planning entity {value!r} is absent from the current scene")
        return name
    planning = PlanningFacts.parse(get("planning_facts"), entities=table.values(), resolve=resolve,
                                   default_destination=get("task_target_container"), constants=("left", "right"))
    planning = replace(planning, monitors=knowledge_monitors(world, get("planning_facts"), resolve, task))
    from .research.config import ResearchConfig
    research_config = ResearchConfig.parse(get("research_config") or getattr(world,"research_config",None))
    if research_config.enabled and getattr(world,"research_knowledge",None) is not None:
        from .research.monitor import ResearchMonitor
        block = (get("planning_facts") or {}).get("research",{}) if isinstance(get("planning_facts"),Mapping) else {}
        planning = replace(planning,monitors=(*planning.monitors,ResearchMonitor(world.research_knowledge.get(),block,research_config,resolve)))
    raw = get("planning_facts")
    if raw is None or isinstance(raw, Mapping):
        try:
            from .articulation import articulation_monitor
        except ImportError:
            from articulation import articulation_monitor
        raw = raw or {}
        if world.articulation_knowledge is not None:
            articulation = articulation_monitor(world, task if task in knowledge.tasks() else None,
                                                raw, resolve, context, tuple(table.values()))
            if articulation is not None:
                planning = replace(planning, initial=articulation.initial(planning.initial, raw, resolve),
                                   monitors=(*planning.monitors, articulation))
        if task in ("texas_holdem", "texas_holdem_explore") and task in knowledge.tasks() and world.poker_knowledge is not None:
            try:
                from .knowledge.poker_lcs import PokerMonitor, parse_poker_selection
            except ImportError:
                from knowledge.poker_lcs import PokerMonitor, parse_poker_selection
            poker = PokerMonitor(world.poker_knowledge.get(), parse_poker_selection(raw.get("poker_selection"), resolve))
            planning = replace(planning, monitors=(*planning.monitors, poker))
    if (knowledge.is_qa(task) and not any(s.action == "press" for s in planning.selections)
            and not any(getattr(m, "action", None) == "press" for m in planning.monitors)):
        from domiknows.generation.grounded import SelectionRequirement
        # QA action shape is known; the answer is unavailable without exposed
        # measurements, option bindings and a declared comparison requirement.
        candidates = tuple(name for name in table.values() if "button" in name.lower())
        planning = replace(planning, selections=(
            *planning.selections, SelectionRequirement("press", candidates, property=task),
        ))
    # Normalize only supplied domain observations before binding or validation.
    try:
        from .semantics import PlanSemantics
    except ImportError:
        from semantics import PlanSemantics
    planning = replace(planning, initial=PlanSemantics(entities, world.skill_arguments, planning).initial_state().grounded)
    prefix = planning.completed_skill_prefix
    if any(skill not in world.skill_arguments for skill in prefix):
        raise ValueError("completed_skill_prefix contains an unknown skill")
    if expected is not None and tuple(expected[:len(prefix)]) != prefix:
        raise ValueError("completed_skill_prefix contradicts the task skill sequence")
    if prefix and "pick" in prefix and not any(obj is not None for _,obj in planning.initial.holdings) and prefix[-1] not in ("place", "insert"):
        raise ValueError("completed grasp progress requires an explicitly held object")
    raw = get("planning_facts")
    cycle = planning.skill_cycle
    if any(skill not in world.skill_arguments for skill in cycle):
        raise ValueError("skill_cycle contains an unknown skill")
    return TaskRequirements(expected, target, knowledge.schema(task), planning, skill_cycle=cycle,
                            task=task or None, interface=interface, explicit_sequence=explicit_sequence,
                            visible_entities=visible)



def request_sequences(world, max_operations, requirements):
    """Syntactic derivatives justified by explicit task progress/held state."""
    from domiknows.generation.dfa.pattern_language import BoundedPatternLanguage
    prefix = requirements.planning.completed_skill_prefix
    base = BoundedPatternLanguage(world.subtask_patterns, max_operations + len(prefix))
    roots = set(world.subtask_patterns)
    if prefix:
        state = base.start
        for skill in prefix:
            state = base.advance(state, skill)
            if state is None:
                raise ValueError("completed_skill_prefix is not a declared subtask prefix")
        roots.update(state.remaining)
    held = any(obj is not None for _, obj in requirements.planning.initial.holdings)
    if held:
        roots.update(seq[1:] for seq in world.subtask_patterns if seq[0] == "pick" and len(seq) > 1)
    expected = requirements.expected_skills
    remaining = None if expected is None else expected[len(prefix):]
    if remaining and remaining[0] in {"place", "insert", "pour", "lift", "pull", "push"}:
        if not held:
            raise ValueError("a continuation task requires an explicitly held object")
        roots.add(tuple(remaining))
    completion = requirements.planning.completion(requirements.planning.initial)
    schema = requirements.schema
    obligations_met = schema is None or (len(prefix) >= schema.min_operations and set(schema.required_skills) <= set(prefix))
    allow_empty = bool(completion.applicable and completion.satisfied is True and not completion.missing and not remaining and obligations_met)
    return BoundedPatternLanguage(world.subtask_patterns, max_operations, tuple(sorted(roots)), allow_empty)
