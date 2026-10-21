"""VLABench predicate bindings and token/state adapter for the generic compiler."""
from dataclasses import dataclass, field, fields, replace
from typing import Any

from domiknows.generation.dfa.predicate_monitor import (
    CompiledPredicateConstraint as CompiledWorldRule,
    PredicateBinding,
    PredicateConstrainedDFA,
    PredicateConstraintCompilationError as WorldConstraintCompilationError,
    PredicateEvent,
    PredicateMonitorDFA,
    SemanticTransition,
    compile_predicate_constraints,
)

try:
    from .semantics import OperationPredicates, PlanSemantics, PlanSemanticState
    from .knowledge.task_lcs import knowledge_values
except ImportError:
    from semantics import OperationPredicates, PlanSemantics, PlanSemanticState
    from knowledge.task_lcs import knowledge_values


@dataclass(frozen=True)
class CompiledPlannerDFA(PredicateConstrainedDFA):
    """VLABench world binding attached to a generic compiled grammar."""

    world_constraint_bundle: Any = None
    token_dfa: Any = None
    max_operations: int = 12
    operation_dfa: Any = None
    completion_bounds_cache: Any = field(default_factory=dict, compare=False, repr=False)

    @property
    def world_constraint_rules(self):
        return self.constraint_program.rules


def compile_world_constraints(world):
    """Declare VLABench predicate meanings and compile using the generic core."""
    bindings = []
    transition_fields = {"repeated_pick_entity", "inconsistent_container", "unopened_container_placement"}
    for field in fields(OperationPredicates):
        domain = "transition" if field.name in transition_fields else "operation"
        bindings.append(PredicateBinding(getattr(world, field.name), field.name, domain))
    bindings.extend((
        PredicateBinding(world.operation, "operation", "operation"),
        PredicateBinding(world.transition, "transition", "transition"),
        PredicateBinding(world.allowed_transition, "allowed_transition", "transition"),
        PredicateBinding(world.premature_composite_eos, "premature_composite_eos", "operation", "completion"),
    ))
    bindings.extend(PredicateBinding(getattr(world, name), name, "operation", phase) for name, phase in (
        ("task_skill_allowed", "task_skill"), ("task_target_allowed", "task_target"),
        ("task_sequence_complete", "completion"),
    ))
    bindings.extend(PredicateBinding(concept, "skill:" + skill, "operation")
                    for skill, concept in world.skills.items())
    if getattr(world, "task_concepts", None) is not None:
        bindings.extend(PredicateBinding(concept, name, "operation", phase)
                        for concept, name, phase in world.task_concepts.bindings())
    from domiknows.generation.grounded import predicate_bindings
    bindings.append(PredicateBinding(world.plan, "episode", "episode"))
    bindings.extend(predicate_bindings(world.grounded_constraints))
    return compile_predicate_constraints(world.graph, bindings)


@dataclass(frozen=True)
class PlanDecodingState:
    semantic: PlanSemanticState = PlanSemanticState()
    skill: str | None = None
    params: tuple = ()
    pending_argument: str | None = None
    last_values: tuple = ()
    ended: bool = False


class VLABenchPredicateAdapter:
    """Interpret plan tokens and emit predicate valuations at operation boundaries."""

    def __init__(self, vocabulary, world, entities, schema=None, requirements=None):
        self.vocabulary = vocabulary
        self.world = world
        try:
            from .task_policy import TaskRequirements
        except ImportError:
            from task_policy import TaskRequirements
        self.requirements = requirements or TaskRequirements(schema=schema)
        self.semantics = PlanSemantics(entities, world.skill_arguments, self.requirements.planning)
        self.schema = self.requirements.schema
        self.entities = self.semantics.entities
        self.start_state = PlanDecodingState(self.semantics.initial_state())

    def _finish(self, state):
        params = dict(state.params)
        facts = self.semantics.evaluate_operation(state.semantic, state.skill, params)
        values = {field.name: getattr(facts, field.name) for field in fields(facts)}
        # The grammar accepts only declared complete subtask patterns; the edge
        # predicate therefore holds on every accepting grammar trajectory.
        from domiknows.generation.grounded import observations
        _successor, grounded_checks = self.semantics.grounded_transition(state.semantic, state.skill, params)
        values.update(observations(grounded_checks))
        values.update(operation=True, transition=True, allowed_transition=True, premature_composite_eos=False)
        values.update({"skill:" + skill: state.skill == skill for skill in self.world.skills})
        values["task_target_allowed"] = self.requirements.target_allowed(
            state.semantic.completed_operations, params, self.entities,
        )
        events = [PredicateEvent("operation", values), PredicateEvent("operation", values, "task_target")]
        if state.semantic.completed_operations:
            events.append(PredicateEvent("transition", values))
        return SemanticTransition(
            PlanDecodingState(self.semantics.advance(state.semantic, state.skill, params),
                              last_values=tuple(sorted(values.items()))), tuple(events),
        )

    def _completion_events(self, state, *, eos):
        values = dict(state.last_values)
        from domiknows.generation.grounded import observations
        root_event = PredicateEvent("episode", {"episode": True, **observations((self.requirements.planning.completion(state.semantic.grounded),))}, "completion")
        if not values:
            return (root_event,) if eos else ()
        values.update(self.task_completion_values(state.semantic, eos=eos))
        return (PredicateEvent("operation", values, "completion"),) + ((root_event,) if eos else ())

    def task_completion_values(self, semantic, *, eos):
        # Declared obligations (minimum operations, required skills) are task LCs over the history concepts.
        return {
            "task_sequence_complete": not eos or self.requirements.sequence_complete(semantic.completed_operations),
            "premature_composite_eos": False,
            **self.knowledge_values(semantic.completed_operations - 1, final=eos, history=semantic.completed_skills),
        }

    def knowledge_values(self, index, *, choice=False, final=False, history=()):
        """Observations for the task concepts of the world graph (empty if the graph declares none)."""
        concepts = getattr(self.world, "task_concepts", None)
        if concepts is None:
            return {}
        from_task = self.requirements
        return knowledge_values(concepts, from_task.task, from_task.interface, index,
                                explicit=from_task.explicit_sequence, choice=choice, final=final, history=history)

    def task_skill_values(self, count, skill):
        return {"operation": True,
                "task_skill_allowed": self.requirements.skill_allowed(count, skill),
                **{"skill:" + name: skill == name for name in self.world.skills},
                **self.knowledge_values(count, choice=True)}

    def step(self, state, symbol):
        token = self.vocabulary.token_for_label(int(symbol))
        if state.skill is not None and (token.startswith("skill:") or token == self.vocabulary.eos_token):
            # Optional roles finish at the next operation/EOS boundary when omitted.
            if state.pending_argument or not all(key in dict(state.params) for key in self.world.skill_arguments[state.skill]):
                return None
            finished = self._finish(state)
            following = self.step(finished.state, symbol)
            return None if following is None else SemanticTransition(following.state, finished.events + following.events)
        if state.ended:
            return SemanticTransition(state) if token == self.vocabulary.eos_token else None
        if token == self.vocabulary.eos_token:
            if state.skill is not None:
                return None
            return SemanticTransition(replace(state, ended=True), self._completion_events(state, eos=True))
        if token.startswith("skill:"):
            if state.skill is not None:
                return None
            skill = token[6:]
            next_state = PlanDecodingState(state.semantic, skill=skill, last_values=state.last_values)
            result = (SemanticTransition(next_state) if self.world.skill_arguments[skill] or self.world.optional_skill_arguments.get(skill)
                      else self._finish(next_state))
            # Finalize the previous operation's deferred predicates with False:
            # only the actual last operation is grounded as a premature EOS.
            skill_values = self.task_skill_values(state.semantic.completed_operations, skill)
            skill_event = PredicateEvent("operation", skill_values, "task_skill")
            return SemanticTransition(result.state, self._completion_events(state, eos=False) + (skill_event,) + result.events)
        if token.startswith("arg:"):
            return SemanticTransition(replace(state, pending_argument=token[4:]))
        if token.startswith("obj:") and state.skill is not None and state.pending_argument:
            params = dict(state.params)
            params[state.pending_argument] = int(token[4:])
            next_state = replace(state, params=tuple(sorted(params.items())), pending_argument=None)
            if all(key in params for key in (*self.world.skill_arguments[state.skill], *self.world.optional_skill_arguments.get(state.skill, ()))):
                return self._finish(next_state)
            return SemanticTransition(next_state)
        return None

    def is_accepting(self, state):
        return state.ended and state.skill is None

    def acceptance_events(self, state):
        return self._completion_events(state, eos=True)


class WorldConstraintDFA(PredicateMonitorDFA):
    """Compatibility entry point binding VLABench to the reusable monitor."""

    def __init__(self, base_dfa, vocabulary, world, program, entities, schema=None, *, requirements=None):
        adapter = VLABenchPredicateAdapter(vocabulary, world, entities, schema, requirements)
        bounds = task_completion_bounds(base_dfa, vocabulary, adapter, program)
        super().__init__(base_dfa, program, adapter,
                         completion_bound=None if bounds is None else lambda state: bounds(state.grammar[1]))


def task_completion_bounds(base_dfa, vocabulary, adapter, program):
    """Relax entity-dependent rules, retaining observable task LCs exactly.

    The grammar builder explicitly composes (token grammar, operation grammar).
    The latter records selected skills and argument phase, independent of entity
    assignments. Any accepted concrete path must also pass this projection.
    """
    from domiknows.generation.dfa.productive import DFACompletionBounds
    grammar = getattr(base_dfa, "operation_dfa", None)
    if grammar is None:
        return None
    names = {"operation", "task_skill_allowed", "task_sequence_complete", "premature_composite_eos"}
    names.update("skill:" + skill for skill in adapter.world.skills)
    if getattr(adapter.world, "task_concepts", None) is not None:
        names.update(name for _concept, name, _phase in adapter.world.task_concepts.bindings())
    projected = program.project(names, domain="operation", phases=("task_skill", "completion"))
    if not projected.rules:
        return None
    requirements = adapter.requirements
    prefix = requirements.planning.completed_skill_prefix
    schema = adapter.schema
    # This cache is structural, not keyed by context identity or entity counts.
    # Entity facts are absent from the projection and remain request-bound.
    key = (projected, requirements.expected_skills, requirements.skill_cycle, prefix,
           None if schema is None else (schema.min_operations, schema.required_skills),
           requirements.task, requirements.interface, requirements.explicit_sequence)
    cache = base_dfa.completion_bounds_cache
    if key in cache:
        return cache[key]
    completions, skill_checks = {}, {}
    skills_by_label = {vocabulary.label_for_token("skill:" + skill): skill for skill in adapter.world.skills}

    def completion_allowed(history, eos):
        if not history:
            # No previous operation event exists at a grounded no-op start.
            return True
        key = (history, eos)
        if key not in completions:
            semantic = PlanSemanticState(completed_operations=len(prefix) + len(history),
                                         completed_skills=frozenset(prefix) | frozenset(history))
            values = {"operation": True, **adapter.task_completion_values(semantic, eos=eos),
                      **{"skill:" + skill: history[-1] == skill for skill in adapter.world.skills}}
            completions[key] = projected.check(PredicateEvent("operation", values, "completion"))
        return completions[key]

    def transition_allowed(source, symbol, target):
        history, _skill, _index, _phase, ended = source
        if ended:
            return True
        if symbol in skills_by_label:
            skill = skills_by_label[symbol]
            count = len(prefix) + len(history)
            key = (count, skill)
            if key not in skill_checks:
                skill_checks[key] = projected.check(PredicateEvent(
                    "operation", adapter.task_skill_values(count, skill), "task_skill"))
            return skill_checks[key] and completion_allowed(history, False)
        if symbol == vocabulary.eos_label:
            return completion_allowed(history, True)
        return True

    bounds = DFACompletionBounds.from_dfa(
        grammar, transition_allowed=transition_allowed,
        accepting_allowed=lambda state: completion_allowed(state[0], True),
    )
    if len(cache) >= 32:
        cache.pop(next(iter(cache)))
    cache[key] = bounds
    return bounds


def requirements_for_policy(policy):
    from domiknows.generation.dfa.predicate_monitor import find_predicate_monitor
    monitor = find_predicate_monitor(policy)
    return getattr(monitor.adapter, "requirements", None) if monitor is not None else None


def bind_task_policy(policy, vocabulary, context):
    """Compatibility entry point: bind task facts using the same compiled LCs."""
    try:
        from .graph import bind_vlabench_contextual_dfa
        from .task_policy import task_requirements
    except ImportError:
        from graph import bind_vlabench_contextual_dfa
        from task_policy import task_requirements
    from domiknows.generation.dfa.predicate_monitor import find_predicate_monitor
    bound = bind_vlabench_contextual_dfa(policy, vocabulary, context)
    monitor = find_predicate_monitor(bound)
    if bound is not monitor:
        raise ValueError("bind task facts before adding outer runtime policies")
    world = monitor.adapter.world
    previous = monitor.adapter.requirements
    requirements = task_requirements(context, world, monitor.adapter.entities)
    if "planning_facts" not in context:
        requirements = replace(requirements, planning=previous.planning, allowed_sequences=previous.allowed_sequences)
    if not context.get("task") and not context.get("task_name"):
        requirements = replace(requirements, schema=monitor.adapter.requirements.schema)
    return WorldConstraintDFA(
        monitor.base_dfa, vocabulary, world, monitor.program, monitor.adapter.entities,
        requirements=requirements,
    )
