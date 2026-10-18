"""VLABench predicate bindings and token/state adapter for the generic compiler."""
from dataclasses import dataclass, fields, replace
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
except ImportError:
    from semantics import OperationPredicates, PlanSemantics, PlanSemanticState


@dataclass(frozen=True)
class CompiledPlannerDFA(PredicateConstrainedDFA):
    """VLABench world binding attached to a generic compiled grammar."""

    world_constraint_bundle: Any = None
    token_dfa: Any = None
    max_operations: int = 8

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
        values["task_sequence_complete"] = not eos or self.requirements.sequence_complete(state.semantic.completed_operations)
        values["premature_composite_eos"] = bool(eos and self.schema and self.semantics.premature_composite_eos(
            state.semantic, min_operations=self.schema.min_operations, required_skills=self.schema.required_skills,
        ))
        return (PredicateEvent("operation", values, "completion"),) + ((root_event,) if eos else ())

    def step(self, state, symbol):
        token = self.vocabulary.token_for_label(int(symbol))
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
            result = (SemanticTransition(next_state) if self.world.skill_arguments[skill]
                      else self._finish(next_state))
            # Finalize the previous operation's deferred predicates with False:
            # only the actual last operation is grounded as a premature EOS.
            skill_values = {"skill:" + name: skill == name for name in self.world.skills}
            skill_values.update(operation=True,
                                task_skill_allowed=self.requirements.skill_allowed(state.semantic.completed_operations, skill))
            skill_event = PredicateEvent("operation", skill_values, "task_skill")
            return SemanticTransition(result.state, self._completion_events(state, eos=False) + (skill_event,) + result.events)
        if token.startswith("arg:"):
            return SemanticTransition(replace(state, pending_argument=token[4:]))
        if token.startswith("obj:") and state.skill is not None and state.pending_argument:
            params = dict(state.params)
            params[state.pending_argument] = int(token[4:])
            next_state = replace(state, params=tuple(sorted(params.items())), pending_argument=None)
            if all(key in params for key in self.world.skill_arguments[state.skill]):
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
        super().__init__(base_dfa, program, VLABenchPredicateAdapter(vocabulary, world, entities, schema, requirements))


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
