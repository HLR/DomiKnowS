"""Compilation, progress annotations, and request binding for the planner graph."""
from collections import deque
from itertools import count
from typing import Mapping

try:
    from .plan_codec import PlanVocabulary, plan_to_tokens
    from .graph import create_planner_generation_graph
except ImportError:
    from plan_codec import PlanVocabulary, plan_to_tokens
    from graph import create_planner_generation_graph

_BUILD_IDS = count()

def _legal_skill_sequences(world, max_operations: int):
    complete: set[tuple[str, ...]] = set()
    queue = deque([()])
    visited = {()}
    while queue:
        prefix = queue.popleft()
        for pattern in world.subtask_patterns:
            candidate = prefix + tuple(pattern)
            if len(candidate) > max_operations:
                continue
            complete.add(candidate)
            if candidate not in visited:
                visited.add(candidate)
                queue.append(candidate)
    prefixes = {sequence[:index] for sequence in complete for index in range(len(sequence) + 1)}
    return complete, prefixes


def _with_operation_history(dfa, vocabulary, max_operations):
    """Annotate accepted graph paths for task-completion pruning, without defining grammar."""
    from domiknows.generation.dfa.core import DFA
    from domiknows.generation.dfa.pattern_language import PatternProgress

    skills = {vocabulary.label_for_token(f"skill:{s}"): s for s in vocabulary.skills}
    history = PatternProgress(0, None, frozenset(), frozenset())
    start = (history, dfa.start_state, 0, "compiled", False)
    states, accepting, transitions = {start}, set(), {}
    queue = deque([start])
    while queue:
        source = queue.popleft()
        history, raw, _, _, ended = source
        if dfa.is_accepting(raw):
            accepting.add(source)
        for label in dfa.alphabet:
            next_raw = dfa.step(raw, label)
            if next_raw is None or next_raw in dfa.dead_states:
                continue
            next_history = history
            if label in skills:
                skill = skills[label]
                # Saturation keeps annotations finite if a caller removes a
                # length LC. It never adds a rejection beyond graph formulas.
                next_history = PatternProgress(min(len(history) + 1, max_operations + 1), skill,
                                               history.used | {skill}, frozenset())
            target = (next_history, next_raw, 0, "compiled", ended or label == vocabulary.eos_label)
            transitions[source, label] = target
            if target not in states:
                states.add(target)
                queue.append(target)
    return DFA(frozenset(states), dfa.alphabet, transitions, start, frozenset(accepting))


def _domain_dfa(world, vocabulary, max_operations, complete_sequences=None):
    """Compatibility helper: build and compile the same declarative graph."""
    from domiknows.generation import constraints_to_dfa_from_graph
    graph, bundle = create_planner_generation_graph(
        world, vocabulary, max_operations=max_operations,
        graph_name=f"vlabench_request_grammar_{next(_BUILD_IDS)}",
        skill_language=complete_sequences,
    )
    return _with_operation_history(
        constraints_to_dfa_from_graph(graph, bundle, on_unsupported="error"),
        vocabulary, max_operations,
    )


def compile_planner_dfa(graph, generation_bundle, world, vocabulary, *, max_operations: int = 12):
    """Intersect graph LCs with the complete language declared by the world."""
    from domiknows.generation.dfa.core import product_dfa, minimize_dfa

    from domiknows.generation.dfa.graph_discovery import analyze_generation_constraints
    from domiknows.generation.dfa._constraints import accept_all_dfa
    analyses = analyze_generation_constraints(graph, generation_bundle, on_unsupported="error")
    all_fragments = [dfa for result in analyses for dfa in result.dfas]
    graph_dfa = minimize_dfa(product_dfa(all_fragments)) if all_fragments else accept_all_dfa(generation_bundle.vocabulary)
    # Keep non-grammar LCs for request-specific grammar replacement.
    token_fragments = [dfa for result in analyses
                       if graph.logicalConstrains[result.lc_name] is not generation_bundle.planner_grammar
                       for dfa in result.dfas]
    token_dfa = product_dfa(token_fragments) if token_fragments else accept_all_dfa(generation_bundle.vocabulary)
    try:
        from .constraint_compiler import CompiledPlannerDFA, compile_world_constraints
    except ImportError:
        from constraint_compiler import CompiledPlannerDFA, compile_world_constraints
    program = compile_world_constraints(world)
    operation_dfa = _with_operation_history(graph_dfa, vocabulary, max_operations)
    grammar = product_dfa([token_dfa, operation_dfa])
    return CompiledPlannerDFA.from_dfa(grammar, program, world_constraint_bundle=world, token_dfa=token_dfa,
                                     max_operations=max_operations, operation_dfa=operation_dfa)


def dfa_accepts_plan(dfa, bundle, plan, entity_table=None, *, world=None) -> bool:
    try:
        labels = [bundle.vocabulary.label_for_token(token) for token in plan_to_tokens(plan, entity_table, world=world)]
    except (KeyError, ValueError):
        return False
    if entity_table is not None:
        dfa = bind_vlabench_contextual_dfa(dfa, bundle.vocabulary, {"entities": entity_table})
    return bool(dfa.accepts(labels))


def bind_vlabench_contextual_dfa(base_dfa, vocabulary, context):
    """Bind explicit task and scene facts to compiled world constraints."""
    try:
        from .constraint_compiler import WorldConstraintDFA
        from .task_policy import task_requirements
    except ImportError:
        from constraint_compiler import WorldConstraintDFA
        from task_policy import task_requirements
    from domiknows.generation.dfa.predicate_monitor import find_predicate_monitor
    if find_predicate_monitor(base_dfa) is not None:
        # Retain the exact bound policy for sampling and replay.
        return base_dfa
    if not hasattr(base_dfa, "constraint_program"):
        raise TypeError("VLABench binding requires a compiled world constraint program")
    get = context.get if isinstance(context, Mapping) else lambda k, default=None: getattr(context, k, default)
    entities = get("entity_table") or get("entities") or ()
    from dataclasses import replace
    from domiknows.generation.dfa.core import product_dfa
    try:
        from .task_policy import request_sequences
    except ImportError:
        from task_policy import request_sequences
    world = base_dfa.world_constraint_bundle
    requirements = task_requirements(context, world, entities)
    sequences = request_sequences(world, base_dfa.max_operations, requirements)
    requirements = replace(requirements, allowed_sequences=sequences)
    grammar = base_dfa
    if sequences.allow_empty or frozenset(sequences.roots) != frozenset(world.subtask_patterns):
        operation_dfa = _domain_dfa(world, vocabulary, base_dfa.max_operations, sequences)
        grammar = type(base_dfa).from_dfa(
            product_dfa([base_dfa.token_dfa, operation_dfa]), base_dfa.constraint_program,
            world_constraint_bundle=world, token_dfa=base_dfa.token_dfa,
            max_operations=base_dfa.max_operations, operation_dfa=operation_dfa,
        )
    return WorldConstraintDFA(grammar, vocabulary, world, base_dfa.constraint_program, entities, requirements=requirements)
