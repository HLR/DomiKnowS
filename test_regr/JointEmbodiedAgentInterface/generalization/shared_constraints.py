"""Transfer profile compiled from existing graph LCs, not task-name aliases."""
from dataclasses import replace
from functools import lru_cache
from collections import Counter

SHARED_RULES = frozenset({
    'operation_exactly_one_skill', 'operation_schema_is_valid',
    'operation_pointer_is_valid', 'anti_self_placement', 'pick_excludes_static_fixtures',
})
EXCLUDED_NONUNIVERSAL = {
    'causal__destinations_stay_put': 'valid plans can pick an earlier destination',
    'causal__distinct_subtask_destinations': 'a valid plan can revisit a destination',
    'no_redundant_repeated_picks': 'regrasping and rearrangement are legitimate',
    'task_patterns': 'benchmark-specific sequences are not external task knowledge',
    'physics_qa': 'external suites do not expose the same property-question protocol',
}


@lru_cache(maxsize=4)
def shared_profile(vocabulary, max_operations):
    from test_regr.VLABenchAgentInterface.world_graph import build_vlabench_world_graph
    from test_regr.VLABenchAgentInterface.knowledge import TaskKnowledge
    from test_regr.VLABenchAgentInterface.graph import create_planner_generation_graph
    from test_regr.VLABenchAgentInterface.generation_runtime import compile_planner_dfa
    world = build_vlabench_world_graph('shared_transfer_world', knowledge_level='scene',
        task_knowledge=TaskKnowledge({}, {}, {}, {}), task_identity='oracle')
    # Any schema-valid skill sequence; no VLABench subtask template restriction.
    world = replace(world, subtask_patterns=tuple((skill,) for skill in vocabulary.skills))
    retained = set()
    def graphs(graph):
        yield graph
        for child in graph.subgraphs.values():
            yield from graphs(child)
    for graph in graphs(world.graph):
        for constraint in graph.logicalConstrains.values():
            if not getattr(constraint, 'headLC', True):
                continue
            name = getattr(constraint, 'name', None)
            constraint.active = name in SHARED_RULES
            if constraint.active:
                retained.add(name)
    if retained != SHARED_RULES:
        raise ValueError(f'shared rule declarations differ: {retained ^ SHARED_RULES}')
    graph, bundle = create_planner_generation_graph(world, vocabulary, max_operations=max_operations,
                                                   graph_name='shared_transfer_grammar')
    return compile_planner_dfa(graph, bundle, world, vocabulary, max_operations=max_operations)


def bind_shared(vocabulary, max_operations, context):
    from test_regr.VLABenchAgentInterface.generation_runtime import bind_vlabench_contextual_dfa
    # Only observational bindings; task identity / obligations / reference goals
    # must never activate task-specific knowledge in the external profile.
    safe = {k: context[k] for k in ('entity_table', 'entities', 'generation_entity_types') if k in context}
    return bind_vlabench_contextual_dfa(shared_profile(vocabulary, max_operations), vocabulary, safe)


def rule_trace(dfa, labels):
    """Per-rule rejected alternatives on the actually generated prefixes."""
    from domiknows.generation.dfa.predicate_monitor import find_predicate_monitor, PredicateConstraintProgram
    monitor = find_predicate_monitor(dfa)
    if monitor is None:
        return {'declared': [], 'applicable_events': {}, 'triggered_rejections': {}, 'events': []}
    applicable = Counter(); triggered = Counter(); rows = []
    programs = {rule.name: PredicateConstraintProgram((rule,)) for rule in monitor.program.rules}
    state = monitor.start_state
    for step, label in enumerate(labels):
        if state is None:
            break
        for candidate in monitor.base_dfa.allowed_tokens(state.grammar):
            transition = monitor.adapter.step(state.semantic, candidate)
            if transition is None:
                continue
            rejected = []
            for rule in monitor.program.rules:
                events = tuple(e for e in transition.events if (e.domain, e.phase) == (rule.domain, rule.phase))
                if not events:
                    continue
                applicable[rule.name] += 1
                if not programs[rule.name].check_events(events):
                    triggered[rule.name] += 1; rejected.append(rule.name)
            if rejected:
                rows.append({'step': step, 'candidate_label': int(candidate), 'rejected_by': rejected})
        state = monitor.step(state, int(label))
    return {'declared': sorted(programs), 'applicable_events': dict(applicable),
            'triggered_rejections': dict(triggered), 'events': rows}
