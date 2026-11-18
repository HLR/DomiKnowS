import pytest
from domiknows.generation.dfa.core import DFA
from domiknows.generation.dfa.predicate_monitor import PredicateMonitorDFA, compile_predicate_constraints
from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary
from .contract import ExecutorContract
from .executor_graph import CAPABILITIES


@pytest.fixture(scope='module')
def compiled():
    vocabulary = PlanVocabulary(skills=('lift', 'pick', 'place', 'pour'),
        argument_keys=('target_container_name', 'target_entity_name'),
        skill_arguments=(('lift', ()), ('pick', ('target_entity_name',)),
                         ('place', ('target_container_name',)), ('pour', ('target_container_name',))),
        max_entities=4)
    alphabet = frozenset(range(vocabulary.label_count))
    base = DFA(frozenset({0}), alphabet, {(0, a): 0 for a in alphabet}, 0, frozenset({0}))
    public = {'entities':['entity_000','entity_001','entity_002'], 'movable':['entity_000','entity_001']}
    return ExecutorContract(base, vocabulary, public, CAPABILITIES, max_operations=4), vocabulary


def labels(v, text):
    return [v.label_for_token(t) for t in text.split()]


def test_acceptance_methods_are_generic_compiled_monitor_methods(compiled):
    d, v = compiled
    for method in ('step', 'allowed_tokens', 'is_accepting'):
        assert getattr(ExecutorContract, method) is getattr(PredicateMonitorDFA, method)
    assert d.declarations.language.vocabulary.tokens == v.tokens
    assert set(d.program.coverage.compiled) == {
        'executor_pointer_range', 'executor_pick_static_fixture', 'executor_anti_self_placement'}


@pytest.mark.parametrize('text', [
    'skill:lift <eos>',
    'skill:place arg:target_container_name obj:2 <eos>',
    'skill:pick arg:target_entity_name obj:0 <eos>',
    'skill:pick arg:target_entity_name obj:0 skill:pick arg:target_entity_name obj:1 <eos>',
    'skill:pick arg:target_container_name obj:0 skill:place arg:target_container_name obj:2 <eos>',
    'skill:pick arg:target_entity_name obj:3 skill:place arg:target_container_name obj:2 <eos>',
    'skill:pick arg:target_entity_name obj:2 skill:place arg:target_container_name obj:0 <eos>',
    'skill:pick arg:target_entity_name obj:0 skill:place arg:target_container_name obj:0 <eos>',
    '<eos>',
])
def test_compiled_rules_reject_semantic_counterexamples(compiled, text):
    d, v = compiled
    assert not d.accepts(labels(v, text))


def test_valid_wrong_object_and_repeated_relocation_remain_possible(compiled):
    d, v = compiled
    pair = 'skill:pick arg:target_entity_name obj:1 skill:place arg:target_container_name obj:0'
    assert d.accepts(labels(v, pair + ' <eos>'))
    assert d.accepts(labels(v, pair + ' ' + pair + ' <eos>'))
    assert not d.accepts(labels(v, pair + ' ' + pair + ' ' + pair + ' <eos>'))
    assert not d.accepts(labels(v, pair + ' <eos> <eos>'))


def test_semantic_policy_changes_only_by_recompiling_the_lc(compiled):
    d, v = compiled
    graph = d.declarations.semantic_graph
    rule = next(lc for lc in graph.logicalConstrains.values()
                if getattr(lc, 'name', None) == 'executor_anti_self_placement')
    sequence = labels(v, 'skill:pick arg:target_entity_name obj:0 skill:place arg:target_container_name obj:0 <eos>')
    assert not d.accepts(sequence)
    try:
        rule.active = False
        program = compile_predicate_constraints(graph, d.declarations.bindings)
        changed = PredicateMonitorDFA(d.base_dfa, program, d.adapter)
        assert changed.accepts(sequence)  # no hidden Python self-placement guard
    finally:
        rule.active = True
    assert not d.accepts(sequence)  # the original compiled program remains frozen


def test_compiler_failure_has_no_procedural_fallback(compiled, monkeypatch):
    from test_regr.VLABenchAgentInterface import generation_runtime
    d, v = compiled
    def unsupported(*args, **kwargs):
        raise ValueError('unsupported LC')
    monkeypatch.setattr(generation_runtime, 'compile_plan_language', unsupported)
    with pytest.raises(ValueError, match='unsupported LC'):
        ExecutorContract(d.base_dfa, v, {'entities':['a','b'], 'movable':['a']},
                         CAPABILITIES, max_operations=6)


def test_legacy_python_settings_cannot_resume_as_compiled_policy(compiled, tmp_path):
    from .contract import executor_policy_metadata
    from ...protocol import freeze
    path = tmp_path/'settings.json'
    freeze(path, {'executor_policy':{'implementation':'legacy-python-contract'}})
    metadata = executor_policy_metadata()
    assert metadata['implementation']=='domiknows-compiled-executor-v1'
    assert 'executor_graph.py' in metadata['source_sha256']
    with pytest.raises(ValueError, match='fingerprint'):
        freeze(path, {'executor_policy':metadata})


def test_budget_checks_use_compiled_completion_paths(compiled):
    d, v = compiled
    sequence = labels(v, 'skill:pick arg:target_entity_name obj:0 skill:place arg:target_container_name obj:2 <eos>')
    assert not d.allowed_tokens(d.start_state, remaining_steps=5)
    state = d.start_state
    for i, label in enumerate(sequence):
        assert label in d.allowed_tokens(state, remaining_steps=len(sequence)-i-1)
        state = d.step(state, label)
    assert d.is_accepting(state) and not d.allowed_tokens(state, remaining_steps=0)


def test_omitting_history_annotations_preserves_the_compiled_shared_language():
    from collections import deque
    from test_regr.VLABenchAgentInterface.world_graph import build_vlabench_world_graph
    from test_regr.VLABenchAgentInterface.graph import create_planner_generation_graph
    from test_regr.VLABenchAgentInterface.generation_runtime import compile_planner_dfa
    from ....generalization.shared_constraints import shared_profile, SHARED_RULES
    v = PlanVocabulary.from_world(build_vlabench_world_graph('executor_history_equivalence'), max_entities=3)
    plain = shared_profile(v, 2)
    assert plain.operation_dfa is None
    assert {r.name for r in plain.constraint_program.rules} == SHARED_RULES
    world = plain.world_constraint_bundle
    graph, bundle = create_planner_generation_graph(world, v, max_operations=2,
        graph_name='executor_history_reference')
    annotated = compile_planner_dfa(graph, bundle, world, v, max_operations=2)
    # Standard paired-state DFA equivalence checks every reachable prefix,
    # including EOS, malformed prefixes and every vocabulary label.
    pending = deque([(plain.start_state, annotated.start_state)])
    seen = set(pending)
    while pending:
        a, b = pending.popleft()
        assert plain.is_accepting(a) == annotated.is_accepting(b)
        for token in plain.alphabet | annotated.alphabet:
            successor = (plain.step(a, token), annotated.step(b, token))
            if successor not in seen:
                seen.add(successor); pending.append(successor)
