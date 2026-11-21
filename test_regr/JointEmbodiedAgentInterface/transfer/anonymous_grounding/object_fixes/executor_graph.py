"""Executor restrictions declared in DomiKnowS, then compiled.

Repeating pick/place pairs express the single-gripper interface, never task
answers. Observation predicates contain no goals or precomputed legality flags.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import count
from domiknows.graph import Graph, Concept
from domiknows.graph.logicalConstrain import andL, atMostAL, exactAL, ifL, notL, orL
from domiknows.generation import mark_for_dfa
from domiknows.generation.dfa.graph_discovery import analyze_generation_constraints
from domiknows.generation.dfa.predicate_monitor import PredicateBinding, compile_predicate_constraints
from domiknows.generation.dfa.productive import DFACompletionBounds

CAPABILITIES = {'pick': 'target_entity_name', 'place': 'target_container_name'}
_IDS = count()


@dataclass(frozen=True)
class ExecutorDeclarations:
    language: object
    grammar_rules: tuple
    semantic_graph: object
    predicates: dict
    bindings: tuple
    program: object


def declare_supported_skills(graph, bundle, vocabulary, supported):
    """Compile the executor's action inventory without changing label IDs."""
    unsupported = [bundle.context.token_value('skill:' + skill, 'position')
                   for skill in vocabulary.skills if skill not in supported]
    if unsupported:
        with graph:
            mark_for_dfa(atMostAL(unsupported[0] if len(unsupported) == 1 else orL(*unsupported), 0,
                                 name='executor_unsupported_skill'))


@lru_cache(maxsize=16)
def executor_declarations(vocabulary, max_operations=12):
    from test_regr.VLABenchAgentInterface.graph import create_plan_grammar_graph
    from test_regr.VLABenchAgentInterface.generation_runtime import compile_plan_language
    serial = next(_IDS)
    # Labels stay identical; the graph declares exactly the executor's schemas.
    schema = replace(vocabulary, skill_arguments=tuple((s, (r,)) for s, r in CAPABILITIES.items()),
                     optional_skill_arguments=())
    if schema.tokens != vocabulary.tokens:
        raise AssertionError('executor grammar changed canonical label IDs')
    graph, bundle = create_plan_grammar_graph(schema, (('pick', 'place'),),
        max_operations=max_operations, graph_name=f'executor_language_{serial}')
    ctx = bundle.context
    declare_supported_skills(graph, bundle, vocabulary, CAPABILITIES)
    with graph:
        mark_for_dfa(exactAL(ctx.token_value(vocabulary.eos_token, 'position'), 1,
                            name='executor_single_termination'))
    language = compile_plan_language(graph, bundle, schema)
    analyses = analyze_generation_constraints(graph, bundle, on_unsupported='error')
    grammar_rules = tuple((getattr(graph.logicalConstrains.get(a.lc_name), 'name', None) or a.lc_name,
                           dfa, frozenset(DFACompletionBounds.from_dfa(dfa).distances))
                          for a in analyses for dfa in a.dfas)
    grammar_rules = tuple(sorted(grammar_rules, key=lambda r: not r[0].startswith('executor_')))
    semantic_graph = Graph(f'executor_observations_{serial}')
    with semantic_graph:
        predicates = {name: Concept(name=name) for name in (
            'operation', 'pick', 'place', 'observed_target', 'movable_target', 'subject_equals_destination')}
        p = lambda name: predicates[name]('op')
        ifL(p('operation'), p('observed_target'), name='executor_pointer_range')
        ifL(andL(p('operation'), p('pick')), p('movable_target'), name='executor_pick_static_fixture')
        ifL(andL(p('operation'), p('place')), notL(p('subject_equals_destination')),
            name='executor_anti_self_placement')
    bindings = tuple(PredicateBinding(concept, name, 'executor_operation')
                     for name, concept in predicates.items())
    program = compile_predicate_constraints(semantic_graph, bindings)
    return ExecutorDeclarations(language, grammar_rules, semantic_graph, predicates, bindings, program)
