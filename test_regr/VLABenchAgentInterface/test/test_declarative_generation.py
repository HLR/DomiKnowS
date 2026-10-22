"""Independent checks of the declared token language, including formula edits."""
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest

from domiknows.generation import constraints_to_dfa_from_graph
from ..graph import create_planner_generation_graph
from ..plan_codec import PlanVocabulary, plan_to_tokens, tokens_to_plan
from ..world_graph import build_vlabench_world_graph


def reference_words(world, vocabulary, maximum):
    """Enumerate complete skill sentences, then independently serialize roles."""
    sentences = set()
    def append(prefix):
        for pattern in world.subtask_patterns:
            sentence = prefix + pattern
            if len(sentence) <= maximum and sentence not in sentences:
                sentences.add(sentence)
                append(sentence)
    append(())
    words = set()
    for sentence in sentences:
        alternatives = []
        for skill in sentence:
            required = world.skill_arguments[skill]
            optional = world.optional_skill_arguments.get(skill, ())
            operations = []
            for extra in range(len(optional) + 1):
                keys = (*required, *optional[:extra])
                for pointers in product(range(vocabulary.max_entities), repeat=len(keys)):
                    operation = [f'skill:{skill}']
                    for key, pointer in zip(keys, pointers):
                        operation.extend((f'arg:{key}', f'obj:{pointer}'))
                    operations.append(tuple(operation))
            alternatives.append(operations)
        for operations in product(*alternatives):
            words.add(sum(operations, ()) + (vocabulary.eos_token,))
    return words


@pytest.fixture(scope='module')
def declared():
    world = build_vlabench_world_graph('declarative_language_world')
    # Shared and complete prefixes, a zero-argument skill with an optional
    # role, and a second start skill exercise nondeterministic interpretation.
    world = replace(world, subtask_patterns=(('pick',), ('pick', 'push'), ('press',)))
    vocabulary = PlanVocabulary.from_world(world, max_entities=2)
    graph, bundle = create_planner_generation_graph(world, vocabulary, max_operations=3,
                                                   graph_name='declarative_language')
    dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported='error')
    return world, vocabulary, graph, bundle, dfa


def accepts(dfa, vocabulary, tokens):
    return dfa.accepts([vocabulary.label_for_token(token) for token in tokens])


def test_language_matches_independent_reference(declared):
    world, vocabulary, _, _, dfa = declared
    expected = reference_words(world, vocabulary, 3)
    assert expected
    for word in expected:
        assert accepts(dfa, vocabulary, word), word
        # A complete operation may end the plan, but an incomplete role pair
        # and an early EOS must never be accepted as if it were complete.
        for index in range(len(word)):
            truncated = word[:index] + (vocabulary.eos_token,)
            assert accepts(dfa, vocabulary, truncated) == (truncated in expected), truncated
    alphabet = ('skill:pick', 'skill:push', 'arg:target_entity_name',
                'arg:target_container_name', 'obj:0', vocabulary.eos_token)
    for length in range(6):
        for word in product(alphabet, repeat=length):
            # EOS padding is permitted after a complete plan.
            canonical = word[:word.index(vocabulary.eos_token) + 1] if vocabulary.eos_token in word else word
            padded = all(token == vocabulary.eos_token for token in word[len(canonical):])
            assert accepts(dfa, vocabulary, word) == (canonical in expected and padded), word


def test_boundaries_unknown_labels_and_operation_limit(declared):
    _, vocabulary, _, _, dfa = declared
    op = ('skill:press', 'arg:target_entity_name', 'obj:0')
    valid = op * 3 + (vocabulary.eos_token,)
    assert accepts(dfa, vocabulary, valid)
    assert accepts(dfa, vocabulary, valid + (vocabulary.eos_token,) * 3)
    assert not accepts(dfa, vocabulary, op * 4 + (vocabulary.eos_token,))
    assert not accepts(dfa, vocabulary, valid[:-1])
    assert not accepts(dfa, vocabulary, valid + ('obj:0',))
    assert not dfa.accepts([vocabulary.other_label, vocabulary.eos_label])
    assert not accepts(dfa, vocabulary, ('skill:pick', 'obj:0', 'arg:target_entity_name', vocabulary.eos_token))
    assert not accepts(dfa, vocabulary, ('skill:pick', 'arg:target_container_name', 'obj:0', vocabulary.eos_token))


def test_real_world_patterns_match_reference():
    world = build_vlabench_world_graph('declarative_real_world')
    vocabulary = PlanVocabulary.from_world(world, max_entities=1)
    graph, bundle = create_planner_generation_graph(world, vocabulary, max_operations=3,
                                                   graph_name='declarative_real_language')
    dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported='error')
    expected = reference_words(world, vocabulary, 3)
    for word in expected:
        assert accepts(dfa, vocabulary, word), word
    for n in range(4):
        for skills in product(world.skills, repeat=n):
            plan = [{'name': skill, 'params': {key: 0 for key in world.skill_arguments[skill]}}
                    for skill in skills]
            tokens = tuple(plan_to_tokens(plan, world=world))
            assert accepts(dfa, vocabulary, tokens) == (tokens in expected), skills


def test_formula_edits_change_acceptance_without_hidden_grammar():
    world = build_vlabench_world_graph('declarative_edit_world')
    vocabulary = PlanVocabulary.from_world(world, max_entities=1)
    graph, bundle = create_planner_generation_graph(world, vocabulary, max_operations=3,
                                                   graph_name='declarative_edit_language')
    def compiled():
        return constraints_to_dfa_from_graph(graph, bundle, on_unsupported='error')
    illegal_start = ('skill:place', 'arg:target_container_name', 'obj:0', vocabulary.eos_token)
    assert not accepts(compiled(), vocabulary, illegal_start)
    family = bundle.planner_grammar
    start_rule = family.e[-2]
    family.e.remove(start_rule)
    assert accepts(compiled(), vocabulary, illegal_start)
    family.e.insert(len(family.e)-1, start_rule)
    assert not accepts(compiled(), vocabulary, illegal_start)
    incomplete = ('skill:pick', vocabulary.eos_token)
    assert not accepts(compiled(), vocabulary, incomplete)
    family.e.pop()  # Removing the adjacency formula allows the incomplete pair.
    assert accepts(compiled(), vocabulary, incomplete)
    # Names are diagnostics; renaming the surviving formulas changes nothing.
    family.name = 'a_different_name'
    assert accepts(compiled(), vocabulary, incomplete)


def test_required_and_multiple_optional_arguments():
    world = build_vlabench_world_graph('declarative_many_roles_world')
    world = SimpleNamespace(domain_checksum=world.domain_checksum, skills={'pick': world.skills['pick']},
                    skill_arguments={'pick': ('first', 'second')},
                    optional_skill_arguments={'pick': ('third', 'fourth')},
                    subtask_patterns=(('pick',),))
    vocabulary = PlanVocabulary.from_world(world, max_entities=1)
    graph, bundle = create_planner_generation_graph(world, vocabulary, max_operations=1,
                                                   graph_name='declarative_many_roles')
    # The legacy 5-token cap is deliberately retained: raise it here to test
    # the independently declared multi-role grammar rather than that cap.
    cap = next(lc for lc in graph.logicalConstrains.values()
               if type(lc).__name__ == 'atMostAL' and lc.e[-1] == 5)
    cap.e[-1] = 9
    dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported='error')
    for keys in (('first','second'), ('first','second','third'), ('first','second','third','fourth')):
        word = ('skill:pick',) + sum(((f'arg:{key}','obj:0') for key in keys), ()) + (vocabulary.eos_token,)
        assert accepts(dfa, vocabulary, word)
    assert not accepts(dfa, vocabulary, ('skill:pick','arg:first','obj:0','arg:fourth','obj:0',vocabulary.eos_token))


def test_vocabulary_checksum_and_legacy_exports(tmp_path):
    import pickle
    from .. import graph as legacy
    vocabulary = PlanVocabulary.default(4)
    assert vocabulary.checksum == 'cc3c979fd236b317fc9dcf86ce9c611e7a351f3199dc4a0e8ce49ed0aa3549ec'
    assert PlanVocabulary.load(vocabulary.save(tmp_path / 'vocabulary.json')) == vocabulary
    assert pickle.loads(pickle.dumps(vocabulary)) == vocabulary
    # Older checkpoints resolve the class through graph.PlanVocabulary.
    module = PlanVocabulary.__module__
    try:
        PlanVocabulary.__module__ = legacy.__name__
        old_checkpoint = pickle.dumps(vocabulary)
    finally:
        PlanVocabulary.__module__ = module
    assert pickle.loads(old_checkpoint) == vocabulary
    assert legacy.PlanVocabulary is PlanVocabulary
    assert legacy.plan_to_tokens is plan_to_tokens
    assert legacy.tokens_to_plan is tokens_to_plan
    assert legacy.canonicalize_plan is not None
