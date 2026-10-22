"""The auxiliary compiler solves shared witnesses, rather than per-rule guesses."""
from itertools import product

import pytest

from domiknows.generation import GenerationEncoder, constraints_to_dfa_from_graph
from domiknows.generation.dfa.auxiliary import AuxiliaryFormulaError
from domiknows.graph import EnumConcept
from domiknows.graph.logicalConstrain import andL, exactL, ifL, notL, orL


def alternating_graph(name):
    graph, bundle = GenerationEncoder(vocab=['a', 'b', '<eos>'], eos_token='<eos>', graph_name=name).build_graph()
    ctx = bundle.context
    with graph:
        state = bundle.token(name='arbitrary_witness_name', ConceptClass=EnumConcept, values=['left','right'])
        family = andL(
            exactL(state.left('x'), state.right('x'), 1),
            ifL(state.left('x'), ctx.token_value('a','x')),
            ifL(state.right('x'), ctx.token_value('b','x')),
            ifL(ctx.sequence_start('s'), state.left('first',path=('s',bundle.sequence_start_token))),
            ifL(ctx.is_before_rel('e'), orL(
                andL(state.left('x',path=('e',bundle.first_token)), state.right('y',path=('e',bundle.second_token))),
                andL(state.right('x',path=('e',bundle.first_token)), state.left('y',path=('e',bundle.second_token))),
            )),
            name='names_do_not_define_semantics',
        )
    return graph, bundle, state, family


def test_projected_local_formulas_match_independent_assignments():
    graph, bundle, _, _ = alternating_graph('auxiliary_alternating')
    dfa = constraints_to_dfa_from_graph(graph,bundle,on_unsupported='error')
    for length in range(7):
        for word in product(('a','b','<eos>','<other>'),repeat=length):
            # Independent finite-model interpretation: choose exactly one
            # witness per position; start left, alternate, match each token.
            expected = any(
                (not witnesses or witnesses[0] == 0)
                and all(a != b for a,b in zip(witnesses,witnesses[1:]))
                and all(token == ('a' if state == 0 else 'b') for token,state in zip(word,witnesses))
                for witnesses in product((0,1),repeat=length)
            )
            labels = [bundle.vocabulary.other_label if token == '<other>' else bundle.vocabulary.label_for_token(token)
                      for token in word]
            assert dfa.accepts(labels) == expected, word


def test_separate_heads_share_one_auxiliary_assignment():
    graph,bundle,state,family = alternating_graph('auxiliary_shared')
    # Remove token compatibility and starts/edges. Each witness is otherwise
    # possible. Two separate head constraints rule out both possibilities.
    family.e[:] = family.e[:1]
    with graph:
        notL(state.left('position'))
        notL(state.right('position'))
    dfa=constraints_to_dfa_from_graph(graph,bundle,on_unsupported='error')
    assert dfa.accepts([])
    assert not dfa.accepts([bundle.vocabulary.label_for_token('a')])


def test_separate_families_share_one_auxiliary_assignment():
    graph,bundle,state,family = alternating_graph('auxiliary_two_families')
    family.e[:] = family.e[:1]
    with graph:
        andL(exactL(state.left('x'),state.right('x'),1),notL(state.left('position')))
        notL(state.right('position'))
    assert not constraints_to_dfa_from_graph(graph,bundle,on_unsupported='error').accepts([0])


def test_unsupported_endpoint_binding_fails_closed():
    graph,bundle,state,family=alternating_graph('auxiliary_invalid_path')
    with graph:
        wrong = ifL(bundle.is_before_rel('edge'), state.left('x',path=('e',bundle.first_token)))
        family.e.append(wrong)
        wrong.headLC=False
    with pytest.raises(AuxiliaryFormulaError,match='endpoint paths'):
        constraints_to_dfa_from_graph(graph,bundle,on_unsupported='error')


def test_removed_unary_rule_changes_the_projected_language():
    graph,bundle,_,family=alternating_graph('auxiliary_remove_unary')
    labels=[bundle.vocabulary.label_for_token('a')]*2
    assert not constraints_to_dfa_from_graph(graph,bundle,on_unsupported='error').accepts(labels)
    family.e.pop(2)  # The right witness no longer requires b.
    assert constraints_to_dfa_from_graph(graph,bundle,on_unsupported='error').accepts(labels)


def test_latent_only_family_does_not_become_a_hard_constraint():
    from domiknows.generation import LatentWindowSpec, mark_for_dfa, mark_for_latent
    graph, bundle, _, family = alternating_graph('auxiliary_latent_only')
    labels = [bundle.vocabulary.label_for_token('b')]
    assert not constraints_to_dfa_from_graph(graph, bundle, on_unsupported='error').accepts(labels)
    mark_for_latent(family, LatentWindowSpec(if_label=0, formula=0, window=1))
    assert constraints_to_dfa_from_graph(graph, bundle, on_unsupported='error').accepts(labels)
    mark_for_dfa(family)
    assert not constraints_to_dfa_from_graph(graph, bundle, on_unsupported='error').accepts(labels)
