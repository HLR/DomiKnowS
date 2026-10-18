"""Reuse predicate compilation in a document workflow with string symbols."""
from dataclasses import dataclass, replace
from itertools import product

import pytest

from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import andL, ifL, notL, orL, sumL
from domiknows.generation import (
    DFA, MissingSemanticBinding, PredicateBinding, PredicateConstrainedDFA,
    PredicateConstraintCompilationError, PredicateEvent, SemanticTransition,
    compile_predicate_constraints,
)


def document_policy():
    with Graph("document_publication_policy") as graph:
        needs_review = Concept(name="needs_review")
        approved = Concept(name="approved")
        ifL(needs_review("document"), approved("document"), name="review_before_publication")
    program = compile_predicate_constraints(graph, (
        PredicateBinding(needs_review, "needs_review", "publication"),
        PredicateBinding(approved, "approved", "publication"),
    ))
    return graph, needs_review, approved, program


@dataclass(frozen=True)
class DocumentState:
    approved: bool = False
    published: bool = False


class DocumentAdapter:
    def __init__(self, requires_review):
        self.requires_review = requires_review
        self.start_state = DocumentState()

    def publication(self, state):
        return PredicateEvent("publication", {"needs_review": self.requires_review, "approved": state.approved})

    def step(self, state, symbol):
        if state.published:
            return None
        if symbol == "approve":
            return SemanticTransition(replace(state, approved=True))
        if symbol == "publish":
            return SemanticTransition(replace(state, published=True), (self.publication(state),))
        return None

    def is_accepting(self, state):
        return state.published

    def acceptance_events(self, state):
        return (self.publication(state),)


def grammar(program):
    base = DFA(frozenset({0}), frozenset({"approve", "publish"}),
               {(0, "approve"): 0, (0, "publish"): 0}, 0, frozenset({0}))
    return PredicateConstrainedDFA.from_dfa(base, program)


def test_same_compiled_program_binds_independent_requests_and_noninteger_symbols():
    _, _, _, program = document_policy()
    compiled = grammar(program)
    restricted = compiled.bind(DocumentAdapter(requires_review=True))
    unrestricted = compiled.bind(DocumentAdapter(requires_review=False))
    assert restricted.allowed_tokens(restricted.start_state) == {"approve"}
    assert unrestricted.allowed_tokens(unrestricted.start_state) == {"approve", "publish"}
    assert not restricted.accepts(["publish"])
    assert restricted.accepts(["approve", "publish"])
    assert unrestricted.accepts(["publish"])
    assert not restricted.accepts(["approve"])
    assert not restricted.accepts(["approve", "publish", "approve"])
    # The compiled grammar is unchanged by per-request binding.
    assert compiled.accepts(["publish"])
    assert compiled.start_state == 0


def test_runtime_acceptance_rechecks_final_predicate_valuations():
    _, _, _, program = document_policy()
    class IncompleteEventAdapter(DocumentAdapter):
        def step(self, state, symbol):
            result = super().step(state, symbol)
            return None if result is None else SemanticTransition(result.state)
    monitored = grammar(program).bind(IncompleteEventAdapter(True))
    assert not monitored.accepts(["publish"])
    assert monitored.accepts(["approve", "publish"])


def test_boolean_expression_semantics_match_independent_truth_table():
    with Graph("boolean_policy_truth_table") as graph:
        a, b, c = (Concept(name=name) for name in ("a", "b", "c"))
        ifL(a("item"), orL(b("item"), notL(andL(a("item"), c("item")))), name="boolean_rule")
    program = compile_predicate_constraints(graph, [PredicateBinding(concept, name, "item")
                                                   for concept, name in ((a, "a"), (b, "b"), (c, "c"))])
    for av, bv, cv in product((False, True), repeat=3):
        expected = (not av) or bv or not (av and cv)
        assert program.check(PredicateEvent("item", {"a": av, "b": bv, "c": cv})) == expected


def test_adapter_defined_evaluation_phases_are_generic():
    with Graph("deferred_document_policy") as graph:
        document, signed = Concept(name="document"), Concept(name="signed")
        ifL(document("d"), signed("d"), name="signature_required")
    program = compile_predicate_constraints(graph, (
        PredicateBinding(document, "document", "file"),
        PredicateBinding(signed, "signed", "file", phase="seal"),
    ))
    values = {"document": True, "signed": False}
    assert program.check(PredicateEvent("file", values))
    assert not program.check(PredicateEvent("file", values, phase="seal"))
    assert program.check(PredicateEvent("file", {**values, "signed": True}, phase="seal"))


@pytest.mark.parametrize("values", [{"needs_review": True}, {"needs_review": True, "approved": 1}])
def test_missing_or_nonboolean_predicate_fails_with_rule_name(values):
    _, _, _, program = document_policy()
    with pytest.raises(MissingSemanticBinding, match="review_before_publication"):
        program.check(PredicateEvent("publication", values))


@pytest.mark.parametrize("case", ["unregistered", "path", "value", "domains", "phases"])
def test_unsupported_shapes_fail_closed_with_constraint_name(case):
    with Graph("unsupported_predicate_policy") as graph:
        a, b = Concept(name="a"), Concept(name="b")
        if case == "value":
            sumL(a("x"), name="unsupported_policy")
        elif case == "path":
            ifL(a("x"), a("x", path=("x",)), name="unsupported_policy")
        else:
            ifL(a("x"), b("x"), name="unsupported_policy")
    bindings = [PredicateBinding(a, "a", "item", phase="first" if case == "phases" else "immediate")]
    if case != "unregistered":
        bindings.append(PredicateBinding(b, "b", "other" if case == "domains" else "item",
                                         phase="second" if case == "phases" else "immediate"))
    with pytest.raises(PredicateConstraintCompilationError, match="unsupported_policy"):
        compile_predicate_constraints(graph, bindings)


def test_compiled_rules_are_snapshots_and_recompilation_observes_deactivation():
    graph, needs_review, approved, program = document_policy()
    rule = next(lc for lc in graph.logicalConstrains.values() if lc.headLC)
    rule.active = False
    rebound = compile_predicate_constraints(graph, (
        PredicateBinding(needs_review, "needs_review", "publication"),
        PredicateBinding(approved, "approved", "publication"),
    ))
    event = PredicateEvent("publication", {"needs_review": True, "approved": False})
    assert not program.check(event)
    assert rebound.check(event)


def test_ambiguous_predicate_registration_is_rejected():
    graph, needs_review, approved, _ = document_policy()
    with pytest.raises(ValueError, match="duplicate predicate"):
        compile_predicate_constraints(graph, (
            PredicateBinding(needs_review, "same", "publication"),
            PredicateBinding(approved, "same", "publication"),
        ))


def test_binding_guard_handles_wrapped_policies_and_plain_grammars():
    from types import SimpleNamespace
    from domiknows.generation import PredicateConstraintProgram, find_predicate_monitor, require_bound_predicate_dfa

    _, _, _, program = document_policy()
    compiled = grammar(program)
    unbound = SimpleNamespace(base_dfa=compiled)
    with pytest.raises(MissingSemanticBinding, match="request-bound"):
        require_bound_predicate_dfa(unbound)
    monitored = compiled.bind(DocumentAdapter(True))
    wrapped = SimpleNamespace(base_dfa=SimpleNamespace(base_dfa=monitored))
    assert find_predicate_monitor(wrapped) is monitored
    assert require_bound_predicate_dfa(wrapped) is wrapped
    plain = grammar(PredicateConstraintProgram())
    assert require_bound_predicate_dfa(plain) is plain
    # A malformed wrapper cycle cannot hang a decoder's binding check.
    cycle = SimpleNamespace()
    cycle.base_dfa = cycle
    assert find_predicate_monitor(cycle) is None


def test_semantic_lookahead_prunes_locally_valid_dead_ends_with_exact_token_budget():
    _, _, _, program = document_policy()
    base = DFA(frozenset(range(5)), frozenset({"shortcut", "route", "approve", "publish"}),
               {(0, "shortcut"): 1, (0, "route"): 2, (1, "publish"): 4,
                (2, "approve"): 3, (3, "publish"): 4}, 0, frozenset({4}))
    class Routes(DocumentAdapter):
        def step(self, state, symbol):
            if symbol in {"shortcut", "route"}:
                return SemanticTransition(state)
            return super().step(state, symbol)
    monitored = PredicateConstrainedDFA.from_dfa(base, program).bind(Routes(True))
    assert monitored.allowed_tokens(monitored.start_state) == {"shortcut", "route"}
    assert monitored.allowed_tokens(monitored.start_state, remaining_steps=2) == {"route"}
    assert monitored.allowed_tokens(monitored.start_state, remaining_steps=1) == set()
    assert monitored.can_reach_accepting(monitored.start_state, max_steps=3)
    assert not monitored.can_reach_accepting(monitored.start_state, max_steps=2)
    state = monitored.step(monitored.step(monitored.start_state, "route"), "approve")
    assert monitored.allowed_tokens(state, remaining_steps=0) == {"publish"}
    # Check all budgets against explicit suffix enumeration, independent of
    # the search algorithm and including the zero-additional-transition case.
    for budget in range(4):
        expected = {symbol for symbol in base.alphabet if any(
            monitored.accepts((symbol, *suffix))
            for length in range(budget + 1) for suffix in product(base.alphabet, repeat=length)
        )}
        assert monitored.allowed_tokens(monitored.start_state, remaining_steps=budget) == expected


def test_outer_productivity_view_includes_additional_runtime_restrictions():
    from domiknows.generation import ProductiveDFA

    _, _, _, program = document_policy()
    monitored = grammar(program).bind(DocumentAdapter(True))
    class PublishingOnly:
        base_dfa = monitored
        start_state = monitored.start_state
        alphabet = monitored.alphabet
        def step(self, state, symbol):
            return monitored.step(state, symbol) if symbol == "publish" else None
        def is_accepting(self, state):
            return monitored.is_accepting(state)
        def allowed_tokens(self, state, remaining_steps=None):
            return monitored.allowed_tokens(state, remaining_steps).intersection({"publish"})
    # The inner policy has a completion via approval; the outer restriction
    # removes that route and must not inherit the inner reachability answer.
    assert monitored.can_reach_accepting(monitored.start_state, max_steps=2)
    outer = ProductiveDFA(PublishingOnly())
    assert not outer.can_reach_accepting(outer.start_state, max_steps=2)
    assert outer.allowed_tokens(outer.start_state, remaining_steps=2) == set()


def test_search_limit_is_explicit_and_does_not_cache_unknown_as_infeasible():
    from domiknows.generation import ProductiveDFA, SemanticSearchLimitExceeded

    loop = DFA(frozenset({0}), frozenset({"wait"}), {(0, "wait"): 0}, 0, frozenset())
    chain = DFA(frozenset(range(22)), frozenset({"wait"}),
                {(i, "wait"): i + 1 for i in range(21)}, 0, frozenset())
    limited = ProductiveDFA(chain, max_state_expansions=2)
    with pytest.raises(SemanticSearchLimitExceeded, match="feasibility is unknown"):
        limited.allowed_tokens(0, remaining_steps=20)
    assert limited.allowed_tokens(0, remaining_steps=1) == set()
    # A large finite depth uses an iterative search rather than Python recursion.
    assert not ProductiveDFA(loop).can_reach_accepting(0, max_steps=1500)
    with pytest.raises(ValueError, match="finite"):
        limited.can_reach_accepting(0)


def test_program_projection_keeps_complete_observable_rules_and_their_snapshot():
    graph, needs_review, approved, program = document_policy()
    # A known predicate is not enough to evaluate a rule containing an unknown one.
    assert not program.project({"needs_review"}).rules
    projected = program.project({"needs_review", "approved"}, domain="publication", phases=("immediate",))
    assert projected.rules == program.rules
    assert not projected.check(PredicateEvent("publication", {"needs_review": True, "approved": False}))
    assert not program.project({"needs_review", "approved"}, domain="other").rules
    assert not program.project({"needs_review", "approved"}, phases=("later",)).rules
    for rule in graph.logicalConstrains.values():
        rule.active = False
    # Both programs retain the compiled LC snapshot, irrespective of later edits.
    assert projected.rules == program.rules


def test_compilation_coverage_accounts_for_every_active_head_and_inactive_rule():
    graph, needs_review, approved, _ = document_policy()
    with graph:
        sumL(needs_review("d"), name="disabled_value", active=False)
    program = compile_predicate_constraints(graph, (
        PredicateBinding(needs_review, "needs_review", "publication"),
        PredicateBinding(approved, "approved", "publication"),
    ))
    assert program.coverage.compiled == ("review_before_publication",)
    assert program.coverage.inactive == ("disabled_value",)
    assert len(program.coverage.compiled) == sum(lc.headLC and lc.active for lc in graph.logicalConstrains.values())
    # Turning on a previously omitted unsupported rule must fail compilation.
    next(lc for lc in graph.logicalConstrains.values() if lc.name == "disabled_value").active = True
    with pytest.raises(PredicateConstraintCompilationError, match="disabled_value"):
        compile_predicate_constraints(graph, (PredicateBinding(needs_review, "needs_review", "publication"),
                                              PredicateBinding(approved, "approved", "publication")))


def test_descendant_constraints_are_included_in_coverage_and_cannot_be_silently_omitted():
    with Graph("parent_policy") as graph:
        a = Concept(name="a")
        with Graph("nested_policy"):
            b = Concept(name="b")
            ifL(a("item"), b("item"), name="nested_rule")
    bindings = (PredicateBinding(a, "a", "item"), PredicateBinding(b, "b", "item"))
    program = compile_predicate_constraints(graph, bindings)
    assert program.coverage.compiled == ("nested_rule",)
    assert not program.check(PredicateEvent("item", {"a": True, "b": False}))
    with pytest.raises(PredicateConstraintCompilationError, match="nested_rule.*unregistered"):
        compile_predicate_constraints(graph, bindings[:1])



@pytest.mark.parametrize("requires_review", [False, True])
def test_generic_bounded_equivalence_audit_uses_independent_document_oracle(requires_review):
    from test_regr.generation.constraint_equivalence import assert_bounded_equivalence

    _, _, _, program = document_policy()
    compiled = grammar(program)
    monitor = compiled.bind(DocumentAdapter(requires_review))
    def oracle(sequence):
        return (bool(sequence) and sequence[-1] == "publish"
                and (not requires_review or "approve" in sequence[:-1]))
    audit = assert_bounded_equivalence(
        compiled, monitor, oracle, max_steps=4, terminal_symbols={"publish"},
    )
    assert audit.accepted == (3 if requires_review else 4)
    assert audit.rejected > 0 and audit.checked_prefixes == 4
