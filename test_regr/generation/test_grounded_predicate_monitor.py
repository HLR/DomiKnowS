"""Truth-table and finite-model oracles for generic LC monitoring."""
from itertools import product

import pytest

from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import (
    andL, atLeastL, atMostL, equivalenceL, exactL, existsL, forAllL,
    ifL, iffL, nandL, norL, notL, orL, xorL,
)
from domiknows.generation import (
    DFA, MissingSemanticBinding, PredicateBinding, PredicateConstrainedDFA,
    PredicateConstraintCompilationError, PredicateEvent, PredicateGroundingLimitExceeded,
    SemanticTransition, compile_predicate_constraints,
)


def compile_boolean(operator, *, count=None, variables=("x", "x", "x"), max_groundings=10000):
    with Graph("extended_boolean_monitor") as graph:
        concepts = [Concept(name=name) for name in ("a", "b", "c")]
        operands = [concept(variable) for concept, variable in zip(concepts, variables)]
        kwargs = {} if count is None else {"limit": count}
        operator(*operands, name="extended_rule", **kwargs)
    return compile_predicate_constraints(graph, [PredicateBinding(c, name, "event")
        for c, name in zip(concepts, ("a", "b", "c"))], max_groundings=max_groundings)


@pytest.mark.parametrize("operator,oracle", [
    (nandL, lambda xs: not all(xs)), (norL, lambda xs: not any(xs)),
    (xorL, lambda xs: sum(xs) % 2 == 1),
    (iffL, lambda xs: len(set(xs)) == 1), (equivalenceL, lambda xs: len(set(xs)) == 1),
])
def test_new_boolean_operators_match_exhaustive_independent_tables(operator, oracle):
    program = compile_boolean(operator)
    for values in product((False, True), repeat=3):
        assert program.check(PredicateEvent("event", dict(zip(("a", "b", "c"), values)))) == oracle(values)


@pytest.mark.parametrize("operator,limit,oracle", [
    (atMostL, 1, lambda n: n <= 1), (atLeastL, 2, lambda n: n >= 2),
    (exactL, 2, lambda n: n == 2), (existsL, None, lambda n: n >= 1),
    (exactL, 0, lambda n: n == 0),
])
def test_counts_over_scalar_operands(operator, limit, oracle):
    program = compile_boolean(operator, count=limit)
    for values in product((False, True), repeat=3):
        assert program.check(PredicateEvent("event", dict(zip(("a", "b", "c"), values)))) == oracle(sum(values))


@pytest.mark.parametrize("operator,limit,oracle", [
    (atMostL, 1, lambda n: n <= 1), (atLeastL, 2, lambda n: n >= 2),
    (exactL, 2, lambda n: n == 2), (existsL, None, lambda n: n >= 1),
])
def test_head_counts_quantify_over_all_entities_in_event(operator, limit, oracle):
    with Graph("count_entities_monitor") as graph:
        approved = Concept(name="approved")
        operator(approved("document"), name="count_documents", **({} if limit is None else {"limit": limit}))
    program = compile_predicate_constraints(graph, [PredicateBinding(approved, "approved", "batch")])
    assert program.rules[0].variables == ("document",)
    assert program.rules[0].free_variables == ()
    for truths in product((False, True), repeat=3):
        values = {("approved", entity): truth for entity, truth in zip(("a", "b", "c"), truths)}
        assert program.check(PredicateEvent("batch", values, variable_domains={"document": ("a", "b", "c")})) == oracle(sum(truths))
    assert program.check(PredicateEvent("batch", {}, variable_domains={"document": ()})) == oracle(0)


@pytest.mark.parametrize("operator", [ifL, forAllL])
def test_two_variables_use_cartesian_universal_grounding_and_distinct_values(operator):
    with Graph("multiple_variable_monitor") as graph:
        a, b = Concept(name="a"), Concept(name="b")
        operator(a("x"), b("y"), name="two_variables")
    program = compile_predicate_constraints(graph, [PredicateBinding(a, "a", "event"), PredicateBinding(b, "b", "event")])
    for truths in product((False, True), repeat=4):
        av, bv = truths[:2], truths[2:]
        event = PredicateEvent("event", {
            ("a", "a0"): av[0], ("a", "a1"): av[1],
            ("b", "b0"): bv[0], ("b", "b1"): bv[1],
        }, variable_domains={"x": ("a0", "a1"), "y": ("b0", "b1")})
        assert program.check(event) == all(not x or y for x in av for y in bv)
    with pytest.raises(MissingSemanticBinding, match="two_variables.*variable_domains"):
        program.check(PredicateEvent("event", {"a": True, "b": False}))
    assert program.check(PredicateEvent("event", {}, variable_domains={"x": (), "y": ("b",)}))


def test_same_predicate_with_different_variables_never_collapses_to_one_boolean():
    with Graph("same_predicate_two_variables") as graph:
        p = Concept(name="p")
        xorL(p("source"), p("target"), name="distinct_truths")
    program = compile_predicate_constraints(graph, [PredicateBinding(p, "p", "event")])
    event = PredicateEvent("event", {("p", "a"): True, ("p", "b"): False},
                           variable_domains={"source": ("a",), "target": ("b",)})
    assert program.check(event)
    assert not program.check(PredicateEvent("event", event.values, variable_domains={"source": ("a", "b"), "target": ("a", "b")}))
    with pytest.raises(MissingSemanticBinding, match="distinct_truths"):
        program.check(PredicateEvent("event", {"p": True}, variable_domains=event.variable_domains))


def test_local_count_keeps_outer_variable_and_quantifies_only_inner_variable():
    with Graph("conditional_counts_monitor") as graph:
        entity = Concept(name="entity")
        request, approved, eligible = (entity(name=name) for name in ("request", "approved", "eligible"))
        ifL(request("x"), exactL(approved("x"), eligible("y"), limit=2), name="conditional_count")
    program = compile_predicate_constraints(graph, [PredicateBinding(c, name, "event")
        for c, name in ((request, "request"), (approved, "approved"), (eligible, "eligible"))])
    assert program.rules[0].free_variables == ("x",)
    for truths in product((False, True), repeat=5):
        r0, r1, a0, a1, eligible = truths
        event = PredicateEvent("event", {
            ("request", "a"): r0, ("request", "b"): r1,
            ("approved", "a"): a0, ("approved", "b"): a1,
            ("eligible", "candidate"): eligible,
        }, variable_domains={"x": ("a", "b"), "y": ("candidate",)})
        assert program.check(event) == ((not r0 or a0 + eligible == 2) and (not r1 or a1 + eligible == 2))


def test_nested_exists_scopes_and_universal_implication():
    with Graph("nested_count_scope_monitor") as graph:
        a, b, c = [Concept(name=name) for name in ("a", "b", "c")]
        forAllL(a("x"), existsL(andL(b("x"), notL(existsL(c("y"))))), name="nested_exists")
    program = compile_predicate_constraints(graph, [PredicateBinding(p, name, "event") for p, name in ((a, "a"), (b, "b"), (c, "c"))])
    for av, bv, c0, c1 in product((False, True), repeat=4):
        event = PredicateEvent("event", {("a", "outer"): av, ("b", "outer"): bv, ("c", 0): c0, ("c", 1): c1},
                               variable_domains={"x": ("outer",), "y": (0, 1)})
        assert program.check(event) == (not av or (bv and not (c0 or c1)))


def test_head_count_sums_operands_without_cartesian_duplicate_counts():
    program = compile_boolean(exactL, count=3, variables=("x", "y", "z"))
    event = PredicateEvent("event", {("a", 1): True, ("a", 2): True, ("b", 3): True, ("c", 4): False},
                           variable_domains={"x": (1, 2), "y": (3,), "z": (4,)})
    assert program.check(event)


def test_three_free_variables_remain_independent():
    program = compile_boolean(orL, variables=("x", "y", "z"))
    assert program.rules[0].free_variables == ("x", "y", "z")
    for truths in product((False, True), repeat=6):
        values = {(predicate, entity): truth for (predicate, entity), truth in zip(
            product(("a", "b", "c"), (0, 1)), truths)}
        event = PredicateEvent("event", values, variable_domains={"x": (0, 1), "y": (0, 1), "z": (0, 1)})
        expected = all(truths[x] or truths[2+y] or truths[4+z] for x, y, z in product((0, 1), repeat=3))
        assert program.check(event) == expected


def test_ternary_relation_and_normalized_role_projection():
    with Graph("ternary_predicate_monitor") as graph:
        entity = Concept(name="entity")
        relation = Concept(name="relation")
        relation.has_a(first=entity, second=entity, third=entity)
        ready = entity(name="ready")
        ifL(relation("x", "y", "z"), ready("z"), name="ternary_rule")
    program = compile_predicate_constraints(graph, [PredicateBinding(relation, "relation", "event"), PredicateBinding(ready, "ready", "event")])
    for relation_value, ready_value in product((False, True), repeat=2):
        event = PredicateEvent("event", {("relation", 1, 2, 3): relation_value, ("ready", 3): ready_value},
                               variable_domains={"x": (1,), "y": (2,), "z": (3,)})
        assert program.check(event) == (not relation_value or ready_value)


def test_mixed_boolean_count_formula_preserves_local_variable_scopes():
    with Graph("mixed_count_boolean_monitor") as graph:
        a, b, c = [Concept(name=name) for name in ("a", "b", "c")]
        iffL(existsL(andL(a("x"), b("x"))), notL(exactL(c("y"), limit=0)), name="mixed_rule")
    program = compile_predicate_constraints(graph, [PredicateBinding(p, name, "event") for p, name in ((a, "a"), (b, "b"), (c, "c"))])
    for a0, a1, b0, b1, c0 in product((False, True), repeat=5):
        values = {("a", 0): a0, ("a", 1): a1, ("b", 0): b0, ("b", 1): b1, ("c", 2): c0}
        event = PredicateEvent("event", values, variable_domains={"x": (0, 1), "y": (2,)})
        assert program.check(event) == (((a0 and b0) or (a1 and b1)) == c0)


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_invalid_count_limits_fail_with_constraint_name(limit):
    with pytest.raises(PredicateConstraintCompilationError, match="extended_rule.*limit"):
        compile_boolean(exactL, count=limit)


def test_missing_observation_after_a_false_grounding_is_still_reported():
    program = compile_boolean(andL)
    event = PredicateEvent("event", {("a", 0): False, ("b", 0): True, ("c", 0): True}, variable_domains={"x": (0, 1)})
    with pytest.raises(MissingSemanticBinding, match="extended_rule"):
        program.check(event)


def test_grounded_counts_execute_only_at_the_declared_phase():
    with Graph("grounded_deferred_count") as graph:
        a, b = Concept(name="a"), Concept(name="b")
        ifL(a("x"), existsL(b("y")), name="deferred_count")
    program = compile_predicate_constraints(graph, [PredicateBinding(a, "a", "event"),
        PredicateBinding(b, "b", "event", phase="seal")])
    assert program.check(PredicateEvent("event", {}))
    values = {("a", "a"): True, ("b", "b"): False}
    assert not program.check(PredicateEvent("event", values, phase="seal", variable_domains={"x": ("a",), "y": ("b",)}))


def test_legacy_positional_and_default_count_limits_are_supported():
    with Graph("count_limit_compatibility") as graph:
        p = Concept(name="p")
        with pytest.warns(DeprecationWarning):
            atMostL(p("x"), 0, name="positional_limit")
        atLeastL(p("x"), name="default_limit")
    program = compile_predicate_constraints(graph, [PredicateBinding(p, "p", "event")])
    assert not program.check(PredicateEvent("event", {"p": True}))
    assert not program.check(PredicateEvent("event", {"p": False}))
    assert program.coverage.compiled == ("positional_limit", "default_limit")


def relation_program():
    with Graph("relational_predicate_monitor") as graph:
        entity = Concept(name="entity")
        link = Concept(name="link")
        link.has_a(source=entity, target=entity)
        approved = entity(name="approved")
        ifL(link("source", "target"), approved("target"), name="approve_link_target")
    return compile_predicate_constraints(graph, [PredicateBinding(link, "link", "event"), PredicateBinding(approved, "approved", "event")])


def test_binary_relation_joins_reuse_the_target_variable():
    program = relation_program()
    assert program.rules[0].variables == ("source", "target")
    for edges in product((False, True), repeat=4):
        for approved in product((False, True), repeat=2):
            values = {("link", source, target): truth for (source, target), truth in zip(product((0, 1), repeat=2), edges)}
            values.update({("approved", i): approved[i] for i in (0, 1)})
            event = PredicateEvent("event", values, variable_domains={"source": (0, 1), "target": (0, 1)})
            assert program.check(event) == all(not edges[2*s+t] or approved[t] for s, t in product((0, 1), repeat=2))


@pytest.mark.parametrize("domains,values", [
    ({"source": (0,)}, {}),
    ({"source": (0,), "target": (1,)}, {("link", 0, 1): False}),
    ({"source": (0,), "target": (1,)}, {("link", 0, 1): True, ("approved", 1): 1}),
    ({"source": (0, 0), "target": ()}, {}),
    ({"source": ([0],), "target": ()}, {}),
    ({"source": iter((0,)), "target": ()}, {}),
])
def test_incomplete_or_invalid_groundings_fail_closed_with_rule_name(domains, values):
    with pytest.raises(MissingSemanticBinding, match="approve_link_target"):
        relation_program().check(PredicateEvent("event", values, variable_domains=domains))


def test_exact_grounding_budget_failure_does_not_become_a_rejection():
    program = compile_boolean(andL, max_groundings=2)
    with pytest.raises(PredicateGroundingLimitExceeded, match="extended_rule.*unknown"):
        program.check(PredicateEvent("event", {"a": True, "b": True, "c": True}))
    program = compile_boolean(existsL, max_groundings=4)
    with pytest.raises(PredicateGroundingLimitExceeded, match="extended_rule.*unknown"):
        program.check(PredicateEvent("event", {}, variable_domains={"x": range(1000000000)}))
    with pytest.raises(PredicateGroundingLimitExceeded, match="extended_rule.*unknown"):
        program.check(PredicateEvent("event", {}, variable_domains={"x": range(2**100)}))
    with pytest.raises(ValueError, match="max_groundings"):
        compile_boolean(andL, max_groundings=0)


def test_relational_monitor_masks_and_replay_agree_with_an_independent_plan_oracle():
    program = relation_program()
    symbols = ("approve_a", "approve_b", "link_a", "link_b")
    base = DFA(frozenset({0}), frozenset(symbols), {(0, symbol): 0 for symbol in symbols}, 0, frozenset({0}))

    class Adapter:
        start_state = frozenset()

        def step(self, state, symbol):
            target = symbol[-1]
            if symbol.startswith("approve"):
                return SemanticTransition(state | {target})
            event = PredicateEvent("event", {("link", "robot", target): True, ("approved", target): target in state},
                                   variable_domains={"source": ("robot",), "target": (target,)})
            return SemanticTransition(state, (event,))

        def is_accepting(self, state):
            return True

        def acceptance_events(self, state):
            return ()

    monitor = PredicateConstrainedDFA.from_dfa(base, program).bind(Adapter())
    for length in range(4):
        for sequence in product(symbols, repeat=length):
            approved, expected = set(), True
            for symbol in sequence:
                if symbol.startswith("approve"):
                    approved.add(symbol[-1])
                else:
                    expected = expected and symbol[-1] in approved
            assert monitor.accepts(sequence) == expected
    assert monitor.allowed_tokens(monitor.start_state, remaining_steps=1) == {"approve_a", "approve_b"}
    state = monitor.step(monitor.start_state, "approve_a")
    assert monitor.allowed_tokens(state, remaining_steps=0) == {"approve_a", "approve_b", "link_a"}
