"""The compiled evaluators, the rule index and the verdict memo decide exactly as the tuple interpreter does."""
import random

import pytest

from domiknows.generation.dfa import predicate_monitor as pm
from domiknows.generation.dfa.semantic import MissingSemanticBinding

NAMES = [f"p{i}" for i in range(6)]


def random_expression(rng, depth=0):
    if depth >= 3 or rng.random() < 0.3:
        return ("predicate", rng.choice(NAMES), ())
    operator = rng.choice(["notL", "ifL", "andL", "orL", "nandL", "norL", "xorL", "iffL", "forAllL", "count"])
    if operator == "notL":
        return (operator, random_expression(rng, depth + 1))
    if operator in ("ifL", "forAllL"):
        return (operator, random_expression(rng, depth + 1), random_expression(rng, depth + 1))
    if operator == "count":
        children = tuple(((), random_expression(rng, depth + 1)) for _ in range(rng.randint(1, 3)))
        return ("count", rng.choice(["atMostL", "atLeastL", "exactL", "existsL"]), rng.randint(0, 2), children)
    return (operator, *(random_expression(rng, depth + 1) for _ in range(rng.randint(1, 3))))


def outcome(call):
    try:
        return ("value", call())
    except (MissingSemanticBinding, pm.PredicateGroundingLimitExceeded) as error:
        return (type(error).__name__, str(error))


def random_values(rng):
    values = {}
    for name in NAMES:
        draw = rng.random()
        if draw < 0.08:
            continue  # missing observation
        values[name] = 1 if draw < 0.14 else (rng.random() < 0.5)  # an int is not a Boolean observation
    return values


@pytest.mark.parametrize("seed", range(5))
def test_compiled_closure_matches_the_interpreter(seed):
    rng = random.Random(seed)
    for _ in range(300):
        expression, values = random_expression(rng), random_values(rng)
        budget = rng.choice([None, 1, 3, 1000])
        reference = outcome(lambda: pm._evaluate(expression, values, None, None,
                                                 _budget=None if budget is None else [budget]))
        compiled = outcome(lambda: pm._compile(expression)(values, None, None, None if budget is None else [budget]))
        assert compiled == reference, (expression, values, budget)


def make_program(rng, count=8):
    rules = tuple(
        pm.CompiledPredicateConstraint(f"r{i}", rng.choice(["a", "b"]), expression, frozenset(NAMES[:]),
                                       rng.choice(["immediate", "late"]), (), ())
        for i, expression in enumerate(random_expression(rng) for _ in range(count)))
    return pm.PredicateConstraintProgram(rules)


def reference_check(program, event):
    for rule in program.rules:
        if rule.domain != event.domain or rule.phase != event.phase:
            continue
        if not pm._evaluate(rule.expression, event.values, None, None, _budget=[program.max_groundings]):
            return False
    return True


@pytest.mark.parametrize("seed", range(5))
def test_indexed_memoized_check_matches_the_reference_cold_and_warm(seed):
    rng = random.Random(100 + seed)
    program = make_program(rng)
    events = [pm.PredicateEvent(rng.choice(["a", "b", "c"]), random_values(rng), rng.choice(["immediate", "late"]))
              for _ in range(400)]
    for _round in range(2):  # second round reads the memo
        for event in events:
            expected = outcome(lambda: reference_check(program, event))
            actual = outcome(lambda: program.check(event))
            if expected[0] != "value":
                assert actual[0] == expected[0]  # same error class; messages carry the rule name here
            else:
                assert actual == expected, event


def test_memo_is_bounded_and_never_stores_unobserved_values(monkeypatch):
    monkeypatch.setattr(pm, "_MEMO_LIMIT", 4)
    rule = pm.CompiledPredicateConstraint("r", "d", ("predicate", "p0", ()), frozenset({"p0"}))
    program = pm.PredicateConstraintProgram((rule,))
    runtime = program._plan[("d", "immediate")].runtimes[0]
    with pytest.raises(MissingSemanticBinding):
        program.check(pm.PredicateEvent("d", {}))
    assert runtime.memo == {}
    assert program.check(pm.PredicateEvent("d", {"p0": True})) and not program.check(pm.PredicateEvent("d", {"p0": False}))
    assert len(runtime.memo) == 2
