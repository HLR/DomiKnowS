"""SimpleTL Boolean, counting and temporal witness knowledge in DomiKnowS.

The parser supplies expression structure; atomic observations and numeric time
comparisons supply evidence. These formulas determine satisfaction. Each
operator application has one complete constraint, regardless of its arity.
"""

from dataclasses import dataclass
from functools import lru_cache
from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import andL, orL, notL, exactL, equivalenceL
from domiknows.generation.dfa.predicate_monitor import (
    PredicateBinding,
    compile_predicate_constraints,
    _evaluate,
)


@dataclass(frozen=True)
class GoalOperator:
    graph: object
    operands: tuple
    result: object
    constant: object


def build_goal_operator(kind, arity, count=0):
    """Declare one satisfaction equation with descriptive concept names."""
    with Graph(f"eai_goal_{kind}_{arity}_{count}") as graph:
        evaluation = Concept(name="goal_evaluation")
        operands = tuple(evaluation(name=f"operand_{index}_holds") for index in range(arity))
        result = evaluation(name="goal_satisfied")
        constant = evaluation(name="true_observation")
        calls = [operand("evaluation") for operand in operands]
        if kind in {"and", "forall"}:
            expression = andL(*calls) if calls else constant("evaluation")
        elif kind in {"or", "exists"}:
            expression = orL(*calls) if calls else notL(constant("evaluation"))
        elif kind == "not":
            expression = notL(calls[0])
        elif kind == "forn":
            expression = (
                exactL(*calls, int(count))
                if calls
                else (constant("evaluation") if count == 0 else notL(constant("evaluation")))
            )
        elif kind == "temporal_witness":
            # operand 0: the term holds at this snapshot.
            # operand 1: the snapshot precedes the permitted starting snapshot.
            expression = andL(calls[0], notL(calls[1]))
        else:
            raise ValueError(f"Unsupported goal operator: {kind}")
        equivalenceL(result("evaluation"), expression, name=f"goal_{kind}_satisfaction")
    return GoalOperator(graph, operands, result, constant)


def compile_goal_operator(bundle):
    bindings = [
        PredicateBinding(bundle.result, "result", "goal"),
        PredicateBinding(bundle.constant, "constant", "goal"),
    ]
    bindings.extend(
        PredicateBinding(concept, f"operand:{index}", "goal")
        for index, concept in enumerate(bundle.operands)
    )
    return compile_predicate_constraints(bundle.graph, bindings)


@lru_cache(maxsize=256)
def _operator_program(kind, arity, count):
    return compile_goal_operator(build_goal_operator(kind, arity, count))


def evaluate_goal_operator(kind, values, count=0, *, program=None):
    """Project a declared satisfaction equation onto its Boolean result."""
    values = tuple(bool(value) for value in values)
    program = program or _operator_program(kind, len(values), count)
    observations = {
        "result": True,
        "constant": True,
        **{f"operand:{index}": value for index, value in enumerate(values)},
    }
    return all(_evaluate(rule.expression, observations) for rule in program.rules)
