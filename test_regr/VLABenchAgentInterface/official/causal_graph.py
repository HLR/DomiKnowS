"""Id-only causal knowledge for the official plan language.

Each operation is grounded against the prefix *before* it executes. Equality
and history membership are observations; these three formulas decide legality.
No scene classification, task identity, or reference plan is needed.
"""
from dataclasses import dataclass

from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import ifL, notL, orL
from domiknows.generation.dfa.predicate_monitor import (
    PredicateBinding,
    compile_predicate_constraints,
)


@dataclass(frozen=True)
class CausalGraph:
    graph: Graph
    skills: dict
    observations: dict

    def compile(self):
        """Snapshot active formulas; rebuild the DFA after editing declarations."""
        return compile_predicate_constraints(
            self.graph,
            [PredicateBinding(concept, "skill_" + name, "causal")
             for name, concept in self.skills.items()]
            + [PredicateBinding(concept, name, "causal")
               for name, concept in self.observations.items()],
        )


def build_causal_graph(name="vlabench_official_causal"):
    with Graph(name) as graph:
        operation = Concept(name="operation")
        skills = {name: operation(name=name) for name in ("pick", "place", "insert", "pour", "push")}
        observations = {
            name: operation(name=name)
            for name in ("target_is_subject", "target_used_in_subtask", "target_was_destination")
        }
        pick = skills["pick"]
        place = skills["place"]
        insert = skills["insert"]
        pour = skills["pour"]
        push = skills["push"]
        target_is_subject = observations["target_is_subject"]
        target_used_in_subtask = observations["target_used_in_subtask"]
        target_was_destination = observations["target_was_destination"]

        # A destination cannot be the object picked in this subtask.
        ifL(
            orL(place("operation"), insert("operation"), pour("operation"), push("operation")),
            notL(target_is_subject("operation")),
            name="anti_self_placement",
        )
        # A subtask may use a destination only once, even across different skills.
        ifL(
            orL(place("operation"), insert("operation"), pour("operation"), push("operation")),
            notL(target_used_in_subtask("operation")),
            name="distinct_subtask_destinations",
        )
        # An earlier destination remains in place for the rest of the plan.
        ifL(
            pick("operation"),
            notL(target_was_destination("operation")),
            name="destinations_stay_put",
        )
    return CausalGraph(graph, skills, observations)
