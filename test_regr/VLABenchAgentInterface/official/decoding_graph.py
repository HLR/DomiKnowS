"""Episode knowledge for constrained plan decoding, declared as logical constraints.

The frozen model's own answers (the press question, the largest tag number it read,
the subtask patterns it named) and the instruction's object candidates are
observations of one episode. These formulas decide which operations they allow;
``KnowledgePlanDFA`` grounds the observations and compiles the formulas.
"""
from dataclasses import dataclass

from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import andL, ifL
from domiknows.generation.dfa.predicate_monitor import (
    PredicateBinding,
    compile_predicate_constraints,
)

OPERATION_OBSERVATIONS = (
    "press",                             # the candidate operation is a press
    "first_operation",                   # it is the plan's first operation
    "press_answer_yes",                  # the model answered Yes to the press question
    "has_target",                        # it names an object id
    "scene_bound_read",                  # the model read the largest tag number in the labeled image
    "target_within_scene_bound",         # its id is at most that number
    "patterns_chosen",                   # the model named subtask patterns for this episode
    "continues_chosen_pattern",          # it extends a named pattern that still fits the operation budget
    "instruction_candidates_known",      # the instruction grounds candidates for this skill and role
    "target_is_instruction_candidate",   # its id is one of them
    "visibility_supplied",              # explicit support set from the initial observation
    "target_seen_initially",            # this id belongs to that set (not merely to an id range)
)
PLAN_OBSERVATIONS = (
    "plan_patterns_chosen",              # the model named subtask patterns for this episode
    "chosen_patterns_complete",          # the plan ends on a complete named pattern
)


@dataclass(frozen=True)
class DecodingGraph:
    graph: Graph
    operation: dict
    plan: dict

    def compile(self):
        """Snapshot active formulas; rebuild the DFA after editing declarations."""
        return compile_predicate_constraints(
            self.graph,
            [PredicateBinding(concept, name, "operation") for name, concept in self.operation.items()]
            + [PredicateBinding(concept, name, "plan") for name, concept in self.plan.items()],
        )


def build_decoding_graph(name="vlabench_decoding_knowledge"):
    with Graph(name) as graph:
        operation = Concept(name="decoded_operation")
        plan = Concept(name="decoded_plan")
        o = {name: operation(name=name) for name in OPERATION_OBSERVATIONS}
        p = {name: plan(name=name) for name in PLAN_OBSERVATIONS}

        # A Yes to the press question makes the plan one press.
        ifL(o["press_answer_yes"]("operation"),
            andL(o["press"]("operation"), o["first_operation"]("operation")),
            name="press_answer_single_press")
        # Object ids stay within the tags the model read in the labeled image.
        ifL(andL(o["scene_bound_read"]("operation"), o["has_target"]("operation")),
            o["target_within_scene_bound"]("operation"),
            name="scene_ids_bound_targets")
        # The plan is made of the subtask patterns the model named.
        ifL(o["patterns_chosen"]("operation"), o["continues_chosen_pattern"]("operation"),
            name="chosen_patterns_continue")
        ifL(p["plan_patterns_chosen"]("plan"), p["chosen_patterns_complete"]("plan"),
            name="chosen_patterns_complete")
        # An object the instruction grounds for a skill and role must be one of its candidates.
        ifL(o["instruction_candidates_known"]("operation"), o["target_is_instruction_candidate"]("operation"),
            name="instruction_candidates_bound_targets")
        ifL(andL(o["visibility_supplied"]("operation"), o["has_target"]("operation")),
            o["target_seen_initially"]("operation"), name="observation_supported_references")
    return DecodingGraph(graph, o, p)
