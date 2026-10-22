"""Task relevance is an explicit equivalence over observed lexical cues."""

from functools import lru_cache
from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import equivalenceL, orL
from domiknows.generation.dfa.predicate_monitor import (
    PredicateBinding,
    compile_predicate_constraints,
    _evaluate,
)

CLEAN_CUES = ("clean", "dust", "lint", "mop", "polish", "rinse", "scrub", "stain", "wash", "wipe")
TASK_FIELDS = (
    "task_name",
    "natural_language_description",
    "original_goal",
    "tl_goal",
    "transition_model",
)


def build_task_relevance_graph():
    with Graph("eai_task_relevance") as graph:
        task = Concept(name="task")
        cues = {cue: task(name="mentions_" + cue) for cue in CLEAN_CUES}
        clean_permitted = task(name="clean_permitted")
        equivalenceL(
            clean_permitted("task"),
            orL(*(cue("task") for cue in cues.values())),
            name="clean_requires_task_evidence",
        )
    return graph, cues, clean_permitted


@lru_cache(maxsize=1)
def _task_program():
    graph, cues, permitted = build_task_relevance_graph()
    bindings = [PredicateBinding(concept, cue, "task") for cue, concept in cues.items()]
    bindings.append(PredicateBinding(permitted, "permitted", "task"))
    return compile_predicate_constraints(graph, bindings)


def task_semantic_action_permissions(row, *, program=None):
    """Ground substring observations from task fields, excluding demonstrations."""
    text = " ".join(str(row.get(key, "")) for key in TASK_FIELDS).lower()
    values = {cue: cue in text for cue in CLEAN_CUES}
    values["permitted"] = True
    allowed = all(_evaluate(rule.expression, values) for rule in (program or _task_program()).rules)
    return ("clean",) if allowed else ()
