"""Detached result values and transparent per-call reuse statistics."""
from dataclasses import dataclass
import copy
import json


@dataclass
class IncrementalInferenceResult:
    status: str
    mode: str
    answers: dict
    assignments: list
    changes: dict
    stats: dict
    objective: float | None = None
    error: str | None = None

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), allow_nan=False, **kwargs)


def new_stats():
    return dict(constraints_reused=0, constraints_rebuilt=0, formula_plan_hits=0,
                formula_plan_misses=0, adjacency_reused=0, adjacency_rebuilt=0,
                ilp_models_reused=0, ilp_models_patched=0, ilp_models_created=0,
                variables_added=0, variables_removed=0, circuit_structures_reused=0,
                circuit_structures_rebuilt=0, full_rebuild_reasons=[],
                detection_ms=0., grounding_ms=0., compilation_ms=0.,
                evaluation_ms=0., population_ms=0., total_ms=0.)
