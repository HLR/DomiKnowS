"""Score one prediction against two references without changing inference inputs."""
import hashlib
import json
from dataclasses import fields, is_dataclass

from test_regr.VLABenchAgentInterface.planning_metrics import (
    PlanningMetrics, score_planning_metrics, strict_plan_match,
)
from test_regr.VLABenchAgentInterface.repair_density_labels import updated_density_examples


def input_fingerprint(example):
    values = ({f.name: getattr(example, f.name) for f in fields(example)}
              if is_dataclass(example) else dict(vars(example)))
    values.pop("operation_sequence", None)
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


class PairedReferenceAudit:
    def __init__(self, output, *, correct_density=True, corrector=updated_density_examples):
        self.output = output
        self.corrector = corrector if correct_density else lambda examples: examples
        self.metrics = {key: PlanningMetrics() for key in ("original", "corrected")}
        self.matches = {key: 0 for key in self.metrics}
        self.episode_ids = []
        self.changed = []

    def observe(self, example, prediction, *, generation_error=None):
        fingerprint = input_fingerprint(example)
        corrected = self.corrector([example])[0]
        if input_fingerprint(corrected) != fingerprint or input_fingerprint(example) != fingerprint:
            raise ValueError("reference correction changed inference inputs")
        entities = example.entities
        record = {"episode_id": example.episode_id, "task": example.task,
                  "dimension": example.dimension, "entity_table": entities,
                  "prediction": prediction, "input_fingerprint": fingerprint,
                  "generation_error": generation_error}
        for name, reference in (("original", example), ("corrected", corrected)):
            scores = score_planning_metrics(prediction, reference.operation_sequence,
                                            reference.dependency, entity_table=entities)
            strict = (0.0 if generation_error else strict_plan_match(
                prediction, reference.operation_sequence, entity_table=entities))
            self.metrics[name].update(scores, task=example.task, dimension=example.dimension)
            self.matches[name] += strict
            record[name] = {"reference": reference.operation_sequence,
                            "scores": scores, "strict_plan_match": strict}
        record["reference_changed"] = not bool(strict_plan_match(
            example.operation_sequence, corrected.operation_sequence, entity_table=entities))
        if record["reference_changed"]:
            self.changed.append(example.episode_id)
        self.episode_ids.append(example.episode_id)
        self.output.write(json.dumps(record, default=str) + "\n")
        self.output.flush()

    def report(self):
        count = len(self.episode_ids)
        if len(set(self.episode_ids)) != count:
            raise ValueError("paired evaluation contains duplicate episode IDs")
        return {"same_predictions_for_both_references": True,
                "inference_inputs_unchanged": True, "examples": count,
                "episode_ids": self.episode_ids, "changed_reference_ids": self.changed,
                **{key: {"planning_metrics": value.result(),
                         "strict_plan_match": self.matches[key] / max(1, count)}
                   for key, value in self.metrics.items()}}
