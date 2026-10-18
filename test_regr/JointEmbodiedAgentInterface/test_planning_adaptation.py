from copy import deepcopy
from dataclasses import asdict
import random
from types import SimpleNamespace
import pytest
import torch
from contextlib import nullcontext
from .checkpoint import validate_planning_schema_migration
from .finetune_planning import balanced_examples
from domiknows.generation.validation import retains_validation_scores
from test_regr.VLABenchAgentInterface.graph import PlanVocabulary
from test_regr.VLABenchAgentInterface.world_graph import LEGACY_VLABENCH_DOMAIN_CHECKSUM, VLABENCH_DOMAIN_CHECKSUM


def test_migration_preserves_all_label_meanings_and_rejects_other_drift():
    current_vocab = asdict(PlanVocabulary.default())
    old_vocab = deepcopy(current_vocab)
    old_vocab.pop("optional_skill_arguments")
    old_vocab["domain_checksum"] = LEGACY_VLABENCH_DOMAIN_CHECKSUM
    old = {"vlabench_domain_checksum": LEGACY_VLABENCH_DOMAIN_CHECKSUM, "vlabench_vocabulary": old_vocab, "vlabench_dfa": {"max_operations": 8}}
    new = {"vlabench_domain_checksum": VLABENCH_DOMAIN_CHECKSUM, "vlabench_vocabulary": current_vocab, "vlabench_dfa": {"max_operations": 12}}
    validate_planning_schema_migration(old, new)
    for key, value in (("max_entities", 32), ("skills", ("different",)), ("eos_token", "other")):
        changed = deepcopy(old)
        changed["vlabench_vocabulary"][key] = value
        with pytest.raises(ValueError, match="label meanings"):
            validate_planning_schema_migration(changed, new)
    changed = deepcopy(old)
    changed["vlabench_domain_checksum"] = "unaudited"
    with pytest.raises(ValueError, match="audited"):
        validate_planning_schema_migration(changed, new)


def test_balancing_uses_only_supplied_training_examples_and_is_seeded():
    examples = [SimpleNamespace(task=task, dimension=dimension, episode_id=f"train/{task}/{i}")
                for task, dimension, count in (("density_qa", "PhysLaw", 1), ("book_rearrange", "Complex", 2), ("other", "Semantic", 20))
                for i in range(count)]
    first = list(balanced_examples(examples, 1000, random.Random(42)))
    second = list(balanced_examples(examples, 1000, random.Random(42)))
    assert [x.episode_id for x in first] == [x.episode_id for x in second]
    assert all(any(x is supplied for supplied in examples) for x in first)
    assert 650 < sum(x.dimension in ("PhysLaw", "Complex") for x in first) < 850


def test_generic_retention_rejects_unknown_nonfinite_and_dimension_regression():
    baseline = {"overall": 75.0, "physics": 65.0, "unavailable": None}
    assert retains_validation_scores({"overall": 76.0, "physics": 65.0}, baseline)
    assert not retains_validation_scores({"overall": 90.0, "physics": 64.0}, baseline)
    assert not retains_validation_scores({"overall": 90.0}, baseline)
    assert not retains_validation_scores({"overall": 90.0, "physics": float("nan")}, baseline)
    assert not retains_validation_scores({}, {"unavailable": None})


@pytest.mark.parametrize("candidate_physics,should_promote", [(70.0, False), (85.0, True)])
def test_training_promotes_only_retained_generated_validation_scores(tmp_path, monkeypatch, candidate_physics, should_promote):
    from . import finetune_planning as module
    class Planner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.label_heads = torch.nn.ModuleDict({"eai": torch.nn.Linear(2, 2), "vlabench": torch.nn.Linear(2, 2)})
        @property
        def device(self):
            return torch.device("cpu")
    planner = Planner()
    controller = torch.nn.Linear(2, 2)
    runtime = SimpleNamespace(eai_vocabulary=object(), vlabench_vocabulary=object(), domain_scope=lambda _: nullcontext())
    train = SimpleNamespace(task="density_qa", dimension="PhysLaw", episode_id="train/density/0")
    valid = SimpleNamespace(task="density_qa", dimension="PhysLaw", episode_id="validation/density/0")
    monkeypatch.setattr(module, "_prepare_eval", lambda *args: (runtime, {"train": [train], "validation": [valid], "test": []}))
    monkeypatch.setattr(module, "JointQwenVLPlanner", SimpleNamespace(from_pretrained=lambda **kwargs: planner))
    monkeypatch.setattr(module, "_controller", lambda *args: controller)
    monkeypatch.setattr(module, "load_joint_checkpoint", lambda *args, **kwargs: {"stage": "stage2"})
    monkeypatch.setattr(module, "loss_for", lambda *args: planner.label_heads["vlabench"].weight.sum() * 0 + 1)
    def measured(overall, physics):
        return {"generation_failures": 0, "planning_metrics": {"overall": {"score": overall}, "Macro": overall,
                "dimensions": {"PhysLaw": {"examples": 1, "score": physics}}}}
    measurements = iter((measured(80.0, 80.0), measured(84.0, candidate_physics)))
    monkeypatch.setattr(module, "validation_plans", lambda *args: next(measurements))
    saved = []
    def save(path, **kwargs):
        path.write_text("baseline" if kwargs["epoch"] == -1 else "candidate")
        saved.append(kwargs)
    monkeypatch.setattr(module, "save_joint_checkpoint", save)
    output = tmp_path / "adaptation"
    module.main(["--checkpoint", str(tmp_path / "champion.pt"), "--output", str(output), "--device", "cpu", "--epochs", "1", "--samples-per-epoch", "1"])
    assert saved[-1]["metrics"]["retention_eligible"] is should_promote
    assert (output / "planner_best.pt").read_text() == ("candidate" if should_promote else "baseline")
