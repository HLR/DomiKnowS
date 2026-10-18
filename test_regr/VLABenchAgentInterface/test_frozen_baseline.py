"""Native baseline routing and scoring, independent of graph generation."""
import json
from types import SimpleNamespace

import pytest
import torch

from . import dataset, main, native_planning, training
from .dataset import PlanningExample


REFERENCE = [
    {"name": "pick", "params": {"target_entity_name": 0}},
    {"name": "push", "params": {"target_container_name": 1}},
]


def example(reference=REFERENCE):
    return PlanningExample("select_fruit/example0", "Put apple in bowl", tuple(reference),
                           (), (), None, entities=("apple", "bowl"))


class Native:
    def __init__(self, outputs):
        self.model = torch.nn.Linear(1, 1)
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, item, **kwargs):
        assert not self.model.training
        assert not torch.is_grad_enabled()
        self.calls.append((item, kwargs))
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value, []


@pytest.mark.parametrize("training_mode", [False, True])
def test_native_shared_evaluator_scores_without_validity_gate(training_mode):
    planner = Native([json.dumps(REFERENCE), "malformed", "[]"])
    planner.model.train(training_mode)
    result = training.evaluate_planner(planner, [example()] * 3,
                                      generation_mode="frozen-vlm", max_new_tokens=37)
    assert result["examples"] == 3
    metrics = result["planning_metrics"]
    assert metrics["overall"]["score"] == pytest.approx(100 / 3)
    assert metrics["overall"]["legacy_score"] == pytest.approx(80 / 3)
    assert metrics["dimensions"]["M&T"]["examples"] == 3
    # Optional push destinations now round-trip through the graph schema too.
    # Malformed and empty outputs remain scored rather than filtered out.
    assert result["valid"] == pytest.approx(1 / 3)
    assert result["native_valid"] == pytest.approx(1 / 3)
    assert planner.model.training == training_mode
    assert all(kwargs == {"mode": "raw", "sample": False, "max_tokens": 37}
               for _, kwargs in planner.calls)


@pytest.mark.parametrize("failure", [RuntimeError("GPU failure"), "reference"])
def test_native_errors_propagate_and_restore_mode(failure):
    planner = Native([failure if isinstance(failure, Exception) else "[]"])
    item = example() if isinstance(failure, Exception) else example([{"broken": True}])
    with pytest.raises((RuntimeError, ValueError)):
        training.evaluate_planner(planner, [item], generation_mode="frozen-vlm")
    assert planner.model.training


def test_native_empty_evaluation_has_no_fabricated_macro():
    planner = Native([])
    result = training.evaluate_planner(planner, [], generation_mode="frozen-vlm")
    assert result["examples"] == 0
    assert result["planning_metrics"]["Macro"] is None
    assert result["planning_metrics"]["overall"]["score"] is None
    assert planner.calls == []


def test_cli_native_uses_scene_loader_and_never_constructs_graph(monkeypatch, capsys):
    parser = main.build_parser()
    args = parser.parse_args(["evaluate-planner", "--planning-dir", "data", "--generation-mode",
                              "frozen-vlm", "--planner-model", "local-model", "--split", "all",
                              "--limit", "2", "--max-new-tokens", "31", "--no-load-in-4bit"])
    def forbidden(*args, **kwargs):
        pytest.fail("native baseline entered the graph generation path")
    for name in ("load_planning_examples", "build_constraint_runtime", "build_vlabench_world_graph", "_load_planner"):
        monkeypatch.setattr(main, name, forbidden)
    def load_scene(root, *, limit, observation_mode):
        assert root == "data" and limit == 2 and observation_mode == "visual"
        return [example(), example()]
    monkeypatch.setattr(dataset, "load_scene_planning_examples", load_scene)
    planner = Native([json.dumps(REFERENCE)] * 2)
    def load(path, *, load_in_4bit):
        assert path == "local-model" and not load_in_4bit
        return planner
    monkeypatch.setattr(native_planning.FrozenNativePlanner, "load", load)
    main.command_evaluate_planner(args)
    result = json.loads(capsys.readouterr().out)
    assert result["generation_mode"] == "frozen-vlm" and not result["dfa_constrained"]
    assert result["examples"] == 2 and result["planning_metrics"]["overall"]["score"] == 100
    assert planner.calls[0][1]["max_tokens"] == 31


@pytest.mark.parametrize("flags,message", [
    (["--generation-mode", "frozen-vlm", "--observation-mode", "metadata-assisted"], "visual observation mode"),
    ([], "requires --resume-adapter"),
    (["--generation-mode", "frozen-vlm", "--resume-adapter", "adapter"], "only the base VLM"),
    (["--generation-mode", "frozen-vlm", "--vocab", "vocab.json"], "only the base VLM"),
    (["--generation-mode", "frozen-vlm", "--max-new-tokens", "0"], "must be positive"),
])
def test_cli_rejects_misleading_baselines(flags, message):
    args = main.build_parser().parse_args(["evaluate-planner", "--planning-dir", "unused", *flags])
    with pytest.raises(ValueError, match=message):
        main.command_evaluate_planner(args)


def test_native_loader_freezes_base_model_without_adapter(monkeypatch):
    import sys
    from . import models
    model = torch.nn.Linear(1, 1)
    calls = []
    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return model
    processor = SimpleNamespace(tokenizer=object())
    class Processor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return processor
    monkeypatch.setattr(models, "resolve_vision_language_loader", lambda: (Loader, Processor))
    monkeypatch.setattr(models, "load_sanitized_auto_config", lambda *a, **kw: "config")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    monkeypatch.setattr(native_planning, "byte_token_map", lambda tokenizer: {})
    monkeypatch.setattr(native_planning, "tokenizer_checksum", lambda tokenizer: "checksum")
    planner = native_planning.FrozenNativePlanner.load("base", load_in_4bit=False)
    assert planner.model is model and planner.processor is processor
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert calls[0][0] == "base"
    assert "adapter_path" not in calls[0][1]
