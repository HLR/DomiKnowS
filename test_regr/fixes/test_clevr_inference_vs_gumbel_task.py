from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

TASK_DIR = Path(__file__).resolve().parents[2] / "Tasks" / "clevr_inference_vs_gumbel"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from clevr_constraints import answer_to_query_label, prepare_logic_fields, translate_program_to_constraint
from graph import create_graph
import main as clevr_main


def test_query_program_translates_to_query_iota_constraint():
    data = json.load(open(TASK_DIR / "data" / "clevr_20_programs.json", encoding="utf-8"))
    program = data["items"][0]["program"]

    constraint = translate_program_to_constraint(program)

    assert "queryL(material" in constraint
    assert "iotaL(" in constraint
    assert "right(" in constraint
    assert "left(" in constraint


def test_answer_to_query_label_maps_all_attribute_groups():
    assert answer_to_query_label("gray", "color") == 0
    assert answer_to_query_label("metal", "material") == 1
    assert answer_to_query_label("sphere", "shape") == 1
    assert answer_to_query_label("large", "size") == 1


def test_compact_dataset_preserves_question_answer_order():
    copied = json.load(open(TASK_DIR / "data" / "20_examples_string_CLEVR.json", encoding="utf-8"))
    compact = json.load(open(TASK_DIR / "data" / "clevr_20_programs.json", encoding="utf-8"))

    compact_questions = [item["question"] for item in compact["items"]]
    copied_questions = [item["question"] for item in copied]

    assert len(compact_questions) == 40
    assert compact_questions == copied_questions


def test_expanded_compact_dataset_covers_templates_and_answer_types():
    compact = json.load(open(TASK_DIR / "data" / "clevr_20_programs.json", encoding="utf-8"))
    generated = [item for item in compact["items"] if item.get("generated_source")]

    assert len(compact["items"]) == 40
    assert len(generated) == 20
    assert {item["template_filename"] for item in generated} == {
        "zero_hop.json",
        "one_hop.json",
        "two_hop.json",
        "three_hop.json",
        "same_relate.json",
        "single_and.json",
        "single_or.json",
        "comparison.json",
        "compare_integer.json",
    }
    final_functions = {
        item["program"][-1]["function"]
        for item in generated
    }
    assert any(fn.startswith("query_") for fn in final_functions)
    assert "count" in final_functions
    assert any(fn in {"exist", "equal_integer", "less_than", "greater_than"} or fn.startswith("equal_") for fn in final_functions)


def test_expanded_compact_dataset_compiles_mixed_answer_types():
    items = clevr_main.load_items()
    results = create_graph(items, include_query_questions=True, relation_syntax="legacy")
    executions = results[0]
    query_types = results[9]

    prepare_logic_fields(items, device="cpu", executions=executions, query_types=query_types)

    assert len(items) == 40
    assert any(item["query_type"] is not None for item in items)
    assert any(item["query_type"] is None and isinstance(item["answer"], bool) for item in items)
    assert any(item["query_type"] is None and isinstance(item["answer"], int) for item in items)


def test_task_defaults_keep_global_constraints_downweighted(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    args = clevr_main.parse_args()

    assert args.train_items is None
    assert args.eval_items is None
    assert args.global_constraint_loss_weight == 0.1
    assert args.executable_constraint_loss_weight == 1.0


def test_post_training_example_uses_ephemeral_ad_hoc_query():
    calls = {}

    class FakeDataNode:
        def inferExecutableResults(self, **kwargs):
            calls.setdefault("inference", []).append(kwargs)
            return {
                "ADHOC0": {
                    "type": "count",
                    "answer": 2,
                    "probability": 0.75,
                    "distribution": None,
                    "mode": "tnorm",
                    "exact": False,
                }
            }

    class FakeBuilder:
        def getDataNode(self, *, device):
            calls["device"] = device
            return FakeDataNode()

    class FakeModel:
        def eval(self):
            calls["model_eval"] = True

        def __call__(self, sample):
            calls["sample"] = sample
            return None, None, None, FakeBuilder()

    class FakeConstraintModel:
        tnorm = "P"

        def eval(self):
            calls["constraint_model_eval"] = True

    sample = {
        "question": "How many red objects are there?",
        "answer": 2,
        "logic_str": 'sumL(red("x"))',
    }
    red = object()
    built = clevr_main.BuiltProgram(
        name="learned",
        program=SimpleNamespace(
            model=FakeModel(),
            cmodel=FakeConstraintModel(),
        ),
        train_dataset=[],
        eval_dataset=[sample],
        query_namespace={"red": red},
    )

    returned_sample, comparison = clevr_main.infer_ad_hoc_example(built, "cpu")

    assert returned_sample is sample
    assert list(comparison.results) == ["tnorm", "circuit", "ilp"]
    assert all(result["answer"] == 2 for result in comparison.results.values())
    assert comparison.answers_agree is True
    assert comparison.types_agree is True
    assert [call["mode"] for call in calls["inference"]] == [
        "tnorm",
        "circuit",
        "ilp",
    ]
    assert all(
        call["queries"] == 'sumL(red("x"))'
        for call in calls["inference"]
    )
    assert all(
        call["queryNamespace"] == {"red": red}
        for call in calls["inference"]
    )
    assert all(call["tnorm"] == "P" for call in calls["inference"])
    assert all(call["populate"] is False for call in calls["inference"])
    assert calls["device"] == "cpu"
    assert calls["model_eval"] is True
    assert calls["constraint_model_eval"] is True


@pytest.mark.gurobi
def test_post_training_ad_hoc_query_supports_circuit_and_ilp_modes():
    args = SimpleNamespace(
        epochs=1,
        train_items=1,
        eval_items=1,
        device="cpu",
        lr=1e-2,
        tnorm="P",
        seed=0,
        global_constraint_loss_weight=0.1,
        executable_constraint_loss_weight=1.0,
        disable_global_constraint_loss=True,
        gumbel_temp_start=1.0,
        gumbel_temp_end=0.3,
        hard_gumbel=False,
    )
    all_items = clevr_main.load_items()
    # Both examples are simple count questions, keeping this backend smoke
    # focused on ad hoc dispatch instead of relation-grounding complexity.
    built = clevr_main.build_program(
        "backend-smoke",
        clevr_main.InferenceProgram,
        [all_items[38], all_items[20]],
        args,
        "cpu",
    )

    _sample, comparison = clevr_main.infer_ad_hoc_example(built, "cpu")
    assert list(comparison.results) == ["tnorm", "circuit", "ilp"]
    assert comparison.types_agree is True

    tnorm = comparison.results["tnorm"]
    assert tnorm["type"] == "count"
    assert isinstance(tnorm["answer"], int)
    assert tnorm["mode"] == "tnorm"
    assert tnorm["exact"] is False
    assert isinstance(tnorm["probability"], float)
    assert tnorm["distribution"] is not None

    circuit = comparison.results["circuit"]
    assert circuit["type"] == "count"
    assert isinstance(circuit["answer"], int)
    assert circuit["mode"] == "circuit"
    assert circuit["exact"] is True
    assert isinstance(circuit["probability"], float)
    assert circuit["distribution"] is not None

    ilp = comparison.results["ilp"]
    assert ilp["type"] == "count"
    assert isinstance(ilp["answer"], int)
    assert ilp["mode"] == "ilp"
    assert ilp["exact"] is None
    assert ilp["probability"] is None
    assert ilp["distribution"] is None
