from __future__ import annotations

import json
from pathlib import Path
import sys

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
