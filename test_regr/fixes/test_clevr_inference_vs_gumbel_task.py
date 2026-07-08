from __future__ import annotations

import json
from pathlib import Path
import sys

TASK_DIR = Path(__file__).resolve().parents[2] / "Tasks" / "clevr_inference_vs_gumbel"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from clevr_constraints import answer_to_query_label, translate_program_to_constraint
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


def test_compact_dataset_preserves_copied_twenty_question_order():
    copied = json.load(open(TASK_DIR / "data" / "20_examples_string_CLEVR.json", encoding="utf-8"))
    compact = json.load(open(TASK_DIR / "data" / "clevr_20_programs.json", encoding="utf-8"))

    compact_questions = [item["question"] for item in compact["items"]]
    copied_questions = [item["question"] for item in copied]

    assert len(compact_questions) == 20
    assert compact_questions == copied_questions


def test_task_defaults_keep_global_constraints_downweighted(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    args = clevr_main.parse_args()

    assert args.global_constraint_loss_weight == 0.1
    assert args.executable_constraint_loss_weight == 1.0
