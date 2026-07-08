from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUN_DIR = Path(__file__).resolve().parent
DATA_DIR = RUN_DIR / "data"
DEFAULT_GENERATOR_OUTPUT = RUN_DIR.parents[2] / "clevr-dataset-gen" / "output" / "domiknows_balanced"
GENERATED_MARKER = "clevr-dataset-gen:domiknows_balanced"
TEMPLATE_ORDER = [
    "zero_hop.json",
    "one_hop.json",
    "two_hop.json",
    "three_hop.json",
    "same_relate.json",
    "single_and.json",
    "single_or.json",
    "comparison.json",
    "compare_integer.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append a balanced set of generated CLEVR examples to the compact task dataset."
    )
    parser.add_argument("--generator-output", type=Path, default=DEFAULT_GENERATOR_OUTPUT)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--compact-json", type=Path, default=DATA_DIR / "clevr_20_programs.json")
    parser.add_argument("--string-json", type=Path, default=DATA_DIR / "20_examples_string_CLEVR.json")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def normalize_program(program: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for node in program:
        normalized.append(
            {
                "inputs": list(node.get("inputs", [])),
                "function": node.get("function", node.get("type")),
                "value_inputs": list(node.get("value_inputs", [])),
            }
        )
    return normalized


def answer_kind(question: dict[str, Any]) -> str:
    final_fn = question["program"][-1].get("function", question["program"][-1].get("type"))
    if final_fn == "count":
        return "count"
    if final_fn and final_fn.startswith("query_"):
        return "query"
    return "bool"


def sort_key(question: dict[str, Any]) -> tuple[int, int, int, int]:
    template = question["template_filename"]
    template_idx = TEMPLATE_ORDER.index(template) if template in TEMPLATE_ORDER else len(TEMPLATE_ORDER)
    return (
        template_idx,
        int(question.get("question_family_index", 0)),
        int(question.get("image_index", 0)),
        int(question.get("question_index", 0)),
    )


def select_questions(questions: list[dict[str, Any]], existing_text: set[str], count: int) -> list[dict[str, Any]]:
    candidates = [
        q for q in sorted(questions, key=sort_key)
        if q["template_filename"] in TEMPLATE_ORDER and q["question"] not in existing_text
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    def add(question: dict[str, Any]) -> bool:
        marker = int(question["question_index"])
        if marker in selected_ids or len(selected) >= count:
            return False
        selected.append(question)
        selected_ids.add(marker)
        return True

    for template in TEMPLATE_ORDER:
        template_candidates = [q for q in candidates if q["template_filename"] == template]
        if not template_candidates:
            raise RuntimeError(f"No generated questions found for template {template}")
        preferred_order = ("query", "bool", "count") if template not in {"zero_hop.json", "two_hop.json", "three_hop.json"} else ("count", "query", "bool")
        for kind in preferred_order:
            match = next((q for q in template_candidates if answer_kind(q) == kind), None)
            if match is not None:
                add(match)
                break

    def balanced_match(kind: str | None = None) -> dict[str, Any] | None:
        template_counts = {
            template: sum(1 for q in selected if q["template_filename"] == template)
            for template in TEMPLATE_ORDER
        }
        kind_counts = {
            current_kind: sum(1 for q in selected if answer_kind(q) == current_kind)
            for current_kind in ("query", "bool", "count")
        }
        available = [
            q for q in candidates
            if int(q["question_index"]) not in selected_ids and (kind is None or answer_kind(q) == kind)
        ]
        if not available:
            return None
        return min(
            available,
            key=lambda q: (
                template_counts[q["template_filename"]],
                kind_counts[answer_kind(q)],
                sort_key(q),
            ),
        )

    minimums = {"query": 6, "bool": 6, "count": 4}
    for kind, minimum in minimums.items():
        while sum(1 for q in selected if answer_kind(q) == kind) < minimum:
            match = balanced_match(kind)
            if match is None:
                break
            add(match)

    while len(selected) < count:
        match = balanced_match()
        if match is None:
            break
        add(match)

    if len(selected) != count:
        raise RuntimeError(f"Selected {len(selected)} questions, expected {count}")
    return selected


def build_item(question: dict[str, Any], scene_by_image: dict[int, dict[str, Any]], index: int) -> dict[str, Any]:
    image_index = int(question["image_index"])
    scene = scene_by_image[image_index]
    return {
        "index": index,
        "question": question["question"],
        "answer": question["answer"],
        "matched_answer": question["answer"],
        "question_index": int(question["question_index"]),
        "image_index": image_index,
        "image_filename": question["image_filename"],
        "template_filename": question["template_filename"],
        "question_family_index": int(question["question_family_index"]),
        "generated_source": GENERATED_MARKER,
        "program": normalize_program(question["program"]),
        "scene": scene,
    }


def main() -> None:
    args = parse_args()
    questions_path = args.generator_output / "CLEVR_questions.json"
    scenes_path = args.generator_output / "CLEVR_scenes.json"

    compact = load_json(args.compact_json)
    question_data = load_json(questions_path)
    scene_data = load_json(scenes_path)

    base_items = [
        item for item in compact["items"]
        if item.get("generated_source") != GENERATED_MARKER
    ]
    existing_text = {item["question"] for item in base_items}
    scene_by_image = {int(scene["image_index"]): scene for scene in scene_data["scenes"]}
    selected = select_questions(question_data["questions"], existing_text, args.count)

    appended = [
        build_item(question, scene_by_image, index)
        for index, question in enumerate(selected, start=len(base_items))
    ]
    compact["source"] = (
        "Exact matches from test_regr/Clever/train plus generated CLEVR examples "
        "from clevr-dataset-gen output/domiknows_balanced"
    )
    compact["items"] = base_items + appended
    dump_json(args.compact_json, compact)

    question_answer_rows = [
        {"question": item["question"], "answer": item["answer"]}
        for item in compact["items"]
    ]
    dump_json(args.string_json, question_answer_rows)

    by_template = {template: 0 for template in TEMPLATE_ORDER}
    by_kind = {"query": 0, "bool": 0, "count": 0}
    for question in selected:
        by_template[question["template_filename"]] += 1
        by_kind[answer_kind(question)] += 1

    print(f"Appended {len(appended)} generated examples; total items={len(compact['items'])}")
    print("Selected templates:", by_template)
    print("Selected answer kinds:", by_kind)


if __name__ == "__main__":
    main()
