from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clevr_constraints import detect_query_type, translate_program_to_constraint


RUN_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = RUN_DIR / "data" / "clevr_20_programs.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print CLEVR English questions with their DomiKnowS executable constraints."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--start", type=int, default=0, help="First dataset item index to print.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of items to print.")
    parser.add_argument(
        "--generated-only",
        action="store_true",
        help="Print only examples appended from clevr-dataset-gen.",
    )
    return parser.parse_args()


def load_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "items" in payload:
        return payload["items"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported dataset format in {path}")


def iter_selected(
    items: list[dict[str, Any]],
    *,
    start: int,
    limit: int | None,
    generated_only: bool,
) -> list[dict[str, Any]]:
    selected = [
        item for item in items
        if int(item.get("index", 0)) >= start and (not generated_only or item.get("generated_source"))
    ]
    if limit is not None:
        selected = selected[:limit]
    return selected


def main() -> None:
    args = parse_args()
    items = load_items(args.dataset)
    selected = iter_selected(
        items,
        start=args.start,
        limit=args.limit,
        generated_only=args.generated_only,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Showing {len(selected)} of {len(items)} examples")
    print()

    for item in selected:
        program = item.get("program", [])
        constraint = translate_program_to_constraint(program)
        final_fn = program[-1].get("function") if program else "<none>"
        query_type = detect_query_type(program)
        template = item.get("template_filename", "original")

        print("=" * 100)
        print(f"Index: {item.get('index')}")
        print(f"Template: {template}")
        print(f"Final function: {final_fn}")
        if query_type is not None:
            print(f"Query type: {query_type}")
        print(f"Answer: {item.get('answer')}")
        print(f"English: {item.get('question')}")
        print("Constraint:")
        print(constraint)
        print()


if __name__ == "__main__":
    main()
