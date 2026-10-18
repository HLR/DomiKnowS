"""Read-only reference representability/constraint audit; never supplies decoder facts."""
import argparse
import json
from collections import Counter
from pathlib import Path
from .dataset import load_planning_examples, deterministic_split
from .graph import plan_to_tokens, tokens_to_plan
from .training import build_constraint_runtime
from .world_graph import validate_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-operations", type=int, default=12)
    args = parser.parse_args(argv)
    examples = load_planning_examples(args.planning_dir)
    runtime = build_constraint_runtime(max_entities=64, max_operations=args.max_operations)
    rows = []
    for split, values in deterministic_split(examples, seed=42).items():
        for index, example in enumerate(values):
            row = {"episode_id": example.episode_id, "split": split, "task": example.task, "errors": []}
            try:
                if len(example.operation_sequence) > args.max_operations:
                    row["errors"].append("operation_budget")
                tokens = plan_to_tokens(example.operation_sequence, example.entities, world=runtime.world_bundle)
                if tokens_to_plan(tokens) != list(example.operation_sequence):
                    row["errors"].append("lossy_roundtrip")
                validation = validate_plan(example.operation_sequence, entity_table=example.entities)
                row["errors"].extend(validation.errors)
                policy = runtime.dfa_for("vlabench", example)
                labels = [runtime.vocabulary.label_for_token(t) for t in tokens]
                if not policy.accepts(labels):
                    row["errors"].append("reference_violates_bound_constraints")
            except (ValueError, KeyError) as error:
                row["errors"].append(str(error))
            rows.append(row)
            if index % 100 == 0:
                print(json.dumps({"split": split, "processed": index + 1, "total": len(values)}), flush=True)
    result = {"examples": len(rows), "issues": sum(bool(r["errors"]) for r in rows),
              "issues_by_task": dict(Counter(r["task"] for r in rows if r["errors"])), "episodes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
