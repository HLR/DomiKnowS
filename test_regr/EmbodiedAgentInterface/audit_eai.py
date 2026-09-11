"""Independent audit for the EAI symbolic benchmark.

This audit is intentionally separate from model training. It checks every loaded
reference example, rejects empty plans, reports coverage by simulator domain and
first action, and runs the focused adversarial semantic checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from dataset import EOS_TOKEN, load_eai_dataset
from reward import TokenVocabulary, evaluate_goal_satisfaction
from test_reward import (
    test_dense_temporal_reward_credits_ordered_prefix_progress,
    test_empty_plan_cannot_satisfy_nonempty_goal,
    test_goal_is_read_from_tl_not_demonstration_side_effects,
    test_objectless_actions_do_not_shift_following_actions,
    test_reward_closure_tensor_contract,
    test_then_requires_action_order,
    test_virtualhome_putback_means_ontop,
    test_wrong_relation_source_is_rejected,
)


FOCUSED_CHECKS = (
    test_goal_is_read_from_tl_not_demonstration_side_effects,
    test_wrong_relation_source_is_rejected,
    test_empty_plan_cannot_satisfy_nonempty_goal,
    test_objectless_actions_do_not_shift_following_actions,
    test_virtualhome_putback_means_ontop,
    test_then_requires_action_order,
    test_dense_temporal_reward_credits_ordered_prefix_progress,
    test_reward_closure_tensor_contract,
)


def domain_for(example: dict) -> str:
    transition_model = str(example.get("transition_model", ""))
    return "VirtualHome" if "(:domain virtualhome" in transition_model.lower() else "BEHAVIOR"


def audit_examples(examples: list[dict]) -> dict:
    vocabulary = TokenVocabulary(examples[0]["generation_vocab"], eos_token=EOS_TOKEN)
    by_domain = defaultdict(lambda: Counter())
    by_action = defaultdict(lambda: Counter())
    failures = []
    total_facts = 0

    for example in examples:
        domain = domain_for(example)
        action = str(example.get("first_action") or "other")
        result = evaluate_goal_satisfaction(
            example["target_action_tokens"], example, vocabulary
        )
        empty = evaluate_goal_satisfaction(
            [vocabulary.eos_label], example, vocabulary
        )
        bucket = by_domain[domain]
        bucket["examples"] += 1
        bucket["gold_success"] += int(result["is_success"] == 1.0)
        bucket["empty_rejected"] += int(empty["is_success"] == 0.0)
        bucket["parse_errors"] += int(result["parse_error"] is not None)
        total_facts += len(result["gold_state"])
        action_bucket = by_action[(domain, action)]
        action_bucket["examples"] += 1
        action_bucket["gold_success"] += int(result["is_success"] == 1.0)
        if result["parse_error"] is not None or result["is_success"] != 1.0 or empty["is_success"] != 0.0:
            failures.append(
                {
                    "task_id": example.get("task_id", "task"),
                    "domain": domain,
                    "first_action": action,
                    "gold_success": result["is_success"],
                    "empty_success": empty["is_success"],
                    "parse_error": result["parse_error"],
                }
            )

    return {
        "examples": len(examples),
        "domains": {name: dict(values) for name, values in sorted(by_domain.items())},
        "first_action": {
            f"{domain}:{action}": dict(values)
            for (domain, action), values in sorted(by_action.items())
        },
        "average_grounded_goal_facts": total_facts / max(1, len(examples)),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the independent EAI semantic audit.")
    parser.add_argument("--dataset", choices=("all", "behavior", "virtualhome"), default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=135)
    parser.add_argument("--data-path", default=None, help="Optional local parquet/csv/jsonl dataset path.")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    print("=== EAI semantic audit ===")
    print("Focused adversarial checks:")
    for check in FOCUSED_CHECKS:
        check()
    print(f"  passed {len(FOCUSED_CHECKS)}/{len(FOCUSED_CHECKS)}")

    examples = load_eai_dataset(
        args.dataset, limit=args.limit, data_path=args.data_path, max_steps=args.max_steps, device="cpu"
    )
    report = audit_examples(examples)
    print(f"Reference examples: {report['examples']}")
    print("| Domain | Examples | Gold success | Empty rejected | Parse errors |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for domain, metrics in report["domains"].items():
        print(
            f"| {domain} | {metrics['examples']} | "
            f"{metrics['gold_success']}/{metrics['examples']} | "
            f"{metrics['empty_rejected']}/{metrics['examples']} | "
            f"{metrics['parse_errors']} |"
        )
    print(f"Average grounded goal facts: {report['average_grounded_goal_facts']:.2f}")
    print("First-action coverage:")
    for key, metrics in report["first_action"].items():
        print(
            f"  {key}: {metrics['gold_success']}/{metrics['examples']} "
            f"gold-success"
        )
    if report["failures"]:
        print(f"FAILURES: {len(report['failures'])}")
        for failure in report["failures"][:20]:
            print("  " + json.dumps(failure, sort_keys=True))
        if args.json_path:
            Path(args.json_path).write_text(json.dumps(report, indent=2) + "\n")
        return 1
    print("Audit passed: all reference plans satisfy their goals and empty plans are rejected.")
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
