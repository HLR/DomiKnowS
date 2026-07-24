import argparse
from pathlib import Path

from .dataset import DEFAULT_TEMPORAL_DATA_ROOT, discover_temporal_datasets, load_temporal_instances
from .execution import create_candidate_event_pairs, validate_dataset_convertible
from .oracle import answer_label, check_oracle, consistency_violations


def oracle_failures_for_all_pairs(instance):
    failures = []
    for pair_index, pair in enumerate(instance.get("event_pairs", [])):
        query_instance = {
            **instance,
            "query_pair": {"e1": pair["e1"], "e2": pair["e2"]},
        }
        expected = pair.get("label")
        actual = answer_label(query_instance)
        if actual != expected or not check_oracle(query_instance, expected):
            failures.append((pair_index, pair, actual))
    return failures


def consistency_failures_for_each_sample(instances):
    failures = []
    for instance_index, instance in enumerate(instances):
        violations = consistency_violations([instance])
        if violations:
            failures.append((instance_index, instance.get("doc_id"), violations))
    return failures


def check_file(path, limit=None, check_consistency=False, count_candidates=False):
    instances = load_temporal_instances(path, limit=limit, group_by_document=True)
    convert_failures = validate_dataset_convertible(instances)
    oracle_failures = []
    total_pairs = 0
    total_events = 0
    total_candidates = 0
    for instance_index, instance in enumerate(instances):
        event_count = len(instance.get("events", []))
        total_events += event_count
        total_pairs += len(instance.get("event_pairs", []))
        if count_candidates:
            total_candidates += len(create_candidate_event_pairs(instance))
        else:
            total_candidates += event_count * max(event_count - 1, 0)
        failures = oracle_failures_for_all_pairs(instance)
        if failures:
            oracle_failures.append((instance_index, failures))
    consistency = consistency_failures_for_each_sample(instances) if check_consistency else None
    return {
        "path": str(path),
        "documents": len(instances),
        "events": total_events,
        "labeled_pairs": total_pairs,
        "candidate_pairs": total_candidates,
        "convert_failures": convert_failures,
        "oracle_failures": oracle_failures,
        "consistency_violations": consistency,
    }


def candidate_files(root):
    discovered = discover_temporal_datasets(root)
    files = []
    for key in ("matres", "tbdense"):
        for path in discovered[key]:
            path = Path(path)
            if path.is_file() and path.suffix.lower() in {".jsonl", ".tsv", ".tab", ".txt"}:
                files.append(path)
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="Smoke-test TemporalRelation dataset conversion and oracle labels.")
    parser.add_argument("paths", nargs="*", type=Path, help="Dataset files to check. If omitted, discover under --root.")
    parser.add_argument("--root", type=Path, default=DEFAULT_TEMPORAL_DATA_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--check-consistency", action="store_true", help="Run expensive global consistency checks.")
    parser.add_argument("--materialize-candidates", action="store_true", help="Actually build all candidate pairs instead of counting them.")
    args = parser.parse_args()

    files = args.paths or candidate_files(args.root)
    if not files:
        print(f"No MATRES/TB-Dense .jsonl/.tsv/.tab/.txt files found under {args.root}")
        return 2

    exit_code = 0
    for path in files:
        result = check_file(
            path,
            limit=args.limit,
            check_consistency=args.check_consistency,
            count_candidates=args.materialize_candidates,
        )
        print(f"file={result['path']}")
        print(f"  documents={result['documents']}")
        print(f"  events={result['events']}")
        print(f"  labeled_pairs={result['labeled_pairs']}")
        print(f"  candidate_pairs={result['candidate_pairs']}")
        print(f"  convert_failures={len(result['convert_failures'])}")
        print(f"  oracle_failures={len(result['oracle_failures'])}")
        if result["consistency_violations"] is not None:
            print(f"  consistency_failure_samples={len(result['consistency_violations'])}")
        else:
            print("  consistency_failure_samples=skipped")
        if result["convert_failures"] or result["oracle_failures"] or result["consistency_violations"]:
            exit_code = 1
            print(f"  first_convert_failures={result['convert_failures'][:3]}")
            print(f"  first_oracle_failures={result['oracle_failures'][:3]}")
            if result["consistency_violations"]:
                print(f"  first_consistency_failures={result['consistency_violations'][:3]}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
