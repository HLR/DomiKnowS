import argparse
import json
from collections import Counter, defaultdict


def load_rows(path):
    rows = []
    summary = None
    with open(path, "r") as data_file:
        for line in data_file:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("type") == "summary":
                summary = item
            elif item.get("type") == "prediction":
                rows.append(item)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Analyze direct LLM baseline prediction errors.")
    parser.add_argument("path")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows, summary = load_rows(args.path)
    total = len(rows)
    wrong = [row for row in rows if not row.get("ok")]
    errors = [row for row in rows if "error" in row]
    print("summary", json.dumps(summary, sort_keys=True) if summary else None)
    print(f"predictions={total} wrong={len(wrong)} parse_errors={len(errors)} wrong_rate={len(wrong)/total if total else 0:.4f}")

    gold_counts = Counter(str(row.get("gold")) for row in rows)
    pred_counts = Counter(str(row.get("pred")) for row in rows if "pred" in row)
    wrong_gold_counts = Counter(str(row.get("gold")) for row in wrong)
    wrong_pred_counts = Counter(str(row.get("pred")) for row in wrong if "pred" in row)
    confusions = Counter((str(row.get("gold")), str(row.get("pred"))) for row in wrong if "pred" in row)

    print("gold_counts", dict(gold_counts.most_common()))
    print("pred_counts", dict(pred_counts.most_common()))
    print("wrong_by_gold", dict(wrong_gold_counts.most_common()))
    print("wrong_by_pred", dict(wrong_pred_counts.most_common()))
    print("top_confusions")
    for (gold, pred), count in confusions.most_common(args.top):
        print(f"  gold={gold} pred={pred} count={count}")

    if errors:
        error_types = Counter(row.get("error", "")[:120] for row in errors)
        print("parse_error_types")
        for error, count in error_types.most_common(args.top):
            print(f"  count={count} error={error}")

    print("first_wrong")
    for row in wrong[: args.top]:
        shown = {key: row.get(key) for key in ["index", "doc_id", "qid", "gold", "pred", "error"] if key in row}
        print(json.dumps(shown, sort_keys=True))


if __name__ == "__main__":
    main()
