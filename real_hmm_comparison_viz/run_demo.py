"""Generate the real HMM comparison visualization files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import webbrowser

try:
    from .flow import CANDIDATES, DEMOS, TWO_CONSTRAINT_CANDIDATES, build_flow
except ImportError:  # pragma: no cover
    from flow import CANDIDATES, DEMOS, TWO_CONSTRAINT_CANDIDATES, build_flow


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = TASK_DIR / "demo_output"


def terminal_file_link(path: Path, label: str = "open index.html") -> str:
    uri = path.resolve().as_uri()
    return f"\033]8;;{uri}\033\\{label}\033]8;;\033\\ ({uri})"


def write_demo(candidate: str, output_dir: Path = DEFAULT_OUTPUT_DIR, *, demo: str = "one") -> dict:
    flow = build_flow(candidate, demo=demo)
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_json = json.dumps(flow, indent=2, allow_nan=False)
    (output_dir / "flow.json").write_text(flow_json + "\n", encoding="utf-8")
    viewer = (TASK_DIR / "viewer.html").read_text(encoding="utf-8")
    embedded = viewer.replace("__FLOW_JSON__", flow_json.replace("</", "<\\/"), 1)
    (output_dir / "index.html").write_text(embedded, encoding="utf-8")
    return flow


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=DEMOS, default="one")
    parser.add_argument(
        "--candidate",
        default="invalid",
        help=(
            "Candidate name. One-constraint: "
            f"{', '.join(CANDIDATES)}. Two-constraint: {', '.join(TWO_CONSTRAINT_CANDIDATES)}."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)

    flow = write_demo(args.candidate, args.output_dir, demo=args.demo)
    html_path = args.output_dir / "index.html"
    json_path = args.output_dir / "flow.json"

    print("Real HMM vs DomiKnowS-aware HMM visualization demo")
    print(f"Demo: {args.demo}")
    print(f"Candidate: {args.candidate}")
    print("Sequence:", " ".join(flow["candidate"]["sequence"]))
    print("Constraint:", flow["constraint"]["text"])
    print("DFA accepted:", flow["dfa"]["accepted"])
    if flow["dfa"]["rejection_reason"]:
        print("Rejection:", flow["dfa"]["rejection_reason"])
    print("DiscreteHMM log-likelihood:", flow["discrete_hmm"]["log_likelihood"])
    print("DomiKnowSAwareHMM log-likelihood:", flow["domiknows_hmm"]["log_likelihood"])
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote HTML: {html_path}")
    print(f"Open HTML: {terminal_file_link(html_path)}")

    if args.open:
        webbrowser.open(os.fspath(html_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
