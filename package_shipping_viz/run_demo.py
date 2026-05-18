"""Generate the package-shipping graph-HMM/DFA visualization files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import webbrowser

try:
    from .flow import DEFAULT_CANDIDATE_SOURCE, SHIPPING_TASKS, build_flow
except ImportError:  # pragma: no cover - direct script execution
    from flow import DEFAULT_CANDIDATE_SOURCE, SHIPPING_TASKS, build_flow


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = TASK_DIR / "demo_output"


def terminal_file_link(path: Path, label: str = "open index.html") -> str:
    uri = path.resolve().as_uri()
    return f"\033]8;;{uri}\033\\{label}\033]8;;\033\\ ({uri})"


def write_demo(
    task: str,
    candidate_source: str,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 0,
    hmm_iterations: int = 20,
) -> dict:
    flow = build_flow(
        task=task,
        candidate_source=candidate_source,
        seed=seed,
        hmm_iterations=hmm_iterations,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_json = json.dumps(flow, indent=2, allow_nan=False)
    (output_dir / "flow.json").write_text(flow_json + "\n", encoding="utf-8")
    viewer = (TASK_DIR / "viewer.html").read_text(encoding="utf-8")
    embedded = viewer.replace("__FLOW_JSON__", flow_json.replace("</", "<\\/"), 1)
    (output_dir / "index.html").write_text(embedded, encoding="utf-8")
    return flow


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=SHIPPING_TASKS, default="ship_fragile_vase")
    parser.add_argument("--candidate-source", default=DEFAULT_CANDIDATE_SOURCE)
    parser.add_argument("--hmm-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)

    flow = write_demo(
        args.task,
        args.candidate_source,
        output_dir=args.output_dir,
        seed=args.seed,
        hmm_iterations=args.hmm_iterations,
    )
    html_path = args.output_dir / "index.html"
    json_path = args.output_dir / "flow.json"

    print("Package shipping planning visualization")
    print("Task:", flow["task"]["selected"])
    print("Candidate source:", flow["candidate"]["source"])
    print("Plan:", " -> ".join(flow["candidate"]["actions"]))
    print("DFA accepted:", flow["dfa"]["accepted"])
    if flow["dfa"]["rejection_reason"]:
        print("Rejection:", flow["dfa"]["rejection_reason"])
    print("Graph-HMM log-likelihood:", flow["graph_hmm"]["log_likelihood"])
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote HTML: {html_path}")
    print(f"Open HTML: {terminal_file_link(html_path)}")

    if args.open:
        webbrowser.open(os.fspath(html_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
