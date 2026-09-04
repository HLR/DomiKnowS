"""Cross-platform launcher for TemporalRelation example commands."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys

if __package__:
    from .config import TEMPORAL_CONFIG
else:
    from config import TEMPORAL_CONFIG


COMMAND_MODULES = {
    "smoke": "test_regr.TemporalRelation.smoke_test_dataset",
    "predicate-train": "test_regr.TemporalRelation.train_predicate_classifier",
    "program-train": "test_regr.TemporalRelation.program_qwen_train",
    "inference": "test_regr.TemporalRelation.run_llm_inference",
}


def configure_python_path(path):
    """Prepend the configured source root for imports and child processes."""
    path = str(Path(path).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

    current = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if path not in current:
        os.environ["PYTHONPATH"] = os.pathsep.join([path, *current])
    return path


def parse_command(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Run a TemporalRelation entry point after applying python_path from "
            f"{TEMPORAL_CONFIG.source}."
        )
    )
    parser.add_argument("command", choices=COMMAND_MODULES)
    return parser.parse_args(argv[:1]).command


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = parse_command(argv)
    module_name = COMMAND_MODULES[command]
    configure_python_path(TEMPORAL_CONFIG.python_path)
    sys.argv = [module_name, *argv[1:]]
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
