"""Smoke test for the hard reinforcement example.

This exercises the full pipeline (BERT + spaCy embeddings + sampling reward
loss).  It is marked ``slow`` and skipped when the heavy optional dependencies
(spaCy model ``en_core_web_sm`` and a downloadable ``bert-base-uncased``) are not
available, so it never breaks a lightweight test run.
"""
from pathlib import Path
import importlib.util
import sys

import pytest

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))


def _deps_available():
    try:
        import spacy  # noqa: F401
        spacy.load("en_core_web_sm")
        print("spaCy and en_core_web_sm are available.")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.slow


@pytest.mark.skipif(not _deps_available(),
                    reason="requires spaCy en_core_web_sm and transformers")
def test_hard_example_runs_and_scores():
    main_path = RUN_DIR / "main.py"
    spec = importlib.util.spec_from_file_location("hard_example_main", main_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    class _Args:
        lr = 1e-4
        epochs = 1
        evaluate = False
        load_previous = False
        train_size = 2
        train_portion = "entities_with_relation"
        previous_portion = "entities_only_with_1_things_YN"
        checked_acc = 0
        counting_tnorm = "G"
        data_path = "data2.json"
        device = "cpu"
        num_samples = 2
        estimator = "importance_weighted"

    assert module.main(_Args()) == 0
