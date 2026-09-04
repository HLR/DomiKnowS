from __future__ import annotations

import importlib


def import_task_module(name: str):
    return importlib.import_module(f"Tasks.hf_generation.{name}")


def test_hf_generation_hybrid_demo_reranks_mock_candidates():
    hybrid_demo = import_task_module("hybrid_demo")

    summary = hybrid_demo.run_hybrid_demo(steps=1, num_candidates=2)

    assert summary["ranked"]
    assert summary["ranked"][0].score.accepted
    assert summary["repair"]["suggestions"]
    assert summary["rejected_score"].accepted is False


def test_hf_generation_hybrid_cli_runs(capsys):
    hybrid_demo = import_task_module("hybrid_demo")

    assert hybrid_demo.main(["--steps", "1", "--num-candidates", "2"]) == 0
    captured = capsys.readouterr().out

    assert "Path: hybrid controller/scorer" in captured
    assert "Ranked candidates" in captured
    assert "Rejected candidate diagnostic" in captured

