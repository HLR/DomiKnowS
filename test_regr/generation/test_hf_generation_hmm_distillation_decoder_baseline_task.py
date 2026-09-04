from __future__ import annotations

import importlib

import torch

from domiknows.generation import HybridController


def import_task_module(name: str):
    return importlib.import_module(f"Tasks.hf_generation.{name}")


def test_hmm_distillation_decoder_baseline_compares_decoder_accuracy(monkeypatch):
    demo = import_task_module("hmm_distillation_decoder_baseline")
    decode_calls = []
    original_decode = HybridController.decode_hmm_dfa

    def spy_decode_hmm_dfa(self, prompt, *args, **kwargs):
        decode_calls.append((prompt, dict(kwargs)))
        return original_decode(self, prompt, *args, **kwargs)

    monkeypatch.setattr(HybridController, "decode_hmm_dfa", spy_decode_hmm_dfa)

    summary = demo.run_hmm_distillation_decoder_baseline(
        prompts=("Once", "Story"),
        steps=2,
        max_new_tokens=4,
    )

    assert len(summary["losses"]) == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in summary["losses"])
    accuracy = summary["accuracy"]
    assert accuracy["raw_lm_greedy"]["accepted_accuracy"] == 0.0
    assert accuracy["dfa_greedy"]["accepted_accuracy"] == 1.0
    assert accuracy["dfa_beam"]["accepted_accuracy"] == 1.0
    assert accuracy["dfa_sample"]["accepted_accuracy"] == 1.0
    assert accuracy["product_compact_learner_dfa"]["accepted_accuracy"] == 1.0
    assert accuracy["product_hmm_dfa"]["accepted_accuracy"] == 1.0
    assert accuracy["raw_lm_greedy"]["dog_avoidance"] == 0.0
    assert accuracy["product_hmm_dfa"]["dog_avoidance"] == 1.0
    assert len(decode_calls) == 2
    assert all(call_kwargs["search"] == "beam" for _prompt, call_kwargs in decode_calls)
    assert all(call_kwargs["max_new_tokens"] == 4 for _prompt, call_kwargs in decode_calls)


def test_hmm_distillation_decoder_baseline_cli_runs(capsys):
    demo = import_task_module("hmm_distillation_decoder_baseline")

    assert demo.main(["--steps", "1", "--prompts", "Once", "--max-new-tokens", "4"]) == 0
    captured = capsys.readouterr().out

    assert "HMM distillation decoder baseline" in captured
    assert "Decoder accuracy" in captured
    assert "raw_lm_greedy" in captured
    assert "product_hmm_dfa" in captured
