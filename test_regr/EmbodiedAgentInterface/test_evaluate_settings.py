import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from dataset import EOS_TOKEN, dummy_dataset
from domiknows.generation.dfa.vocabulary import TokenVocabulary

_spec = importlib.util.spec_from_file_location("eai_evaluate_settings", HERE / "evaluate_settings.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


class _Dfa:
    def __init__(self, accepted):
        self.accepted = accepted
        self.seen = []

    def accepts(self, labels):
        self.seen.append(list(labels))
        return self.accepted


def _gold_scores(dfa, policy_for=None):
    examples = dummy_dataset(max_steps=8)
    vocabulary = TokenVocabulary(examples[0]["generation_vocab"], eos_token=EOS_TOKEN)
    predictions = [[int(x) for x in sample["target_action_labels"]] for sample in examples]
    return examples, ev.score_predictions("t", predictions, examples, vocabulary, dfa=dfa, policy_for=policy_for)


def test_dfa_valid_uses_the_per_sample_policy_and_the_padded_sequence():
    global_dfa, sample_dfa = _Dfa(True), _Dfa(False)
    examples, score = _gold_scores(global_dfa, policy_for=lambda sample: sample_dfa)
    assert score["dfa_valid"] == 0.0
    assert not global_dfa.seen
    assert len(sample_dfa.seen) == len(examples)
    assert all(len(labels) == 8 for labels in sample_dfa.seen)


def test_dfa_valid_falls_back_to_the_global_dfa_without_a_policy():
    _, score = _gold_scores(_Dfa(True))
    assert score["dfa_valid"] == 1.0
    _, score = _gold_scores(None, policy_for=lambda sample: _Dfa(False))
    assert score["dfa_valid"] == 1.0 and score["dfa_checked"] is False
