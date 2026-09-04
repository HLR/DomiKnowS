import importlib
import sys
from pathlib import Path

import torch

from domiknows.generation.learners import GraphHMMGenerationHead, GraphSpectralGenerationHead


COLLIE_DIR = Path(__file__).resolve().parents[2] / "Tasks" / "collie"


class FakeTokenizer:
    eos_token = "<|endoftext|>"
    eos_token_id = 0

    def __init__(self):
        self.token_to_id = {
            "<|endoftext|>": 0,
            " The": 1,
            " slide": 2,
            " At": 3,
            " the": 4,
            ".": 5,
        }
        self.id_to_token = {value: key for key, value in self.token_to_id.items()}

    def encode(self, token):
        return [self.token_to_id.get(token, 99)]

    def decode(self, token_id):
        return self.id_to_token.get(int(token_id), "?")

    def __call__(self, text, return_tensors=None):
        class Output:
            pass

        output = Output()
        output.input_ids = torch.tensor([[3, 4]], dtype=torch.long)
        return output


class DummyBackbone(torch.nn.Module):
    def forward(self, *_args, **_kwargs):
        raise AssertionError("graph_hmm learner path should not call the HF backbone")


class ScriptedBackbone(torch.nn.Module):
    def __init__(self, generated_token_ids=(1, 2, 0, 0), vocab_size=128, prompt_len=2):
        super().__init__()
        self.generated_token_ids = tuple(generated_token_ids)
        self.vocab_size = int(vocab_size)
        self.prompt_len = int(prompt_len)

    def forward(self, input_ids, *_args, **_kwargs):
        generated_index = max(0, int(input_ids.shape[1]) - self.prompt_len)
        token_id = self.generated_token_ids[min(generated_index, len(self.generated_token_ids) - 1)]
        logits = torch.full((1, int(input_ids.shape[1]), self.vocab_size), -20.0)
        logits[0, -1, int(token_id)] = 20.0

        class Output:
            pass

        output = Output()
        output.logits = logits
        return output


def import_collie_program():
    sys.path.insert(0, str(COLLIE_DIR))
    try:
        return importlib.import_module("program"), importlib.import_module("tokens")
    finally:
        try:
            sys.path.remove(str(COLLIE_DIR))
        except ValueError:
            pass


def build_collie_program(kind, *, graph_hmm_source="target", backbone=None):
    program, tokens = import_collie_program()
    tokenizer = FakeTokenizer()
    label_map = tokens.TokenMap({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 99: 6})
    return program.build_program(
        label_map,
        backbone if backbone is not None else DummyBackbone(),
        tokenizer,
        ["<|endoftext|>", " The", " slide"],
        pad_size=4,
        graph_hmm_learner=kind,
        graph_hmm_source=graph_hmm_source,
    )


def sample_data():
    return {
        "target_tokens": torch.tensor([[1, 2, 0]], dtype=torch.long),
        "instruction_tokens": torch.tensor([[3, 4]], dtype=torch.long),
        "_testing_generated_tokens": torch.tensor([[1, 2, 0]], dtype=torch.long),
    }


def assert_pmd_step_is_finite(collie_program):
    output = collie_program.model(sample_data())
    model_loss = output[0]
    constraint_loss, *_ = collie_program.cmodel(output[3])
    assert torch.isfinite(model_loss)
    assert torch.isfinite(constraint_loss)


def test_collie_can_attach_graph_hmm_generation_head():
    collie_program = build_collie_program("hmm")

    assert collie_program.graph_hmm_learner == "hmm"
    assert any(isinstance(module, GraphHMMGenerationHead) for module in collie_program.model.modules())
    assert_pmd_step_is_finite(collie_program)


def test_collie_graph_hmm_can_learn_from_hf_generated_labels():
    collie_program = build_collie_program("hmm", graph_hmm_source="generated", backbone=ScriptedBackbone())

    output = collie_program.model(sample_data())
    generated_loss = collie_program.graph_hmm_model.last_imitation_loss

    assert collie_program.graph_hmm_source == "generated"
    assert any(isinstance(module, GraphHMMGenerationHead) for module in collie_program.model.modules())
    assert collie_program.graph_hmm_model.last_teacher_labels.tolist() == [1, 2, 0, 0]
    assert torch.isfinite(output[0])
    assert torch.isfinite(generated_loss)


def test_collie_graph_spectral_can_learn_from_hf_generated_labels():
    collie_program = build_collie_program("spectral", graph_hmm_source="generated", backbone=ScriptedBackbone())

    output = collie_program.model(sample_data())
    generated_loss = collie_program.graph_hmm_model.last_imitation_loss

    assert collie_program.graph_hmm_source == "generated"
    assert any(isinstance(module, GraphSpectralGenerationHead) for module in collie_program.model.modules())
    assert collie_program.graph_hmm_model.last_teacher_labels.tolist() == [1, 2, 0, 0]
    assert torch.isfinite(output[0])
    assert torch.isfinite(generated_loss)


def test_collie_can_attach_graph_spectral_generation_head():
    collie_program = build_collie_program("spectral")

    assert collie_program.graph_hmm_learner == "spectral"
    assert any(isinstance(module, GraphSpectralGenerationHead) for module in collie_program.model.modules())
    assert_pmd_step_is_finite(collie_program)


def test_collie_default_hf_path_remains_default():
    collie_program = build_collie_program("none")

    assert collie_program.graph_hmm_learner == "none"
    assert not any(isinstance(module, (GraphHMMGenerationHead, GraphSpectralGenerationHead)) for module in collie_program.model.modules())
