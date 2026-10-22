from types import SimpleNamespace

import torch

from test_regr.EmbodiedAgentInterface.modules import CausalLMActionObjectGenerator
from test_regr.EmbodiedAgentInterface.dataset import (
    VLABENCH_AUX_DATASET,
    VLABenchAuxiliaryPlanningExample,
    ensure_vlabench_auxiliary_planning_data,
    load_vlabench_auxiliary_planning_examples,
)
from test_regr.EmbodiedAgentInterface.vlabench_auxiliary import (
    VLABENCH_PROMPT_KEY,
    auxiliary_selection_key,
    prepare_vlabench_text_examples,
    train_vlabench_text_auxiliary,
)
from test_regr.VLABenchAgentInterface.graph import plan_to_tokens
from test_regr.VLABenchAgentInterface.training import build_constraint_runtime


def _press_example(index=0):
    return VLABenchAuxiliaryPlanningExample(
        episode_id=f"press/{index}",
        instruction="Press the red button.",
        operation_sequence=(
            {"name": "press", "params": {"target_entity_name": "red button"}},
        ),
        entities=("red button",),
    )


def test_eai_data_layer_downloads_only_planning_snapshot(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    resolved = ensure_vlabench_auxiliary_planning_data(tmp_path)
    assert resolved == tmp_path.resolve()
    assert calls == [
        {
            "repo_id": VLABENCH_AUX_DATASET,
            "repo_type": "dataset",
            "local_dir": tmp_path.resolve(),
            "token": None,
            "max_workers": 1,
        }
    ]
    assert (tmp_path / ".eai_vlabench_aux_complete").exists()

    ensure_vlabench_auxiliary_planning_data(tmp_path)
    assert len(calls) == 1


def test_eai_dataset_loads_vlabench_text_without_opening_images(tmp_path, monkeypatch):
    episode = tmp_path / "task" / "episode"
    (episode / "input").mkdir(parents=True)
    (episode / "output").mkdir()
    (episode / "env_config").mkdir()
    (episode / "input" / "instruction.txt").write_text(
        "Press the red button.", encoding="utf-8"
    )
    (episode / "input" / "camera.png").write_bytes(b"not-an-image")
    (episode / "output" / "operation_sequence.json").write_text(
        '{"skill_sequence":[{"name":"press","params":{"target_entity_name":"red button"}}]}',
        encoding="utf-8",
    )
    (episode / "env_config" / "episode.json").write_text(
        '{"entities":["red button"]}', encoding="utf-8"
    )
    monkeypatch.setattr(
        "PIL.Image.open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("EAI auxiliary loader opened an image")
        ),
    )

    examples = load_vlabench_auxiliary_planning_examples(tmp_path)
    assert len(examples) == 1
    assert examples[0].instruction == "Press the red button."
    assert examples[0].entities == ("red button",)


def test_text_preparation_uses_numbered_entities_and_graph_dfa(monkeypatch):
    runtime = build_constraint_runtime(
        max_entities=4, max_operations=2, name_prefix="test_eai_aux_prepare"
    )
    monkeypatch.setattr(
        "PIL.Image.open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("text-only preparation opened an image")
        ),
    )
    prepared = prepare_vlabench_text_examples([_press_example()], runtime)

    assert len(prepared) == 1
    prompt = prepared[0][VLABENCH_PROMPT_KEY]
    assert "Instruction: Press the red button." in prompt
    assert "0: red button" in prompt
    assert "image" not in prepared[0]
    labels = prepared[0]["target_plan_labels"].tolist()
    first_eos = labels.index(runtime.vocabulary.eos_label)
    assert runtime.dfa.accepts(labels[: first_eos + 1])


def test_causal_prompt_builder_can_supply_domain_prompt():
    head = object.__new__(CausalLMActionObjectGenerator)
    head.prompt_builder = lambda value: f"domain::{value}"
    assert head._prompt_user_content("plan") == "domain::plan"


def test_auxiliary_selection_is_semantic_then_loss():
    base = {
        "exact_graph_match": 0.5,
        "skill_with_entity_match": 0.7,
        "entity_match": 0.8,
        "skill_match": 0.9,
        "valid": 1.0,
        "loss": 2.0,
    }
    lower_loss = dict(base, loss=1.0)
    more_exact = dict(base, exact_graph_match=0.6, loss=100.0)
    assert auxiliary_selection_key(lower_loss) > auxiliary_selection_key(base)
    assert auxiliary_selection_key(more_exact) > auxiliary_selection_key(lower_loss)


class _FakeSharedBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor([[0.25]]))


class _FakeAuxiliaryHead(torch.nn.Module):
    def __init__(self, eai_head, runtime):
        super().__init__()
        self.model = eai_head.model
        self.output = torch.nn.Linear(1, runtime.vocabulary.label_count)
        self.label_count = runtime.vocabulary.label_count
        self.eos_label = runtime.vocabulary.eos_label
        self.reference_labels = [
            runtime.vocabulary.label_for_token(token)
            for token in plan_to_tokens(
                _press_example().operation_sequence,
                _press_example().entities,
                world=runtime.world_bundle,
            )
        ]

    def forward(self, _contains, _prompt, target):
        feature = self.model.lora_weight.expand(target.numel(), 1)
        return self.output(feature)

    def next_label_logits(self, input_ids, text=""):
        del text
        step = int(torch.as_tensor(input_ids).numel()) - 1
        label = self.reference_labels[min(step, len(self.reference_labels) - 1)]
        logits = torch.full((self.label_count,), -10.0)
        logits[label] = 10.0
        return logits

    def token_id_for_label(self, label):
        return int(label)


def test_auxiliary_updates_shared_and_temporary_head_only(tmp_path):
    runtime = build_constraint_runtime(
        max_entities=4, max_operations=2, name_prefix="test_eai_aux_train"
    )
    eai_head = SimpleNamespace(
        model=_FakeSharedBackbone(),
        tokenizer=None,
        output=torch.nn.Linear(1, 3),
    )
    shared_before = eai_head.model.lora_weight.detach().clone()
    eai_output_before = {
        name: value.detach().clone() for name, value in eai_head.output.state_dict().items()
    }
    checkpoint = tmp_path / "pilot.vlabench_aux.pth"

    result = train_vlabench_text_auxiliary(
        eai_head,
        {"train": [_press_example(0)], "validation": [_press_example(1)]},
        runtime,
        epochs=1,
        lr=0.05,
        checkpoint_path=checkpoint,
        auxiliary_head_factory=lambda head, active_runtime, **_kwargs: (
            _FakeAuxiliaryHead(head, active_runtime)
        ),
    )

    assert result.selected_epoch == 1
    assert result.validation_metrics["exact_graph_match"] == 1.0
    assert result.validation_metrics["valid"] == 1.0
    assert not torch.equal(shared_before, eai_head.model.lora_weight.detach())
    assert checkpoint.exists()
    assert all(
        torch.equal(value, eai_head.output.state_dict()[name])
        for name, value in eai_output_before.items()
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["selected_epoch"] == 1
    assert payload["vocabulary_checksum"] == runtime.vocabulary.checksum
    assert payload["domain_checksum"] == runtime.world_bundle.domain_checksum
