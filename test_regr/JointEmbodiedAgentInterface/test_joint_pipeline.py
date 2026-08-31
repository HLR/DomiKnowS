from __future__ import annotations

import copy
import random
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from domiknows.generation.dfa.vocabulary import TokenVocabulary
from domiknows.graph import Concept, Graph
from domiknows.program import SolverPOIProgram
from domiknows.reinforcement.reinforcement_program import ReinforcementProgram
from test_regr.EmbodiedAgentInterface.dataset import dummy_dataset
from test_regr.EmbodiedAgentInterface.reward import make_eai_reward_function
from test_regr.VLABenchAgentInterface.models import MultiViewController

from .checkpoint import (
    _checkpoint_staging_location,
    _cpu_rng_state,
    _dfa_configuration,
    _normalize_dfa_configuration,
    load_joint_checkpoint,
    save_joint_checkpoint,
)
from .main import build_parser, stage1_selection_key, stage2_selection_key
from .models import JointQwenVLPlanner
from .program import JointReinforcementProgram, JointSolverPOIProgram, _TrainingProgress
from .world_graph import build_joint_runtime, build_joint_world_graph


class FakeProcessor:
    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, *_args, **_kwargs):
        messages = _args[0]
        return messages[0]["content"][-1]["text"]

    def __call__(self, **kwargs):
        self.prompts.extend(kwargs["text"])
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 8)
        self.forward_calls = 0
        self.config = SimpleNamespace(
            hidden_size=8,
            model_type="fake-qwen-vl",
            _name_or_path="fake-qwen-vl",
        )

    def forward(self, input_ids, **_kwargs):
        self.forward_calls += 1
        return SimpleNamespace(hidden_states=(self.embedding(input_ids),))


class TinyController(torch.nn.Module):
    def __init__(self, horizon=2):
        super().__init__()
        self.horizon = horizon
        self.action = torch.nn.Parameter(torch.zeros(7))
        self.value_head = torch.nn.Linear(1, 1)

    def forward(self, images, state, task_index):
        return self.action.expand(images.shape[0], self.horizon, 7)


class TinyImageEncoder(torch.nn.Module):
    output_dim = 8

    def forward(self, images):
        return torch.zeros(images.shape[0], self.output_dim, device=images.device)


def test_training_progress_is_newline_based_and_flushed(capsys):
    progress = _TrainingProgress("test rounds", 2, interval_seconds=0)
    progress.update(1, loss=1.0)
    progress.update(2, loss=0.5)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[joint-training] test rounds: 0/2 started\n" in captured.err
    assert "test rounds: 1/2" in captured.err
    assert "test rounds: 2/2" in captured.err
    assert "loss=0.5000" in captured.err
    assert "\r" not in captured.err


@pytest.fixture(scope="module")
def joint_fixture():
    examples = dummy_dataset(max_steps=4)
    vocabulary = TokenVocabulary(examples[0]["generation_vocab"], eos_token="<eos>")
    world = build_joint_world_graph("joint_acceptance_world")
    runtime = build_joint_runtime(
        world,
        vocabulary,
        max_eai_steps=4,
        eai_object_tokens=examples[0]["object_tokens"],
        eai_action_tokens=examples[0]["action_tokens"],
        eai_action_sequence_tokens=examples[0]["action_tokens"],
        eai_openable_object_tokens=examples[0]["openable_object_tokens"],
        max_vlabench_entities=2,
        max_vlabench_operations=1,
    )
    return examples, runtime


def make_planner(runtime):
    return JointQwenVLPlanner(
        FakeBackbone(),
        FakeProcessor(),
        eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary,
    )


def test_joint_sequence_loss_encodes_backbone_once(joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    planner.train()
    non_eos = next(
        label for label in range(runtime.eai_vocabulary.label_count)
        if label != runtime.eai_vocabulary.eos_label
    )
    labels = torch.tensor(
        [non_eos] * 31 + [runtime.eai_vocabulary.eos_label],
        dtype=torch.long,
    )

    loss = planner.supervised_loss(
        "eai",
        context={"instruction": "test", "goal": "test"},
        target_labels=labels,
    )
    calls_before_backward = planner.model.forward_calls
    loss.backward()

    assert calls_before_backward == 1
    assert planner.model.forward_calls == 1
    assert planner.model.embedding.weight.grad is not None
    assert planner.context_projections["eai"].weight.grad is not None
    assert planner.token_embeddings["eai"].weight.grad is not None
    assert planner.graph_decoders["eai"].weight_hh_l0.grad is not None
    assert planner.label_heads["eai"].weight.grad is not None


def test_autoregressive_sample_reuses_one_context_with_differentiable_logprob(joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    planner.train()
    calls_before = planner.model.forward_calls
    encoded_context = planner.encode_context("eai", {"instruction": "find the object"})

    labels, logprob = planner.sample_labels_from_context(
        "eai",
        encoded_context,
        runtime.eai_dfa,
        max_steps=runtime.max_eai_steps,
    )
    second_labels, second_logprob = planner.sample_labels_from_context(
        "eai",
        encoded_context,
        runtime.eai_dfa,
        max_steps=runtime.max_eai_steps,
    )

    assert labels and second_labels
    assert logprob.requires_grad and second_logprob.requires_grad
    assert planner.model.forward_calls - calls_before == 1
    (-(logprob + second_logprob)).backward()
    assert planner.model.forward_calls - calls_before == 1
    assert planner.model.embedding.weight.grad is not None
    assert planner.token_embeddings["eai"].weight.grad is not None


def test_vlabench_replay_keeps_collection_graph_free_and_restores_lora_gradient(joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    context = {
        "instruction": "Pick the apple.",
        "images": (),
        "entity_table": ("apple", "bowl"),
    }
    max_steps = runtime.max_vlabench_operations * 5 + 1
    prepared = planner.prepare_replay_context("vlabench", context)
    assert all(not torch.is_tensor(value) or value.device.type == "cpu" for value in prepared.values())

    with torch.no_grad():
        encoded = planner.encode_replay_context("vlabench", prepared)
        labels, collected_logprob = planner.sample_labels_from_context(
            "vlabench",
            encoded,
            runtime.vlabench_dfa,
            max_steps=max_steps,
        )
    assert not collected_logprob.requires_grad

    calls_before = planner.model.forward_calls
    replayed_logprob = planner.replay_labels_logprob(
        "vlabench",
        prepared,
        labels,
        runtime.vlabench_dfa,
        max_steps=max_steps,
    )
    assert replayed_logprob.requires_grad
    assert planner.model.forward_calls - calls_before == 1
    (-replayed_logprob).backward()
    assert planner.model.embedding.weight.grad is not None
    assert planner.label_heads["vlabench"].weight.grad is not None


def test_joint_pretrained_loader_uses_transformers_compatibility_resolver(joint_fixture, monkeypatch):
    _examples, runtime = joint_fixture
    calls = []

    class ModelLoader:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            return FakeBackbone()

    class ProcessorLoader:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("processor", model_id, kwargs))
            return FakeProcessor()

    monkeypatch.setattr(
        "test_regr.JointEmbodiedAgentInterface.models.resolve_vision_language_loader",
        lambda: (ModelLoader, ProcessorLoader),
    )
    planner = JointQwenVLPlanner.from_pretrained(
        eai_vocabulary=runtime.eai_vocabulary,
        vlabench_vocabulary=runtime.vlabench_vocabulary,
        model_id="fake-qwen-vl",
        use_lora=False,
        load_in_4bit=False,
        gradient_checkpointing=False,
    )

    assert isinstance(planner.model, FakeBackbone)
    assert [entry[0] for entry in calls] == ["model", "processor"]
    assert "dtype" in calls[0][2]
    assert "torch_dtype" not in calls[0][2]


def test_joint_root_hierarchy_activation_ancestors_and_exception_restore(joint_fixture):
    _examples, runtime = joint_fixture
    assert runtime.world.eai.graph.sup is runtime.root
    assert runtime.world.vlabench.graph.sup is runtime.root
    assert runtime.eai_generation_graph.sup is runtime.root
    assert runtime.vlabench_generation_graph.sup is runtime.root

    eai_probe = runtime.activation_profiles["eai"][0]
    vla_probe = runtime.activation_profiles["vlabench"][0]
    with pytest.raises(RuntimeError):
        with runtime.domain_scope("eai", extra_concepts=[runtime.world.operation.name]):
            assert runtime.root.is_concept_active(eai_probe)
            assert not runtime.root.is_concept_active(vla_probe)
            assert runtime.root.is_concept_active(runtime.world.operation)
            assert runtime.root.is_concept_active(runtime.world.eai.graph.constraint)
            assert runtime.root.is_concept_active(runtime.world.vlabench.graph.constraint)
            raise RuntimeError("restore")
    assert runtime.active_domain is None
    assert len(runtime.root.get_active_concepts()) == len(runtime.root._activation_concepts())

    runtime.activate_domain("eai")
    with runtime.domain_scope("vlabench"):
        assert runtime.active_domain == "vlabench"
    assert runtime.active_domain == "eai"
    runtime.activate_domain(None)


def test_activation_rejects_unknown_and_foreign_concepts(joint_fixture):
    _examples, runtime = joint_fixture
    with pytest.raises(ValueError, match="Unknown concept"):
        runtime.activate_domain("eai", ["does_not_exist"])
    with Graph("joint_foreign"):
        foreign = Concept(name="foreign")
    with pytest.raises(ValueError, match="does not belong"):
        runtime.activate_domain("eai", [foreign])
    with pytest.raises(ValueError, match="Unknown concept"):
        with runtime.domain_scope("eai", ["still_missing"]):
            pass
    assert runtime.active_domain is None
    runtime.activate_domain(None)


def test_domain_scope_serializes_concurrent_callers(joint_fixture):
    _examples, runtime = joint_fixture
    events = []
    first_entered = threading.Event()
    release_first = threading.Event()

    def first():
        with runtime.domain_scope("eai"):
            events.append("eai-enter")
            first_entered.set()
            assert release_first.wait(2)
            events.append("eai-exit")

    def second():
        assert first_entered.wait(2)
        with runtime.domain_scope("vlabench"):
            events.append("vla-enter")
            events.append("vla-exit")

    left = threading.Thread(target=first)
    right = threading.Thread(target=second)
    left.start()
    right.start()
    assert first_entered.wait(2)
    release_first.set()
    left.join(2)
    right.join(2)
    assert events == ["eai-enter", "eai-exit", "vla-enter", "vla-exit"]
    assert runtime.active_domain is None


def test_shared_planner_routes_prefixes_and_only_active_head_gets_gradient(joint_fixture):
    examples, runtime = joint_fixture
    planner = make_planner(runtime)
    eai_before = planner.label_heads["eai"].weight.detach().clone()
    vla_before = planner.label_heads["vlabench"].weight.detach().clone()
    vla_modules = {
        "projection": planner.context_projections["vlabench"],
        "embedding": planner.token_embeddings["vlabench"],
        "decoder": planner.graph_decoders["vlabench"],
    }
    vla_decoder_before = {
        f"{module_name}.{name}": parameter.detach().clone()
        for module_name, module in vla_modules.items()
        for name, parameter in module.named_parameters()
    }
    optimizer = torch.optim.SGD(planner.parameters(), lr=0.1)
    controller = TinyController()
    program = JointSolverPOIProgram(
        runtime,
        planner,
        planner_optimizer=optimizer,
        controller=controller,
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.1),
    )
    program._planner_step("eai", examples[0])
    assert not torch.equal(eai_before, planner.label_heads["eai"].weight)
    assert torch.equal(vla_before, planner.label_heads["vlabench"].weight)
    vla_decoder_after = {
        f"{module_name}.{name}": parameter
        for module_name, module in vla_modules.items()
        for name, parameter in module.named_parameters()
    }
    assert all(
        torch.equal(value, vla_decoder_after[name])
        for name, value in vla_decoder_before.items()
    )
    assert runtime.active_domain is None

    labels, logprob = planner.for_domain("eai").sample_labels(
        {"instruction": examples[0]["causal_prompt_text"]},
        runtime.eai_dfa,
        max_steps=runtime.max_eai_steps,
    )
    assert labels
    assert logprob.requires_grad

    non_eos = next(
        label for label in range(runtime.eai_vocabulary.label_count)
        if label != runtime.eai_vocabulary.eos_label
    )
    planner.eval()
    prompts_before = len(planner.processor.prompts)
    calls_before = planner.model.forward_calls
    conditioned = planner.sequence_logits(
        "eai",
        {"instruction": "test prefix"},
        torch.tensor([[runtime.eai_vocabulary.eos_label, non_eos]]),
    )
    alternative = planner.sequence_logits(
        "eai",
        {"instruction": "test prefix"},
        torch.tensor([[runtime.eai_vocabulary.eos_label, runtime.eai_vocabulary.eos_label]]),
    )
    assert planner.model.forward_calls - calls_before == 2
    assert len(planner.processor.prompts) - prompts_before == 2
    assert planner.processor.prompts[-1].endswith("Plan:")
    assert not torch.allclose(conditioned[:, -1], alternative[:, -1])


def test_stage1_controller_updates_only_on_vlabench_turn(joint_fixture):
    examples, runtime = joint_fixture
    planner = make_planner(runtime)
    controller = TinyController()
    program = JointSolverPOIProgram(
        runtime,
        planner,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.05),
        controller=controller,
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.1),
    )
    before = controller.action.detach().clone()
    planner.eval()
    program._planner_step("eai", examples[0])
    assert planner.training
    assert torch.equal(before, controller.action)
    batch = {
        "images": torch.zeros(1, 2, 1, 3, 4, 4),
        "state": torch.zeros(1, 2, 7),
        "task_index": torch.zeros(1, dtype=torch.long),
        "actions": torch.ones(1, 2, 7),
    }
    program._controller_step(batch)
    assert not torch.equal(before, controller.action)


def test_controller_bc_warmup_runs_requested_steps_and_only_updates_controller(joint_fixture):
    examples, runtime = joint_fixture
    planner = make_planner(runtime)
    controller = TinyController()
    program = JointSolverPOIProgram(
        runtime,
        planner,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.05),
        controller=controller,
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.1),
    )
    batch = {
        "images": torch.zeros(1, 2, 1, 3, 4, 4),
        "state": torch.zeros(1, 2, 7),
        "task_index": torch.zeros(1, dtype=torch.long),
        "actions": torch.ones(1, 2, 7),
    }
    planner_before = planner.label_heads["vlabench"].weight.detach().clone()
    controller_before = controller.action.detach().clone()
    metrics = program.train_controller_warmup([batch], steps=3)
    assert metrics["steps"] == 3 and metrics["loss"] > 0
    assert not torch.equal(controller_before, controller.action)
    assert torch.equal(planner_before, planner.label_heads["vlabench"].weight)
    assert runtime.active_domain is None


def test_program_types_share_identical_planner_and_defaults(joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    controller = TinyController()
    planner_optimizer = torch.optim.SGD(planner.parameters(), lr=0.01)
    controller_optimizer = torch.optim.SGD(controller.parameters(), lr=0.01)
    stage1 = JointSolverPOIProgram(
        runtime,
        planner,
        planner_optimizer=planner_optimizer,
        controller=controller,
        controller_optimizer=controller_optimizer,
    )
    stage2 = JointReinforcementProgram(
        runtime,
        planner,
        controller,
        planner_optimizer=planner_optimizer,
        controller_optimizer=controller_optimizer,
        env_factory=lambda **_kwargs: None,
        supervised_weight=0.1,
        controller_bc_weight=0.05,
        ppo_epochs=4,
    )
    assert isinstance(stage1, SolverPOIProgram)
    assert isinstance(stage2, ReinforcementProgram)
    assert stage1.planner_head is stage2.planner_head is planner
    assert stage2.eai_supervised_weight == 0.5
    assert stage2.supervised_weight == 0.1
    assert stage2.controller_bc_weight == 0.05
    assert stage2.gamma == 0.99 and stage2.gae_lambda == 0.95
    assert stage2.max_position_step == pytest.approx(0.02)
    assert stage2.max_rotation_step == pytest.approx(0.10)
    assert stage2.ik_tolerance == pytest.approx(1e-3)
    assert stage2.ik_max_steps == 200


def test_stage2_eai_update_uses_domain_reward_and_shared_planner_only(joint_fixture):
    examples, runtime = joint_fixture
    planner = make_planner(runtime)
    controller = TinyController()
    item = dict(examples[0])
    item["reward_function"] = make_eai_reward_function(
        item,
        vocabulary=runtime.eai_vocabulary,
        mode="dense",
        world_bundle=runtime.world.eai,
    )
    program = JointReinforcementProgram(
        runtime,
        planner,
        controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.05),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.01),
        env_factory=lambda **_kwargs: None,
        eai_supervised_examples=[item],
        eai_num_samples=2,
        ppo_epochs=1,
    )
    eai_before = planner.label_heads["eai"].weight.detach().clone()
    vla_before = planner.label_heads["vlabench"].weight.detach().clone()
    calls_before = planner.model.forward_calls
    planner.eval()
    metrics = program.train_eai_update(item)
    assert planner.training
    assert metrics["samples"] == 2
    assert 0.0 <= metrics["reward"] <= 1.0
    assert 0.0 <= metrics["goal_recall"] <= 1.0
    assert not torch.equal(eai_before, planner.label_heads["eai"].weight)
    assert torch.equal(vla_before, planner.label_heads["vlabench"].weight)
    assert planner.model.forward_calls - calls_before == 2
    assert runtime.active_domain is None


def test_joint_stage2_invalid_vlabench_plan_never_executes_controller(joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    controller = TinyController()

    class FakeSimulator:
        def __init__(self):
            self.steps = 0
            self.task = SimpleNamespace(
                entities={"apple": SimpleNamespace(geoms=())},
                get_instruction=lambda: "Pick the apple.",
            )

        def reset(self):
            return SimpleNamespace(last=lambda: False, observation=self.get_observation())

        def get_observation(self, require_pcd=False):
            return {"rgb": np.zeros((1, 4, 4, 3), dtype=np.uint8), "ee_state": np.zeros(7)}

        def step(self, _action):
            self.steps += 1
            raise AssertionError("invalid plans must not execute")

        def close(self):
            pass

    simulator = FakeSimulator()
    program = JointReinforcementProgram(
        runtime,
        planner,
        controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.01),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.01),
        env_factory=lambda **_kwargs: simulator,
        num_samples=2,
        ppo_epochs=1,
    )

    def invalid_sample(**_kwargs):
        score = torch.nn.functional.logsigmoid(planner.label_heads["vlabench"].weight[0, 0])
        return [{"name": "pick", "params": {}}], score

    program.vlabench_planner.sample_with_logprob = invalid_sample
    calls_before = planner.model.forward_calls
    episode = program.collect_episode({"task": "select_book"})
    assert not episode.valid
    assert episode.total_return == 0.0
    assert episode.controller == []
    assert simulator.steps == 0
    assert planner.model.forward_calls - calls_before == 1
    assert runtime.active_domain is None


def test_joint_checkpoint_roundtrip_and_compatibility_rejection(tmp_path, joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    controller = TinyController()
    planner_optimizer = torch.optim.Adam(planner.parameters(), lr=0.01)
    controller_optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    planner.label_heads["eai"].weight.sum().backward()
    planner_optimizer.step()
    planner_optimizer.zero_grad(set_to_none=True)
    controller.action.sum().backward()
    controller_optimizer.step()
    controller_optimizer.zero_grad(set_to_none=True)
    expected = planner.label_heads["eai"].weight.detach().clone()
    expected_embedding = planner.token_embeddings["eai"].weight.detach().clone()
    path = save_joint_checkpoint(
        tmp_path / "joint.pt",
        runtime=runtime,
        planner=planner,
        controller=controller,
        planner_optimizer=planner_optimizer,
        controller_optimizer=controller_optimizer,
        stage="stage1",
        epoch=4,
        round_robin_cursor=17,
    )
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(1)
    with torch.no_grad():
        planner.label_heads["eai"].weight.add_(10)
        planner.token_embeddings["eai"].weight.add_(10)
    planner_optimizer.param_groups[0]["lr"] = 0.9
    payload = load_joint_checkpoint(
        path,
        runtime=runtime,
        planner=planner,
        controller=controller,
        planner_optimizer=planner_optimizer,
        controller_optimizer=controller_optimizer,
        # Reproduce the full GPU command's Stage 1 -> Stage 2 restore path even
        # on CPU-only CI.  CUDA requests must stage optimizer state on CPU.
        map_location=torch.device("cuda"),
    )
    assert torch.equal(expected, planner.label_heads["eai"].weight)
    assert torch.equal(expected_embedding, planner.token_embeddings["eai"].weight)
    assert planner_optimizer.param_groups[0]["lr"] == 0.01
    assert all(
        state["step"].device.type == "cpu"
        for state in planner_optimizer.state.values()
        if "step" in state
    )
    assert payload["round_robin_cursor"] == 17
    assert runtime.active_domain is None
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)

    incompatible = copy.copy(runtime)
    incompatible.runtime_checksum = "different"
    with pytest.raises(ValueError, match="runtime_checksum"):
        load_joint_checkpoint(path, runtime=incompatible, planner=planner, controller=controller)


def test_legacy_controller_checkpoint_requires_stage1_migration(tmp_path, joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    controller = MultiViewController(
        TinyImageEncoder(), hidden_dim=8, action_horizon=1, max_views=1
    )
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    path = save_joint_checkpoint(
        tmp_path / "legacy_controller.pt",
        runtime=runtime,
        planner=planner,
        controller=controller,
        planner_optimizer=None,
        controller_optimizer=optimizer,
        stage="stage1",
        epoch=0,
        round_robin_cursor=0,
    )
    payload = torch.load(path, weights_only=False)
    payload["compatibility"].pop("controller_configuration")
    torch.save(payload, path)
    restored = load_joint_checkpoint(
        path,
        runtime=runtime,
        planner=planner,
        controller=controller,
        controller_optimizer=optimizer,
    )
    assert restored["controller_migration_required"] is True
    assert optimizer.state == {}

    payload["stage"] = "controller_warmup"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="resume a Stage 1 checkpoint"):
        load_joint_checkpoint(path, runtime=runtime, planner=planner, controller=controller)


def test_joint_checkpoint_loads_legacy_bitsandbytes_auxiliary_keys(tmp_path, joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    controller = TinyController()
    path = save_joint_checkpoint(
        tmp_path / "legacy_quantized.pt",
        runtime=runtime,
        planner=planner,
        controller=controller,
        planner_optimizer=None,
        controller_optimizer=None,
        stage="stage1",
        epoch=0,
        round_robin_cursor=1,
    )
    payload = torch.load(path, weights_only=False)
    payload["joint_checkpoint_version"] = 1
    payload["planner"]["model.visual.blocks.0.attn.qkv.weight.absmax"] = torch.ones(1)
    payload["planner"]["model.visual.blocks.0.attn.qkv.weight.quant_map"] = torch.ones(1)
    payload["planner"][
        "model.visual.blocks.0.attn.qkv.weight.quant_state.bitsandbytes__nf4"
    ] = torch.ones(1)
    torch.save(payload, path)

    restored = load_joint_checkpoint(path, runtime=runtime, planner=planner, controller=controller)
    assert restored["round_robin_cursor"] == 1


def test_joint_checkpoint_loads_legacy_process_local_dfa_numbering(tmp_path, joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    controller = TinyController()
    path = save_joint_checkpoint(
        tmp_path / "legacy_dfa.pt",
        runtime=runtime,
        planner=planner,
        controller=controller,
        planner_optimizer=None,
        controller_optimizer=None,
        stage="stage1",
        epoch=4,
        round_robin_cursor=5,
    )
    payload = torch.load(path, weights_only=False)
    # Recreate the old repr-based payload with deliberately different labels
    # for an isomorphic two-state automaton.
    legacy = {
        "max_operations": 8,
        "start_state": "4",
        "states": ["4", "9"],
        "accepting_states": ["9"],
        "transitions": [("4", 1, "9"), ("9", 1, "9")],
    }
    renumbered = {
        "max_operations": 8,
        "start_state": "7",
        "states": ["2", "7"],
        "accepting_states": ["2"],
        "transitions": [("2", 1, "2"), ("7", 1, "2")],
    }
    assert _normalize_dfa_configuration(legacy) == _normalize_dfa_configuration(renumbered)

    # A real checkpoint's legacy fingerprint also upgrades to the new stable
    # format; its graph/model payload remains otherwise untouched.
    for key in ("eai_dfa", "vlabench_dfa"):
        domain_dfa = runtime.eai_dfa if key == "eai_dfa" else runtime.vlabench_dfa
        current = payload["compatibility"][key]
        payload["compatibility"][key] = {
            name: value
            for name, value in current.items()
            if name not in {"format_version", "state_count"}
        } | {
            "start_state": repr(domain_dfa.start_state),
            "states": sorted(
                repr(value)
                for value in domain_dfa.states
            ),
            "accepting_states": sorted(
                repr(value)
                for value in domain_dfa.accepting_states
            ),
            "transitions": sorted(
                (repr(source), int(symbol), repr(target))
                for (source, symbol), target in domain_dfa.transitions.items()
            ),
        }
    torch.save(payload, path)
    restored = load_joint_checkpoint(path, runtime=runtime, planner=planner, controller=controller)
    assert restored["epoch"] == 4


def test_dfa_checkpoint_configuration_ignores_state_labels():
    from domiknows.generation.dfa.core import DFA

    first = DFA(
        states=frozenset({4, 9}), alphabet=frozenset({1}),
        transitions={(4, 1): 9, (9, 1): 9}, start_state=4,
        accepting_states=frozenset({9}),
    )
    second = DFA(
        states=frozenset({2, 7}), alphabet=frozenset({1}),
        transitions={(7, 1): 2, (2, 1): 2}, start_state=7,
        accepting_states=frozenset({2}),
    )
    assert _dfa_configuration(first, max_operations=8) == _dfa_configuration(
        second, max_operations=8
    )


def test_joint_checkpoint_rejects_prefix_reprompt_architecture(tmp_path, joint_fixture):
    _examples, runtime = joint_fixture
    planner = make_planner(runtime)
    controller = TinyController()
    path = save_joint_checkpoint(
        tmp_path / "old_architecture.pt",
        runtime=runtime,
        planner=planner,
        controller=controller,
        planner_optimizer=None,
        controller_optimizer=None,
        stage="stage1",
        epoch=0,
        round_robin_cursor=1,
    )
    payload = torch.load(path, weights_only=False)
    payload["compatibility"]["model_configuration"].pop("graph_decoder_version")
    payload["compatibility"]["model_configuration"].pop("graph_decoder_hidden_size")
    torch.save(payload, path)

    with pytest.raises(ValueError, match="model_configuration"):
        load_joint_checkpoint(path, runtime=runtime, planner=planner, controller=controller)


def test_checkpoint_rng_state_is_normalized_to_cpu_byte_tensor():
    state = _cpu_rng_state(torch.tensor([1, 2, 3], dtype=torch.int64))
    assert state.device.type == "cpu"
    assert state.dtype == torch.uint8


def test_cuda_checkpoint_restore_uses_cpu_staging():
    assert _checkpoint_staging_location("cuda").type == "cpu"
    assert _checkpoint_staging_location(torch.device("cuda:1")).type == "cpu"
    assert torch.device(_checkpoint_staging_location("cpu")).type == "cpu"


def test_balanced_checkpoint_keys_and_cli_defaults():
    weak_eai = {
        "eai": {"goal_success": 0.1, "goal_recall": 0.9},
        "vlabench": {"exact_graph_match": 0.8, "valid": 1.0},
        "validation_loss": 0.1,
    }
    balanced = {
        "eai": {"goal_success": 0.5, "goal_recall": 0.5},
        "vlabench": {"exact_graph_match": 0.5, "valid": 0.8},
        "validation_loss": 0.3,
    }
    assert stage1_selection_key(balanced) > stage1_selection_key(weak_eai)
    first = {"eai": {"success": 0.4, "reward": 0.4}, "vlabench": {"success_rate": 0.4, "return": 0.8}}
    second = {"eai": {"success": 0.1, "reward": 0.9}, "vlabench": {"success_rate": 0.9, "return": 0.9}}
    assert stage2_selection_key(first) > stage2_selection_key(second)
    args = build_parser().parse_args(["train-agent", "--two-stage"])
    assert (args.stage1_epochs, args.stage2_epochs) == (5, 3)
    assert (args.eai_samples, args.vlabench_planner_samples, args.vlabench_rollouts) == (8, 4, 8)
    assert args.stage2_rounds_per_epoch == 10
    assert args.planner_decoder_hidden_dim == 512
    assert args.video_decoder_cache_size == 8
    assert args.controller_warmup_steps == 2000
    assert args.max_position_step == pytest.approx(0.02)
    assert args.max_rotation_step == pytest.approx(0.10)
    assert args.ik_tolerance == pytest.approx(1e-3)
    assert args.ik_max_steps == 200
