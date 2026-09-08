import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from test_regr.VLABenchAgentInterface.dataset import (
    CONTROL_DATASET_ID,
    PLANNING_DATASET_ID,
    LeRobotWindowDataset,
    build_numbered_segmentation_view,
    deterministic_split,
    download_processed_datasets,
    control_task_index_for_instruction,
    load_control_task_instructions,
    load_planning_examples,
    _video_tensor,
)
from test_regr.VLABenchAgentInterface.environment import (
    bound_ee_action,
    create_environment,
    ee_action_to_env_action,
    euler_to_quaternion,
    quaternion_to_euler,
    reset_reward_tracking,
    robot_frame_position,
    robot_to_world_ee_action,
    world_to_robot_ee_state,
)
from test_regr.VLABenchAgentInterface.diagnostics import RolloutDiagnostics
from test_regr.VLABenchAgentInterface.graph import PlanVocabulary, plan_to_tokens
from test_regr.VLABenchAgentInterface.models import (
    FrozenSigLIPEncoder,
    MultiViewController,
    QwenVLPlanner,
    TinyImageEncoder,
    controller_loss,
    clear_siglip_special_token_ids,
    load_sanitized_auto_config,
    prepare_kbit_model,
    resolve_vision_language_loader,
    sanitize_special_token_ids,
    special_token_id_summary,
    vision_language_hidden_size,
)
from test_regr.VLABenchAgentInterface.main import (
    _aggregate_task_metrics,
    _reinforcement_resume_position,
    build_parser,
    reinforcement_checkpoint_eligible,
    reinforcement_preflight_eligible,
    reinforcement_selection_key,
)
from test_regr.VLABenchAgentInterface.program import (
    ControllerTransition,
    JointEpisode,
    PlannerReplayDecision,
    VLABenchHierarchicalReinforcementProgram,
    _controller_inputs,
    _entity_pointer_dfa,
    _observation_state,
    _signal,
    _task_signals,
    generalized_advantage_estimate,
    ppo_clipped_loss,
)
from test_regr.VLABenchAgentInterface.training import (
    build_constraint_runtime,
    create_stage1_program,
    create_stage2_program,
    evaluate_controller,
    evaluate_planner,
    load_joint_checkpoint,
    load_checkpoint,
    prepare_planner_program_examples,
    save_joint_checkpoint,
    save_checkpoint,
    train_controller_epoch,
    train_controller_steps,
    train_planner_reinforcement_epoch,
)
from test_regr.VLABenchAgentInterface.world_graph import (
    build_vlabench_world_graph,
    condition_index_for_task,
)


class RateLimitError(Exception):
    def __init__(self):
        super().__init__("too many requests")
        self.response = SimpleNamespace(status_code=429, headers={"Retry-After": "0"})


def test_environment_imports_registration_modules_before_load_env(monkeypatch):
    calls = []

    def import_module(name):
        calls.append(name)
        if name == "VLABench.envs":
            return SimpleNamespace(
                load_env=lambda task, **kwargs: {"task": task, **kwargs},
            )
        return SimpleNamespace()

    monkeypatch.setattr(
        "test_regr.VLABenchAgentInterface.environment.importlib.import_module",
        import_module,
    )
    result = create_environment("select_fruit", robot="franka", time_limit=4)

    assert calls == ["VLABench.robots", "VLABench.tasks", "VLABench.envs"]
    assert result == {
        "task": "select_fruit",
        "robot": "franka",
        "time_limit": 4,
        "run_mode": "eval",
    }


def test_missing_upstream_reward_tracking_state_is_initialized_once():
    calls = []
    task = SimpleNamespace()

    def reset_progress():
        calls.append("progress")
        task.target_is_grasped = {"flower": False}

    def reset_intention():
        calls.append("intention")
        task.intention_distance = {"flower": np.inf}

    task.reset_task_progress = reset_progress
    task.reset_intention_distance = reset_intention
    env = SimpleNamespace(task=task)

    reset_reward_tracking(env)
    reset_reward_tracking(env)

    assert calls == ["progress", "intention"]
    assert task.target_is_grasped == {"flower": False}
    assert np.isinf(task.intention_distance["flower"])


def test_partial_upstream_reward_tracking_adds_missing_target_and_signal_falls_back():
    task = SimpleNamespace(
        target_entity="target_book",
        target_is_grasped={"other_book": False},
        intention_distance={"other_book": 0.5},
    )
    reset_reward_tracking(SimpleNamespace(task=task))
    assert task.target_is_grasped["target_book"] is False
    assert np.isinf(task.intention_distance["target_book"])

    env = SimpleNamespace(get_intention_score=lambda **_kwargs: {}["missing_target"])
    assert _signal(env, "get_intention_score") == 0.0


def test_intention_shaping_uses_monotone_discrete_upstream_signal():
    seen = {}

    def intention(**kwargs):
        seen.update(kwargs)
        return 1.0

    assert _signal(SimpleNamespace(get_intention_score=intention), "get_intention_score") == 1.0
    assert seen == {"threshold": 0.1, "discrete": True}


def test_signal_passes_upstream_physics_argument():
    physics = object()
    seen = {}

    def progress(received):
        seen["physics"] = received
        return 0.5

    env = SimpleNamespace(physics=physics, get_task_progress=progress)
    assert _signal(env, "get_task_progress") == 0.5
    assert seen["physics"] is physics


def test_task_signals_use_target_distance_when_upstream_progress_is_flat():
    target = SimpleNamespace(get_xpos=lambda _physics: np.array([1.0, 0.0, 0.0]))
    env = SimpleNamespace(
        physics=object(),
        task=SimpleNamespace(target_entity="apple", entities={"apple": target}),
        get_task_progress=lambda *_args, **_kwargs: 0.0,
        get_intention_score=lambda *_args, **_kwargs: 0.0,
    )
    diagnostics = RolloutDiagnostics()
    diagnostics.observe(env, np.zeros(7))
    diagnostics.observe(env, np.array([0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    progress, intention, source = _task_signals(env, diagnostics)
    assert progress == pytest.approx(0.25)
    assert intention == 0.0
    assert source == "target_distance"


def test_online_entity_pointer_dfa_masks_unknown_observation_pointers():
    world = build_vlabench_world_graph("test_online_pointer_world")
    runtime = build_constraint_runtime(
        world, max_entities=4, max_operations=2, name_prefix="test_online_pointer"
    )
    conditioned = _entity_pointer_dfa(runtime.dfa, runtime.vocabulary, entity_count=2)
    valid = [
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}},
    ]
    unknown = [
        {"name": "pick", "params": {"target_entity_name": 3}},
        {"name": "place", "params": {"target_container_name": 1}},
    ]

    def labels(plan):
        return [
            runtime.vocabulary.label_for_token(token)
            for token in plan_to_tokens(plan, ("apple", "bowl"), world=world)
        ]

    assert conditioned.accepts(labels(valid))
    assert runtime.dfa.accepts(labels(unknown))
    assert not conditioned.accepts(labels(unknown))


def test_vision_language_loader_supports_current_and_legacy_transformers(monkeypatch):
    current_model = type("CurrentImageTextModel", (), {})
    legacy_model = type("LegacyVision2SeqModel", (), {})
    processor = type("Processor", (), {})
    current = SimpleNamespace(
        __version__="5.0",
        AutoModelForImageTextToText=current_model,
        AutoModelForVision2Seq=legacy_model,
        AutoProcessor=processor,
    )
    monkeypatch.setitem(sys.modules, "transformers", current)
    assert resolve_vision_language_loader() == (current_model, processor)

    legacy = SimpleNamespace(
        __version__="4.56",
        AutoModelForVision2Seq=legacy_model,
        AutoProcessor=processor,
    )
    monkeypatch.setitem(sys.modules, "transformers", legacy)
    assert resolve_vision_language_loader() == (legacy_model, processor)


def test_vision_language_hidden_size_supports_nested_config_and_adapter_wrapper():
    class NestedBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2048))

    class AdapterWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(peft_type="LORA")
            self.base_model = NestedBackbone()

    assert vision_language_hidden_size(NestedBackbone()) == 2048
    assert vision_language_hidden_size(AdapterWrapper()) == 2048


def test_sanitize_special_token_ids_clears_invalid_nested_ids():
    config = SimpleNamespace(
        vocab_size=32000,
        bos_token_id=49406,
        eos_token_id=49407,
        text_config=SimpleNamespace(vocab_size=10, bos_token_id=1, eos_token_id=12),
    )
    assert sanitize_special_token_ids(config) is config
    assert config.bos_token_id is None
    assert config.eos_token_id is None
    assert config.text_config.bos_token_id == 1
    assert config.text_config.eos_token_id is None

    missing = SimpleNamespace(vocab_size=32000, text_config=SimpleNamespace(vocab_size=32000))
    sanitize_special_token_ids(missing)
    assert missing.bos_token_id is None
    assert missing.eos_token_id is None
    assert missing.text_config.bos_token_id is None
    assert missing.text_config.eos_token_id is None

    siglip_raw = {"model_type": "siglip_text_model"}
    sanitize_special_token_ids(siglip_raw)
    assert siglip_raw["bos_token_id"] is None
    assert siglip_raw["eos_token_id"] is None

    summary = special_token_id_summary(
        SimpleNamespace(
            model_type="siglip",
            vocab_size=None,
            text_config=SimpleNamespace(
                model_type="siglip_text_model", vocab_size=32000, bos_token_id=None, eos_token_id=None
            ),
        )
    )
    assert "root:class=SimpleNamespace model_type='siglip'" in summary
    assert "root.text_config:class=SimpleNamespace model_type='siglip_text_model'" in summary
    assert "bos=None" in summary and "eos=None" in summary


def test_load_sanitized_auto_config_cleans_raw_nested_ids_before_validation():
    raw = {
        "model_type": "fake",
        "vocab_size": 32000,
        "bos_token_id": 49406,
        "eos_token_id": 49407,
        "text_config": {"vocab_size": 32000, "bos_token_id": 49406, "eos_token_id": 49407},
    }

    class PretrainedConfig:
        @staticmethod
        def get_config_dict(*_args, **_kwargs):
            return raw, {}

    calls = {}

    class AutoConfig:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            raise AssertionError("raw sanitized configs must use AutoConfig.for_model")

        @staticmethod
        def for_model(_model_type, **kwargs):
            calls.update(kwargs)
            if _model_type == "siglip":
                text_config = dict(kwargs["text_config"])
                text_config.setdefault("model_type", "siglip_text_model")
                return SimpleNamespace(model_type="siglip", text_config=text_config)
            return SimpleNamespace(**kwargs)

    config = load_sanitized_auto_config(
        SimpleNamespace(AutoConfig=AutoConfig, PreTrainedConfig=PretrainedConfig),
        "fake-model",
        local_files_only=True,
    )
    assert config.bos_token_id is None
    assert config.eos_token_id is None
    assert config.text_config["bos_token_id"] is None
    assert config.text_config["eos_token_id"] is None
    assert calls["bos_token_id"] is None
    assert calls["eos_token_id"] is None
    assert calls["text_config"]["bos_token_id"] is None
    assert calls["text_config"]["eos_token_id"] is None

    sparse_siglip = {
        "model_type": "siglip",
        "text_config": {
            "hidden_size": 768,
            "vocab_size": 32000,
            "bos_token_id": 49406,
            "eos_token_id": 49407,
        },
    }

    class SparsePretrainedConfig:
        @staticmethod
        def get_config_dict(*_args, **_kwargs):
            return sparse_siglip, {}

    sparse = load_sanitized_auto_config(
        SimpleNamespace(AutoConfig=AutoConfig, PreTrainedConfig=SparsePretrainedConfig),
        "sparse-siglip",
        local_files_only=True,
    )
    assert sparse.text_config["bos_token_id"] is None
    assert sparse.text_config["eos_token_id"] is None
    assert calls["text_config"]["bos_token_id"] == 0
    assert calls["text_config"]["eos_token_id"] == 0

    restored = SimpleNamespace(
        model_type="siglip",
        text_config=SimpleNamespace(model_type="siglip_text_model", bos_token_id=0, eos_token_id=0),
    )
    clear_siglip_special_token_ids(restored)
    assert restored.text_config.bos_token_id is None
    assert restored.text_config.eos_token_id is None


def test_qwen_loader_sanitizes_model_and_processor_special_tokens(monkeypatch):
    from test_regr.VLABenchAgentInterface import models

    config = SimpleNamespace(vocab_size=32000, bos_token_id=49406, eos_token_id=49407)
    processor_calls = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(hidden_size=4, vocab_size=32000, bos_token_id=None, eos_token_id=None)

    class ModelClass:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            assert kwargs["config"].bos_token_id is None
            assert kwargs["config"].eos_token_id is None
            assert "bos_token_id" not in kwargs
            assert "eos_token_id" not in kwargs
            assert kwargs["generation_config"].bos_token_id is None
            assert kwargs["generation_config"].eos_token_id is None
            return FakeModel()

    class ProcessorClass:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            processor_calls.update(kwargs)
            return SimpleNamespace()

    class AutoConfig:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return config

    class GenerationConfig:
        @classmethod
        def from_model_config(cls, _config):
            return cls()

    monkeypatch.setattr(models, "resolve_vision_language_loader", lambda: (ModelClass, ProcessorClass))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=AutoConfig, GenerationConfig=GenerationConfig),
    )
    vocabulary = PlanVocabulary(
        skills=("pick",), argument_keys=("target_entity_name",),
        skill_arguments=(("pick", ("target_entity_name",)),), max_entities=2,
    )
    models.QwenVLPlanner.from_pretrained(vocabulary, "fake-qwen", use_lora=False)
    assert processor_calls["bos_token_id"] is None
    assert processor_calls["eos_token_id"] is None


def test_kbit_preparation_preserves_non_reentrant_checkpointing(monkeypatch):
    calls = []
    model = torch.nn.Linear(2, 2)

    def prepare(candidate, **kwargs):
        calls.append((candidate, kwargs))
        return candidate

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(prepare_model_for_kbit_training=prepare),
    )

    assert prepare_kbit_model(model, gradient_checkpointing=True) is model
    assert calls[-1] == (
        model,
        {
            "use_gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        },
    )

    assert prepare_kbit_model(model, gradient_checkpointing=False) is model
    assert calls[-1] == (model, {"use_gradient_checkpointing": False})


@pytest.mark.parametrize("output_kind", ["tensor", "pooled", "hidden"])
def test_frozen_siglip_encoder_accepts_current_and_legacy_outputs(monkeypatch, output_kind):
    class FakeSigLIP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(
                vision_config=SimpleNamespace(hidden_size=3),
            )

        def get_image_features(self, *, pixel_values):
            batch = pixel_values.shape[0]
            pooled = torch.ones(batch, 3)
            if output_kind == "tensor":
                return pooled
            if output_kind == "pooled":
                return SimpleNamespace(
                    pooler_output=pooled,
                    last_hidden_state=torch.zeros(batch, 4, 3),
                )
            return SimpleNamespace(
                pooler_output=None,
                last_hidden_state=torch.ones(batch, 4, 3),
            )

    auto_model = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: FakeSigLIP(),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=auto_model))

    encoder = FrozenSigLIPEncoder("fake-siglip")
    features = encoder(torch.zeros(2, 3, 32, 32))

    assert features.shape == (2, 3)
    assert features.dtype == torch.float32
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_frozen_siglip_encoder_sanitizes_config_before_loading(monkeypatch):
    config = SimpleNamespace(
        vocab_size=32000,
        bos_token_id=49406,
        eos_token_id=49407,
        vision_config=SimpleNamespace(hidden_size=3),
    )
    calls = {}

    class FakeSigLIP(torch.nn.Module):
        def __init__(self, loaded_config):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = loaded_config

    class AutoConfig:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return config

    class AutoModel:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            calls.update(kwargs)
            return FakeSigLIP(kwargs["config"])

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoConfig=AutoConfig, AutoModel=AutoModel))
    encoder = FrozenSigLIPEncoder("fake-siglip")
    assert calls["config"].bos_token_id is None
    assert calls["config"].eos_token_id is None
    assert encoder.output_dim == 3


def test_dataset_download_retries_429_and_resumes(tmp_path, monkeypatch):
    calls = []
    delays = []
    progress_targets = (
        (importlib.import_module("huggingface_hub.utils.tqdm"), "tqdm"),
        (importlib.import_module("huggingface_hub.utils"), "tqdm"),
        (importlib.import_module("huggingface_hub.utils._xet_progress_reporting"), "tqdm"),
        (importlib.import_module("huggingface_hub.file_download"), "tqdm"),
        (importlib.import_module("huggingface_hub._snapshot_download"), "hf_tqdm"),
    )
    original_progress = [getattr(module, attribute) for module, attribute in progress_targets]

    def fake_snapshot_download(**kwargs):
        assert all(
            getattr(module, attribute) is kwargs["tqdm_class"]
            for module, attribute in progress_targets
        )
        calls.append(kwargs)
        if len(calls) == 1:
            raise RateLimitError()
        return str(kwargs["local_dir"])

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("test_regr.VLABenchAgentInterface.dataset.time.sleep", delays.append)
    planning, control = download_processed_datasets(
        tmp_path / "planning",
        tmp_path / "control",
        max_workers=1,
        retries=2,
        retry_delay=0,
    )

    assert planning == (tmp_path / "planning").resolve()
    assert control == (tmp_path / "control").resolve()
    assert [call["repo_id"] for call in calls] == [
        PLANNING_DATASET_ID,
        PLANNING_DATASET_ID,
        CONTROL_DATASET_ID,
    ]
    assert all(call["max_workers"] == 1 for call in calls)
    progress_class = calls[0]["tqdm_class"]
    assert all(call["tqdm_class"] is progress_class for call in calls)
    assert all(base.__module__ != "tqdm.std" for base in progress_class.__mro__)
    progress = progress_class(total=0, disable=True, name="huggingface.test")
    progress.close()
    assert [getattr(module, attribute) for module, attribute in progress_targets] == original_progress
    assert delays == [0.0]


def test_download_progress_is_newline_based_and_tqdm_independent(capsys):
    from test_regr.VLABenchAgentInterface.dataset import _TerminalDownloadProgress

    progress = _TerminalDownloadProgress(total=2, desc="Reconstructing", unit="B", unit_scale=True)
    progress.update(1)
    progress.update(1)
    progress.close()

    rendered = capsys.readouterr().err
    assert "Reconstructing: 0B/2B" in rendered
    assert "Reconstructing: 2B/2B (100.0%)" in rendered
    assert "\r" not in rendered


def test_planning_folder_loader_and_deterministic_split(tmp_path):
    example = tmp_path / "Spatial" / "task" / "example0"
    (example / "input").mkdir(parents=True)
    (example / "output").mkdir()
    (example / "env_config").mkdir()
    (example / "input" / "instruction.txt").write_text("Put the apple in the bowl.", encoding="utf-8")
    Image.new("RGB", (16, 16), "white").save(example / "input" / "rgb.png")
    Image.new("RGB", (16, 16), "black").save(example / "input" / "segmented_prompt.png")
    plan = [
        {"name": "pick", "params": {"target_entity_name": "apple"}},
        {"name": "place", "params": {"target_container_name": "bowl"}},
    ]
    (example / "output" / "operation_sequence.json").write_text(
        json.dumps({"skill_sequence": plan}), encoding="utf-8",
    )
    (example / "env_config" / "episode.json").write_text(json.dumps({"entities": ["apple", "bowl"]}), encoding="utf-8")
    loaded = load_planning_examples(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].instruction == "Put the apple in the bowl."
    assert loaded[0].entities == ("apple", "bowl")
    assert len(loaded[0].image_paths) == len(loaded[0].segmented_image_paths) == 1
    assert deterministic_split(list(range(20)), seed=42) == deterministic_split(list(range(20)), seed=42)
    one_episode = deterministic_split([7], seed=42)
    assert one_episode == {"train": [7], "validation": [], "test": []}
    three_episodes = deterministic_split([1, 2, 3], seed=42)
    assert len(three_episodes["train"]) >= 1
    assert set().union(*map(set, three_episodes.values())) == {1, 2, 3}
    world = build_vlabench_world_graph("test_dataset_vocabulary_world")
    vocabulary = PlanVocabulary.from_plans(({"skill_sequence": plan},), world, max_entities=12)
    vocabulary_path = tmp_path / "vocab.json"
    vocabulary.save(vocabulary_path)
    restored = PlanVocabulary.load(vocabulary_path)
    assert restored.checksum == vocabulary.checksum
    assert restored.skill_argument_map["pick"] == ("target_entity_name",)
    assert restored.skill_argument_map["lift"] == ()


def test_control_loader_with_one_limited_episode_keeps_training_windows(monkeypatch):
    from test_regr.VLABenchAgentInterface.main import _control_loaders

    records = [{"episode_index": 0} for _ in range(10)]
    monkeypatch.setattr(
        "test_regr.VLABenchAgentInterface.main.load_hf_control_records",
        lambda *_args, **_kwargs: records,
    )
    args = SimpleNamespace(
        task="add_condiment",
        control_source="not-a-local-path",
        limit=10,
        action_horizon=2,
        batch_size=1,
        workers=0,
    )
    loaders = _control_loaders(args)

    assert len(loaders["train"].dataset) == 10
    assert len(loaders["validation"].dataset) == 0
    assert len(loaders["test"].dataset) == 0


def test_numbered_segmentation_overlay():
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    segmentation = np.zeros((20, 30), dtype=np.int32)
    segmentation[2:8, 2:8] = 5
    segmentation[10:18, 20:28] = 9
    image, centers = build_numbered_segmentation_view(rgb, segmentation)
    assert image.size == (30, 20)
    assert set(centers) == {0, 1}
    assert centers[0] == (4, 4)


def _records():
    result = []
    for episode in range(2):
        for frame in range(4):
            result.append({
                "episode_index": episode,
                "frame_index": frame,
                "state": np.full(7, frame, dtype=np.float32),
                "actions": np.array([frame] * 6 + [frame % 2], dtype=np.float32),
                "images": torch.rand(3, 3, 24, 24),
                "task_index": episode,
            })
    return result


def test_lerobot_v3_external_video_columns_are_reconstructed(tmp_path, monkeypatch):
    info = {
        "chunks_size": 1000,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "image": {"dtype": "video"},
            "second_image": {"dtype": "video"},
            "wrist_image": {"dtype": "video"},
            "state": {"dtype": "float32"},
        },
    }
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    expected = []
    for key in ("image", "second_image", "wrist_image"):
        path = tmp_path / "videos" / key / "chunk-001" / "file-001.mp4"
        path.parent.mkdir(parents=True)
        path.touch()
        expected.append(path.resolve())

    decoded = []

    def fake_video_tensor(value, **_kwargs):
        decoded.append(Path(value["path"]).resolve())
        return torch.zeros(3, 12, 12)

    monkeypatch.setattr("test_regr.VLABenchAgentInterface.dataset._video_tensor", fake_video_tensor)
    records = [{
        "episode_index": 1001,
        "frame_index": 0,
        "timestamp": 0.0,
        "state": np.zeros(7, dtype=np.float32),
        "actions": np.zeros(7, dtype=np.float32),
    }]
    dataset = LeRobotWindowDataset(
        records,
        observation_horizon=1,
        action_horizon=1,
        video_root=tmp_path,
    )
    item = dataset[0]

    assert item["images"].shape == (1, 3, 3, 12, 12)
    assert decoded == expected


def test_torchcodec_decoder_cache_is_bounded_lru_and_released(tmp_path, monkeypatch):
    created = []
    closed = []

    class FakeDecoder:
        def __init__(self, path):
            self.path = path
            created.append(path)

        def get_frame_played_at(self, _timestamp):
            return SimpleNamespace(data=torch.zeros(3, 4, 4))

        def close(self):
            closed.append(self.path)

    decoders = SimpleNamespace(VideoDecoder=FakeDecoder)
    monkeypatch.setitem(sys.modules, "torchcodec", SimpleNamespace(decoders=decoders))
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    paths = [tmp_path / f"video-{index}.mp4" for index in range(3)]
    cache = {}

    for index in (0, 1, 0, 2):
        _video_tensor(
            {"path": str(paths[index]), "timestamp": 0.0},
            timestamp=0.0,
            video_root=None,
            cache=cache,
            cache_size=2,
        )

    keys = [str(path.resolve()) for path in paths]
    assert created == [keys[0], keys[1], keys[2]]
    assert list(cache) == [keys[0], keys[2]]
    assert closed == [keys[1]]

    dataset = LeRobotWindowDataset([], video_decoder_cache_size=2)
    dataset._video_cache = cache
    dataset.close()
    assert cache == {}
    assert closed == [keys[1], keys[2], keys[0]]

    with pytest.raises(ValueError, match="cache size"):
        LeRobotWindowDataset([], video_decoder_cache_size=0)


def test_control_windows_controller_loss_and_training(tmp_path):
    condition_index = condition_index_for_task("select_poker")
    dataset = LeRobotWindowDataset(
        _records(), observation_horizon=2, action_horizon=3,
        condition_index=condition_index, plan_pattern=("pick", "lift"),
    )
    item = dataset[0]
    assert item["state"].shape == (2, 7)
    assert item["images"].shape == (2, 3, 3, 24, 24)
    assert item["actions"].shape == (3, 7)
    assert item["task_index"].item() == condition_index
    assert item["plan_context"].tolist() == [1, 0, 1]
    language_dataset = LeRobotWindowDataset(
        _records(), observation_horizon=2, action_horizon=3, condition_index=None
    )
    assert language_dataset[language_dataset.index.index((1, 0))]["task_index"].item() == 1
    model = MultiViewController(TinyImageEncoder(16), hidden_dim=24, action_horizon=3, max_views=3)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    output = model(batch["images"], batch["state"], batch["task_index"])
    assert output.shape == (2, 3, 7)
    loss, metrics = controller_loss(output, batch["actions"])
    assert torch.isfinite(loss) and metrics["pose_loss"] >= 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trained = train_controller_epoch(model, [batch], optimizer, device="cpu", mixed_precision=False)
    assert trained["loss"] >= 0
    stepped = train_controller_steps(
        model,
        [batch],
        optimizer,
        steps=3,
        device="cpu",
        mixed_precision=False,
    )
    assert stepped["steps"] == 3
    assert stepped["loss"] >= 0
    checkpoint = tmp_path / "controller.pt"
    save_checkpoint(checkpoint, model=model, optimizer=optimizer, epoch=0, metrics=trained)
    restored = MultiViewController(TinyImageEncoder(16), hidden_dim=24, action_horizon=3, max_views=3)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    payload = load_checkpoint(checkpoint, model=restored, optimizer=restored_optimizer)
    assert payload["epoch"] == 0


def test_controller_delta_loss_uses_execution_scales_and_wraps_euler_angles():
    state = torch.zeros(1, 1, 7)
    state[..., 3] = torch.pi - 0.05
    target = torch.zeros(1, 1, 7)
    target[..., 0] = 0.02
    target[..., 3] = -torch.pi + 0.05
    target[..., -1] = 1.0
    prediction = target.clone()
    prediction[..., -1] = 20.0

    loss, metrics = controller_loss(
        prediction,
        target,
        state=state,
        pose_step_scale=(0.02, 0.02, 0.02, 0.10, 0.10, 0.10),
    )

    assert metrics["pose_loss"] == pytest.approx(0.0, abs=1e-7)
    assert loss.item() < 1e-6


def test_control_task_metadata_preserves_language_condition_ids(tmp_path):
    metadata = tmp_path / "meta"
    metadata.mkdir()
    (metadata / "tasks.jsonl").write_text(
        '{"task_index": 7, "task": "Insert the rose into the vase."}\n'
        '{"task_index": 91, "task": "Pick up the mahjong of 1 pin"}\n',
        encoding="utf-8",
    )
    instructions = load_control_task_instructions(tmp_path)
    assert instructions == {
        7: "Insert the rose into the vase.",
        91: "Pick up the mahjong of 1 pin",
    }
    assert control_task_index_for_instruction(
        "  insert THE rose into the vase! ", instructions
    ) == 7
    with pytest.raises(KeyError, match="not uniquely represented"):
        control_task_index_for_instruction("Insert the tulip into the vase", instructions)


@dataclass
class FakeExample:
    operation_sequence: tuple
    entities: tuple = ("apple", "bowl")
    instruction: str = "Put the apple in the bowl."
    image_paths: tuple = ()
    segmented_image_paths: tuple = ()
    dependency: str = "Sequential"

    def as_reward_item(self):
        return {
            "operation_sequence": list(self.operation_sequence),
            "entities": self.entities,
            "instruction": self.instruction,
        }


class FakePlanner(torch.nn.Module):
    def __init__(self, good, bad):
        super().__init__()
        self.preference = torch.nn.Parameter(torch.tensor(0.0))
        self.good = json.dumps(good)
        self.bad = json.dumps(bad)
        self.calls = 0

    def sample_with_logprob(self, **_kwargs):
        good = self.calls % 2 == 0
        self.calls += 1
        logprob = torch.nn.functional.logsigmoid(self.preference if good else -self.preference)
        return (self.good if good else self.bad), logprob


def test_reward_driven_planner_epoch_uses_domiknows_runtime():
    good = (
        {"name": "pick", "params": {"target_entity_name": "apple"}},
        {"name": "place", "params": {"target_container_name": "bowl"}},
    )
    bad = ({"name": "pick", "params": {}},)
    world = build_vlabench_world_graph("test_rl_pipeline_world")
    runtime = build_constraint_runtime(world, max_entities=4, max_operations=3, name_prefix="test_rl_pipeline")
    planner = FakePlanner(good, bad)
    optimizer = torch.optim.SGD(planner.parameters(), lr=0.1)
    before = planner.preference.item()
    metrics = train_planner_reinforcement_epoch(
        planner, [FakeExample(good)], optimizer, runtime,
        num_samples=4, estimator="reinforce",
    )
    assert metrics["reward"] == 0.5
    assert planner.preference.item() > before


class FakeProcessor:
    tokenizer = SimpleNamespace(pad_token_id=0)

    def apply_chat_template(self, *_args, **_kwargs):
        return "prompt"

    def __call__(self, **_kwargs):
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}

class FakeGenerationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.preference = torch.nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(hidden_size=4)
        self.calls = 0

    def forward(self, input_ids, **_kwargs):
        self.calls += 1
        hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4, device=input_ids.device)
        hidden[..., 0] = self.preference
        return SimpleNamespace(hidden_states=(hidden,))


def test_qwen_sample_is_rescored_with_differentiable_log_probability():
    world = build_vlabench_world_graph("test_compact_qwen_world")
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="test_compact_qwen")
    planner = QwenVLPlanner(FakeGenerationModel(), FakeProcessor(), runtime.vocabulary)
    _plan, logprob = planner.sample_with_logprob(
        instruction="test",
        images=[],
        entity_table=["apple", "bowl"],
        dfa=runtime.dfa,
        world=world,
        max_steps=runtime.max_tokens,
    )
    assert logprob.requires_grad
    assert planner.model.calls == 1
    (-logprob).backward()
    assert planner.output.weight.grad is not None
    assert planner.graph_decoder.weight_hh_l0.grad is not None


def test_qwen_teacher_forcing_encodes_context_once_and_uses_prefixes():
    world = build_vlabench_world_graph("test_teacher_forced_qwen_world")
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="test_teacher_forced_qwen")
    planner = QwenVLPlanner(FakeGenerationModel(), FakeProcessor(), runtime.vocabulary)
    context = {"instruction": "test", "images": [], "entity_table": ["apple", "bowl"]}
    first = torch.tensor([[runtime.vocabulary.eos_label, 1, 2]])
    second = torch.tensor([[runtime.vocabulary.eos_label, 2, 1]])
    first_logits = planner.sequence_logits(context, first)
    second_logits = planner.sequence_logits(context, second)
    assert planner.model.calls == 2
    assert first_logits.shape == second_logits.shape == (1, 3, runtime.vocabulary.label_count)
    assert not torch.allclose(first_logits[:, -1], second_logits[:, -1])


def test_standalone_agent_uses_bounded_report_defaults():
    args = build_parser().parse_args([
        "train-agent",
        "--two-stage",
        "--planning-dir", "planning",
        "--control-source", "control",
        "--output", "output",
    ])
    assert args.planner_decoder_hidden_dim == 512
    assert args.controller_warmup_steps == 20000
    assert args.validation_limit == 32
    assert args.rl_epochs == 3
    assert args.rl_rounds_per_epoch == 10
    assert args.rl_min_success_rate == pytest.approx(0.10)
    assert args.rl_min_successful_tasks == 3
    assert args.rl_max_ik_truncation_rate == pytest.approx(0.25)
    assert args.rl_preflight_min_success_rate == pytest.approx(0.0)
    assert args.rl_preflight_min_successful_tasks == 0
    assert args.rl_preflight_min_positive_return_rate == pytest.approx(0.01)
    assert args.rl_preflight_max_ik_truncation_rate == pytest.approx(0.50)
    assert args.eval_rollouts_per_task == 1
    assert args.ik_tolerance == pytest.approx(5e-3)
    assert args.max_consecutive_ik_rejections == 3


def test_standalone_reinforcement_gates_use_fixed_seed_evaluation():
    metrics = {
        "training": {
            "success_rate": 0.75,
            "successful_task_count": 8,
            "ik_truncation_rate": 0.05,
        },
        "evaluation": {
            "success_rate": 0.0,
            "successful_task_count": 0,
            "return": 0.0,
            "steps": 200.0,
            "execution_complete_rate": 0.1,
            "ik_truncation_rate": 0.9,
        },
    }
    assert not reinforcement_checkpoint_eligible(
        metrics,
        min_success_rate=0.10,
        min_successful_tasks=3,
        max_ik_truncation_rate=0.25,
    )
    assert not reinforcement_preflight_eligible(metrics["evaluation"])
    assert not reinforcement_preflight_eligible({
        "success_rate": 0.0,
        "successful_task_count": 0,
        "ik_truncation_rate": 16 / 30,
    })
    assert reinforcement_preflight_eligible({
        "success_rate": 0.0,
        "successful_task_count": 0,
        "positive_return_rate": 0.1,
        "ik_truncation_rate": 0.50,
    })
    assert not reinforcement_preflight_eligible({
        "success_rate": 0.0,
        "successful_task_count": 0,
        "positive_return_rate": 0.0,
        "ik_truncation_rate": 0.10,
    })
    better = {
        "success_rate": 0.20,
        "successful_task_count": 3,
        "return": 0.1,
        "steps": 100.0,
        "execution_complete_rate": 0.8,
        "ik_truncation_rate": 0.2,
    }
    assert reinforcement_checkpoint_eligible(
        better,
        min_success_rate=0.10,
        min_successful_tasks=3,
        max_ik_truncation_rate=0.25,
    )
    assert reinforcement_selection_key(better) > reinforcement_selection_key(
        metrics["evaluation"]
    )


def test_reinforcement_round_metrics_aggregate_repeated_tasks():
    rounds = [
        {"per_task": {"select_book": {
            "episodes": 2, "successes": 1, "valid_rate": 1.0,
            "positive_return_rate": 0.5,
            "return": 0.5, "steps": 10.0, "ik_failures": 4,
            "ik_recoveries": 1, "ik_truncation_rate": 0.5,
            "execution_complete_rate": 0.5,
        }}},
        {"per_task": {"select_book": {
            "episodes": 1, "successes": 1, "valid_rate": 0.0,
            "positive_return_rate": 1.0,
            "return": 0.2, "steps": 4.0, "ik_failures": 2,
            "ik_recoveries": 1, "ik_truncation_rate": 0.0,
            "execution_complete_rate": 1.0,
        }}},
    ]

    metrics = _aggregate_task_metrics(rounds)["select_book"]

    assert metrics["episodes"] == 3
    assert metrics["success_rate"] == pytest.approx(2 / 3)
    assert metrics["positive_return_rate"] == pytest.approx(2 / 3)
    assert metrics["valid_rate"] == pytest.approx(2 / 3)
    assert metrics["return"] == pytest.approx(0.4)
    assert metrics["ik_failures"] == 6
    assert metrics["ik_recovery_rate"] == pytest.approx(2 / 6)
    assert metrics["execution_complete_rate"] == pytest.approx(2 / 3)


def test_reinforcement_resume_supports_round_and_legacy_epoch_checkpoints():
    partial = {
        "stage": "reinforcement",
        "epoch": 1,
        "next_round": 4,
        "metrics": {"rounds": [{"episodes": 2}] * 4},
    }
    assert _reinforcement_resume_position(partial, 10) == (
        1, 4, partial["metrics"]["rounds"]
    )
    complete_rounds = {**partial, "next_round": 10, "metrics": {"rounds": [{}] * 10}}
    assert _reinforcement_resume_position(complete_rounds, 10)[0:2] == (1, 10)
    assert _reinforcement_resume_position(
        {"stage": "reinforcement", "epoch": 1}, 10
    ) == (2, 0, [])
    with pytest.raises(ValueError, match="inconsistent"):
        _reinforcement_resume_position({**partial, "next_round": 5}, 10)


class TinyCompactPlanner(torch.nn.Module):
    def __init__(self, vocabulary):
        super().__init__()
        self.vocabulary = vocabulary
        self.preference = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, _contains, _context, target_labels):
        return self.preference.expand(len(target_labels), self.vocabulary.label_count)

    def sample_with_logprob(self, **_kwargs):
        positive = self.calls % 2 == 0
        self.calls += 1
        logprob = torch.nn.functional.logsigmoid(self.preference if positive else -self.preference)
        return [
            {"name": "pick", "params": {"target_entity_name": 0}},
            {"name": "place", "params": {"target_container_name": 1}},
        ], logprob

    def supervised_loss(self, **_kwargs):
        return (self.preference - 1.0).square()


class ModeCheckedReplayPlanner(TinyCompactPlanner):
    def replay_labels_logprob(self, _context, _labels, _dfa, *, max_steps):
        assert max_steps > 0
        if not self.training:
            raise RuntimeError("replay forward was executed in evaluation mode")
        return torch.nn.functional.logsigmoid(self.preference)


class EvaluationPlanner(torch.nn.Module):
    def __init__(self, plan):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor(0.0))
        self.plan = plan

    def generate_plan(self, **_kwargs):
        return self.plan


def test_evaluators_restore_the_callers_training_mode():
    world = build_vlabench_world_graph("test_evaluation_mode_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_evaluation_mode"
    )
    plan = (
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}},
    )
    planner = EvaluationPlanner(plan)
    planner.train()
    evaluate_planner(planner, [FakeExample(plan)], runtime)
    assert planner.training

    controller = MultiViewController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    controller.train()
    batch = {
        "images": torch.rand(1, 2, 1, 3, 16, 16),
        "state": torch.rand(1, 2, 7),
        "task_index": torch.zeros(1, dtype=torch.long),
        "actions": torch.rand(1, 1, 7),
    }
    evaluate_controller(controller, [batch], device="cpu", max_batches=1)
    assert controller.training


def test_stage2_planner_replay_enters_training_mode_after_evaluation():
    world = build_vlabench_world_graph("test_replay_training_mode_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_replay_training_mode"
    )
    planner = ModeCheckedReplayPlanner(runtime.vocabulary)
    controller = MultiViewController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    program = create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.1),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.1),
        env_factory=lambda **_kwargs: None,
        supervised_weight=0.0,
    )
    replay = PlannerReplayDecision({}, (1,), None, runtime.max_tokens)
    planner.eval()
    loss = program._update_planner([
        JointEpisode([replay], [], 1.0, True, True, 1),
        JointEpisode([replay], [], 0.0, False, True, 1),
    ])
    assert planner.training
    assert torch.isfinite(torch.tensor(loss))


class TinySolverPlanner(torch.nn.Module):
    def __init__(self, max_tokens, label_count):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(max_tokens, label_count))

    def forward(self, _contains, _context, target_labels):
        return self.logits[: len(target_labels)]


def test_stage1_solver_program_performs_supervised_update():
    world = build_vlabench_world_graph("test_stage1_train_world")
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="test_stage1_train")
    reference = (
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}},
    )
    planner = TinySolverPlanner(runtime.max_tokens, runtime.vocabulary.label_count)
    program = create_stage1_program(runtime, planner)
    before = planner.logits.detach().clone()
    program.train(
        prepare_planner_program_examples([FakeExample(reference)], runtime),
        valid_set=None,
        test_set=None,
        train_epoch_num=1,
        Optim=lambda params: torch.optim.SGD(params, lr=0.1),
        test_every_epoch=False,
    )
    assert not torch.equal(planner.logits, before)


def test_program_types_share_exact_planner_head():
    from domiknows.program import SolverPOIProgram
    from domiknows.reinforcement.reinforcement_program import ReinforcementProgram

    world = build_vlabench_world_graph("test_program_types_world")
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="test_program_types")
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1)
    stage1 = create_stage1_program(runtime, planner)
    stage2 = create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.1),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.1),
        env_factory=lambda **_kwargs: None,
    )
    assert isinstance(stage1, SolverPOIProgram)
    assert isinstance(stage2, ReinforcementProgram)
    assert isinstance(stage2, VLABenchHierarchicalReinforcementProgram)
    assert stage1.planner_head is stage2.planner_head is planner


def test_stage2_supervised_anchor_contributes_when_returns_have_no_advantage():
    world = build_vlabench_world_graph("test_anchor_world")
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="test_anchor")
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1)
    optimizer = torch.optim.SGD(planner.parameters(), lr=0.1)
    reference = (
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}},
    )
    program = create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=optimizer,
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.1),
        env_factory=lambda **_kwargs: None,
        supervised_examples=[FakeExample(reference, entities=("apple", "bowl"))],
        supervised_weight=0.1,
    )
    logprob = torch.nn.functional.logsigmoid(planner.preference)
    episodes = [JointEpisode([logprob], [], 1.0, True, True, 1)]
    before = planner.preference.item()
    program._update_planner(episodes)
    assert planner.preference.item() > before


def test_controller_actor_critic_gae_and_ppo_contracts():
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=2, max_views=1)
    inputs = (torch.rand(1, 2, 1, 3, 16, 16), torch.rand(1, 2, 7), torch.zeros(1, dtype=torch.long))
    actions, logprob, entropy, value = controller.sample_action_chunk(*inputs)
    assert actions.shape == (1, 2, 7)
    assert logprob.shape == entropy.shape == (1, 2)
    assert value.shape == (1,)
    evaluated, _, evaluated_value = controller.evaluate_action_chunk(*inputs, actions.detach())
    torch.testing.assert_close(evaluated, logprob, rtol=1e-4, atol=1e-4)
    previous = torch.cat((inputs[1][:, -1:, :6], actions[:, :-1, :6]), dim=1)
    delta = actions[..., :6] - previous
    angular_delta = torch.atan2(torch.sin(delta[..., 3:]), torch.cos(delta[..., 3:]))
    bounded_delta = torch.cat((delta[..., :3], angular_delta), -1)
    scales = torch.tensor(controller.pose_step_scale).view(1, 1, 6)
    assert bool((bounded_delta.abs() <= scales + 1e-6).all())
    loss = ppo_clipped_loss(evaluated.sum(1), logprob.detach().sum(1), torch.ones(1)) + evaluated_value.mean()
    loss.backward()
    assert controller.value_head.weight.grad is not None
    advantages, returns = generalized_advantage_estimate([0.1, 1.0], [0.2, 0.3], [False, True])
    assert len(advantages) == len(returns) == 2
    assert returns[-1] == pytest.approx(1.0)


def test_controller_critic_is_bounded():
    controller = MultiViewController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    controller.value_head.weight.data.fill_(1.0e6)
    controller.value_head.bias.data.fill_(1.0e6)
    output = controller.policy(
        torch.rand(1, 2, 1, 3, 16, 16),
        torch.rand(1, 2, 7),
        torch.zeros(1, dtype=torch.long),
    )
    assert torch.isfinite(output.value).all()
    assert bool((output.value.abs() <= 1.0).all())


def test_controller_features_are_conditioned_on_active_graph_operation():
    controller = MultiViewController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    images = torch.rand(1, 2, 1, 3, 16, 16)
    state = torch.rand(1, 2, 7)
    task = torch.zeros(1, dtype=torch.long)
    pick = controller._features(images, state, task, torch.tensor([[1, 1, 1]]))
    place = controller._features(images, state, task, torch.tensor([[2, 2, 2]]))
    assert not torch.allclose(pick, place)


def test_controller_pose_chunk_is_local_cumulative_and_uses_physical_noise_scales():
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=3, max_views=1)
    controller.policy_head.weight.data.zero_()
    controller.policy_head.bias.data.zero_()
    state = torch.zeros(1, 2, 7)
    state[0, -1, :6] = torch.tensor([0.2, -0.1, 0.4, 3.13, 0.2, -0.3])
    inputs = (torch.rand(1, 2, 1, 3, 16, 16), state, torch.zeros(1, dtype=torch.long))
    output = controller.policy(*inputs)
    torch.testing.assert_close(output.pose_mean, state[:, -1:, :6].expand(-1, 3, -1))
    torch.testing.assert_close(output.pose_std[0, 0], torch.tensor(controller.exploration_std))

    # A saturated x/roll increment remains local per action and accumulates
    # across the chunk; wrapped angles stay on the principal branch.
    bias = controller.policy_head.bias.view(3, 7)
    bias.data[:, 0] = 100.0
    bias.data[:, 3] = 100.0
    moved = controller.policy(*inputs).pose_mean[0]
    torch.testing.assert_close(
        torch.diff(torch.cat((state[0, -1, 0:1], moved[:, 0]))),
        torch.full((3,), controller.pose_step_scale[0]),
    )
    assert bool((moved[:, 3].abs() <= torch.pi).all())


def test_controller_rejects_nonfinite_policy_before_cuda_distribution_sampling():
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1)
    controller.policy_head.bias.data.fill_(float("nan"))
    inputs = (torch.rand(1, 2, 1, 3, 16, 16), torch.rand(1, 2, 7), torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match="non-finite"):
        controller.sample_action_chunk(*inputs)


def test_ppo_log_ratio_is_finite_for_extreme_likelihood_change():
    new = torch.tensor([1.0e4], requires_grad=True)
    loss = ppo_clipped_loss(new, torch.tensor([-1.0e4]), torch.tensor([-1.0]))
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(float(torch.exp(torch.tensor(2.0))))
    loss.backward()
    assert torch.isfinite(new.grad).all()


class FakeRobot:
    def get_qpos_from_ee_pos(self, *, physics, pos, quat, **kwargs):
        assert physics is not None and len(pos) == 3 and len(quat) == 4
        assert kwargs.get("tol") == pytest.approx(1e-3)
        assert kwargs.get("max_steps") == 200
        return True, np.arange(7, dtype=np.float64)


class FailedRobot(FakeRobot):
    def get_qpos_from_ee_pos(self, *, physics, pos, quat, **kwargs):
        return False, np.arange(7, dtype=np.float64)


class RecoveringRobot(FakeRobot):
    def __init__(self):
        self.calls = 0

    def get_qpos_from_ee_pos(self, *, physics, pos, quat, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return False, np.arange(7, dtype=np.float64)
        return super().get_qpos_from_ee_pos(
            physics=physics, pos=pos, quat=quat, **kwargs
        )


class RecordingRobot(FakeRobot):
    def __init__(self):
        self.positions = []

    def get_qpos_from_ee_pos(self, *, physics, pos, quat, **kwargs):
        self.positions.append(np.asarray(pos, dtype=np.float64).copy())
        return super().get_qpos_from_ee_pos(
            physics=physics, pos=pos, quat=quat, **kwargs
        )


def test_ee_action_conversion_uses_two_finger_gripper():
    env = SimpleNamespace(
        robot=FakeRobot(),
        physics=object(),
        action_spec=SimpleNamespace(minimum=np.full(9, -0.5), maximum=np.full(9, 0.5)),
    )
    opened = ee_action_to_env_action(env, [0, 0, 0, 0, 0, 0, 1])
    closed = ee_action_to_env_action(env, [0, 0, 0, 0, 0, 0, 0])
    assert opened.shape == (9,)
    assert np.all(opened <= 0.5) and np.all(opened >= -0.5)
    assert np.allclose(opened[-2:], 0.04)
    assert np.allclose(closed[-2:], 0.0)
    with pytest.raises(ValueError, match="inverse-kinematics"):
        ee_action_to_env_action(SimpleNamespace(robot=FailedRobot(), physics=object()), np.zeros(7))


def test_vlabench_quaternion_convention_and_observed_pose_conversion():
    half = np.sqrt(0.5)
    quaternion = euler_to_quaternion(np.pi / 2.0, 0.0, 0.0)
    assert quaternion == pytest.approx([half, half, 0.0, 0.0])
    assert quaternion_to_euler(quaternion) == pytest.approx([np.pi / 2.0, 0.0, 0.0])

    observed = _observation_state({
        "ee_state": np.asarray([0.1, 0.2, 0.3, *quaternion, 1.0]),
        "q_state": np.full(7, 99.0),
    })
    assert observed == pytest.approx([0.1, 0.2, 0.3, np.pi / 2.0, 0.0, 0.0, 1.0])


def test_vlabench_controller_frame_roundtrip_and_input_conversion():
    frame = np.asarray([0.0, -0.4, 0.78])
    world_state = np.asarray([0.1, -0.2, 1.2, 0.2, -0.3, 0.4, 1.0])
    robot_state = world_to_robot_ee_state(world_state, frame)
    assert robot_state == pytest.approx([0.1, 0.2, 0.42, 0.2, -0.3, 0.4, 1.0])
    assert robot_to_world_ee_action(robot_state, frame) == pytest.approx(world_state)

    observation = {
        "rgb": np.zeros((3, 8, 8, 3), dtype=np.uint8),
        "ee_state": world_state,
    }
    _images, state, _task, _plan = _controller_inputs(
        [observation], 0, "cpu", robot_frame=frame
    )
    assert state[0, -1].numpy() == pytest.approx(robot_state)


def test_robot_frame_position_uses_official_environment_accessor():
    env = SimpleNamespace(
        get_robot_frame_position=lambda: np.asarray([0.1, -0.5, 0.8]),
        robot=SimpleNamespace(robot_config={"position": [9.0, 9.0, 9.0]}),
    )
    assert robot_frame_position(env) == pytest.approx([0.1, -0.5, 0.8])


def test_ee_action_safety_envelope_limits_cartesian_and_wrapped_rotation_steps():
    current = np.asarray([0.0, 0.4, 0.2, 3.10, 0.0, -3.10, 0.0])
    target = np.asarray([1.0, -1.0, 0.9, -3.10, 1.0, 3.10, 1.0])
    bounded = bound_ee_action(target, current)
    assert np.max(np.abs(bounded[:3] - current[:3])) <= 0.02 + 1e-9
    angular_delta = (bounded[3:6] - current[3:6] + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(angular_delta)) <= 0.10 + 1e-9
    assert bounded[6] == 1.0


class FakeTimeStep:
    def __init__(self, terminal=False):
        self.terminal = terminal

    def last(self):
        return self.terminal


class FakeSimulator:
    def __init__(self, success=True):
        self.success = success
        self.count = 0
        self.robot = FakeRobot()
        self.physics = object()
        self.task = SimpleNamespace(
            entities={"apple": SimpleNamespace(), "bowl": SimpleNamespace()},
            get_instruction=lambda: "Put the apple in the bowl.",
            should_terminate_episode=lambda _physics: self.success,
        )

    def reset(self):
        self.count = 0
        return FakeTimeStep(False)

    def get_observation(self, require_pcd=False):
        assert not require_pcd
        return {"rgb": np.zeros((1, 16, 16, 3), dtype=np.uint8), "ee_state": np.zeros(7, dtype=np.float32)}

    def step(self, action):
        assert np.asarray(action).shape == (9,)
        self.count += 1
        return FakeTimeStep(self.success)

    def get_task_progress(self):
        return float(self.success and self.count > 0)

    def get_intention_score(self, **_kwargs):
        return float(self.success and self.count > 0)

    def close(self):
        pass


class InvalidCompactPlanner(TinyCompactPlanner):
    def sample_with_logprob(self, **_kwargs):
        self.calls += 1
        return [{"name": "pick", "params": {}}], torch.nn.functional.logsigmoid(self.preference)


def _joint_program(runtime, planner, controller, factory, *, num_samples=4):
    return create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.1),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.01),
        env_factory=factory,
        execute_horizon=1,
        max_steps=1,
        num_samples=num_samples,
        ppo_epochs=1,
        supervised_weight=0.0,
        controller_bc_weight=0.0,
    )


def test_joint_rewards_telescope_and_planner_uses_return_to_go():
    world = build_vlabench_world_graph("test_joint_reward_world")
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="test_joint_reward")
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1)
    program = _joint_program(runtime, planner, controller, lambda **_kwargs: FakeSimulator(success=True))
    episode = program.collect_episode({"task": "select_book"})
    assert episode.total_return == pytest.approx(0.95)
    assert sum(item.reward for item in episode.controller) == pytest.approx(episode.total_return)
    assert episode.planner_returns == pytest.approx([episode.total_return])
    assert episode.controller[0].plan_context.tolist() == [[1, 0, 1]]
    assert torch.max(torch.abs(episode.controller[0].actions[0, 0, :3])).item() <= 0.05 + 1e-6
    assert torch.isfinite(episode.controller[0].old_logprob)


def test_rollout_stores_sampled_policy_action_before_execution_safety_transform():
    world = build_vlabench_world_graph("test_policy_action_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_policy_action"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)

    class LatentActionController(MultiViewController):
        def sample_action_chunk(self, images, state, task_index):
            batch = images.shape[0]
            action = torch.tensor(
                [0.8, -0.8, 0.8, 1.0, -1.0, 1.0, 1.0],
                device=images.device,
            ).view(1, 1, 7).expand(batch, 1, 7)
            return (
                action,
                torch.full((batch, 1), -3.25, device=images.device),
                torch.zeros((batch, 1), device=images.device),
                torch.full((batch,), 0.125, device=images.device),
            )

        def evaluate_action_chunk(self, *_args, **_kwargs):
            raise AssertionError("rollout collection must not relabel transformed actions")

    controller = LatentActionController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    program = _joint_program(
        runtime, planner, controller, lambda **_kwargs: FakeSimulator(success=True),
        num_samples=1,
    )
    episode = program.collect_episode({"task": "select_book"})
    transition = episode.controller[0]
    torch.testing.assert_close(
        transition.actions[0, 0],
        torch.tensor([0.8, -0.8, 0.8, 1.0, -1.0, 1.0, 1.0]),
    )
    assert transition.old_logprob.item() == pytest.approx(-3.25)
    assert transition.old_value.item() == pytest.approx(0.125)


def test_feasibility_credit_targets_rejected_action_not_successful_prefix():
    world = build_vlabench_world_graph("test_feasibility_index_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_feasibility_index"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)

    class IndexedController(MultiViewController):
        def sample_action_chunk(self, images, state, task_index):
            batch = images.shape[0]
            actions = torch.zeros(batch, 4, 7, device=images.device)
            logprobs = torch.tensor(
                [[-1.0, -2.0, -3.0, -4.0]], device=images.device
            ).expand(batch, -1)
            return (
                actions,
                logprobs,
                torch.zeros_like(logprobs),
                torch.zeros(batch, device=images.device),
            )

    class ThirdActionFailedRobot(FakeRobot):
        def __init__(self):
            self.calls = 0

        def get_qpos_from_ee_pos(self, *, physics, pos, quat, **kwargs):
            self.calls += 1
            if self.calls <= 2:
                return super().get_qpos_from_ee_pos(
                    physics=physics, pos=pos, quat=quat, **kwargs
                )
            return False, np.arange(7, dtype=np.float64)

    controller = IndexedController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=4, max_views=1
    )
    simulator = FakeSimulator(success=False)
    simulator.robot = ThirdActionFailedRobot()
    program = create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.1),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.01),
        env_factory=lambda **_kwargs: simulator,
        execute_horizon=4,
        max_steps=4,
        num_samples=1,
        ppo_epochs=1,
        supervised_weight=0.0,
        controller_bc_weight=0.0,
        max_consecutive_ik_rejections=1,
    )

    episode = program.collect_episode({"task": "select_book"})
    transition = episode.controller[0]

    assert transition.executed == 2
    assert transition.old_logprob.item() == pytest.approx(-3.0)
    assert transition.feasibility_cost == pytest.approx(1.0)
    assert transition.feasibility_index == 2
    assert transition.old_feasibility_logprob.item() == pytest.approx(-3.0)


def test_feasibility_gradient_changes_only_rejected_action_likelihood():
    world = build_vlabench_world_graph("test_feasibility_gradient_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_feasibility_gradient"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)

    class IndexedLogprobController(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logprobs = torch.nn.Parameter(torch.zeros(4))
            self.value = torch.nn.Parameter(torch.zeros(()))

        def evaluate_action_chunk(self, images, state, task_index, actions, plan_context=None):
            return (
                self.logprobs.unsqueeze(0),
                torch.zeros(1, 4),
                self.value.unsqueeze(0),
            )

    controller = IndexedLogprobController()
    program = create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.1),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.01),
        env_factory=lambda **_kwargs: None,
        ppo_epochs=1,
        value_weight=0.0,
        entropy_weight=0.0,
        feasibility_weight=0.05,
        supervised_weight=0.0,
        controller_bc_weight=0.0,
    )
    transition = ControllerTransition(
        images=torch.zeros(1, 1),
        state=torch.zeros(1, 1, 7),
        task_index=torch.zeros(1, dtype=torch.long),
        actions=torch.zeros(1, 4, 7),
        old_logprob=torch.zeros(()),
        old_value=torch.zeros(()),
        reward=1.0,
        done=True,
        executed=2,
        feasibility_cost=1.0,
        feasibility_index=2,
        old_feasibility_logprob=torch.zeros(()),
        advantage=0.0,
        return_value=1.0,
    )
    episode = JointEpisode([], [transition], 1.0, False, True, 2)

    program._update_controller([episode])

    assert controller.logprobs[2].item() < 0.0
    torch.testing.assert_close(
        controller.logprobs[[0, 1, 3]], torch.zeros(3)
    )


def test_single_positive_transition_preserves_ppo_policy_signal():
    world = build_vlabench_world_graph("test_single_positive_transition_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_single_positive_transition"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)

    class SingleLogprobController(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logprob = torch.nn.Parameter(torch.zeros(()))
            self.value = torch.nn.Parameter(torch.zeros(()))

        def evaluate_action_chunk(self, images, state, task_index, actions, plan_context=None):
            return (
                self.logprob.reshape(1, 1),
                torch.zeros(1, 1),
                self.value.reshape(1),
            )

    controller = SingleLogprobController()
    program = create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.1),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.01),
        env_factory=lambda **_kwargs: None,
        ppo_epochs=1,
        value_weight=0.0,
        entropy_weight=0.0,
        feasibility_weight=0.0,
        supervised_weight=0.0,
        controller_bc_weight=0.0,
    )
    transition = ControllerTransition(
        images=torch.zeros(1, 1),
        state=torch.zeros(1, 1, 7),
        task_index=torch.zeros(1, dtype=torch.long),
        actions=torch.zeros(1, 1, 7),
        old_logprob=torch.zeros(()),
        old_value=torch.zeros(()),
        reward=1.0,
        done=True,
        executed=1,
        advantage=1.0,
        return_value=1.0,
    )
    episode = JointEpisode([], [transition], 1.0, True, True, 1)

    program._update_controller([episode])

    assert controller.logprob.item() > 0.0
    assert program.last_controller_update["actor_update_attempted"] is True
    assert program.last_controller_update["rolled_back"] is False


def test_mixed_return_batch_uses_zero_return_rollout_as_negative_evidence():
    world = build_vlabench_world_graph("test_mixed_return_evidence_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_mixed_return_evidence"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)

    class IndexedEpisodeController(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logprobs = torch.nn.Parameter(torch.zeros(2))
            self.value = torch.nn.Parameter(torch.zeros(()))

        def evaluate_action_chunk(self, images, state, task_index, actions, plan_context=None):
            index = int(task_index.reshape(-1)[0])
            return (
                self.logprobs[index].reshape(1, 1),
                torch.zeros(1, 1),
                self.value.reshape(1),
            )

    controller = IndexedEpisodeController()
    program = create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.1),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.01),
        env_factory=lambda **_kwargs: None,
        ppo_epochs=1,
        value_weight=0.0,
        entropy_weight=0.0,
        feasibility_weight=0.0,
        supervised_weight=0.0,
        controller_bc_weight=0.0,
    )

    def transition(task_index, advantage, reward):
        return ControllerTransition(
            images=torch.zeros(1, 1),
            state=torch.zeros(1, 1, 7),
            task_index=torch.tensor([task_index]),
            actions=torch.zeros(1, 1, 7),
            old_logprob=torch.zeros(()),
            old_value=torch.zeros(()),
            reward=reward,
            done=True,
            executed=1,
            advantage=advantage,
            return_value=reward,
        )

    positive = JointEpisode([], [transition(0, 1.0, 1.0)], 1.0, True, True, 1)
    unsuccessful = JointEpisode([], [transition(1, 0.0, 0.0)], 0.0, False, True, 1)

    program._update_controller([positive, unsuccessful])

    assert controller.logprobs[0].item() > 0.0
    assert controller.logprobs[1].item() < 0.0


def test_recoverable_simulator_initialization_failure_retries_one_rollout():
    world = build_vlabench_world_graph("test_simulator_retry_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_simulator_retry"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1)
    physics_error = type(
        "PhysicsError",
        (RuntimeError,),
        {"__module__": "dm_control.mujoco.engine"},
    )
    calls = {"count": 0}

    def factory(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise physics_error("Physics state is invalid: mjWARN_BADQACC")
        return FakeSimulator(success=True)

    program = _joint_program(runtime, planner, controller, factory, num_samples=1)
    episode = program.collect_episode({"task": "select_book"})
    assert calls["count"] == 2
    assert episode.valid and episode.success


def test_online_controller_converts_dataset_frame_target_to_world_before_ik():
    world = build_vlabench_world_graph("test_controller_robot_frame_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_controller_robot_frame"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    simulator = FakeSimulator(success=True)
    simulator.robot = RecordingRobot()
    simulator.get_robot_frame_position = lambda: np.asarray([0.0, -0.4, 0.78])
    world_position = np.asarray([0.2, -0.1, 1.0])
    robot_position = np.asarray([0.2, 0.3, 0.22])
    simulator.get_observation = lambda require_pcd=False: {
        "rgb": np.zeros((1, 16, 16, 3), dtype=np.uint8),
        "ee_state": np.asarray([*world_position, 1.0, 0.0, 0.0, 0.0, 1.0]),
    }

    def hold_in_robot_frame(images, state, task_index, plan_context=None):
        assert state[0, -1, :3].cpu().numpy() == pytest.approx(robot_position)
        actions = torch.tensor(
            [[[*robot_position, 0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32
        )
        zeros = torch.zeros((1, 1), dtype=torch.float32)
        return actions, zeros, zeros, torch.zeros(1, dtype=torch.float32)

    controller.sample_action_chunk = hold_in_robot_frame
    program = _joint_program(
        runtime, planner, controller, lambda **_kwargs: simulator, num_samples=1
    )

    episode = program.collect_episode({"task": "select_book"})

    assert episode.success
    assert len(simulator.robot.positions) == 1
    # A robot-frame hold target must become the exact observed world pose.
    # Treating it directly as world-frame would instead trigger a 2 cm move.
    assert simulator.robot.positions[0] == pytest.approx(world_position)


def test_invalid_plan_never_executes_controller_or_environment():
    world = build_vlabench_world_graph("test_joint_invalid_world")
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="test_joint_invalid")
    planner = InvalidCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1)
    simulator = FakeSimulator(success=True)
    program = _joint_program(runtime, planner, controller, lambda **_kwargs: simulator)
    episode = program.collect_episode({"task": "select_book"})
    assert not episode.valid
    assert episode.total_return == 0.0
    assert episode.controller == []
    assert simulator.count == 0
    assert planner.calls == program.num_samples == 4
    assert episode.planner_returns == [0.0] * 4


def test_failed_ik_action_receives_zero_and_is_not_executed():
    world = build_vlabench_world_graph("test_joint_failed_ik_world")
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="test_joint_failed_ik")
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1)
    simulator = FakeSimulator(success=True)
    simulator.robot = FailedRobot()
    program = _joint_program(runtime, planner, controller, lambda **_kwargs: simulator)
    episode = program.collect_episode({"task": "select_book"})
    assert episode.valid
    assert episode.total_return == 0.0
    assert simulator.count == 0
    assert len(episode.controller) == 3
    assert all(item.executed == 0 for item in episode.controller)
    assert all(item.feasibility_cost == 1.0 for item in episode.controller)
    assert all(item.feasibility_index == 0 for item in episode.controller)
    assert all(item.old_logprob.item() == pytest.approx(0.0) for item in episode.controller)
    assert all(torch.isfinite(item.old_feasibility_logprob) for item in episode.controller)
    assert episode.ik_failures == 12
    assert episode.termination_reason == "ik_failure"


def test_flat_progress_uses_episode_phase_to_advance_operation_context():
    world = build_vlabench_world_graph("test_phase_context_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_phase_context"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)

    class RecordingController(MultiViewController):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.contexts = []

        def sample_action_chunk(self, images, state, task_index, plan_context=None):
            self.contexts.append(plan_context.detach().clone())
            return super().sample_action_chunk(images, state, task_index, plan_context)

    controller = RecordingController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    program = create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.1),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.01),
        env_factory=lambda **_kwargs: FakeSimulator(success=False),
        execute_horizon=1,
        max_steps=4,
        num_samples=1,
        ppo_epochs=1,
        supervised_weight=0.0,
        controller_bc_weight=0.0,
    )

    episode = program.collect_episode({"task": "select_book"})

    assert episode.steps == 4
    assert len(controller.contexts) == 4
    assert torch.equal(controller.contexts[0], controller.contexts[1])
    assert not torch.equal(controller.contexts[1], controller.contexts[2])
    assert torch.equal(controller.contexts[2], controller.contexts[3])


def test_geometric_progress_does_not_switch_pick_to_place_before_grasp():
    world = build_vlabench_world_graph("test_geometric_phase_context_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_geometric_phase_context"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)

    class RecordingController(MultiViewController):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.contexts = []

        def sample_action_chunk(self, images, state, task_index, plan_context=None):
            self.contexts.append(plan_context.detach().clone())
            return super().sample_action_chunk(images, state, task_index, plan_context)

    class GeometricProgressSimulator(FakeSimulator):
        def __init__(self):
            super().__init__(success=False)
            self.task.target_entity = "apple"
            self.task.entities["apple"] = SimpleNamespace(
                get_xpos=lambda _physics: np.asarray([1.0, 0.0, 0.0])
            )

    controller = RecordingController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    program = create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.1),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.01),
        env_factory=lambda **_kwargs: GeometricProgressSimulator(),
        execute_horizon=1,
        max_steps=4,
        num_samples=1,
        ppo_epochs=1,
        supervised_weight=0.0,
        controller_bc_weight=0.0,
    )

    episode = program.collect_episode({"task": "select_book"})

    assert episode.steps == 4
    assert len(controller.contexts) == 4
    assert all(torch.equal(controller.contexts[0], context) for context in controller.contexts)


def test_ik_recovery_retries_a_smaller_bounded_delta():
    world = build_vlabench_world_graph("test_joint_recovered_ik_world")
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="test_joint_recovered_ik")
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1)
    simulator = FakeSimulator(success=True)
    simulator.robot = RecoveringRobot()
    program = _joint_program(runtime, planner, controller, lambda **_kwargs: simulator)
    episode = program.collect_episode({"task": "select_book"})
    assert episode.success and episode.valid
    assert episode.ik_failures == 1
    assert episode.ik_recoveries == 1
    assert episode.termination_reason == "success"
    assert episode.controller[0].feasibility_cost == 0.0
    assert episode.controller[0].feasibility_index is None


def test_online_controller_uses_language_task_id_not_skill_pattern():
    world = build_vlabench_world_graph("test_language_control_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_language_control"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)

    class RecordingController(MultiViewController):
        seen_task_index = None

        def sample_action_chunk(self, images, state, task_index):
            self.seen_task_index = int(task_index.item())
            return super().sample_action_chunk(images, state, task_index)

    controller = RecordingController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    program = create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.1),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=0.01),
        env_factory=lambda **_kwargs: FakeSimulator(success=True),
        controller_task_instructions={73: "Put the apple in the bowl."},
        execute_horizon=1,
        max_steps=1,
        num_samples=1,
        ppo_epochs=1,
        supervised_weight=0.0,
        controller_bc_weight=0.0,
    )
    episode = program.collect_episode({"task": "select_book"})
    assert episode.success
    assert controller.seen_task_index == 73


def test_zero_return_rollout_does_not_apply_ppo_or_entropy_to_actor():
    world = build_vlabench_world_graph("test_zero_return_actor_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_zero_return_actor"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    program = _joint_program(
        runtime, planner, controller, lambda **_kwargs: FakeSimulator(success=False),
        num_samples=1,
    )
    episode = program.collect_episode({"task": "select_book"})
    assert episode.total_return == 0.0 and episode.controller
    transition = episode.controller[0]
    transition.feasibility_cost = 1.0
    transition.feasibility_index = 0
    transition.old_feasibility_logprob = transition.old_logprob.detach().clone()
    program.controller_anchor_loader = [{
        "images": transition.images,
        "state": transition.state,
        "task_index": transition.task_index,
        "plan_context": transition.plan_context,
        "actions": transition.actions,
    }]
    program.controller_bc_weight = 1.0
    actor_before = controller.policy_head.weight.detach().clone()
    task_before = controller.task_embedding.weight.detach().clone()
    value_before = controller.value_head.weight.detach().clone()
    loss = program._update_controller([episode])
    assert torch.isfinite(torch.tensor(loss))
    assert torch.equal(actor_before, controller.policy_head.weight)
    assert torch.equal(task_before, controller.task_embedding.weight)
    assert not torch.equal(value_before, controller.value_head.weight)
    assert program.last_controller_update["actor_update_attempted"] is False
    assert program.last_controller_update["rolled_back"] is False


def test_controller_ppo_rolls_back_complete_update_when_policy_drift_exceeds_limit():
    world = build_vlabench_world_graph("test_controller_rollback_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_controller_rollback"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    program = create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=torch.optim.SGD(planner.parameters(), lr=0.1),
        controller_optimizer=torch.optim.SGD(controller.parameters(), lr=1.0),
        env_factory=lambda **_kwargs: FakeSimulator(success=True),
        execute_horizon=1,
        max_steps=1,
        num_samples=1,
        ppo_epochs=2,
        ppo_target_action_log_ratio=1e-4,
        entropy_weight=1.0,
        supervised_weight=0.0,
        controller_bc_weight=0.0,
    )
    episode = program.collect_episode({"task": "select_book"})
    before = {
        name: value.detach().clone()
        for name, value in controller.state_dict().items()
    }

    loss = program._update_controller([episode])

    assert loss == pytest.approx(0.0)
    assert program.last_controller_update["rolled_back"] is True
    assert program.last_controller_update["ppo_epochs_completed"] == 0
    for name, value in controller.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_joint_simulator_training_updates_planner_and_controller():
    world = build_vlabench_world_graph("test_joint_world")
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="test_joint")
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1)
    planner_optimizer = torch.optim.SGD(planner.parameters(), lr=0.1)
    controller_optimizer = torch.optim.SGD(controller.parameters(), lr=0.01)
    counter = {"value": 0}

    def factory(**_kwargs):
        counter["value"] += 1
        return FakeSimulator(success=counter["value"] % 2 == 1)

    program = create_stage2_program(
        runtime,
        planner,
        controller,
        planner_optimizer=planner_optimizer,
        controller_optimizer=controller_optimizer,
        env_factory=factory,
        execute_horizon=1,
        max_steps=1,
        num_samples=1,
        ppo_epochs=1,
        ppo_target_action_log_ratio=10.0,
        ppo_max_log_ratio=10.0,
        supervised_weight=0.0,
        controller_bc_weight=0.0,
    )
    planner_before = planner.preference.detach().clone()
    controller_before = controller.value_head.weight.detach().clone()
    metrics = program.train_joint_epoch([{"task": "select_book"}], rollouts_per_update=2)
    assert metrics["success_rate"] == 0.5
    task_metrics = metrics["per_task"]["select_book"]
    assert task_metrics["episodes"] == 2
    assert task_metrics["successes"] == 1
    assert task_metrics["success_rate"] == 0.5
    assert task_metrics["valid_rate"] == 1.0
    assert task_metrics["return"] == pytest.approx(0.475)
    assert task_metrics["steps"] == 1.0
    assert task_metrics["ik_truncation_rate"] == 0.0
    assert task_metrics["execution_complete_rate"] == 1.0
    assert planner.preference.item() != planner_before.item()
    assert not torch.equal(controller.value_head.weight, controller_before)


def test_fixed_seed_rollout_evaluation_does_not_update_models():
    world = build_vlabench_world_graph("test_rollout_evaluation_world")
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="test_rollout_evaluation")
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1)
    program = _joint_program(runtime, planner, controller, lambda **_kwargs: FakeSimulator(success=True), num_samples=1)
    planner_before = planner.preference.detach().clone()
    controller_before = {name: value.detach().clone() for name, value in controller.state_dict().items()}
    metrics = program.evaluate_rollouts(
        [{"task": "select_book"}, {"task": "select_fruit"}],
        rollouts_per_task=1,
        seed=17,
    )
    assert metrics["episodes"] == 2
    assert metrics["successful_task_count"] == 2
    assert metrics["success_rate"] == 1.0
    assert len(metrics["episode_diagnostics"]) == 2
    evidence = metrics["episode_diagnostics"][0]["diagnostics"]
    assert evidence["observations"] == 2
    assert evidence["commands"] == 1
    assert evidence["target_status"] == "unavailable"  # The fake has no target geometry.
    assert evidence["cameras"]["status"] == "unverified"
    torch.testing.assert_close(planner.preference, planner_before)
    for name, value in controller.state_dict().items():
        torch.testing.assert_close(value, controller_before[name])


def test_joint_checkpoint_restores_rng_and_rejects_domain_mismatch(tmp_path):
    world = build_vlabench_world_graph("test_joint_checkpoint_world")
    runtime = build_constraint_runtime(world, max_entities=2, max_operations=2, name_prefix="test_joint_checkpoint")
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1)
    planner_optimizer = torch.optim.SGD(planner.parameters(), lr=0.1)
    controller_optimizer = torch.optim.SGD(controller.parameters(), lr=0.1)
    torch.manual_seed(1234)
    saved_preference = planner.preference.detach().clone()
    path = save_joint_checkpoint(
        tmp_path / "agent.pt",
        planner=planner,
        controller=controller,
        planner_optimizer=planner_optimizer,
        controller_optimizer=controller_optimizer,
        runtime=runtime,
        stage="reinforcement",
        epoch=3,
        next_round=4,
        metrics={"rounds": [{"episodes": 2}] * 4},
    )
    expected_random = torch.rand(4)
    planner.preference.data.add_(5.0)
    torch.rand(7)
    payload = load_joint_checkpoint(
        path,
        planner=planner,
        controller=controller,
        planner_optimizer=planner_optimizer,
        controller_optimizer=controller_optimizer,
        runtime=runtime,
    )
    assert payload["stage"] == "reinforcement" and payload["epoch"] == 3
    assert payload["next_round"] == 4
    assert len(payload["metrics"]["rounds"]) == 4
    assert torch.equal(planner.preference, saved_preference)
    assert torch.equal(torch.rand(4), expected_random)
    payload["domain_checksum"] = "bad"
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="domain checksum"):
        load_joint_checkpoint(bad, planner=planner, controller=controller, runtime=runtime)


def test_standalone_checkpoint_versions_controller_semantics_and_migrates_supervised(tmp_path):
    world = build_vlabench_world_graph("test_standalone_controller_migration_world")
    runtime = build_constraint_runtime(
        world,
        max_entities=2,
        max_operations=2,
        name_prefix="test_standalone_controller_migration",
    )
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    path = save_joint_checkpoint(
        tmp_path / "supervised.pt",
        planner=planner,
        controller=controller,
        planner_optimizer=None,
        controller_optimizer=optimizer,
        runtime=runtime,
        stage="supervised",
        epoch=0,
    )
    payload = torch.load(path, weights_only=False)
    assert payload["standalone_checkpoint_version"] == 5
    assert payload["controller_configuration"]["behavior_cloning_version"] == 3

    payload["standalone_checkpoint_version"] = 4
    torch.save(payload, path)
    compatible_supervised = load_joint_checkpoint(
        path,
        planner=planner,
        controller=controller,
        controller_optimizer=optimizer,
        runtime=runtime,
    )
    assert "controller_migration_required" not in compatible_supervised

    payload["standalone_checkpoint_version"] = 5
    # V6 used local deltas but decoded video bytes without unit normalization.
    payload["controller_configuration"]["behavior_cloning_version"] = 2
    torch.save(payload, path)
    restored = load_joint_checkpoint(
        path,
        planner=planner,
        controller=controller,
        controller_optimizer=optimizer,
        runtime=runtime,
    )
    assert restored["controller_migration_required"] is True
    assert restored["controller_migration_reason"] == "behavior_cloning"
    assert optimizer.state == {}

    payload["stage"] = "reinforcement"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="controller_configuration"):
        load_joint_checkpoint(
            path,
            planner=planner,
            controller=controller,
            runtime=runtime,
        )


def test_standalone_version2_checkpoint_migrates_only_before_reinforcement(tmp_path):
    world = build_vlabench_world_graph("test_standalone_v2_migration_world")
    runtime = build_constraint_runtime(
        world,
        max_entities=2,
        max_operations=2,
        name_prefix="test_standalone_v2_migration",
    )
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    path = save_joint_checkpoint(
        tmp_path / "legacy.pt",
        planner=planner,
        controller=controller,
        planner_optimizer=None,
        controller_optimizer=None,
        runtime=runtime,
        stage="supervised",
        epoch=0,
    )
    payload = torch.load(path, weights_only=False)
    payload["standalone_checkpoint_version"] = 2
    payload.pop("controller_configuration")
    torch.save(payload, path)
    restored = load_joint_checkpoint(
        path,
        planner=planner,
        controller=controller,
        runtime=runtime,
    )
    assert restored["controller_migration_required"] is True

    payload["stage"] = "reinforcement"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="robot-frame rollout contract"):
        load_joint_checkpoint(
            path,
            planner=planner,
            controller=controller,
            runtime=runtime,
        )


def test_standalone_version4_reinforcement_checkpoint_is_rejected(tmp_path):
    world = build_vlabench_world_graph("test_standalone_v3_rl_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_standalone_v3_rl"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    path = save_joint_checkpoint(
        tmp_path / "legacy_rl.pt",
        planner=planner,
        controller=controller,
        planner_optimizer=None,
        controller_optimizer=None,
        runtime=runtime,
        stage="reinforcement",
        epoch=0,
    )
    payload = torch.load(path, weights_only=False)
    payload["standalone_checkpoint_version"] = 4
    torch.save(payload, path)

    with pytest.raises(ValueError, match="robot-frame rollout contract"):
        load_joint_checkpoint(
            path,
            planner=planner,
            controller=controller,
            runtime=runtime,
        )


def test_standalone_failed_retention_checkpoint_is_rejected(tmp_path):
    world = build_vlabench_world_graph("test_standalone_failed_retention_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="test_standalone_failed_retention"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    path = save_joint_checkpoint(
        tmp_path / "failed_retention.pt",
        planner=planner,
        controller=controller,
        planner_optimizer=None,
        controller_optimizer=None,
        runtime=runtime,
        stage="reinforcement",
        epoch=0,
        metrics={"retention_eligible": False},
    )

    with pytest.raises(ValueError, match="failed its fixed-seed retention gate"):
        load_joint_checkpoint(
            path,
            planner=planner,
            controller=controller,
            runtime=runtime,
        )
