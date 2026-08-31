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
)
from test_regr.VLABenchAgentInterface.graph import PlanVocabulary, plan_to_tokens
from test_regr.VLABenchAgentInterface.models import (
    FrozenSigLIPEncoder,
    MultiViewController,
    QwenVLPlanner,
    TinyImageEncoder,
    controller_loss,
    prepare_kbit_model,
    resolve_vision_language_loader,
    vision_language_hidden_size,
)
from test_regr.VLABenchAgentInterface.program import (
    JointEpisode,
    VLABenchHierarchicalReinforcementProgram,
    _entity_pointer_dfa,
    _observation_state,
    _signal,
    generalized_advantage_estimate,
    ppo_clipped_loss,
)
from test_regr.VLABenchAgentInterface.training import (
    build_constraint_runtime,
    create_stage1_program,
    create_stage2_program,
    load_joint_checkpoint,
    load_checkpoint,
    prepare_planner_program_examples,
    save_joint_checkpoint,
    save_checkpoint,
    train_controller_epoch,
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
    assert result == {"task": "select_fruit", "robot": "franka", "time_limit": 4}


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
        condition_index=condition_index,
    )
    item = dataset[0]
    assert item["state"].shape == (2, 7)
    assert item["images"].shape == (2, 3, 3, 24, 24)
    assert item["actions"].shape == (3, 7)
    assert item["task_index"].item() == condition_index
    model = MultiViewController(TinyImageEncoder(16), hidden_dim=24, action_horizon=3, max_views=3)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    output = model(batch["images"], batch["state"], batch["task_index"])
    assert output.shape == (2, 3, 7)
    loss, metrics = controller_loss(output, batch["actions"])
    assert torch.isfinite(loss) and metrics["pose_loss"] >= 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trained = train_controller_epoch(model, [batch], optimizer, device="cpu", mixed_precision=False)
    assert trained["loss"] >= 0
    checkpoint = tmp_path / "controller.pt"
    save_checkpoint(checkpoint, model=model, optimizer=optimizer, epoch=0, metrics=trained)
    restored = MultiViewController(TinyImageEncoder(16), hidden_dim=24, action_horizon=3, max_views=3)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    payload = load_checkpoint(checkpoint, model=restored, optimizer=restored_optimizer)
    assert payload["epoch"] == 0


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

    def forward(self, input_ids, **_kwargs):
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
    (-logprob).backward()
    assert planner.output.weight.grad is not None


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
    loss = ppo_clipped_loss(evaluated.sum(1), logprob.detach().sum(1), torch.ones(1)) + evaluated_value.mean()
    loss.backward()
    assert controller.value_head.weight.grad is not None
    advantages, returns = generalized_advantage_estimate([0.1, 1.0], [0.2, 0.3], [False, True])
    assert len(advantages) == len(returns) == 2
    assert returns[-1] == pytest.approx(1.0)


class FakeRobot:
    def get_qpos_from_ee_pos(self, *, physics, pos, quat):
        assert physics is not None and len(pos) == 3 and len(quat) == 4
        return True, np.arange(7, dtype=np.float64)


class FailedRobot(FakeRobot):
    def get_qpos_from_ee_pos(self, *, physics, pos, quat):
        return False, np.arange(7, dtype=np.float64)


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


def test_ee_action_safety_envelope_limits_cartesian_and_wrapped_rotation_steps():
    current = np.asarray([0.0, 0.4, 0.2, 3.10, 0.0, -3.10, 0.0])
    target = np.asarray([1.0, -1.0, 0.9, -3.10, 1.0, 3.10, 1.0])
    bounded = bound_ee_action(target, current)
    assert np.max(np.abs(bounded[:3] - current[:3])) <= 0.05 + 1e-9
    angular_delta = (bounded[3:6] - current[3:6] + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(angular_delta)) <= 0.25 + 1e-9
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
    assert torch.max(torch.abs(episode.controller[0].actions[0, 0, :3])).item() <= 0.05 + 1e-6
    assert torch.isfinite(episode.controller[0].old_logprob)


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
    assert not episode.valid
    assert episode.total_return == 0.0
    assert simulator.count == 0


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
        supervised_weight=0.0,
        controller_bc_weight=0.0,
    )
    planner_before = planner.preference.detach().clone()
    controller_before = controller.value_head.weight.detach().clone()
    metrics = program.train_joint_epoch([{"task": "select_book"}], rollouts_per_update=2)
    assert metrics["success_rate"] == 0.5
    assert planner.preference.item() != planner_before.item()
    assert not torch.equal(controller.value_head.weight, controller_before)


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
    assert torch.equal(planner.preference, saved_preference)
    assert torch.equal(torch.rand(4), expected_random)
    payload["domain_checksum"] = "bad"
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="domain checksum"):
        load_joint_checkpoint(bad, planner=planner, controller=controller, runtime=runtime)
