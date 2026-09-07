from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from PIL import Image

from .dataset import LeRobotWindowDataset, _video_tensor
from .diagnostics import ControllerMetrics, RolloutDiagnostics
from .models import MultiViewController, TinyImageEncoder
from .observations import camera_indices, camera_report, image_tensor
from .program import _controller_inputs
from .replay import replay_heldout_demo


def test_decoded_byte_pil_and_live_images_have_identical_siglip_inputs():
    # Include a dark byte value of 1: dtype, not max alone, determines byte scaling.
    pixels = np.zeros((8, 9, 3), dtype=np.uint8)
    pixels[..., 0], pixels[..., 1], pixels[..., 2] = 1, 128, 255
    decoded = torch.from_numpy(pixels.copy()).permute(2, 0, 1)
    decoder = SimpleNamespace(get_frame_played_at=lambda _t: SimpleNamespace(data=decoded))
    actual = _video_tensor(decoder, timestamp=0, video_root=None, cache={})
    expected = decoded.float() / 255
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(image_tensor(Image.fromarray(pixels)), expected)
    torch.testing.assert_close(image_tensor(decoded.float()), expected)
    torch.testing.assert_close(image_tensor(expected), expected)
    observation = {"rgb": pixels[None], "ee_state": np.zeros(7)}
    live = _controller_inputs([observation], 0, "cpu")[0][0, 0, 0]
    torch.testing.assert_close(actual, live)
    assert image_tensor(torch.ones(3, 8, 9, dtype=torch.uint8)).max() == pytest.approx(1 / 255)


@pytest.mark.parametrize("value", [-1.0, 256.0, float("nan")])
def test_image_contract_rejects_invalid_pixels(value):
    with pytest.raises(ValueError):
        image_tensor(torch.full((3, 8, 9), value))


def test_anonymous_inline_views_are_normalized_without_changing_states():
    row = {"images": torch.full((2, 3, 8, 9), 255, dtype=torch.uint8),
           "state": np.full(7, 2.0), "actions": np.full(7, 3.0)}
    item = LeRobotWindowDataset([row], action_horizon=1)[0]
    assert item["images"].min() == 1
    assert item["state"].min() == 2
    assert item["actions"].min() == 3


def test_camera_names_select_content_and_missing_names_fail_closed():
    obs = {"rgb": np.stack([np.full((8, 9, 3), i, np.uint8) for i in (0, 64, 128, 255)]),
           "camera_names": ["overhead", "side", "front", "wrist"], "ee_state": np.zeros(7)}
    ids = camera_indices(None, obs, ["front", "side", "wrist"])
    inputs = _controller_inputs([obs], 0, "cpu", selected_camera_indices=ids)
    torch.testing.assert_close(inputs[0][0, 0, :, 0, 0, 0], torch.tensor([128, 64, 255]) / 255)
    report = camera_report(None, obs, indices=ids, dataset_keys=["image", "second_image", "wrist_image"])
    assert report["selected_indices"] == [2, 1, 3]
    assert report["status"] == "unverified"
    with pytest.raises(ValueError, match="not unique"):
        camera_indices(None, obs, ["unknown"])


def test_view_embeddings_then_mean_do_not_bind_content_to_camera_slot():
    torch.manual_seed(19)
    model = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=2).eval()
    images = torch.rand(1, 2, 3, 3, 12, 12)
    state, task = torch.zeros(1, 2, 7), torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        original = model(images, state, task)
        permuted = model(images[:, :, [2, 0, 1]], state, task)
    torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-6)


def test_metrics_expose_unreachable_deltas_hold_baseline_and_gripper_imbalance():
    metrics = ControllerMetrics()
    state = torch.zeros(1, 2, 7)
    state[..., 6] = 1
    target = torch.zeros(1, 3, 7)
    target[0, :, 0] = torch.tensor([0.04, 0.04, 0.04])
    target[0, :, 6] = torch.tensor([1., 1., 0.])
    prediction = torch.zeros_like(target)
    prediction[..., 6] = 20  # Always open, including the required closure.
    metrics.update(prediction, target, state, torch.zeros(1, 2, 1, 3, 8, 9))
    result = metrics.result()
    assert result["axis_metrics"]["normalized_delta_mae"]["x"] == pytest.approx(2 / 3)
    assert result["axis_metrics"]["first_normalized_delta_mae"]["x"] == pytest.approx(2)
    assert result["axis_metrics"]["target_step_exceedance_rate"]["x"] == pytest.approx(1 / 3)
    assert result["axis_metrics"]["hold_normalized_delta_mae"]["x"] == pytest.approx(2 / 3)
    assert result["gripper_accuracy"] == pytest.approx(2 / 3)
    assert result["gripper"]["balanced_accuracy"] == pytest.approx(0.5)
    assert result["gripper"]["transitions"]["recall"] == 0
    assert result["gripper"]["transitions"]["precision"] is None


def test_metrics_wrap_euler_boundary_and_weight_batches_by_samples():
    metrics = ControllerMetrics()
    state = torch.zeros(1, 2, 7)
    state[..., 3] = torch.pi - 0.01
    target = state[:, -1:].clone()
    target[..., 3] = -torch.pi + 0.01
    prediction = state[:, -1:].clone()
    images = torch.zeros(1, 2, 1, 3, 8, 9)
    metrics.update(prediction, target, state, images)
    # Three perfect examples should divide the one error by four, not by two batches.
    metrics.update(target.repeat(3, 1, 1), target.repeat(3, 1, 1), state.repeat(3, 1, 1), images.repeat(3, 1, 1, 1, 1, 1))
    result = metrics.result()
    assert result["axis_metrics"]["pose_mae"]["roll"] == pytest.approx(0.005, abs=1e-6)
    assert result["axis_metrics"]["normalized_delta_mae"]["roll"] == pytest.approx(0.05, abs=1e-6)


def test_target_distance_uses_task_target_and_observed_gripper():
    target = SimpleNamespace(get_xpos=lambda _: np.array([1., 0., 0.]))
    distractor = SimpleNamespace(get_xpos=lambda _: np.zeros(3))
    env = SimpleNamespace(physics=object(), task=SimpleNamespace(target_entity="apple", entities={"apple": target, "bowl": distractor}))
    metrics = RolloutDiagnostics()
    metrics.observe(env, [0, 0, 0, 0, 0, 0, 1])
    metrics.observe(env, [.25, 0, 0, 0, 0, 0, 1], command_gripper=0)
    result = metrics.result()
    assert list(result["targets"]) == ["apple"]
    assert result["targets"]["apple"]["improvement_m"] == .25
    assert result["commanded_gripper_transitions"] == 1
    assert result.get("observed_gripper_transitions", 0) == 0
    assert RolloutDiagnostics().result()["target_status"] == "unavailable"


def test_target_distance_resolves_vlabench_style_suffix_and_case():
    target = SimpleNamespace(get_xpos=lambda _: np.array([1., 0., 0.]))
    env = SimpleNamespace(
        physics=object(),
        task=SimpleNamespace(
            target_entity="Rococo",
            entities={"rococo_painting": target, "baroque_painting": object()},
        ),
    )
    diagnostics = RolloutDiagnostics()
    diagnostics.observe(env, np.array([0.75, 0., 0., 0., 0., 0., 0.]))
    result = diagnostics.result()
    assert result["target_status"] == "available"
    assert result["targets"]["rococo_painting"]["samples"] == 1


class ReplayEnv:
    def __init__(self):
        self.physics = object()
        self.task = SimpleNamespace(target_entity="apple", entities={
            "apple": SimpleNamespace(get_xpos=lambda _: np.array([1., 0., 0.]))})
        self.state = np.zeros(7)
        self.closed = False

    def reset(self):
        self.state[:] = 0

    def get_observation(self, **_):
        # Image changes with the actual simulated state, exposing teacher forcing.
        image = np.full((1, 8, 9, 3), round(self.state[0] * 1000), np.uint8)
        return {"rgb": image, "ee_state": self.state.copy(), "camera_names": ["front"]}

    def step(self, command):
        self.state = np.asarray(command).copy()
        return SimpleNamespace(last=lambda: False)

    def close(self):
        self.closed = True


class ReplayController(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.observed = []

    def predict_action_chunk(self, images, state, *_):
        self.observed.append((images.clone(), state.clone()))
        action = state[:, -1:].clone()
        action[..., 0] += .01
        return action


def replay_dataset():
    return LeRobotWindowDataset([
        {"episode_index": 7, "state": np.zeros(7), "actions": np.zeros(7),
         "image": np.zeros((8, 9, 3), np.uint8)} for _ in range(5)
    ], action_horizon=1)


def test_replay_uses_fresh_observations_restores_each_arm_and_closes(tmp_path, monkeypatch):
    monkeypatch.setattr("test_regr.VLABenchAgentInterface.replay.ee_action_to_env_action", lambda _env, action, **_: action)
    environments = []
    def factory():
        env = ReplayEnv()
        environments.append(env)
        return env
    restores = []
    controller = ReplayController().train()
    result = replay_heldout_demo(controller, replay_dataset(), {"episode_index": 7},
        env_factory=factory, restore=lambda env, _: restores.append(env), camera_mapping={"image": "front"},
        output_dir=tmp_path, steps=3, execute_horizon=1)
    assert len(restores) == 2 and all(env.closed for env in environments)
    assert controller.training
    assert result["arms"]["controller"]["steps"] == 3
    assert result["arms"]["demonstration"]["trace"][-1]["observed"][0] == 0
    assert result["arms"]["controller"]["trace"][-1]["observed"][0] == pytest.approx(.03)
    assert controller.observed[1][1][0, -1, 0] == pytest.approx(.01)
    assert controller.observed[1][0][0, -1, 0, 0, 0, 0] == pytest.approx(10 / 255)
    assert result["arms"]["controller"]["cameras"]["status"] == "same_frame_pixel_match"


def test_replay_refuses_random_reset_as_matched_demonstration(tmp_path):
    def wrong_restore(env, _):
        env.state[0] = .1
    controller = ReplayController().train()
    result = replay_heldout_demo(controller, replay_dataset(), {"episode_index": 7},
        env_factory=ReplayEnv, restore=wrong_restore, camera_mapping={"image": "front"},
        output_dir=tmp_path, steps=3, execute_horizon=1)
    assert result["status"] == "restore_or_camera_mismatch"
    assert not controller.observed
    assert controller.training


def test_diagnostic_command_preserves_checkpoint_and_rejects_nonheldout_episode(tmp_path, monkeypatch):
    from torch.utils.data import ConcatDataset, DataLoader, Subset
    from . import main

    model = MultiViewController(TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1)
    checkpoint = tmp_path / "controller.pt"
    torch.save({"controller": model.state_dict(), "controller_configuration": {
        "action_representation_version": 4, "behavior_cloning_version": 2}}, checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    dataset = replay_dataset()
    loader = DataLoader(ConcatDataset([Subset(dataset, range(len(dataset)))]), batch_size=2)
    monkeypatch.setattr(main, "_control_loaders", lambda _: {"validation": loader})
    args = main.build_parser().parse_args([
        "diagnose-controller", "--checkpoint", str(checkpoint), "--control-source", "unused",
        "--output", str(tmp_path / "report"), "--task", "select_fruit", "--tiny-vision",
        "--hidden-dim", "8", "--vision-dim", "8", "--max-views", "1", "--action-horizon", "1",
        "--device", "cpu", "--max-batches", "1",
    ])
    main.command_diagnose_controller(args)
    result = json.loads((tmp_path / "report" / "diagnostics.json").read_text())
    assert result["offline_per_task"]["select_fruit"]["action_samples"] == 5
    assert result["replay"]["status"] == "unavailable"
    assert checkpoint.read_bytes() == checkpoint_bytes
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"task": "select_fruit", "episode_index": 99}]))
    camera_map = tmp_path / "cameras.json"
    camera_map.write_text(json.dumps({"image": "front"}))
    args.replay_manifest, args.camera_map, args.replay_restore = str(manifest), str(camera_map), "unused:restore"
    monkeypatch.setattr(main, "_factory", lambda _: lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="not in the selected held-out split"):
        main.command_diagnose_controller(args)
