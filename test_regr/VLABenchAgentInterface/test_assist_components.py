"""Tests for selectable execution-assistance components and the paired ablation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from .assist_ablation import (
    mcnemar_exact_p,
    paired_comparison,
    parse_assist_modes,
    run_assist_ablation,
    wilson_interval,
)
from .environment import euler_to_quaternion
from .models import MultiViewController, TinyImageEncoder
from .program import ASSIST_COMPONENTS, _orientation_step, parse_assist_components
from .test_pipeline import FakeSimulator, TinyCompactPlanner, _joint_program
from .training import build_constraint_runtime
from .world_graph import build_vlabench_world_graph


def test_parse_assist_components_normalizes_and_rejects_unknown_names():
    everything = frozenset(ASSIST_COMPONENTS)
    assert parse_assist_components(None) == everything
    assert parse_assist_components("all") == everything
    assert parse_assist_components("") == everything
    assert parse_assist_components("none") == frozenset()
    assert parse_assist_components("approach, skills") == {"approach", "skills"}
    assert parse_assist_components("orientation+latch") == {"orientation", "latch"}
    assert parse_assist_components(["Latch"]) == {"latch"}
    with pytest.raises(ValueError, match="unknown assist component"):
        parse_assist_components("orientaton")


def test_wilson_interval_and_exact_mcnemar():
    low, high = wilson_interval(2, 30)
    assert low == pytest.approx(0.0184, abs=2e-3)
    assert high == pytest.approx(0.2132, abs=2e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact_p(3, 3) == 1.0


def test_parse_assist_modes_accepts_names_and_custom_component_lists():
    modes = parse_assist_modes("unassisted,orientation_only,approach+skills,full")
    assert [name for name, _, _ in modes] == [
        "unassisted", "orientation_only", "approach+skills", "full",
    ]
    assert modes[0][1] is False and modes[0][2] == frozenset()
    assert modes[1][1] is True and modes[1][2] == {"orientation"}
    assert modes[2][2] == {"approach", "skills"}
    assert modes[3][2] == frozenset(ASSIST_COMPONENTS)
    assert parse_assist_modes("no_orientation")[0][2] == frozenset(ASSIST_COMPONENTS) - {"orientation"}
    with pytest.raises(ValueError):
        parse_assist_modes("bogus")


class _FakeEvaluator:
    """Evaluates a fixed per-mode success pattern on identical scenes."""

    PATTERNS = {
        (False, frozenset()): [True, False, False, False, False, False],
        (True, frozenset({"orientation"})): [True, True, True, False, False, False],
        (True, frozenset(ASSIST_COMPONENTS)): [True, True, True, True, True, False],
    }

    def __init__(self):
        self.calls = []

    def evaluate_rollouts(self, descriptors, *, rollouts_per_task, seed, assistance, assist_components):
        components = frozenset(assist_components)
        self.calls.append((assistance, components, rollouts_per_task, seed))
        outcomes = self.PATTERNS[(assistance, components)]
        return {
            "unassisted_success_rate": 0.0,
            "assisted_episode_rate": float(assistance),
            "ik_truncation_rate": 0.0,
            "episode_diagnostics": [
                {"task": "select_book" if index < 3 else "select_fruit",
                 "success": outcome, "return": 0.5 if outcome else 0.0}
                for index, outcome in enumerate(outcomes)
            ],
        }


def test_run_assist_ablation_reports_paired_statistics_and_writes_json(tmp_path):
    evaluator = _FakeEvaluator()
    path = tmp_path / "ablation.json"
    report = run_assist_ablation(
        evaluator,
        [{"task": "select_book"}, {"task": "select_fruit"}],
        rollouts_per_task=3,
        seed=11,
        modes=parse_assist_modes("unassisted,orientation_only,full"),
        output=path,
    )
    assert [call[3] for call in evaluator.calls] == [11, 11, 11]  # identical scenes in every mode
    assert report["reference_mode"] == "unassisted"
    unassisted = report["modes"]["unassisted"]
    assert unassisted["success"]["count"] == 1 and unassisted["success"]["n"] == 6
    assert unassisted["per_task"]["select_book"]["successes"] == 1
    orientation = report["modes"]["orientation_only"]
    assert orientation["components"] == ["orientation"]
    paired = orientation["paired_vs_unassisted"]
    assert paired == {
        "only_reference_success": 0,
        "only_other_success": 2,
        "both_success": 1,
        "mcnemar_exact_p": pytest.approx(0.5),
    }
    full = report["modes"]["full"]["paired_vs_unassisted"]
    assert full["only_other_success"] == 4 and full["mcnemar_exact_p"] == pytest.approx(0.125)
    assert json.loads(path.read_text(encoding="utf-8"))["rollouts_per_task"] == 3
    with pytest.raises(ValueError, match="equally sized"):
        paired_comparison({"episode_diagnostics": [{"success": True}]}, {"episode_diagnostics": []})


def _fruit_program(components):
    world = build_vlabench_world_graph("assist_components_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="assist_components"
    )

    class Task(SimpleNamespace):
        __module__ = "VLABench.tasks.tests"

    class Simulator(FakeSimulator):
        start_euler = np.asarray([-np.pi + 0.5, 0.0, 0.0])

        def __init__(self):
            super().__init__(success=False)
            self.task = Task(**vars(self.task))
            self.task.target_entity = "apple"
            self.task.entities["apple"] = SimpleNamespace(
                get_xpos=lambda _: np.array([0.2, 0.0, 0.0]),
                get_grasped_keypoints=lambda _: [np.array([0.2, 0.0, 0.0])],
                is_grasped=lambda *_: False,
            )

        def get_observation(self, require_pcd=False):
            observation = super().get_observation(require_pcd)
            observation["ee_state"][0] = 0.1
            # Held away from the expert top-down grasp orientation.
            observation["ee_state"][3:6] = self.start_euler
            return observation

    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    program = _joint_program(runtime, planner, controller, lambda **_: Simulator())
    program.assist_components = parse_assist_components(components)
    return program, Simulator.start_euler


def _same_rotation(first, second):
    return abs(float(np.dot(euler_to_quaternion(*first), euler_to_quaternion(*second)))) == pytest.approx(
        1.0, abs=1e-5
    )


def test_default_components_keep_expert_orientation_and_record_no_orientation_label():
    program, _start = _fruit_program("all")
    episode = program.collect_episode({"task": "select_fruit"})
    diagnostics = episode.diagnostics
    assert diagnostics["assist_components"] == sorted(ASSIST_COMPONENTS)
    assert diagnostics["orientation_label_steps"] == 0
    assert diagnostics["pick_assist_steps"] >= 1


def test_without_orientation_component_policy_orientation_runs_and_expert_step_is_the_label():
    program, start = _fruit_program("approach,skills,latch")
    episode = program.collect_episode({"task": "select_fruit"})
    diagnostics = episode.diagnostics
    assert "orientation" not in diagnostics["assist_components"]
    assert diagnostics["orientation_label_steps"] >= 1
    label = episode.controller[0].assisted_actions[0, 3:6].numpy()
    expected, _ = _orientation_step(start, np.asarray([-np.pi, 0.0, 0.0]), program.max_rotation_step)
    assert _same_rotation(label, expected)
    # The expert orientation is no longer what executed, so it is also not
    # silently counted as an executed assist step when approach is off.
    program.assist_components = parse_assist_components("none")
    quiet = program.collect_episode({"task": "select_fruit"}).diagnostics
    assert quiet["assist_components"] == []
    assert quiet["pick_assist_steps"] == 0
    assert quiet["orientation_label_steps"] >= 1


def test_evaluate_rollouts_assistance_and_component_overrides_are_scoped():
    world = build_vlabench_world_graph("assist_override_world")
    runtime = build_constraint_runtime(
        world, max_entities=2, max_operations=2, name_prefix="assist_override"
    )
    planner = TinyCompactPlanner(runtime.vocabulary)
    controller = MultiViewController(
        TinyImageEncoder(8), hidden_dim=8, action_horizon=1, max_views=1
    )
    program = _joint_program(
        runtime, planner, controller, lambda **_: FakeSimulator(success=True), num_samples=1
    )
    default = program.evaluate_rollouts([{"task": "select_book"}], rollouts_per_task=1, seed=3)
    assert default["episode_diagnostics"][0]["diagnostics"]["assist_components"] == []
    assert default["episode_diagnostics"][0]["success"] is True
    assert default["episode_diagnostics"][0]["return"] > 0.0
    assisted = program.evaluate_rollouts(
        [{"task": "select_book"}],
        rollouts_per_task=1,
        seed=3,
        assistance=True,
        assist_components="orientation",
    )
    assert assisted["episode_diagnostics"][0]["diagnostics"]["assist_components"] == ["orientation"]
    assert program._execution_assistance_override is None
    assert program._assist_components_override is None
