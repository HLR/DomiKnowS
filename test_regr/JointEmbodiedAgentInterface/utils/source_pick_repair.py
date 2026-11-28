"""Collect native-success source pick demonstrations with pre-action inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .matched_vlabench import TASKS, create_environment, record_control, reset_seed, scene_id
from test_regr.VLABenchAgentInterface.research.collection import ExecutedStepRecorder
from test_regr.VLABenchAgentInterface.simulation.environment import reset_reward_tracking
from test_regr.VLABenchAgentInterface.simulation.program import _task_success
from test_regr.VLABenchAgentInterface.world_domain import controller_skill_index


def capture_pick(row, path):
    reset_seed(row["seed"])
    env = create_environment(row["task"], simulator_seed=row["seed"])
    recorder = None
    try:
        env.reset()
        reset_reward_tracking(env)
        if scene_id(env) != row["scene"]:
            raise ValueError("BC source reset changed before replay")
        operations = row["operations"]
        if len(operations) < 2 or operations[0]["skill"] != "pick" or not operations[0]["contact"]:
            return False
        boundary = operations[1]["start_step"]
        recorder = ExecutedStepRecorder(env)
        with np.load(row["demonstration"]) as data:
            for step, command in enumerate(data["commands"]):
                if step == boundary:
                    recorder.close()
                ts = env.step(command)
                env.physics.forward()
                if _task_success(env):
                    if step < boundary:
                        return False
                    np.savez_compressed(path, images=recorder.frames, state=recorder.states, actions=recorder.actions)
                    return True
                if ts.last():
                    return False
        return False
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def collect(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "picks.jsonl"
    with manifest.open("x") as stream:
        for index, task in enumerate(TASKS):
            accepted = 0
            for attempt in range(40):
                seed = 831000 + 100 * index + attempt
                row = record_control(manifest, task, seed)
                archive = output / f"pick-{seed}.npz"
                if row["native_success"] and capture_pick(row, archive):
                    row.update(archive=archive.name, split="train" if accepted < 4 else "validation")
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    accepted += 1
                print(task, seed, "verified picks", accepted, flush=True)
                if accepted == 6:
                    break
            if accepted != 6:
                raise RuntimeError(f"Insufficient verified source pick demonstrations for {task}")


class PickWindows(Dataset):
    def __init__(self, root, split, instructions, horizon):
        from test_regr.VLABenchAgentInterface.dataset import control_task_index_for_instruction
        self.episodes = []
        self.windows = []
        self.horizon = horizon
        for line in (root / "picks.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["split"] != split:
                continue
            index = control_task_index_for_instruction(row["instruction"], instructions)
            with np.load(root / row["archive"]) as data:
                states, actions, images = data["state"], data["actions"], data["images"]
            self.windows.extend((len(self.episodes), int(offset))
                                for offset in np.linspace(0, len(states)-1, 16))
            self.episodes.append((states, actions, images, index))
        if len(self.episodes) != (16 if split == "train" else 8):
            raise ValueError(f"Incomplete source {split} cohort")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        from test_regr.VLABenchAgentInterface.observations import image_tensor
        episode, offset = self.windows[index]
        states, actions, images, instruction = self.episodes[episode]
        history = [max(0, offset-1), offset]
        forward = np.minimum(offset + np.arange(self.horizon), len(actions)-1)
        return dict(state=torch.tensor(states[history], dtype=torch.float32),
                    images=image_tensor(images[history], channels_last=True),
                    actions=torch.tensor(actions[forward], dtype=torch.float32),
                    task_index=instruction, plan_context=torch.tensor([controller_skill_index("pick"), 0, 1]))


def train_picks(stage2, root, save):
    from torch.utils.data import DataLoader
    from test_regr.VLABenchAgentInterface.training import evaluate_controller, train_controller_steps
    horizon = stage2.controller.action_horizon
    loaders = {split: DataLoader(PickWindows(root, split, stage2.controller_task_instructions, horizon),
                                 batch_size=4, shuffle=split == "train")
               for split in ("train", "validation")}
    def evaluate():
        return evaluate_controller(stage2.controller, loaders["validation"], device=stage2.device_name)

    initial = evaluate()
    best = initial["pose_mae"]
    report = {"initial": initial, "epochs": []}
    for epoch in range(1, 5):
        train_controller_steps(stage2.controller, loaders["train"], stage2.controller_optimizer,
                               device=stage2.device_name, steps=len(loaders["train"]))
        metrics = evaluate()
        report["epochs"].append(metrics)
        if metrics["pose_mae"] < best:
            best = metrics["pose_mae"]
            save({**report, "selected_epoch": epoch})
    return report


def train_grounding(stage2, save):
    """Fit only the VLABench linear head on numbered views of independent source scenes."""
    from test_regr.VLABenchAgentInterface.plan_codec import plan_to_tokens
    from test_regr.VLABenchAgentInterface.simulation.environment import numbered_views_from_observation
    from .constrained_stage2 import _active_head, _pilot_policy
    from .matched_vlabench import expert_plan
    from .planner_context import planner_context
    planner = stage2.joint_planner
    vocabulary = planner.vocabulary("vlabench")
    cache = {"train": [], "validation": []}
    with _pilot_policy(stage2), _active_head(stage2, "vlabench") as head:
        with torch.no_grad():
            for index, task in enumerate(TASKS):
                for offset in range(24):
                    seed = 843000 + 100 * index + offset
                    reset_seed(seed)
                    env = create_environment(task, simulator_seed=seed)
                    try:
                        env.reset()
                        views, names = numbered_views_from_observation(env, env.get_observation(require_pcd=False))
                        try:
                            context = planner_context("vlabench", {"task": task}, context={
                                "instruction": env.task.get_instruction(), "images": views, "entity_table": names})
                            tokens = plan_to_tokens(expert_plan(env, task, names), names, world=stage2.runtime.world_bundle)
                            hidden = planner.causal_label_features("vlabench", context, tokens[:-1]).detach()
                            labels = torch.tensor([vocabulary.label_for_token(t) for t in tokens], device=hidden.device)
                            cache["train" if offset < 18 else "validation"].append((hidden, labels))
                        finally:
                            for image in views:
                                image.close()
                    finally:
                        env.close()
        # Scene-balanced token cross entropy includes each visible distractor in the output vocabulary.
        def loss(row):
            hidden, labels = row
            return torch.nn.functional.cross_entropy(head(hidden), labels)

        def evaluate():
            with torch.no_grad():
                return float(torch.stack([loss(row) for row in cache["validation"]]).mean())

        initial = best = evaluate()
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.)
        report = {"initial_loss": initial, "validation_losses": []}
        for epoch in range(1, 9):
            order = np.random.default_rng(37 + epoch).permutation(len(cache["train"]))
            for indices in np.array_split(order, 9):
                optimizer.zero_grad(set_to_none=True)
                torch.stack([loss(cache["train"][i]) for i in indices]).mean().backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                optimizer.step()
            metric = evaluate()
            report["validation_losses"].append(metric)
            if metric < best:
                best = metric
                save({**report, "selected_epoch": epoch})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    collect(parser.parse_args().output)
