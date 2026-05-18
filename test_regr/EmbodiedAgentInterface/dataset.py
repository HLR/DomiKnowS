import ast
import re
from pathlib import Path

import torch


EOS_TOKEN = "<eos>"

ACTION_LABELS = (
    "open",
    "close",
    "walk",
    "grasp",
    "place",
    "put",
    "switch",
    "navigate",
    "other",
)
ACTION_VOCAB = (EOS_TOKEN, *ACTION_LABELS)

HF_DATASET = "Inevitablevalor/EmbodiedAgentInterface"


def _normalize_action_name(action):
    """
    Simplify the baseline vocabulary size
    """
    if isinstance(action, dict):
        action = action.get("action", "")
    action = str(action or "").lower()
    action = action.replace("right_", "").replace("left_", "")
    action = re.sub(r"[^a-z0-9_]+", "_", action).strip("_")

    if "open" in action:
        return "open"
    if "close" in action:
        return "close"
    if "walk" in action:
        return "walk"
    if "grasp" in action or "grab" in action or "pick" in action:
        return "grasp"
    if "place" in action:
        return "place"
    if "put" in action:
        return "put"
    if "switch" in action or "turn" in action:
        return "switch"
    if "navigate" in action or "move" in action:
        return "navigate"
    return "other"


def parse_action_trajectory(value):
    """Parse the EAI action_trajectory field into a Python list."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def action_sequence_labels(row, max_steps=8):
    trajectory = parse_action_trajectory(row.get("action_trajectory"))
    labels = [_normalize_action_name(action) for action in trajectory]
    labels = labels[: max(0, max_steps - 1)]
    labels.append(EOS_TOKEN)
    while len(labels) < max_steps:
        labels.append(EOS_TOKEN)
    return labels


def action_sequence_ids(labels):
    label_to_id = {label: idx for idx, label in enumerate(ACTION_VOCAB)}
    return [label_to_id.get(label, label_to_id["other"]) for label in labels]


def first_action_label(row):
    trajectory = parse_action_trajectory(row.get("action_trajectory"))
    if not trajectory:
        return "other"
    return _normalize_action_name(trajectory[0])


def row_to_example(row, device="cpu", max_steps=8):
    sequence_labels = action_sequence_labels(row, max_steps=max_steps)
    sequence_ids = action_sequence_ids(sequence_labels)
    label = next((item for item in sequence_labels if item != EOS_TOKEN), "other")
    text_parts = [
        row.get("task_name", ""),
        row.get("natural_language_description", ""),
        row.get("tl_goal", ""),
    ]
    text = " ".join(str(part) for part in text_parts if part)
    task_id = row.get("task_id") or row.get("scene_id") or row.get("task_name") or "task"
    return {
        "task_id": str(task_id),
        "task_name": str(row.get("task_name", "")),
        "natural_language_description": str(row.get("natural_language_description", "")),
        "tl_goal": str(row.get("tl_goal", "")),
        "transition_model": str(row.get("transition_model", "")),
        "text": text,
        "first_action": label,
        "target_action_tokens": sequence_labels,
        "target_action_labels": torch.LongTensor(sequence_ids).to(device),
        "token_positions": torch.arange(max_steps, dtype=torch.long, device=device),
        "logic_label": torch.LongTensor([1]).to(device),
    }


def add_action_concept_labels(examples, device="cpu"):
    for sample in examples:
        gold = sample["first_action"]
        for label in ACTION_LABELS:
            sample[f"{label}_label"] = torch.LongTensor([gold == label]).to(device)
    return examples


def dummy_dataset(device="cpu", max_steps=8):
    rows = [
        {
            "task_id": "dummy_open_0",
            "task_name": "turn_on_light",
            "natural_language_description": "Open the cabinet and switch on the light.",
            "tl_goal": "switchon(light)",
            "action_trajectory": "[{'action': 'OPEN', 'object': 'cabinet_1'}]",
        },
        {
            "task_id": "dummy_grasp_0",
            "task_name": "pack_bag",
            "natural_language_description": "Pick up the toothbrush and put it in the backpack.",
            "tl_goal": "inside(toothbrush, backpack)",
            "action_trajectory": "[{'action': 'RIGHT_GRASP', 'object': 'toothbrush_1'}]",
        },
        {
            "task_id": "dummy_place_0",
            "task_name": "set_table",
            "natural_language_description": "Place the plate on the table.",
            "tl_goal": "ontop(plate, table)",
            "action_trajectory": "[{'action': 'RIGHT_PLACE_ON_TOP', 'object': 'table_1'}]",
        },
        {
            "task_id": "dummy_walk_0",
            "task_name": "go_to_sofa",
            "natural_language_description": "Walk towards the sofa before sitting down.",
            "tl_goal": "near(agent, sofa)",
            "action_trajectory": "[{'action': 'WALK_TOWARDS', 'object': 'sofa_1'}]",
        },
        {
            "task_id": "dummy_close_0",
            "task_name": "close_fridge",
            "natural_language_description": "Close the refrigerator after taking the apple.",
            "tl_goal": "closed(fridge)",
            "action_trajectory": "[{'action': 'CLOSE', 'object': 'fridge_1'}]",
        },
    ]
    return add_action_concept_labels([row_to_example(row, device=device, max_steps=max_steps) for row in rows], device=device)


def load_eai_dataset(dataset_name="all", split=None, limit=None, data_path=None, device="cpu", max_steps=8):
    """Load EAI rows from local parquet/csv/jsonl or Hugging Face datasets."""
    if data_path:
        rows = _load_local_rows(Path(data_path))
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Install the `datasets` package or pass --data-path to local EAI parquet/csv/jsonl files."
            ) from exc

        split_name = split or dataset_name
        if dataset_name == "all":
            rows = []
            for split_name in ("behavior", "virtualhome"):
                rows.extend(load_dataset(HF_DATASET, split=split_name))
        else:
            rows = list(load_dataset(HF_DATASET, split=split_name))

    if limit is not None and limit >= 0:
        rows = rows[:limit]

    examples = [row_to_example(dict(row), device=device, max_steps=max_steps) for row in rows]
    return add_action_concept_labels(examples, device=device)


def _load_local_rows(path):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    if suffix == ".csv":
        import pandas as pd

        return pd.read_csv(path).to_dict("records")
    if suffix in {".json", ".jsonl"}:
        import json

        if suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else data.get("data", [])
    raise ValueError(f"Unsupported data file: {path}")


def split_train_dev(examples, dev_fraction=0.2):
    if not examples:
        return [], []
    cut = max(1, int(len(examples) * (1.0 - dev_fraction)))
    cut = min(cut, len(examples))
    return examples[:cut], examples[cut:]

