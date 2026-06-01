import ast
import re
from pathlib import Path

import torch


EOS_TOKEN = "<eos>"

FALLBACK_ACTION = "other"
DEFAULT_ACTION_LABELS = (FALLBACK_ACTION,)
BASE_VOCAB = (EOS_TOKEN, *DEFAULT_ACTION_LABELS)
ACTION_VOCAB = BASE_VOCAB

HF_DATASET = "Inevitablevalor/EmbodiedAgentInterface"


def _normalize_surface_token(value):
    value = str(value or "").lower()
    return re.sub(r"[^a-z0-9_]+", "_", value).strip("_")


def _normalize_action_name(action):
    """Normalize raw EAI action names while removing object/id payloads."""
    raw_action = action.get("action", "") if isinstance(action, dict) else action
    token = _normalize_surface_token(raw_action)
    token = re.sub(r"_\d+$", "", token)
    if isinstance(action, dict):
        obj = _normalize_object_name(action.get("object"))
        if obj:
            suffix = f"_{obj}"
            if token.endswith(suffix):
                token = token[: -len(suffix)]
            token = re.sub(rf"_{re.escape(obj)}(_|$).*", "", token)

    preserve_prefixes = ("left_", "right_", "switch_", "toggle_", "turn_")
    if "_" in token and not token.startswith(preserve_prefixes):
        token = token.split("_", 1)[0]
    return token or FALLBACK_ACTION


def _normalize_object_name(value):
    if isinstance(value, dict):
        value = value.get("object", "") or value.get("name", "")
    value = _normalize_surface_token(value)
    return value or None


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


def action_object_tokens_from_step(step):
    if not isinstance(step, dict):
        action = _normalize_action_name(step)
        return action, None
    action = _normalize_action_name(step.get("action", ""))
    obj = _normalize_object_name(step.get("object"))
    return action, obj


def trajectory_action_object_tokens(row):
    pairs = []
    for step in parse_action_trajectory(row.get("action_trajectory")):
        action, obj = action_object_tokens_from_step(step)
        if action:
            pairs.append((action, obj))
    return pairs


def object_tokens_from_row(row):
    return [obj for _action, obj in trajectory_action_object_tokens(row) if obj]


def action_tokens_from_row(row):
    return [action for action, _obj in trajectory_action_object_tokens(row) if action]


def action_tokens_requiring_object_from_row(row):
    return [action for action, obj in trajectory_action_object_tokens(row) if action and obj]


def build_generation_vocab(rows):
    actions = sorted({action for row in rows for action in action_tokens_from_row(row)})
    objects = sorted({obj for row in rows for obj in object_tokens_from_row(row)})
    vocab = [EOS_TOKEN]
    for token in (*actions, *DEFAULT_ACTION_LABELS, *objects):
        if token not in vocab:
            vocab.append(token)
    return tuple(vocab)


def action_sequence_labels(row, max_steps=8):
    labels = []
    for action, obj in trajectory_action_object_tokens(row):
        labels.append(action)
        if obj:
            labels.append(obj)
    labels = labels[: max(0, max_steps - 1)]
    labels.append(EOS_TOKEN)
    while len(labels) < max_steps:
        labels.append(EOS_TOKEN)
    return labels


def action_sequence_ids(labels, vocab=None):
    vocab = vocab or ACTION_VOCAB
    label_to_id = {label: idx for idx, label in enumerate(vocab)}
    fallback = label_to_id.get("other", len(vocab))
    return [label_to_id.get(label, fallback) for label in labels]


def first_action_label(row):
    pairs = trajectory_action_object_tokens(row)
    if not pairs:
        return FALLBACK_ACTION
    return pairs[0][0]


def row_to_example(row, device="cpu", max_steps=8, vocab=None):
    vocab = vocab or build_generation_vocab([row])
    sequence_labels = action_sequence_labels(row, max_steps=max_steps)
    sequence_ids = action_sequence_ids(sequence_labels, vocab=vocab)
    label = next((item for item in sequence_labels if item != EOS_TOKEN), "other")
    text_parts = [
        row.get("task_name", ""),
        row.get("natural_language_description", ""),
        row.get("tl_goal", ""),
    ]
    text = " ".join(str(part) for part in text_parts if part)
    task_id = row.get("task_id") or row.get("scene_id") or row.get("task_name") or "task"
    action_tokens = set(action_tokens_from_row(row))
    action_requires_object_tokens = set(action_tokens_requiring_object_from_row(row))
    object_tokens = set(object_tokens_from_row(row))
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
        "generation_vocab": vocab,
        "action_tokens": tuple(action for action in vocab if action in action_tokens),
        "action_requires_object_tokens": tuple(
            action for action in vocab if action in action_requires_object_tokens
        ),
        "object_tokens": tuple(token for token in vocab if token in object_tokens),
        "logic_label": torch.LongTensor([1]).to(device),
    }


def add_action_concept_labels(examples, device="cpu"):
    action_labels = sorted({label for sample in examples for label in sample.get("action_tokens", ())})
    action_labels.append(FALLBACK_ACTION) if FALLBACK_ACTION not in action_labels else None
    for sample in examples:
        gold = sample["first_action"]
        for label in action_labels:
            sample[f"{label}_label"] = torch.LongTensor([gold == label]).to(device)
    return examples


def dummy_dataset(device="cpu", max_steps=8):
    rows = [
        {
            "task_id": "dummy_open_0",
            "task_name": "turn_on_light",
            "natural_language_description": "Open the cabinet and switch on the light.",
            "tl_goal": "switchon(light)",
            "action_trajectory": "[{'action': 'OPEN', 'object': 'cabinet_1'}, {'action': 'SWITCH_ON', 'object': 'light_1'}]",
        },
        {
            "task_id": "dummy_grasp_0",
            "task_name": "pack_bag",
            "natural_language_description": "Pick up the toothbrush and put it in the backpack.",
            "tl_goal": "inside(toothbrush, backpack)",
            "action_trajectory": "[{'action': 'RIGHT_GRASP', 'object': 'toothbrush_1'}, {'action': 'PUT', 'object': 'backpack_1'}]",
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
    vocab = build_generation_vocab(rows)
    return add_action_concept_labels(
        [row_to_example(row, device=device, max_steps=max_steps, vocab=vocab) for row in rows],
        device=device,
    )


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

    rows = [dict(row) for row in rows]
    vocab = build_generation_vocab(rows)
    examples = [row_to_example(row, device=device, max_steps=max_steps, vocab=vocab) for row in rows]
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
