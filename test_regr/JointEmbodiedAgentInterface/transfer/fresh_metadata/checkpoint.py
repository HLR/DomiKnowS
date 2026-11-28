"""Load a verified planner-only Stage 1 snapshot into a fresh runtime."""

from pathlib import Path

import torch

from ..protocol import read, sha
from ..repair.common import fresh_planner


FORMAT = "fresh-joint-metadata-preference-v1"


def checkpoint_record(archive, epoch, *, verify_base=True):
    archive = Path(archive)
    config_file = archive / "config.json"
    config, protocol = read(config_file), read(archive / "protocol.json")
    path = protocol["checkpoint"]
    settings = Path(protocol["training_root"]) / "settings.json"
    if (protocol["epoch"] != epoch or sha(config_file) != protocol["config_sha256"]
            or sha(path) != protocol["checkpoint_sha256"]
            or sha(settings) != protocol["training_settings_sha256"]):
        raise ValueError("Stage 1 checkpoint or archived ancestry differs")
    model_source = read(config["model_source"])
    base_files = model_source["planner_files"]
    if model_source["planner_model"] != config["model"] or not base_files:
        raise ValueError("pretrained base model identity differs")
    if verify_base:
        for relative, expected in base_files.items():
            if sha(Path(config["model"]) / relative) != expected:
                raise ValueError(f"pretrained base model file changed: {relative}")
    return config, {"path": path, "epoch": epoch, "sha256": protocol["checkpoint_sha256"],
                    "settings_sha256": protocol["training_settings_sha256"],
                    "config_sha256": protocol["config_sha256"], "pretrained_base_files": base_files}


def load_planner_snapshot(planner, record):
    if sha(record["path"]) != record["sha256"]:
        raise ValueError("Stage 1 weights changed before loading")
    snapshot = torch.load(record["path"], map_location="cpu", weights_only=True)
    if (snapshot.get("format"), snapshot.get("epoch"), snapshot.get("settings_sha256")) != (
            FORMAT, record["epoch"], record["settings_sha256"]):
        raise ValueError("incompatible Stage 1 format, epoch, or lineage")
    expected = {name for name, _ in planner.named_parameters()
                if "lora_" in name or name.startswith("label_heads.")}
    if not expected or set(snapshot["planner"]) != expected:
        raise ValueError("Stage 1 LoRA or label-head parameters differ")
    loaded = planner.load_state_dict(snapshot["planner"], strict=False)
    if loaded.unexpected_keys or expected.intersection(loaded.missing_keys):
        raise ValueError("Stage 1 weights did not load")
    for name, parameter in planner.named_parameters():
        if name not in expected and parameter.requires_grad:
            raise ValueError(f"pretrained base parameter is trainable: {name}")
    planner.eval()
    return planner


def load_frozen_planner(config, record, runtime):
    planner = load_planner_snapshot(fresh_planner(config, runtime, "joint"), record)
    planner.requires_grad_(False)
    planner._base_parameter_requires_grad = {parameter: False for parameter in planner.parameters()}
    return planner
