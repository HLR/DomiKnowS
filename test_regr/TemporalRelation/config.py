"""Portable configuration for the TemporalRelation example."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


CONFIG_ENV_VAR = "TEMPORAL_RELATION_CONFIG"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
_REQUIRED_KEYS = {"python_path", "data_root", "training_model", "inference_model", "output_dir"}


@dataclass(frozen=True)
class TemporalRelationConfig:
    source: Path
    python_path: Path
    data_root: Path
    training_model: str
    inference_model: str
    output_dir: Path

    def output_path(self, filename):
        return self.output_dir / filename


def load_temporal_config(path=None):
    """Load configuration and resolve filesystem paths relative to its JSON file."""
    source = Path(path or os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)).expanduser().resolve()
    try:
        values = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"TemporalRelation config not found: {source}. "
            f"Set {CONFIG_ENV_VAR} to a valid JSON config file."
        ) from exc

    if not isinstance(values, dict):
        raise ValueError(f"TemporalRelation config must contain a JSON object: {source}")
    missing = sorted(_REQUIRED_KEYS - values.keys())
    if missing:
        raise ValueError(f"TemporalRelation config is missing required keys {missing}: {source}")

    return TemporalRelationConfig(
        source=source,
        python_path=_resolve_path(values["python_path"], source),
        data_root=_resolve_path(values["data_root"], source),
        training_model=_model_reference(values["training_model"], source),
        inference_model=_model_reference(values["inference_model"], source),
        output_dir=_resolve_path(values["output_dir"], source),
    )


def _resolve_path(value, source):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def _model_reference(value, source):
    value = str(value)
    path = Path(value).expanduser()
    if path.is_absolute() or value.startswith((".", "~")):
        return str(_resolve_path(value, source))
    return value


TEMPORAL_CONFIG = load_temporal_config()
