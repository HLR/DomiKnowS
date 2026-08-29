"""Download and load the processed VLABench planning/control datasets."""

from __future__ import annotations

import importlib
import json
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from tqdm.std import tqdm as _TerminalTqdm

try:
    from .world_graph import canonicalize_plan
except ImportError:
    from world_graph import canonicalize_plan


PLANNING_DATASET_ID = "VLABench/vlm_evaluation_v1.0"
CONTROL_DATASET_ID = "VLABench/vlabench_primitive_ft_lerobot_video"


class _TerminalDownloadProgress(_TerminalTqdm):
    """Plain terminal bar compatible with old and new huggingface_hub.

    ``huggingface_hub`` supplies a progress-group ``name`` keyword that stock
    tqdm does not accept.  Removing it lets us bypass ``tqdm.auto`` (which can
    incorrectly select the notebook/async renderer in redirected server
    terminals) without hiding useful multi-hour download progress.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)


@contextmanager
def _terminal_huggingface_progress():
    """Route snapshot and per-file Hub bars away from ``tqdm.auto``.

    ``snapshot_download(tqdm_class=...)`` only controls the outer file-count
    bar in several huggingface_hub releases. Xet reconstruction and HTTP byte
    bars continue to use module-level Hub tqdm aliases, so temporarily replace
    those aliases as well and restore them even when a download fails.
    """

    targets = (
        ("huggingface_hub.utils.tqdm", "tqdm"),
        ("huggingface_hub.utils", "tqdm"),
        ("huggingface_hub.file_download", "tqdm"),
        ("huggingface_hub._snapshot_download", "hf_tqdm"),
    )
    patched = []
    try:
        for module_name, attribute in targets:
            module = importlib.import_module(module_name)
            if not hasattr(module, attribute):
                continue
            original = getattr(module, attribute)
            patched.append((module, attribute, original))
            setattr(module, attribute, _TerminalDownloadProgress)
        yield
    finally:
        for module, attribute, original in reversed(patched):
            setattr(module, attribute, original)


@dataclass(frozen=True)
class PlanningExample:
    episode_id: str
    instruction: str
    operation_sequence: tuple[dict[str, Any], ...]
    image_paths: tuple[Path, ...]
    segmented_image_paths: tuple[Path, ...]
    env_config_path: Path | None
    dependency: Any = "Sequential"
    entities: tuple[str, ...] = ()

    def as_reward_item(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "operation_sequence": list(self.operation_sequence),
            "dependency": self.dependency,
            "entities": self.entities,
        }


def download_processed_datasets(
    planning_dir: str | Path,
    control_dir: str | Path,
    *,
    token: str | None = None,
    max_workers: int = 1,
    retries: int = 8,
    retry_delay: float = 5.0,
) -> tuple[Path, Path]:
    """Download both processed repositories, resuming after transient failures."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if retries < 0:
        raise ValueError("retries cannot be negative")
    if retry_delay < 0:
        raise ValueError("retry_delay cannot be negative")

    planning_dir = Path(planning_dir).resolve()
    control_dir = Path(control_dir).resolve()
    _snapshot_download_with_retry(
        PLANNING_DATASET_ID,
        planning_dir,
        token=token,
        max_workers=max_workers,
        retries=retries,
        retry_delay=retry_delay,
    )
    _snapshot_download_with_retry(
        CONTROL_DATASET_ID,
        control_dir,
        token=token,
        max_workers=max_workers,
        retries=retries,
        retry_delay=retry_delay,
    )
    return planning_dir, control_dir


def _http_status(error: Exception) -> int | None:
    """Find an HTTP status on an exception or its chained cause."""
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
        current = current.__cause__ or current.__context__
    return None


def _retry_after(error: Exception) -> float | None:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", {}) or {}
        value = headers.get("Retry-After")
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                pass
        current = current.__cause__ or current.__context__
    return None


def _is_retryable_download_error(error: Exception) -> bool:
    status = _http_status(error)
    if status == 429 or (status is not None and 500 <= status <= 599):
        return True
    # huggingface_hub uses requests/httpx depending on its version. Avoid a
    # hard dependency on either exception hierarchy while still retrying their
    # transport failures.
    return error.__class__.__name__ in {
        "ConnectError", "ConnectionError", "ProxyError", "ReadTimeout",
        "ReadTimeoutError", "SSLError", "Timeout", "TimeoutError",
    }


def _snapshot_download_with_retry(
    repo_id: str,
    local_dir: Path,
    *,
    token: str | None,
    max_workers: int,
    retries: int,
    retry_delay: float,
) -> None:
    """Run a resumable snapshot download with bounded 429/5xx backoff."""
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} to {local_dir}", flush=True)
    for attempt in range(retries + 1):
        try:
            with _terminal_huggingface_progress():
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=local_dir,
                    token=token,
                    max_workers=max_workers,
                    tqdm_class=_TerminalDownloadProgress,
                )
            return
        except Exception as error:
            if attempt >= retries or not _is_retryable_download_error(error):
                raise
            server_delay = _retry_after(error)
            delay = server_delay if server_delay is not None else retry_delay * (2 ** attempt)
            delay = min(60.0, max(0.0, delay))
            status = _http_status(error)
            reason = f"HTTP {status}" if status is not None else error.__class__.__name__
            print(
                f"{repo_id}: {reason}; resuming in {delay:g}s "
                f"(retry {attempt + 1}/{retries})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_plan(example_dir: Path) -> tuple[Path, Any] | None:
    candidates = sorted((example_dir / "output").glob("*.json")) if (example_dir / "output").is_dir() else []
    candidates.extend(path for path in sorted(example_dir.glob("*.json")) if "operation" in path.name.lower())
    for path in candidates:
        try:
            payload = _read_json(path)
            plan = (
                payload.get("operation_sequence", payload.get("skill_sequence", payload))
                if isinstance(payload, Mapping) else payload
            )
            canonicalize_plan(plan)
            return path, payload
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def _read_instruction(input_dir: Path) -> str:
    for path in sorted(input_dir.glob("*")) if input_dir.is_dir() else ():
        if "instruction" not in path.name.lower() or not path.is_file():
            continue
        if path.suffix.lower() == ".json":
            payload = _read_json(path)
            if isinstance(payload, Mapping):
                return str(payload.get("instruction", payload.get("text", ""))).strip()
            return str(payload).strip()
        return path.read_text(encoding="utf-8").strip()
    return ""


def _extract_entities(config: Any) -> tuple[str, ...]:
    found: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child, key)
        elif "entit" in key.lower() or key.lower() in {"name", "target_entity", "target_container"}:
            text = str(value).strip()
            if text and text not in found:
                found.append(text)

    visit(config)
    return tuple(found)


def load_planning_examples(root: str | Path, *, limit: int | None = None) -> list[PlanningExample]:
    """Group the official image-folder layout back into complete episodes."""
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"planning dataset does not exist: {root}")
    examples: list[PlanningExample] = []
    for output_dir in sorted(path for path in root.rglob("output") if path.is_dir()):
        example_dir = output_dir.parent
        found = _find_plan(example_dir)
        if found is None:
            continue
        _plan_path, payload = found
        plan_payload = (
            payload.get("operation_sequence", payload.get("skill_sequence", payload))
            if isinstance(payload, Mapping) else payload
        )
        input_dir = example_dir / "input"
        image_paths = tuple(sorted(
            path for path in input_dir.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            and not any(word in path.name.lower() for word in ("segment", "mask", "prompt"))
        ))
        segmented_paths = tuple(sorted(
            path for path in input_dir.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            and any(word in path.name.lower() for word in ("segment", "mask", "prompt"))
        ))
        config_candidates = sorted((example_dir / "env_config").rglob("*.json")) if (example_dir / "env_config").exists() else []
        config_path = config_candidates[0] if config_candidates else None
        config = _read_json(config_path) if config_path is not None else {}
        dependency = payload.get("dependency", config.get("dependency", "Sequential")) if isinstance(payload, Mapping) else config.get("dependency", "Sequential")
        episode_id = str(example_dir.relative_to(root)).replace("\\", "/")
        entities = list(_extract_entities(config))
        for operation in canonicalize_plan(plan_payload):
            for key in ("target_entity_name", "target_container_name"):
                value = operation["params"].get(key)
                if value is not None and str(value) not in entities:
                    entities.append(str(value))
        examples.append(PlanningExample(
            episode_id=episode_id,
            instruction=_read_instruction(input_dir),
            operation_sequence=tuple(canonicalize_plan(plan_payload)),
            image_paths=image_paths,
            segmented_image_paths=segmented_paths,
            env_config_path=config_path,
            dependency=dependency,
            entities=tuple(entities),
        ))
        if limit is not None and len(examples) >= limit:
            break
    if not examples:
        raise ValueError(f"no VLABench planning episodes were found under {root}")
    return examples


def deterministic_split(
    items: Sequence[Any],
    *,
    seed: int = 42,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, list[Any]]:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("split fractions are invalid")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_fraction)
    validation_end = train_end + int(len(shuffled) * validation_fraction)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def build_numbered_segmentation_view(
    rgb: Image.Image | np.ndarray,
    segmentation: np.ndarray,
    *,
    background_ids: Iterable[int] = (0,),
) -> tuple[Image.Image, dict[int, tuple[int, int]]]:
    """Overlay stable numeric object pointers at segmentation centroids."""
    image = rgb.convert("RGB") if isinstance(rgb, Image.Image) else Image.fromarray(np.asarray(rgb).astype(np.uint8)).convert("RGB")
    mask = np.asarray(segmentation)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != (image.height, image.width):
        raise ValueError(f"segmentation shape {mask.shape} does not match RGB {(image.height, image.width)}")
    background = {int(value) for value in background_ids}
    centers: dict[int, tuple[int, int]] = {}
    draw = ImageDraw.Draw(image)
    pointer = 0
    for segment_id in sorted(int(value) for value in np.unique(mask) if int(value) not in background):
        ys, xs = np.where(mask == segment_id)
        if not len(xs):
            continue
        center = (int(np.median(xs)), int(np.median(ys)))
        centers[pointer] = center
        text = str(pointer)
        radius = 11
        draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=(255, 230, 0), outline=(0, 0, 0), width=2)
        draw.text((center[0] - 4 * len(text), center[1] - 7), text, fill=(0, 0, 0))
        pointer += 1
    return image, centers


def load_hf_control_records(
    source: str | Path = CONTROL_DATASET_ID,
    *,
    task: str | None = None,
    split: str = "train",
    streaming: bool = False,
):
    """Load LeRobot parquet records from Hugging Face or a downloaded snapshot."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the project dev extra to load Hugging Face datasets") from exc
    source_path = Path(source)
    if source_path.exists():
        # A complete Hub snapshot carries named dataset configurations in its
        # metadata. Let ``datasets`` honor those and preserve Video features
        # before falling back to raw parquet discovery.
        try:
            return load_dataset(
                str(source_path),
                name=task,
                split=split,
                streaming=streaming,
            )
        except (ValueError, FileNotFoundError, RuntimeError):
            pass
        parquet = sorted(source_path.rglob("*.parquet"))
        if task:
            task_files = [path for path in parquet if task.lower() in str(path).lower()]
            if not task_files:
                raise RuntimeError(
                    f"the local snapshot does not expose config {task!r}; load by Hub repo ID "
                    "or install LeRobot so v3 task metadata can be applied"
                )
            parquet = task_files
        if not parquet:
            raise FileNotFoundError(f"no parquet control records found under {source_path}")
        return load_dataset("parquet", data_files={split: [str(path) for path in parquet]}, split=split, streaming=streaming)
    return load_dataset(str(source), name=task, split=split, streaming=streaming)


def _tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().clone().float()
    if isinstance(value, Image.Image):
        array = np.asarray(value.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)
    array = np.asarray(value)
    result = torch.as_tensor(array).float()
    if result.ndim == 3 and result.shape[-1] in {1, 3, 4}:
        result = result[..., :3].permute(2, 0, 1)
    return result


def _video_tensor(
    value: Any,
    *,
    timestamp: float,
    video_root: Path | None,
    cache: dict[str, Any],
) -> torch.Tensor:
    """Decode a LeRobot v3 video reference or an already decoded frame."""
    if hasattr(value, "get_frame_played_at"):
        frame = value.get_frame_played_at(float(timestamp))
        return _tensor(getattr(frame, "data", frame))
    if isinstance(value, Mapping) and ("path" in value or "video_path" in value):
        raw_path = value.get("path", value.get("video_path"))
        path = Path(str(raw_path))
        if not path.is_absolute() and video_root is not None:
            path = video_root / path
        frame_time = float(value.get("timestamp", timestamp))
        key = str(path.resolve())
        decoder = cache.get(key)
        if decoder is None:
            try:
                from torchcodec.decoders import VideoDecoder
            except ImportError as exc:
                raise RuntimeError(
                    "LeRobot v3 video rows require torchcodec (or a datasets build that decodes Video features)"
                ) from exc
            decoder = cache[key] = VideoDecoder(key)
        frame = decoder.get_frame_played_at(frame_time)
        return _tensor(getattr(frame, "data", frame))
    return _tensor(value)


class LeRobotWindowDataset(Dataset):
    """Convert frame records into history/action-horizon training windows."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        observation_horizon: int = 2,
        action_horizon: int = 16,
        image_keys: Sequence[str] | None = None,
        video_root: str | Path | None = None,
        condition_index: int | None = None,
    ):
        if observation_horizon <= 0 or action_horizon <= 0:
            raise ValueError("window horizons must be positive")
        self.records = records if hasattr(records, "__len__") and hasattr(records, "__getitem__") else list(records)
        self.observation_horizon = int(observation_horizon)
        self.action_horizon = int(action_horizon)
        self.image_keys = tuple(image_keys or ())
        self.video_root = Path(video_root).resolve() if video_root is not None else None
        self._video_cache: dict[str, Any] = {}
        self.condition_index = None if condition_index is None else int(condition_index)
        episodes: dict[int, list[int]] = {}
        if hasattr(self.records, "column_names") and "episode_index" in self.records.column_names:
            episode_values = self.records["episode_index"]
            for index, episode in enumerate(episode_values):
                episodes.setdefault(int(episode), []).append(index)
        else:
            for index in range(len(self.records)):
                row = self.records[index]
                episodes.setdefault(int(row.get("episode_index", 0)), []).append(index)
        self.episodes = {episode: tuple(indices) for episode, indices in episodes.items()}
        self.index = [(episode, offset) for episode, indices in self.episodes.items() for offset in range(len(indices))]

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _value(row: Mapping[str, Any], *keys: str):
        for key in keys:
            if key in row:
                return row[key]
        raise KeyError(f"record has none of the required keys {keys}")

    def _images(self, row: Mapping[str, Any]) -> torch.Tensor:
        if "images" in row:
            images = _tensor(row["images"])
            return images.unsqueeze(0) if images.ndim == 3 else images
        keys = self.image_keys or tuple(sorted(key for key in row if "image" in key.lower()))
        if not keys:
            raise KeyError("control record contains no image views")
        timestamp = float(np.asarray(row.get("timestamp", 0.0)).reshape(-1)[0])
        return torch.stack([
            _video_tensor(
                row[key],
                timestamp=timestamp,
                video_root=self.video_root,
                cache=self._video_cache,
            )
            for key in keys
        ])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode, offset = self.index[item]
        indices = self.episodes[episode]
        obs_offsets = [max(0, offset - self.observation_horizon + 1 + delta) for delta in range(self.observation_horizon)]
        action_offsets = [min(len(indices) - 1, offset + delta) for delta in range(self.action_horizon)]
        obs_rows = [self.records[indices[position]] for position in obs_offsets]
        action_rows = [self.records[indices[position]] for position in action_offsets]
        state = torch.stack([
            _tensor(self._value(row, "state", "observation.state", "q_state")).reshape(-1)[:7]
            for row in obs_rows
        ])
        images = torch.stack([self._images(row) for row in obs_rows])
        actions = torch.stack([
            _tensor(self._value(row, "actions", "action", "trajectory")).reshape(-1)[:7]
            for row in action_rows
        ])
        task_index = self.condition_index
        if task_index is None:
            task_index = int(obs_rows[-1].get("task_index", 0))
        return {
            "state": state,
            "images": images,
            "actions": actions,
            "task_index": torch.tensor(task_index, dtype=torch.long),
            "episode_index": torch.tensor(episode, dtype=torch.long),
        }
