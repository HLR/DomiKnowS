"""Download and load the processed VLABench planning/control datasets."""

from __future__ import annotations

import importlib
import json
import random
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

try:
    from .world_graph import canonicalize_plan, controller_skill_index
except ImportError:
    from world_graph import canonicalize_plan, controller_skill_index


PLANNING_DATASET_ID = "VLABench/vlm_evaluation_v1.0"
CONTROL_DATASET_ID = "VLABench/vlabench_primitive_ft_lerobot_video"


def _instruction_key(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return " ".join(str(value).strip().lower().rstrip(".!?").split())


def load_control_task_instructions(
    source: str | Path = CONTROL_DATASET_ID,
) -> dict[int, str]:
    """Load the canonical LeRobot ``task_index -> instruction`` metadata."""

    root = Path(source)
    if not root.exists():
        try:
            from huggingface_hub import hf_hub_download

            metadata = Path(hf_hub_download(
                repo_id=str(source),
                filename="meta/tasks.parquet",
                repo_type="dataset",
            ))
        except Exception as exc:
            raise RuntimeError(
                "control task metadata is required to preserve language-conditioned task_index values"
            ) from exc
        candidates = [metadata]
    else:
        candidates = [root / "meta" / "tasks.parquet"]
        candidates.extend(sorted((root / "meta" / "tasks").glob("**/*.parquet")))

    rows: list[Mapping[str, Any]] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read LeRobot task metadata") from exc
        rows.extend(parquet.read_table(path).to_pylist())

    legacy = root / "meta" / "tasks.jsonl"
    if not rows and legacy.is_file():
        rows = [json.loads(line) for line in legacy.read_text(encoding="utf-8").splitlines() if line.strip()]
    result: dict[int, str] = {}
    for row in rows:
        index = row.get("task_index")
        instruction = row.get("task", row.get("instruction", row.get("__index_level_0__")))
        if index is not None and instruction is not None and _instruction_key(instruction):
            result[int(index)] = str(instruction)
    if not result:
        raise RuntimeError(f"no task_index/instruction mappings found under {source}")
    return result


def control_task_index_for_instruction(
    instruction: Any,
    task_instructions: Mapping[int, str],
) -> int:
    """Resolve an environment instruction to its demonstration task ID."""

    requested = _instruction_key(instruction)
    matches = [
        int(index)
        for index, value in task_instructions.items()
        if _instruction_key(value) == requested
    ]
    if len(matches) != 1:
        raise KeyError(f"instruction is not uniquely represented in control metadata: {instruction!r}")
    return matches[0]


class _TerminalDownloadProgress:
    """Dependency-free progress reporter implementing the tqdm surface Hub uses.

    It deliberately prints periodic newline-delimited updates instead of using
    cursor control. This remains visible in redirected server logs and avoids
    terminal/notebook renderer failures in ``tqdm.auto``.
    """

    _lock = threading.RLock()

    def __init__(
        self,
        iterable=None,
        *,
        total=None,
        initial=0,
        desc=None,
        unit="it",
        unit_scale=False,
        disable=False,
        mininterval=2.0,
        **_kwargs,
    ):
        self.iterable = iterable
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                pass
        self.total = total
        self.n = initial or 0
        self.desc = desc or "Progress"
        self.unit = unit
        self.unit_scale = unit_scale
        self.disable = bool(disable)
        self.mininterval = max(0.1, float(mininterval or 2.0))
        self.start_t = time.monotonic()
        self.last_print_t = self.start_t
        self._last_print_state = None
        self._postfix = ""
        self._closed = False
        self._render(force=True)

    @classmethod
    def get_lock(cls):
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock

    @staticmethod
    def format_sizeof(value, suffix="", divisor=1000):
        if value is None:
            return "???"
        value = float(value)
        for prefix in ("", "k", "M", "G", "T", "P"):
            if abs(value) < divisor:
                return f"{value:3.1f}{prefix}{suffix}" if prefix else f"{value:g}{suffix}"
            value /= divisor
        return f"{value:.1f}E{suffix}"

    @property
    def format_dict(self):
        elapsed = max(time.monotonic() - self.start_t, 1e-9)
        return {"rate": self.n / elapsed, "elapsed": elapsed, "n": self.n, "total": self.total}

    def __iter__(self):
        if self.iterable is None:
            return
        try:
            for item in self.iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def update(self, n=1):
        self.n += n or 0
        self._render()

    def refresh(self, *_, **__):
        self._render()
        return True

    def set_description(self, desc=None, refresh=True):
        self.desc = desc or "Progress"
        if refresh:
            self._render()

    set_description_str = set_description

    def set_postfix_str(self, postfix="", refresh=True):
        self._postfix = str(postfix).strip()
        if refresh:
            self._render()

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        values = dict(ordered_dict or {})
        values.update(kwargs)
        self.set_postfix_str(", ".join(f"{key}={value}" for key, value in values.items()), refresh)

    def reset(self, total=None):
        self.n = 0
        if total is not None:
            self.total = total
        self.start_t = time.monotonic()
        self.last_print_t = self.start_t
        self._last_print_state = None
        self._closed = False
        self._render(force=True)

    def close(self):
        if not self._closed:
            self._render(force=True)
            self._closed = True

    @classmethod
    def write(cls, message, file=None, end="\n", nolock=False):
        output = file or sys.stderr
        lock = None if nolock else cls.get_lock()
        if lock is None:
            output.write(f"{message}{end}")
            output.flush()
            return
        with lock:
            output.write(f"{message}{end}")
            output.flush()

    def _render(self, force=False):
        if self.disable:
            return
        now = time.monotonic()
        complete = self.total not in (None, 0) and self.n >= self.total
        if not force and not complete and now - self.last_print_t < self.mininterval:
            return
        state = (self.desc, self.n, self.total, self._postfix)
        if state == self._last_print_state:
            return
        elapsed = max(now - self.start_t, 1e-9)
        rate = self.n / elapsed
        current = self._format_amount(self.n)
        if self.total not in (None, 0):
            total = self._format_amount(self.total)
            percent = 100.0 * self.n / self.total
            amount = f"{current}/{total} ({percent:5.1f}%)"
        else:
            amount = current
        rate_text = f"{self._format_amount(rate)}/s" if self.n else ""
        extras = " ".join(value for value in (rate_text, self._postfix) if value)
        line = f"{self.desc}: {amount}"
        if extras:
            line = f"{line} [{extras}]"
        self.write(line)
        self.last_print_t = now
        self._last_print_state = state

    def _format_amount(self, value):
        if self.unit_scale:
            suffix = "B" if self.unit == "B" else self.unit
            return self.format_sizeof(value, suffix=suffix)
        return f"{value:g} {self.unit}"


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
        ("huggingface_hub.utils._xet_progress_reporting", "tqdm"),
        ("huggingface_hub.file_download", "tqdm"),
        ("huggingface_hub._snapshot_download", "hf_tqdm"),
    )
    resolved_targets = []
    for module_name, attribute in targets:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name:
                continue
            raise
        resolved_targets.append((module, attribute))
    patched = []
    try:
        for module, attribute in resolved_targets:
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
    count = len(shuffled)
    train_end = max(1, int(count * train_fraction)) if count else 0
    remaining = count - train_end
    validation_count = min(int(count * validation_fraction), max(0, remaining - 1))
    if validation_fraction and remaining >= 2:
        validation_count = max(1, validation_count)
    validation_end = train_end + validation_count
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


def _release_video_decoder(decoder: Any) -> None:
    """Release future closeable decoders; dropping the wrapper frees current TorchCodec handles."""

    close = getattr(decoder, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _video_tensor(
    value: Any,
    *,
    timestamp: float,
    video_root: Path | None,
    cache: dict[str, Any],
    cache_size: int = 8,
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
        decoder = cache.pop(key, None)
        if decoder is None:
            try:
                from torchcodec.decoders import VideoDecoder
            except ImportError as exc:
                raise RuntimeError(
                    "LeRobot v3 video rows require torchcodec (or a datasets build that decodes Video features)"
                ) from exc
            decoder = VideoDecoder(key)
        cache[key] = decoder
        while len(cache) > max(1, int(cache_size)):
            oldest_key = next(iter(cache))
            evicted = cache.pop(oldest_key)
            _release_video_decoder(evicted)
            del evicted
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
        plan_pattern: Sequence[str] = (),
        video_decoder_cache_size: int = 8,
    ):
        if observation_horizon <= 0 or action_horizon <= 0:
            raise ValueError("window horizons must be positive")
        self.records = records if hasattr(records, "__len__") and hasattr(records, "__getitem__") else list(records)
        self.observation_horizon = int(observation_horizon)
        self.action_horizon = int(action_horizon)
        self.image_keys = tuple(image_keys or ())
        self.video_root = Path(video_root).resolve() if video_root is not None else None
        if int(video_decoder_cache_size) <= 0:
            raise ValueError("video decoder cache size must be positive")
        self.video_decoder_cache_size = int(video_decoder_cache_size)
        self._video_cache: dict[str, Any] = {}
        self.video_keys: tuple[str, ...] = ()
        self.video_path_template = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        self.video_chunk_size = 1000
        if self.video_root is not None:
            info_path = self.video_root / "meta" / "info.json"
            if info_path.is_file():
                info = _read_json(info_path)
                self.video_path_template = str(info.get("video_path", self.video_path_template))
                self.video_chunk_size = int(info.get("chunks_size", self.video_chunk_size))
                features = info.get("features", {})
                self.video_keys = tuple(
                    key for key, feature in features.items()
                    if isinstance(feature, Mapping) and feature.get("dtype") == "video"
                )
            if not self.video_keys:
                videos = self.video_root / "videos"
                if videos.is_dir():
                    self.video_keys = tuple(sorted(path.name for path in videos.iterdir() if path.is_dir()))
        self.condition_index = None if condition_index is None else int(condition_index)
        self.plan_pattern = tuple(str(skill) for skill in plan_pattern)
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

    def close(self) -> None:
        while self._video_cache:
            _key, decoder = self._video_cache.popitem()
            _release_video_decoder(decoder)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
        timestamp = float(np.asarray(row.get("timestamp", 0.0)).reshape(-1)[0])
        if keys:
            return torch.stack([
                _video_tensor(
                    row[key],
                    timestamp=timestamp,
                    video_root=self.video_root,
                    cache=self._video_cache,
                    cache_size=self.video_decoder_cache_size,
                )
                for key in keys
            ])
        if self.video_root is None or not self.video_keys:
            available = ", ".join(sorted(map(str, row.keys())))
            raise KeyError(
                "control record contains no inline image views and no LeRobot video metadata; "
                f"available columns: {available}"
            )
        episode = int(np.asarray(row.get("episode_index", 0)).reshape(-1)[0])
        paths = [
            self.video_root / self.video_path_template.format(
                video_key=key,
                chunk_index=episode // self.video_chunk_size,
                file_index=episode % self.video_chunk_size,
                episode_index=episode,
            )
            for key in self.video_keys
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            raise FileNotFoundError(
                "LeRobot control video files are missing from the downloaded snapshot: "
                f"{preview}. Resume the VLABench control dataset download."
            )
        return torch.stack([
            _video_tensor(
                {"path": str(path), "timestamp": timestamp},
                timestamp=timestamp,
                video_root=self.video_root,
                cache=self._video_cache,
                cache_size=self.video_decoder_cache_size,
            )
            for path in paths
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
        operation_index = 0
        skill_index = 0
        if self.plan_pattern:
            # Demonstrations do not carry graph-operation boundaries.  Use a
            # deterministic normalized episode phase for operation-aware BC;
            # online rollout replaces it with the planner's actual cursor.
            operation_index = min(
                len(self.plan_pattern) - 1,
                int(offset * len(self.plan_pattern) / max(1, len(indices))),
            )
            skill_index = controller_skill_index(self.plan_pattern[operation_index])
        return {
            "state": state,
            "images": images,
            "actions": actions,
            "task_index": torch.tensor(task_index, dtype=torch.long),
            "plan_context": torch.tensor(
                [skill_index, 0, operation_index + 1 if self.plan_pattern else 0],
                dtype=torch.long,
            ),
            "episode_index": torch.tensor(episode, dtype=torch.long),
        }
