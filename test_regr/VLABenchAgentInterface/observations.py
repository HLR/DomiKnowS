"""Explicit image range and live camera selection for controller inputs."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


# The LeRobot control export uses these names in this stable slot order.  The
# live VLABench camera names are simulator-specific, so they are resolved by
# semantic aliases below instead of assuming that renderer order is stable.
DEFAULT_CONTROLLER_CAMERA_KEYS = ("image", "second_image", "wrist_image")


def image_tensor(value, *, channels_last=None):
    """Convert RGB byte images or unit-range floats to [..., C, H, W].

    Tensor/decoder inputs are CHW; PIL and numpy camera frames are HWC.
    Float byte images are accepted, but negative/standardized images are not.
    Keep this separate from numeric state/action conversion.
    """
    if isinstance(value, Image.Image):
        value = np.asarray(value.convert("RGB")).copy()
        channels_last = True
    if channels_last is None:
        channels_last = not torch.is_tensor(value)
        shape = np.shape(value)
        if len(shape) >= 3 and shape[-3] == 3 and shape[-1] != 3:
            channels_last = False
    result = torch.as_tensor(value)
    is_byte = result.dtype == torch.uint8
    if result.ndim < 3:
        raise ValueError("RGB images must have at least three dimensions")
    if channels_last:
        result = result.movedim(-1, -3)
    if result.shape[-3] != 3:
        raise ValueError("controller images must have exactly three RGB channels")
    result = result.float()
    if not bool(torch.isfinite(result).all()) or result.numel() == 0:
        raise ValueError("controller images must be nonempty and finite")
    low, high = float(result.min()), float(result.max())
    if low < 0 or high > 255:
        raise ValueError("controller RGB input must be in [0, 1] or [0, 255]")
    if is_byte or high > 1:
        result = result / 255.0
    return result.contiguous()


def live_camera_names(env, observation):
    count = len(observation["rgb"])
    names = observation.get("camera_names")
    if names is not None:
        if len(names) != count:
            raise ValueError("camera_names does not match the RGB view count")
        return list(map(str, names))
    model = getattr(getattr(env, "physics", None), "model", None)
    if model is not None and callable(getattr(model, "id2name", None)):
        return [model.id2name(index, "camera") for index in range(count)]
    return [None] * count


def camera_indices(env, observation, names=None, *, max_views=3):
    if names is None:
        return list(range(min(max_views, len(observation["rgb"]))))
    available = live_camera_names(env, observation)
    if not names or len(set(names)) != len(names):
        raise ValueError("controller camera names must be nonempty and unique")
    if any(available.count(name) != 1 for name in names):
        raise ValueError(f"controller camera names {names!r} are not unique in {available!r}")
    return [available.index(name) for name in names]


def resolve_controller_camera_names(
    env,
    observation,
    *,
    dataset_keys=DEFAULT_CONTROLLER_CAMERA_KEYS,
    requested=None,
):
    """Resolve learned camera slots to live names using semantic aliases.

    Positional renderer order is not a camera contract: on VLABench the
    third view is commonly ``forward`` while the learned third slot is
    ``wrist_image``.  Explicit CLI names still take precedence, while the
    default path chooses the unique right/left/wrist aliases and fails closed
    when a required slot cannot be identified.
    """
    keys = tuple(str(key) for key in (dataset_keys or DEFAULT_CONTROLLER_CAMERA_KEYS))
    if not keys:
        raise ValueError("at least one dataset camera key is required")
    if requested is not None:
        names = tuple(str(name) for name in requested)
        if len(names) != len(keys):
            raise ValueError(
                f"controller camera names ({len(names)}) must match dataset slots ({len(keys)})"
            )
        # camera_indices performs the uniqueness/availability validation.
        camera_indices(env, observation, names)
        return names, "explicit"

    available = live_camera_names(env, observation)
    # Synthetic/legacy environments may expose RGB views without camera
    # identities.  Preserve their positional contract; there is no semantic
    # pairing claim to make in that case.
    if available and all(name is None for name in available):
        return None, "positional-fallback"
    unused = set(range(len(available)))
    normalized = ["".join(ch for ch in key.lower() if ch.isalnum()) for key in keys]

    def candidates(key):
        if "wrist" in key or "hand" in key:
            return [
                index for index, name in enumerate(available)
                if index in unused and name is not None
                and ("wrist" in name.lower() or "hand" in name.lower())
            ]
        if key in {"image", "mainimage", "primaryimage"}:
            preferred = ("right", "front", "forward", "overhead", "image")
        elif "second" in key or key in {"leftimage", "sideimage"}:
            preferred = ("left", "side", "forward", "front", "image")
        else:
            preferred = ()
        result = []
        for token in preferred:
            result.extend(
                index for index, name in enumerate(available)
                if index in unused and name is not None and name.lower() == token
            )
        if result:
            return result
        return [
            index for index, name in enumerate(available)
            if index in unused and name is not None and any(token in name.lower() for token in preferred)
        ]

    selected = []
    for key in normalized:
        matches = candidates(key)
        if not matches:
            if "wrist" in key or "hand" in key:
                raise ValueError(
                    f"cannot resolve required wrist camera slot {key!r} from {available!r}"
                )
            # Preserve a useful fallback for anonymous/synthetic observations,
            # but never steal a camera already assigned to another slot.
            matches = sorted(unused)
        if not matches:
            raise ValueError(f"cannot resolve dataset camera slot {key!r} from {available!r}")
        index = matches[0]
        selected.append(available[index])
        unused.remove(index)
    camera_indices(env, observation, selected)
    return tuple(selected), "dataset-alias"


def camera_report(env, observation, *, indices=None, dataset_keys=()):
    names = live_camera_names(env, observation)
    indices = list(indices if indices is not None else range(min(3, len(names))))
    rgb = image_tensor(np.asarray(observation["rgb"]), channels_last=True)
    result = {
        "status": "unverified",  # Names and matching counts alone do not prove semantics.
        "mapping_status": "configured" if dataset_keys and all(names[index] is not None for index in indices) else "unverified",
        "dataset_keys": list(dataset_keys),
        "selected_indices": indices,
        "selected_names": [names[index] for index in indices],
        "all_live_names": names,
        "rgb_range": [float(rgb.min()), float(rgb.max())],
        "rgb_shape": list(rgb.shape),
        "views": [],
    }
    for slot, index in enumerate(indices):
        view = {"slot": slot, "live_index": index, "live_name": names[index]}
        if slot < len(dataset_keys):
            view["dataset_key"] = dataset_keys[slot]
        for source, name in (("instrinsic", "intrinsic"), ("extrinsic", "extrinsic")):
            if source in observation:
                view[name] = np.asarray(observation[source])[index].tolist()
        result["views"].append(view)
    return result
