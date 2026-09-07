"""Explicit image range and live camera selection for controller inputs."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


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


def camera_report(env, observation, *, indices=None, dataset_keys=()):
    names = live_camera_names(env, observation)
    indices = list(indices if indices is not None else range(min(3, len(names))))
    rgb = image_tensor(np.asarray(observation["rgb"]), channels_last=True)
    result = {
        "status": "unverified",  # Names and matching counts alone do not prove semantics.
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
