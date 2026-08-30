"""Atomic, compatibility-checked checkpoints for the joint workflow."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .world_graph import JointDomainRuntime


JOINT_CHECKPOINT_VERSION = 3
SUPPORTED_JOINT_CHECKPOINT_VERSIONS = frozenset({1, 2, JOINT_CHECKPOINT_VERSION})


def _planner_trainable_state(planner: torch.nn.Module) -> Mapping[str, Any]:
    """Return LoRA/other trainable planner parameters without frozen NF4 state.

    bitsandbytes adds serialization-only quantization tensors (``absmax``,
    ``quant_map`` and ``quant_state``) to a module state dict.  They are not
    registered load targets on a newly quantized model, and the immutable base
    model is already reconstructed from the compatibility-checked model id.
    """
    trainable = {
        name for name, parameter in planner.named_parameters()
        if parameter.requires_grad
    }
    state = planner.state_dict()
    return {name: value for name, value in state.items() if name in trainable}


def _is_bitsandbytes_auxiliary_key(name: str) -> bool:
    return any(
        marker in name
        for marker in (".weight.absmax", ".weight.quant_map", ".weight.quant_state.")
    )


def _cpu_rng_state(value: Any) -> torch.Tensor:
    """Normalize RNG tensors before restoring process-global RNG state."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.detach().to(device="cpu", dtype=torch.uint8)


def _checkpoint_staging_location(map_location: str | torch.device):
    """Keep CUDA checkpoint tensors on CPU until their owners restore them.

    ``Optimizer.load_state_dict`` moves parameter-shaped Adam moments to each
    parameter's device and dtype.  Non-capturable Adam step counters are the
    exception: PyTorch deliberately leaves them on their loaded device.  If a
    checkpoint is loaded wholesale with ``map_location="cuda"``, those scalar
    counters become CUDA float32 tensors and foreach Adam rejects them when a
    corresponding parameter is BF16/FP16.  CPU staging preserves the required
    counter placement while module and optimizer loading still move every
    parameter-owned tensor to its actual device.
    """
    try:
        requested = torch.device(map_location)
    except (TypeError, RuntimeError):
        return map_location
    return torch.device("cpu") if requested.type == "cuda" else map_location


def _load_planner_state(planner: torch.nn.Module, state: Mapping[str, Any]) -> None:
    required = {
        name for name, parameter in planner.named_parameters()
        if parameter.requires_grad
    }
    absent = sorted(required.difference(state))
    if absent:
        preview = ", ".join(absent[:5])
        raise RuntimeError(f"joint checkpoint is missing trainable planner state: {preview}")
    result = planner.load_state_dict(state, strict=False)
    unexpected = [
        name for name in result.unexpected_keys
        if not _is_bitsandbytes_auxiliary_key(name)
    ]
    if unexpected:
        preview = ", ".join(unexpected[:5])
        raise RuntimeError(f"joint checkpoint has unexpected planner state: {preview}")


def _vocabulary_payload(value: Any) -> Mapping[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    return {
        "tokens": list(value.tokens),
        "eos_token": value.eos_token,
        "other_token": getattr(value, "other_token", "_other"),
    }


def _model_configuration(planner) -> Mapping[str, Any]:
    model = planner.model
    config = getattr(model, "config", None)
    quantization = getattr(config, "quantization_config", None)
    if hasattr(quantization, "to_dict"):
        quantization = quantization.to_dict()
    elif quantization is not None:
        quantization = repr(quantization)
    peft = getattr(model, "peft_config", None)
    if isinstance(peft, Mapping):
        peft = {
            str(name): value.to_dict() if hasattr(value, "to_dict") else repr(value)
            for name, value in peft.items()
        }
    return {
        "model_type": getattr(config, "model_type", type(model).__name__),
        "name_or_path": getattr(config, "_name_or_path", None),
        "hidden_size": getattr(config, "hidden_size", getattr(config, "d_model", None)),
        "quantization": quantization,
        "peft": peft,
        "head_sizes": {
            name: int(head.out_features) for name, head in planner.label_heads.items()
        },
        "graph_decoder_version": getattr(planner, "graph_decoder_version", None),
        "graph_decoder_hidden_size": getattr(planner, "decoder_hidden_size", None),
    }


def _dfa_configuration(dfa: Any, **limits) -> Mapping[str, Any]:
    transitions = getattr(dfa, "transitions", {})
    states = getattr(dfa, "states", ())
    accepting = getattr(dfa, "accepting_states", ())
    return {
        **limits,
        "start_state": repr(dfa.start_state),
        "states": sorted(repr(value) for value in states),
        "accepting_states": sorted(repr(value) for value in accepting),
        "transitions": sorted(
            (repr(state), int(symbol), repr(target))
            for (state, symbol), target in transitions.items()
        ),
    }


def _compatibility(runtime: JointDomainRuntime, planner) -> Mapping[str, Any]:
    return {
        "combined_domain_checksum": runtime.world.combined_checksum,
        "eai_domain_checksum": runtime.world.eai_domain_checksum,
        "vlabench_domain_checksum": runtime.world.vlabench_domain_checksum,
        "runtime_checksum": runtime.runtime_checksum,
        "activation_profile_version": runtime.activation_profile_version,
        "eai_vocabulary": _vocabulary_payload(runtime.eai_vocabulary),
        "vlabench_vocabulary": _vocabulary_payload(runtime.vlabench_vocabulary),
        "eai_dfa": _dfa_configuration(runtime.eai_dfa, max_steps=runtime.max_eai_steps),
        "vlabench_dfa": _dfa_configuration(
            runtime.vlabench_dfa,
            max_operations=runtime.max_vlabench_operations,
        ),
        "model_configuration": _model_configuration(planner),
    }


def save_joint_checkpoint(
    path: str | Path,
    *,
    runtime: JointDomainRuntime,
    planner: torch.nn.Module,
    controller: torch.nn.Module,
    planner_optimizer: torch.optim.Optimizer | None,
    controller_optimizer: torch.optim.Optimizer | None,
    stage: str,
    epoch: int,
    round_robin_cursor: int,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Save backbone/LoRA once, both heads, PPO state, optimizers, and RNG."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "joint_checkpoint_version": JOINT_CHECKPOINT_VERSION,
        "compatibility": _compatibility(runtime, planner),
        "stage": str(stage),
        "epoch": int(epoch),
        "round_robin_cursor": int(round_robin_cursor),
        "planner": _planner_trainable_state(planner),
        "controller": controller.state_dict(),
        "planner_optimizer": planner_optimizer.state_dict() if planner_optimizer is not None else None,
        "controller_optimizer": controller_optimizer.state_dict() if controller_optimizer is not None else None,
        "metrics": dict(metrics or {}),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def load_joint_checkpoint(
    path: str | Path,
    *,
    runtime: JointDomainRuntime,
    planner: torch.nn.Module,
    controller: torch.nn.Module,
    planner_optimizer: torch.optim.Optimizer | None = None,
    controller_optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    """Restore an exact joint run, rejecting any schema/model drift."""
    payload = torch.load(
        Path(path),
        map_location=_checkpoint_staging_location(map_location),
        weights_only=False,
    )
    if payload.get("joint_checkpoint_version") not in SUPPORTED_JOINT_CHECKPOINT_VERSIONS:
        raise ValueError("unsupported or standalone checkpoint; a joint checkpoint is required")
    expected = _compatibility(runtime, planner)
    actual = payload.get("compatibility", {})
    for key in (
        "combined_domain_checksum",
        "eai_domain_checksum",
        "vlabench_domain_checksum",
        "runtime_checksum",
        "activation_profile_version",
        "eai_vocabulary",
        "vlabench_vocabulary",
        "eai_dfa",
        "vlabench_dfa",
        "model_configuration",
    ):
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"joint checkpoint {key} differs from the current runtime: "
                f"saved={actual.get(key)!r}, current={expected.get(key)!r}"
            )
    _load_planner_state(planner, payload["planner"])
    controller.load_state_dict(payload["controller"])
    if planner_optimizer is not None and payload.get("planner_optimizer") is not None:
        planner_optimizer.load_state_dict(payload["planner_optimizer"])
    if controller_optimizer is not None and payload.get("controller_optimizer") is not None:
        controller_optimizer.load_state_dict(payload["controller_optimizer"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(_cpu_rng_state(payload["torch_rng"]))
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all([
            _cpu_rng_state(state) for state in payload["cuda_rng"]
        ])
    runtime.activate_domain(None)
    runtime._domain_stack.clear()
    return payload
