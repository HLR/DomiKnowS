"""Text-only VLABench planning warm-up for the EAI causal-LM backbone."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

try:
    from .modules import CausalLMActionObjectGenerator, EOSMaskedCrossEntropyLoss
except ImportError:  # Direct execution through EmbodiedAgentInterface/main.py.
    from modules import CausalLMActionObjectGenerator, EOSMaskedCrossEntropyLoss


VLABENCH_PROMPT_KEY = "vlabench_planner_prompt_text"


@dataclass(frozen=True)
class AuxiliaryTrainingResult:
    """Best text-only auxiliary epoch and its restored shared LoRA state."""

    selected_epoch: int
    validation_metrics: Mapping[str, float]
    shared_trainable_snapshot: Mapping[str, torch.Tensor] = field(repr=False)
    vocabulary_checksum: str = ""
    domain_checksum: str = ""
    checkpoint_path: Path | None = None


def _example_value(example: Any, key: str, default=None):
    if isinstance(example, Mapping):
        return example.get(key, default)
    return getattr(example, key, default)


def prepare_vlabench_text_examples(examples, runtime) -> list[dict[str, Any]]:
    """Encode VLABench plans and prompts without opening any image path."""
    from test_regr.VLABenchAgentInterface.graph import plan_to_tokens
    from test_regr.VLABenchAgentInterface.models import planner_prompt

    prepared: list[dict[str, Any]] = []
    for example in examples:
        entities = tuple(_example_value(example, "entities", ()))
        operation_sequence = _example_value(example, "operation_sequence", ())
        tokens = plan_to_tokens(
            operation_sequence,
            entities,
            world=runtime.world_bundle,
        )
        labels = [runtime.vocabulary.label_for_token(token) for token in tokens]
        if len(labels) > runtime.max_tokens:
            episode_id = _example_value(example, "episode_id", "<unknown>")
            raise ValueError(
                f"VLABench episode {episode_id} requires {len(labels)} labels, "
                f"but the graph runtime permits {runtime.max_tokens}"
            )
        if not runtime.dfa.accepts(labels):
            episode_id = _example_value(example, "episode_id", "<unknown>")
            raise ValueError(
                f"VLABench reference plan {episode_id} is rejected by its graph DFA"
            )
        padded = labels + [runtime.vocabulary.eos_label] * (
            runtime.max_tokens - len(labels)
        )
        instruction = str(_example_value(example, "instruction", ""))
        prepared.append(
            {
                "episode_id": str(_example_value(example, "episode_id", "")),
                VLABENCH_PROMPT_KEY: planner_prompt(
                    instruction, entities, runtime.vocabulary
                ),
                "target_plan_labels": torch.tensor(padded, dtype=torch.long),
                "operation_sequence": operation_sequence,
                "dependency": _example_value(example, "dependency", "Sequential"),
                "instruction": instruction,
                "entities": entities,
            }
        )
    return prepared


def _identity_prompt(value: Any) -> str:
    return str(value)


def _resolve_plan_pointers(plan, entity_table):
    """Convert compact obj:N pointers back to names for semantic metrics."""
    entities = dict(entity_table) if isinstance(entity_table, Mapping) else dict(
        enumerate(entity_table)
    )
    resolved = []
    for operation in plan:
        params = {}
        for key, value in operation.get("params", {}).items():
            params[key] = entities.get(value, value) if isinstance(value, int) else value
        resolved.append({"name": operation["name"], "params": params})
    return resolved


def _snapshot_named_trainable(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _restore_named_trainable(
    module: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]
) -> None:
    parameters = dict(module.named_parameters())
    missing = sorted(set(snapshot) - set(parameters))
    if missing:
        raise ValueError(f"auxiliary checkpoint has unknown shared parameters: {missing[:3]}")
    with torch.no_grad():
        for name, value in snapshot.items():
            parameter = parameters[name]
            if tuple(parameter.shape) != tuple(value.shape):
                raise ValueError(
                    f"auxiliary shared parameter shape mismatch for {name}: "
                    f"saved={tuple(value.shape)}, current={tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _independent_parameters(*groups: Iterable[torch.nn.Parameter]):
    seen: set[int] = set()
    result = []
    for group in groups:
        for parameter in group:
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result


def _build_auxiliary_head(
    eai_head,
    runtime,
    *,
    device,
    max_length,
    label_head,
    label_adapter_rank,
):
    return CausalLMActionObjectGenerator(
        label_count=runtime.vocabulary.label_count,
        eos_label=runtime.vocabulary.eos_label,
        device=device,
        max_length=max_length,
        freeze=False,
        vocabulary=runtime.vocabulary,
        shared_model=eai_head.model,
        shared_tokenizer=eai_head.tokenizer,
        label_head=label_head,
        label_adapter_rank=label_adapter_rank,
        prompt_builder=_identity_prompt,
        prompt_key=VLABENCH_PROMPT_KEY,
    )


@torch.no_grad()
def _evaluate_auxiliary(auxiliary_head, examples, runtime, loss_function):
    from domiknows.generation import constrained_label_greedy_decode
    from test_regr.VLABenchAgentInterface.graph import labels_to_plan
    from test_regr.VLABenchAgentInterface.reward import score_vlabench_plan

    auxiliary_head.eval()
    totals = {
        "loss": 0.0,
        "reward": 0.0,
        "skill_match": 0.0,
        "entity_match": 0.0,
        "skill_with_entity_match": 0.0,
        "exact_graph_match": 0.0,
        "valid": 0.0,
    }
    count = 0
    for count, item in enumerate(examples, start=1):
        prompt = item[VLABENCH_PROMPT_KEY]
        targets = item["target_plan_labels"].to(auxiliary_head.output.weight.device)
        logits = auxiliary_head(None, prompt, targets)
        totals["loss"] += float(loss_function(logits, targets).detach())
        decoded = constrained_label_greedy_decode(
            auxiliary_head,
            [runtime.vocabulary.eos_label],
            runtime.vocabulary,
            runtime.dfa,
            max_new_tokens=runtime.max_tokens,
            eos_label=runtime.vocabulary.eos_label,
            next_label_kwargs={"text": prompt},
        )
        prediction = labels_to_plan(
            decoded.labels,
            runtime.vocabulary,
            world=runtime.world_bundle,
        )
        prediction = _resolve_plan_pointers(prediction, item["entities"])
        score = score_vlabench_plan(
            prediction,
            item["operation_sequence"],
            item["dependency"],
            entity_table=item["entities"],
            world_bundle=runtime.world_bundle,
        )
        totals["reward"] += score.total
        totals["valid"] += float(decoded.accepted and score.valid)
        for key in (
            "skill_match",
            "entity_match",
            "skill_with_entity_match",
            "exact_graph_match",
        ):
            totals[key] += float(getattr(score, key))
    return {key: value / max(1, count) for key, value in totals.items()} | {
        "examples": count
    }


def auxiliary_selection_key(metrics: Mapping[str, float]):
    """Order epochs by graph semantics first and validation loss last."""
    return (
        float(metrics.get("exact_graph_match", 0.0)),
        float(metrics.get("skill_with_entity_match", 0.0)),
        float(metrics.get("entity_match", 0.0)),
        float(metrics.get("skill_match", 0.0)),
        float(metrics.get("valid", 0.0)),
        -float(metrics.get("loss", float("inf"))),
    )


def _save_auxiliary_checkpoint(
    path: str | Path,
    *,
    selected_epoch: int,
    metrics: Mapping[str, float],
    shared_snapshot: Mapping[str, torch.Tensor],
    auxiliary_head,
    runtime,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "vlabench_aux_checkpoint_version": 1,
            "selected_epoch": int(selected_epoch),
            "validation_metrics": dict(metrics),
            "vocabulary_checksum": runtime.vocabulary.checksum,
            "domain_checksum": runtime.world_bundle.domain_checksum,
            "shared_trainable": dict(shared_snapshot),
            "auxiliary_output": auxiliary_head.output.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)
    return path


def load_vlabench_auxiliary_checkpoint(path, eai_head, auxiliary_head, runtime):
    """Restore a compatible auxiliary checkpoint or fail with a clear cause."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("vlabench_aux_checkpoint_version") != 1:
        raise ValueError(f"unsupported VLABench auxiliary checkpoint: {path}")
    expected = {
        "vocabulary_checksum": runtime.vocabulary.checksum,
        "domain_checksum": runtime.world_bundle.domain_checksum,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"incompatible VLABench auxiliary {key}: "
                f"saved={payload.get(key)!r}, current={value!r}"
            )
    _restore_named_trainable(eai_head.model, payload["shared_trainable"])
    auxiliary_head.output.load_state_dict(payload["auxiliary_output"])
    return payload


def train_vlabench_text_auxiliary(
    eai_head,
    examples,
    runtime,
    *,
    epochs: int,
    lr: float,
    device="cpu",
    max_length: int = 256,
    label_head: str = "pretrained-adapter",
    label_adapter_rank: int = 64,
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    auxiliary_head_factory: Callable[..., torch.nn.Module] | None = None,
) -> AuxiliaryTrainingResult:
    """Warm the shared LoRA on text-only VLABench plans, then restore the best epoch."""
    if int(epochs) <= 0:
        raise ValueError("VLABench auxiliary epochs must be positive")
    if float(lr) <= 0:
        raise ValueError("VLABench auxiliary learning rate must be positive")
    if isinstance(examples, Mapping):
        train_source = examples.get("train", ())
        validation_source = examples.get("validation", ())
    else:
        try:
            from .dataset import split_vlabench_auxiliary_examples
        except ImportError:
            from dataset import split_vlabench_auxiliary_examples

        split = split_vlabench_auxiliary_examples(list(examples))
        train_source = split["train"]
        validation_source = split["validation"]
    train_examples = prepare_vlabench_text_examples(train_source, runtime)
    validation_examples = prepare_vlabench_text_examples(validation_source, runtime)
    if not train_examples:
        raise ValueError("VLABench auxiliary split contains no training episodes")
    if not validation_examples:
        raise ValueError("VLABench auxiliary split contains no validation episodes")

    factory = auxiliary_head_factory or _build_auxiliary_head
    auxiliary_head = factory(
        eai_head,
        runtime,
        device=device,
        max_length=max_length,
        label_head=label_head,
        label_adapter_rank=label_adapter_rank,
    )
    if auxiliary_head.model is not eai_head.model:
        raise ValueError("the VLABench auxiliary head must share the EAI Qwen/LoRA object")

    eai_adapter_before = {
        name: value.detach().cpu().clone()
        for name, value in eai_head.output.state_dict().items()
    }
    if resume_path is not None:
        load_vlabench_auxiliary_checkpoint(
            resume_path, eai_head, auxiliary_head, runtime
        )

    shared_parameters = [
        parameter
        for parameter in eai_head.model.parameters()
        if parameter.requires_grad
    ]
    if not shared_parameters:
        raise ValueError(
            "VLABench auxiliary warm-up requires trainable shared backbone parameters"
        )
    parameters = _independent_parameters(
        shared_parameters, auxiliary_head.output.parameters()
    )
    optimizer = torch.optim.AdamW(parameters, lr=float(lr))
    loss_function = EOSMaskedCrossEntropyLoss(runtime.vocabulary.eos_label)
    best_key = None
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_shared: dict[str, torch.Tensor] | None = None
    best_output: dict[str, torch.Tensor] | None = None

    try:
        for epoch in range(1, int(epochs) + 1):
            auxiliary_head.train()
            for item in train_examples:
                prompt = item[VLABENCH_PROMPT_KEY]
                target = item["target_plan_labels"].to(
                    auxiliary_head.output.weight.device
                )
                optimizer.zero_grad(set_to_none=True)
                logits = auxiliary_head(None, prompt, target)
                loss = loss_function(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
            metrics = _evaluate_auxiliary(
                auxiliary_head, validation_examples, runtime, loss_function
            )
            print(
                f"VLABench Auxiliary Epoch {epoch} Eval: "
                f"examples={metrics['examples']} loss={metrics['loss']:.6f} "
                f"exact_graph_match={metrics['exact_graph_match']:.3f} "
                f"skill_with_entity_match={metrics['skill_with_entity_match']:.3f} "
                f"entity_match={metrics['entity_match']:.3f} "
                f"skill_match={metrics['skill_match']:.3f} "
                f"dfa_valid={metrics['valid']:.3f}"
            )
            key = auxiliary_selection_key(metrics)
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch
                best_metrics = dict(metrics)
                best_shared = _snapshot_named_trainable(eai_head.model)
                best_output = {
                    name: value.detach().cpu().clone()
                    for name, value in auxiliary_head.output.state_dict().items()
                }

        if best_shared is None or best_output is None or best_metrics is None:
            raise RuntimeError("VLABench auxiliary training produced no checkpoint")
        _restore_named_trainable(eai_head.model, best_shared)
        auxiliary_head.output.load_state_dict(best_output)
        print(f"Restored best VLABench auxiliary epoch {best_epoch}.")

        for name, before in eai_adapter_before.items():
            after = eai_head.output.state_dict()[name].detach().cpu()
            if not torch.equal(before, after):
                raise RuntimeError(
                    "VLABench auxiliary training modified the EAI label adapter"
                )

        saved_path = None
        if checkpoint_path is not None:
            saved_path = _save_auxiliary_checkpoint(
                checkpoint_path,
                selected_epoch=best_epoch,
                metrics=best_metrics,
                shared_snapshot=best_shared,
                auxiliary_head=auxiliary_head,
                runtime=runtime,
            )
            print(f"Saved VLABench auxiliary checkpoint: {saved_path}")
        return AuxiliaryTrainingResult(
            selected_epoch=best_epoch,
            validation_metrics=best_metrics,
            shared_trainable_snapshot=best_shared,
            vocabulary_checksum=runtime.vocabulary.checksum,
            domain_checksum=runtime.world_bundle.domain_checksum,
            checkpoint_path=saved_path,
        )
    finally:
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        del auxiliary_head
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
