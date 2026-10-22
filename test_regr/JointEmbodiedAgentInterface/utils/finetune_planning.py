"""Balanced planner-head adaptation with fixed official splits and validation selection.

Shared Qwen/LoRA, EAI head, and controller remain frozen. This isolates pointer
and sequence supervision from changes to the robot controller or EAI behavior.
"""
import hashlib
import json
from pathlib import Path
import random
import torch
import torch.nn.functional as F

from ..eval_planning import build_eval_parser, _prepare_eval
from ..models import JointQwenVLPlanner
from ..checkpoint import load_joint_checkpoint, save_joint_checkpoint
from ..program import _runtime_adapter
from domiknows.generation.validation import retains_validation_scores
from test_regr.VLABenchAgentInterface.main import _controller
from test_regr.VLABenchAgentInterface.plan_codec import plan_to_tokens
from test_regr.VLABenchAgentInterface.planning_augmentation import permute_entities
from test_regr.VLABenchAgentInterface.training import _planner_images, evaluate_planner


def selection_scores(metrics):
    planning = metrics["planning_metrics"]
    return {"overall": planning["overall"]["score"], "macro": planning["Macro"],
            "negative_generation_failures": -metrics["generation_failures"],
            **{name: values["score"] for name, values in planning["dimensions"].items() if values["examples"]}}


def validation_plans(planner, runtime, examples):
    print("[adaptation] Generating full validation plans for checkpoint retention", flush=True)
    with runtime.domain_scope("vlabench"):
        return evaluate_planner(planner.for_domain("vlabench"), examples, _runtime_adapter(runtime, "vlabench"))


def frozen_digest(planner, controller):
    digest = hashlib.sha256()
    for name, parameter in list(planner.named_parameters()) + [("controller." + n, p) for n, p in controller.named_parameters()]:
        if name.startswith("label_heads.vlabench."):
            continue
        if name.startswith(("label_heads.eai.", "controller.")) or "lora_" in name:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def balanced_examples(examples, count, rng, focus_fraction=0.75):
    groups = {}
    for example in examples:
        groups.setdefault(example.task, []).append(example)
    focus = [task for task, values in groups.items() if values[0].dimension in ("Complex", "PhysLaw")]
    other = [task for task in groups if task not in focus]
    for _ in range(count):
        pool = focus if rng.random() < focus_fraction else other
        task = rng.choice(pool or list(groups))
        yield rng.choice(groups[task])


def loss_for(planner, runtime, example, rng=None):
    images = _planner_images(example)
    try:
        context = {**example.as_reward_item(), "entity_table": example.entities, "images": images}
        plan = example.operation_sequence
        if rng is not None:
            order = list(range(len(example.entities)))
            rng.shuffle(order)
            context, plan = permute_entities(context, plan, order)
        labels = torch.tensor([runtime.vlabench_vocabulary.label_for_token(token) for token in
                               plan_to_tokens(plan, context["entity_table"], world=runtime.world.vlabench)], device=planner.device)
        logits = planner.sequence_logits("vlabench", context, planner.shift_right("vlabench", labels))[0]
        losses = F.cross_entropy(logits, labels, reduction="none")
        weights = torch.tensor([2.0 if runtime.vlabench_vocabulary.token_for_label(int(label)).startswith("obj:") else 1.0 for label in labels], device=planner.device)
        return (losses * weights).sum() / weights.sum()
    finally:
        for image in images:
            image.close()


def main(argv=None):
    parser = build_eval_parser()
    parser.description = __doc__
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume-adaptation", type=Path)
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.samples_per_epoch <= 0 or args.learning_rate <= 0:
        parser.error("epochs, samples-per-epoch and learning-rate must be positive")
    if args.output is None:
        parser.error("--output must name a checkpoint directory")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.vlabench_planning_dir = str(args.planning_dir)
    device = torch.device(args.device)
    runtime, splits = _prepare_eval(args, device)
    planner = JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary, vlabench_vocabulary=runtime.vlabench_vocabulary,
                                                model_id=args.planner_model, load_in_4bit=args.load_in_4bit,
                                                decoder_hidden_size=args.planner_decoder_hidden_dim)
    controller = _controller(args, device)
    payload = load_joint_checkpoint(args.resume_adaptation or args.checkpoint, runtime=runtime, planner=planner, controller=controller,
                                     map_location=device, allow_planning_schema_migration=not bool(args.resume_adaptation))
    # Retain all learned adapters in checkpoints even though this experiment freezes them.
    planner._checkpoint_parameter_names = {name for name, param in planner.named_parameters() if param.requires_grad}
    for param in planner.parameters():
        param.requires_grad_(False)
    for param in controller.parameters():
        param.requires_grad_(False)
    head = planner.label_heads["vlabench"]
    for param in head.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    protected_digest = frozen_digest(planner, controller)
    start = 0
    best_loss = float("inf")
    best_macro = float("-inf")
    baseline_planning = None
    if args.resume_adaptation:
        if "best_validation_macro" not in payload["metrics"]:
            raise ValueError("loss-selected prototype checkpoints require a fresh warm start via --checkpoint, not exact --resume-adaptation")
        optimizer.load_state_dict(payload["planner_optimizer"])
        start = int(payload["epoch"]) + 1
        best_loss = float(payload["metrics"]["best_validation_loss"])
        best_macro = float(payload["metrics"]["best_validation_macro"])
        baseline_planning = payload["metrics"]["baseline_planning"]
        rng.setstate(payload["metrics"]["sampler_rng"])
        if not (args.output / "planner_best.pt").exists():
            import shutil
            previous_best = args.resume_adaptation.parent / "planner_best.pt"
            if not previous_best.exists():
                raise ValueError("resume requires the previous planner_best.pt to preserve validation selection")
            shutil.copy2(previous_best, args.output / "planner_best.pt")
    provenance = {"source_checkpoint": str(args.checkpoint.resolve()), "train_ids": [e.episode_id for e in splits["train"]],
                  "validation_ids": [e.episode_id for e in splits["validation"]], "test_used_for_selection": False,
                  "observation_mode": args.observation_mode, "max_operations": args.max_operations,
                  "seed": args.seed, "backbone_frozen": True, "eai_head_frozen": True, "controller_frozen": True,
                  "focus_sampling_fraction": 0.75, "pointer_loss_weight": 2.0}
    provenance["selection"] = "generated_validation_macro_with_retention_v1"
    provenance["frozen_parameter_sha256"] = protected_digest
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    planner.eval()
    if baseline_planning is None:
        baseline_planning = validation_plans(planner, runtime, splits["validation"])
        best_macro = float(baseline_planning["planning_metrics"]["Macro"])
        save_joint_checkpoint(args.output / "planner_best.pt", runtime=runtime, planner=planner, controller=controller,
            planner_optimizer=optimizer, controller_optimizer=None, stage="planning_head_adaptation", epoch=-1,
            round_robin_cursor=0, metrics={"baseline_planning": baseline_planning, "best_validation_macro": best_macro,
                                          "best_validation_loss": best_loss, "sampler_rng": rng.getstate(), "selection": provenance["selection"]})
    for epoch in range(start, args.epochs):
        total = 0.0
        with runtime.domain_scope("vlabench"):
            for step, example in enumerate(balanced_examples(splits["train"], args.samples_per_epoch, rng), 1):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for(planner, runtime, example, rng)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss: {example.episode_id}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach())
                if step == 1 or step % 16 == 0:
                    print(json.dumps({"phase": "train", "epoch": epoch, "step": step, "steps": args.samples_per_epoch, "loss": total / step}), flush=True)
            with torch.no_grad():
                validation = 0.0
                for step, example in enumerate(splits["validation"], 1):
                    validation += float(loss_for(planner, runtime, example))
                    if step % 50 == 0:
                        print(json.dumps({"phase": "validation", "epoch": epoch, "step": step, "steps": len(splits["validation"])}), flush=True)
            validation /= len(splits["validation"])
        measured = validation_plans(planner, runtime, splits["validation"])
        eligible = retains_validation_scores(selection_scores(measured), selection_scores(baseline_planning))
        macro = float(measured["planning_metrics"]["Macro"])
        improved = eligible and macro > best_macro
        if frozen_digest(planner, controller) != protected_digest:
            raise RuntimeError("frozen EAI/backbone/controller parameters changed")
        best_loss = min(best_loss, validation)
        best_macro = max(best_macro, macro) if eligible else best_macro
        metrics = {"train_loss": total / args.samples_per_epoch, "validation_loss": validation, "best_validation_loss": best_loss,
                   "sampler_rng": rng.getstate(), "selection": provenance["selection"], "source_stage": payload.get("stage"),
                   "baseline_planning": baseline_planning, "validation_planning": measured, "retention_eligible": eligible,
                   "best_validation_macro": best_macro}
        target = args.output / f"planner_epoch_{epoch:03d}.pt"
        save_joint_checkpoint(target, runtime=runtime, planner=planner, controller=controller, planner_optimizer=optimizer,
                              controller_optimizer=None, stage="planning_head_adaptation", epoch=epoch,
                              round_robin_cursor=0, metrics=metrics)
        if improved:
            import shutil
            shutil.copy2(target, args.output / "planner_best.pt")
        print(json.dumps({"phase": "epoch_complete", "epoch": epoch, "train_loss": metrics["train_loss"], "validation_loss": validation, "best": improved, "checkpoint": str(target)}), flush=True)


if __name__ == "__main__":
    main()
