"""Load each embodied planner with the common VLM and run a tiny forward/backward pass."""

from __future__ import annotations

import argparse
import json

import torch
from PIL import Image

from domiknows.generation.dfa.vocabulary import TokenVocabulary
from test_regr.common_backbone import COMMON_VLM_MODEL_ID


def _gradient_check(planner, logits) -> dict:
    logits.float().square().mean().backward()
    gradients = [p.grad for name, p in planner.named_parameters()
                 if "lora_" in name and p.grad is not None]
    if not gradients or not any(bool(torch.isfinite(g).all() and g.abs().sum() > 0) for g in gradients):
        raise RuntimeError("common VLM LoRA gradient did not reach the backbone")
    return {"lora_gradient_tensors": len(gradients)}


def _eai(model_path: str, backward: bool) -> dict:
    from test_regr.EmbodiedAgentInterface.modules import CausalLMActionObjectGenerator

    vocabulary = TokenVocabulary(("walk", "chair_1", "sit", "<eos>"), eos_token="<eos>")
    planner = CausalLMActionObjectGenerator(
        model_path=model_path,
        label_count=vocabulary.label_count,
        eos_label=vocabulary.eos_label,
        vocabulary=vocabulary,
        device="cuda:0",
        device_map="auto",
        use_lora=True,
        max_length=128,
    )
    planner.train(mode=backward)
    with torch.enable_grad() if backward else torch.no_grad():
        logits = planner.next_label_logits(
            [vocabulary.eos_label],
            text="Walk to the chair and sit on it.",
        )
    gradient_result = _gradient_check(planner, logits) if backward else {}
    return {
        "component": "eai",
        "model_type": planner.model.config.model_type,
        "hidden_size": planner.output.weight.shape[-1],
        "logit_shape": list(logits.shape),
        "finite": bool(torch.isfinite(logits).all()),
        "trainable_parameters": planner.trainable_parameter_count(),
        **gradient_result,
    }


def _vlabench(model_path: str, backward: bool) -> dict:
    from test_regr.VLABenchAgentInterface.graph import PlanVocabulary
    from test_regr.VLABenchAgentInterface.models import QwenVLPlanner
    from test_regr.VLABenchAgentInterface.world_graph import build_vlabench_world_graph

    world = build_vlabench_world_graph()
    vocabulary = PlanVocabulary.from_world(world, max_entities=4)
    planner = QwenVLPlanner.from_pretrained(
        vocabulary, model_id=model_path,
        load_in_4bit=True, local_files_only=True,
    )
    planner.train(mode=backward)
    with torch.enable_grad() if backward else torch.no_grad():
        logits = planner.next_label_logits(
            {"instruction": "Put the apple in the bowl.",
             "entity_table": ("apple", "bowl"),
             "images": (Image.new("RGB", (224, 224), "white"),)},
            (),
        )
    gradient_result = _gradient_check(planner, logits) if backward else {}
    return {
        "component": "vlabench",
        "model_type": planner.model.config.model_type,
        "hidden_size": planner.backbone_hidden_size,
        "logit_shape": list(logits.shape),
        "finite": bool(torch.isfinite(logits).all()),
        "trainable_parameters": sum(p.numel() for p in planner.parameters() if p.requires_grad),
        **gradient_result,
    }


def _joint(model_path: str, backward: bool) -> dict:
    from test_regr.JointEmbodiedAgentInterface.models import JointQwenVLPlanner
    from test_regr.JointEmbodiedAgentInterface.world_graph import build_joint_world_graph
    from test_regr.VLABenchAgentInterface.graph import PlanVocabulary

    world = build_joint_world_graph()
    eai_vocabulary = TokenVocabulary(("walk", "chair_1", "sit", "<eos>"), eos_token="<eos>")
    vlabench_vocabulary = PlanVocabulary.from_world(world.vlabench, max_entities=4)
    planner = JointQwenVLPlanner.from_pretrained(
        eai_vocabulary=eai_vocabulary,
        vlabench_vocabulary=vlabench_vocabulary,
        model_id=model_path,
        load_in_4bit=True, local_files_only=True,
    )
    planner.train(mode=backward)
    with torch.enable_grad() if backward else torch.no_grad():
        eai_logits = planner.next_label_logits(
            "eai", {"instruction": "Walk to the chair and sit on it.", "goal": "sitting"}, (),
        )
        vlabench_logits = planner.next_label_logits(
            "vlabench",
            {"instruction": "Put the apple in the bowl.",
             "entity_table": ("apple", "bowl"),
             "images": (Image.new("RGB", (224, 224), "white"),)},
            (),
        )
    gradient_result = _gradient_check(planner, eai_logits + vlabench_logits[:eai_logits.numel()]) if backward else {}
    return {
        "component": "joint",
        "model_type": planner.model.config.model_type,
        "hidden_size": planner.backbone_hidden_size,
        "eai_logit_shape": list(eai_logits.shape),
        "vlabench_logit_shape": list(vlabench_logits.shape),
        "finite": bool(torch.isfinite(eai_logits).all() and torch.isfinite(vlabench_logits).all()),
        "trainable_parameters": sum(p.numel() for p in planner.parameters() if p.requires_grad),
        **gradient_result,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("eai", "vlabench", "joint"), required=True)
    parser.add_argument("--model-path", default=COMMON_VLM_MODEL_ID)
    parser.add_argument("--backward", action="store_true", help="also verify LoRA gradient flow")
    args = parser.parse_args(argv)
    result = {"eai": _eai, "vlabench": _vlabench, "joint": _joint}[args.component](args.model_path, args.backward)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["model_type"] != "qwen3_vl" or not result["finite"]:
        raise RuntimeError("common Qwen3-VL backbone smoke test failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
