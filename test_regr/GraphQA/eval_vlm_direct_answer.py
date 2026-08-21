#!/usr/bin/env python3
"""Direct VLM baseline for VQAR/GraphQA.

This baseline gives a vision-language model the original Visual Genome image,
object ids with bounding boxes, and a rendered version of the VQAR executable
clauses.  It does not use the DomiKnowS executable graph; it scores candidate
object-id continuations directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from .dataset import DEFAULT_VQAR_ROOT, load_vqar_tasks

DEFAULT_QWEN_VL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_CACHE = Path("/egr/research-hlr2/premsrit/VQAR_data/image_cache")


def parse_args():
    p = argparse.ArgumentParser(description="Direct Qwen-VL/InternVL image+question baseline for VQAR GraphQA.")
    p.add_argument("--root", type=Path, default=DEFAULT_VQAR_ROOT)
    p.add_argument("--task-path", type=Path, required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--model-path", default=DEFAULT_QWEN_VL)
    p.add_argument("--backend", choices=["auto", "qwen-vl", "internvl"], default="auto")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--image-cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--draw-boxes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true", help="Only print rendered prompts; do not load a VLM or download images.")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def render_question(task):
    q = task.get("question", task)
    clauses = q.get("clauses", []) or []
    parts = []
    for clause in clauses:
        fn = clause.get("function")
        text = clause.get("text_input")
        if fn == "Initial":
            continue
        if fn == "Find_Attr":
            parts.append(f"has attribute {text}")
        elif fn == "Find_Name":
            parts.append(f"is named {text}")
        elif fn == "Hypernym_Find":
            parts.append(f"belongs to semantic class {text}")
        elif fn == "KG_Find":
            if isinstance(text, (list, tuple)) and len(text) >= 3:
                left, rel, right = text[:3]
                parts.append(f"has knowledge relation {rel} to {right if right not in ('', 'BLANK', None) else left}")
            else:
                parts.append(f"satisfies knowledge relation {text}")
        elif fn == "Relate":
            parts.append(f"is the object reached by relation {text} from the previous candidate set")
        elif fn == "Relate_Reverse":
            parts.append(f"has relation {text} to an object in the previous candidate set")
        elif fn in {"And", "Or"}:
            parts.append(fn.lower())
        else:
            parts.append(f"{fn}({text})")
    if not parts:
        return "Which candidate object answers the visual question?"
    return "Which candidate object " + " and ".join(parts) + "?"


def candidate_text(task, max_candidates=80):
    q = task.get("question", task)
    objects = [str(o) for o in q.get("input", []) or task.get("object_ids", [])]
    bboxes = (task.get("scene_graph", {}) or {}).get("bboxes", {})
    rows = []
    for obj in objects[:max_candidates]:
        box = bboxes.get(obj)
        if box is None:
            try:
                box = bboxes.get(int(obj))
            except Exception:
                box = None
        rows.append(f"{obj}: bbox={box}")
    if len(objects) > max_candidates:
        rows.append(f"... {len(objects) - max_candidates} more candidates omitted")
    return "\n".join(rows)


def prompt_for_task(task):
    return "\n".join([
        "Task: answer a knowledge-based visual question from the image.",
        "Use the image and the listed object bounding boxes. Return exactly one object id from the candidates.",
        f"Question: {render_question(task)}",
        "Candidate object ids and boxes:",
        candidate_text(task),
        "Answer object id:",
    ])


def image_cache_path(task, cache_dir):
    image_id = task.get("image_id") or (task.get("question", {}) or {}).get("image_id")
    return Path(cache_dir) / f"{image_id}.jpg"


def load_image(task, cache_dir, draw_boxes=True):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = image_cache_path(task, cache_dir)
    if not path.exists():
        import requests
        url = task.get("url")
        if not url:
            raise FileNotFoundError(f"No cached image and no URL for image_id={task.get('image_id')}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
    image = Image.open(path).convert("RGB")
    if draw_boxes:
        image = image.copy()
        draw = ImageDraw.Draw(image)
        w, h = image.size
        bboxes = (task.get("scene_graph", {}) or {}).get("bboxes", {})
        for obj in (task.get("question", {}) or {}).get("input", [])[:40]:
            box = bboxes.get(str(obj), bboxes.get(obj))
            if box is None:
                continue
            x, y, bw, bh = [float(v) for v in box]
            # VQAR bboxes are normalized x,y,w,h.
            xyxy = [x * w, y * h, (x + bw) * w, (y + bh) * h]
            draw.rectangle(xyxy, outline="red", width=2)
            draw.text((xyxy[0], max(0, xyxy[1] - 10)), str(obj), fill="red")
    return image


class InternVLDirect:
    def __init__(self, model_path, device="cuda", max_length=2048):
        import sys
        clever_dir = Path(__file__).resolve().parents[1] / "Clever"
        if str(clever_dir) not in sys.path:
            sys.path.insert(0, str(clever_dir))
        from peftvllm import InternVLHF
        self.model = InternVLHF(model_path=model_path, device=device, max_num_patches=1)

    @torch.no_grad()
    def score_choices(self, image, prompt, choices: Iterable[str]):
        scores = []
        for choice in choices:
            question = "\n".join([
                prompt,
                f"Candidate answer object id: {choice}",
                "Is this candidate the correct answer? Answer Yes or No.",
            ])
            score = self.model._score(image, question, target_tokens=["No", "Yes"], max_num=1)
            scores.append(float(score[1].detach().cpu().item()))
        return torch.tensor(scores)


class QwenVLDirect:
    def __init__(self, model_path, device="cuda", max_length=2048):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if str(device).startswith("cuda"):
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs).to(device)
        self.model.eval()
        self.device = device
        self.max_length = int(max_length)

    def _messages(self, image, prompt):
        return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    @torch.no_grad()
    def score_choices(self, image, prompt, choices: Iterable[str]):
        scores = []
        for choice in choices:
            messages = self._messages(image, prompt + " " + str(choice))
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True).to(self.device)
            out = self.model(**inputs, labels=inputs["input_ids"])
            # Length-normalized negative loss. This scores the whole prompt+choice,
            # so it is a simple direct baseline, not a calibrated executor.
            scores.append(-float(out.loss.detach().cpu().item()))
        return torch.tensor(scores)


def main():
    args = parse_args()
    tasks = load_vqar_tasks(args.task_path, limit=args.limit)
    if args.dry_run:
        for task in tasks[: min(3, len(tasks))]:
            print("image_url=", task.get("url"))
            print(prompt_for_task(task))
            print("gold=", (task.get("question", {}) or {}).get("output"))
            print("---")
        return 0
    backend = args.backend
    if backend == "auto":
        backend = "internvl" if "internvl" in str(args.model_path).lower() else "qwen-vl"
    if backend == "internvl":
        model = InternVLDirect(args.model_path, args.device, args.max_length)
    else:
        model = QwenVLDirect(args.model_path, args.device, args.max_length)
    total = 0
    correct = 0
    iterator = tqdm(tasks, desc="GraphQA direct VLM", dynamic_ncols=True) if args.progress else tasks
    for task in iterator:
        gold = [str(x) for x in (task.get("question", {}) or {}).get("output", [])]
        choices = [str(x) for x in (task.get("question", {}) or {}).get("input", [])]
        if len(gold) != 1 or not choices:
            continue
        image = load_image(task, args.image_cache, draw_boxes=args.draw_boxes)
        prompt = prompt_for_task(task)
        scores = model.score_choices(image, prompt, choices)
        pred = choices[int(torch.argmax(scores).item())]
        correct += int(pred == gold[0])
        total += 1
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(acc=f"{correct / max(total, 1):.3f}", n=total)
    print(json.dumps({"total": total, "correct": correct, "accuracy": correct / total if total else 0.0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
