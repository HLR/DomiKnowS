"""Cache frozen VLM image features for GraphQA predicate scoring.

The cache contains image-dependent tensors only. Predicate text and bounding-box
coordinates remain prompt inputs, so one cached image can serve every relevant
Name, Attribute, and Relation query for that image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_IMAGE_DIR = Path("/egr/research-hlr2/premsrit/VQAR_data/image_cache")
DEFAULT_CACHE_ROOT = Path("/egr/research-hlr2/premsrit/GraphQA/image_embedding_cache")


def _safe_model_name(model_path: str) -> str:
    name = Path(model_path.rstrip("/")).name or model_path
    digest = hashlib.sha1(model_path.encode("utf-8")).hexdigest()[:10]
    return f"{name.replace('/', '_')}-{digest}"


def _cpu_bf16(value):
    if torch.is_tensor(value):
        value = value.detach().to("cpu")
        return value.to(torch.bfloat16) if value.is_floating_point() else value
    if isinstance(value, (list, tuple)):
        return [_cpu_bf16(item) for item in value]
    return value


def _atomic_torch_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class QwenVLExtractor:
    def __init__(self, model_path: str, device: str):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def __call__(self, image: Image.Image):
        inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.model.dtype)
        image_grid_thw = inputs["image_grid_thw"].to(self.device)
        output = self.model.model.get_image_features(
            pixel_values,
            image_grid_thw,
            return_dict=True,
        )
        return {
            "pooler_output": _cpu_bf16(output.pooler_output),
            "deepstack_features": _cpu_bf16(output.deepstack_features),
            "image_grid_thw": image_grid_thw.cpu(),
        }


class InternVLExtractor:
    def __init__(self, model_path: str, device: str, image_size: int, max_patches: int):
        # Reuse the project's compatibility shims for InternVL remote code.
        from test_regr.Clever import peftvllm as internvl_helpers

        self._helpers = internvl_helpers
        self.model = internvl_helpers.AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            use_flash_attn=False,
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.image_size = image_size
        self.max_patches = max_patches

    @torch.inference_mode()
    def __call__(self, image: Image.Image):
        pixel_values = self._helpers.load_image_to_tiles(
            image,
            input_size=self.image_size,
            max_num=self.max_patches,
            use_thumbnail=True,
        ).to(self.device, dtype=torch.bfloat16)
        features = self.model.extract_feature(pixel_values)
        return {
            "visual_features": _cpu_bf16(features),
            "num_patches": int(pixel_values.shape[0]),
        }


def _image_files(image_dir: Path, limit: int | None):
    files = sorted(image_dir.glob("*.jpg"))
    if limit is not None:
        files = files[:limit]
    return files


def build_cache(args):
    model_cache = args.output_root / args.backend / _safe_model_name(args.model_path)
    feature_dir = model_cache / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    files = _image_files(args.image_dir, args.limit)
    selected = files[args.shard_index :: args.num_shards]
    extractor = (
        QwenVLExtractor(args.model_path, args.device)
        if args.backend == "qwen-vl"
        else InternVLExtractor(
            args.model_path,
            args.device,
            args.internvl_image_size,
            args.internvl_max_patches,
        )
    )

    manifest = {
        "schema_version": 1,
        "backend": args.backend,
        "model_path": args.model_path,
        "image_dir": str(args.image_dir),
        "dtype": "bfloat16",
        "bbox_grounding": "text_coordinates",
        "num_shards": args.num_shards,
        "internvl_image_size": args.internvl_image_size,
        "internvl_max_patches": args.internvl_max_patches,
    }
    (model_cache / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    written = skipped = failed = 0
    failures = model_cache / f"failures-shard-{args.shard_index:03d}.jsonl"
    progress = tqdm(selected, desc=f"{args.backend} shard {args.shard_index}", unit="image")
    for image_path in progress:
        output_path = feature_dir / f"{image_path.stem}.pt"
        if output_path.is_file() and not args.overwrite:
            skipped += 1
            continue
        try:
            with Image.open(image_path) as raw_image:
                image = raw_image.convert("RGB")
                payload = extractor(image)
                payload.update({
                    "schema_version": 1,
                    "backend": args.backend,
                    "model_path": args.model_path,
                    "image_id": image_path.stem,
                    "image_size": list(image.size),
                })
            _atomic_torch_save(payload, output_path)
            written += 1
        except Exception as error:
            failed += 1
            with failures.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "image": str(image_path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }) + "\n")
        progress.set_postfix(written=written, skipped=skipped, failed=failed)

    summary = {
        "selected": len(selected),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "cache": str(model_cache),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if failed == 0 else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["qwen-vl", "internvl"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--internvl-image-size", type=int, default=448)
    parser.add_argument("--internvl-max-patches", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    return args


if __name__ == "__main__":
    raise SystemExit(build_cache(parse_args()))
