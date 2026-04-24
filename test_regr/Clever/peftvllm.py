from __future__ import annotations

import os
import json
import time
import logging
import traceback
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
from contextlib import nullcontext

# Suppress noisy logs from transformers
for noisy_module in [
    "transformers_modules",
    "transformers",
    "transformers.configuration_utils",
    "transformers.modeling_utils",
    "transformers.tokenization_utils",
]:
    logging.getLogger(noisy_module).setLevel(logging.WARNING)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Monkey-patch torch.utils.checkpoint.checkpoint to default to use_reentrant=False.
# InternVL's custom model code calls checkpoint() directly without passing use_reentrant,
# which defaults to True. With use_reentrant=True, gradient checkpointing requires inputs
# to have requires_grad=True, but with QLoRA only internal LoRA weights need grad —
# so gradients die silently. use_reentrant=False handles this correctly.
import torch.utils.checkpoint as _ckpt
_orig_checkpoint_fn = _ckpt.checkpoint
def _patched_checkpoint(*args, use_reentrant=False, **kwargs):
    return _orig_checkpoint_fn(*args, use_reentrant=use_reentrant, **kwargs)
_ckpt.checkpoint = _patched_checkpoint

# Monkey-patch transformers.verify_tp_plan to tolerate a string _tp_plan.
# InternVL's remote modeling code declares ``_tp_plan = ''`` as a "disabled"
# sentinel (for transformers 4.51+ compat), but transformers 4.56.1's
# ``verify_tp_plan`` calls ``.copy()`` on it and crashes. Treating any
# non-dict plan as "no plan" restores the pre-4.56 behavior. We patch the
# original module *and* any modules that have already imported the name.
try:
    import transformers.integrations.tensor_parallel as _tp_mod
    import transformers.modeling_utils as _mu_mod
    _orig_verify_tp_plan = _tp_mod.verify_tp_plan
    def _safe_verify_tp_plan(expected_keys, tp_plan):
        if tp_plan is not None and not isinstance(tp_plan, dict):
            tp_plan = None
        return _orig_verify_tp_plan(expected_keys, tp_plan)
    _tp_mod.verify_tp_plan = _safe_verify_tp_plan
    if getattr(_mu_mod, "verify_tp_plan", None) is _orig_verify_tp_plan:
        _mu_mod.verify_tp_plan = _safe_verify_tp_plan
except Exception:
    pass

# Monkey-patch _move_missing_keys_from_meta_to_device to tolerate a missing
# ``all_tied_weights_keys`` attribute. Transformers 4.56.1 expects every
# PreTrainedModel to have that dict populated by ``post_init()``, but
# InternVL's remote ``InternVLChatModel`` only defines the legacy
# ``_tied_weights_keys`` list and doesn't trigger the new init path, so the
# loader crashes in ``_finalize_model_loading``. An empty dict means "no
# tied weights to exclude", which matches the older transformers behaviour.
try:
    import transformers.modeling_utils as _mu_mod  # noqa: F811
    _orig_move_missing = _mu_mod.PreTrainedModel._move_missing_keys_from_meta_to_device
    def _move_missing_with_tied_fallback(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig_move_missing(self, *args, **kwargs)
    _mu_mod.PreTrainedModel._move_missing_keys_from_meta_to_device = _move_missing_with_tied_fallback
except Exception:
    pass

from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, get_scheduler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Compatibility shim: newer transformers (>=4.50) bnb-4bit quantizer calls
#     `model.all_tied_weights_keys`, but trust_remote_code models (like
#     InternVLChatModel) only define `_tied_weights_keys`. Expose an alias on
#     PreTrainedModel so the quantizer path resolves it without AttributeError.
try:
    from transformers import PreTrainedModel as _PTM
    if not hasattr(_PTM, "all_tied_weights_keys"):
        def _all_tied_weights_keys_compat(self):
            keys = getattr(self, "_tied_weights_keys", None)
            return list(keys) if keys else []
        _PTM.all_tied_weights_keys = property(_all_tied_weights_keys_compat)
except Exception as _shim_err:
    logging.warning("Failed to install all_tied_weights_keys compat shim: %s", _shim_err)

# Optional imports for training with probabilistic tensors
try:
    from probabilistic_tensor import ProbabilisticTensor, and_op, or_op
    PROB_TENSOR_AVAILABLE = True
except ImportError:
    PROB_TENSOR_AVAILABLE = False
    ProbabilisticTensor = None

# =============================================================================
# 1. Configuration Dataclasses
# =============================================================================

@dataclass
class PathConfig:
    """Paths for data and model outputs."""
    annotation_dir: str = "/egr/research-hlr2/kamalida/refcoco/refcocog"
    image_dir: str = "/egr/research-hlr2/kamalida/data/refgta"
    output_dir: str = "/egr/research-hlr2/kamalida/model_weights/refcocog_lora_adapters_spatial457"
    metadata_file: str = "/localscratch/kamalida/projects/NeSyPython/training_spatial457.json"

@dataclass
class ModelConfig:
    """Model configuration."""
    model_path: str = "OpenGVLab/InternVL2_5-1B-MPO"
    load_4bit: bool = False
    device: str = "cuda"

@dataclass
class LoraConfigWrap:
    """LoRA configuration wrapper."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 1
    lr: float = 4e-5
    batch_size: int = 1  # MUST be 1 due to exec() per-sample logic
    grad_accum_steps: int = 4  # Simulates per-device batch size of 4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_interval_steps: int = 100
    report_every_steps: int = 1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    num_samples: Optional[int] = None

@dataclass
class ExecConfig:
    """Code execution template for logic-based training."""
    code_template: str = (
        """
def logic_executor(query, score_fn, query_fn ,all_bounding_boxes, image, history):

    score = lambda x, num_objects, type=None: score_fn(image, all_bounding_boxes, x, num_objects, draw_bbox=True, use_batch=False, pass_og_image=False, overlay_masks=False, mask=False, contrastive_mask=False, ultimate_score=False, type=type, history=history)
    query = lambda x, object_id=None, mask=False, mask_outside=False, draw_bbox=False, extract_object=False : query_fn(image, all_bounding_boxes, object_id, x, draw_bbox=True, pass_og_image=False, overlay_masks=False)
    objects_count = len(all_bounding_boxes)
    {code}
    """
    )

# =============================================================================
# 2. Utilities
# =============================================================================

def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging format."""
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )

def freeze(module: torch.nn.Module, requires_grad: bool = False) -> None:
    """Freeze or unfreeze module parameters."""
    for p in module.parameters():
        p.requires_grad = requires_grad

def count_trainable_params(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================================
# 3. Image preprocessing (InternVL-ish)
# =============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def dynamic_preprocess(image: Image.Image, image_size=448, use_thumbnail=True, max_num=12):
    """
    InternVL2.5-style tiling (heuristic grid selection).
    NOTE: If different images yield different tile counts, batching can break.
    In the shared wrapper below we resize bbox images to fixed size to keep tile count stable.
    """
    w, h = image.size
    ratio = w / h

    candidates = []
    for gw in range(1, max_num + 1):
        for gh in range(1, max_num + 1):
            blocks = gw * gh
            if blocks > max_num:
                continue
            target_ratio = gw / gh
            # Sort by ratio diff (ascending), then by -blocks (prefer more tiles on tie)
            candidates.append((abs(target_ratio - ratio), -blocks, gw, gh))
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, _, gw, gh = candidates[0]
    blocks = gw * gh

    target_width = gw * image_size
    target_height = gh * image_size
    resized = image.resize((target_width, target_height))

    processed = []
    for i in range(blocks):
        box = (
            (i % gw) * image_size,
            (i // gw) * image_size,
            ((i % gw) + 1) * image_size,
            ((i // gw) + 1) * image_size,
        )
        processed.append(resized.crop(box))

    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))

    return processed

def load_image_to_tiles(img_or_path, input_size=448, max_num=12, use_thumbnail=True):
    if isinstance(img_or_path, str):
        image = Image.open(img_or_path).convert("RGB")
    else:
        image = img_or_path.convert("RGB")

    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_thumbnail, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles], dim=0)  # [num_tiles, 3, H, W]
    return pixel_values


# =============================================================================
# Prompt building (InternVL conversation template)
# =============================================================================
def get_conv_template_from_model(model):
    import importlib
    pkg = model.__module__.rsplit(".", 1)[0]
    conv_mod = importlib.import_module(pkg + ".conversation")
    return conv_mod.get_conv_template

def build_internvl_query(model, tokenizer, question: str, num_patches: int = 1):
    """
    Matches InternVL chat-style prompt:
      <img> <IMG_CONTEXT>*(num_image_token * num_patches) </img>\n{question}
    num_patches = number of tiles for this image (from dynamic_preprocess).
    """
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    get_conv_template = get_conv_template_from_model(model)
    template = get_conv_template(model.template)

    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * model.num_image_token * num_patches) + IMG_END_TOKEN
    template.append_message(template.roles[0], image_tokens + "\n" + question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return query


# =============================================================================
# 4. Shared singleton: InternVL HF (trainable) with InternVL-like API
# =============================================================================
class InternVLHF:
    """
    HF/PyTorch implementation that mirrors your vLLM InternVL API:
      - _score(image_paths, question, target_tokens=...)
      - _score_batch(image_paths, questions, ...)
    Returns probabilities for target tokens (default ["Yes","No"]).
    Supports backprop if you use it under a module that computes a loss.
    
    Updated to match reference implementation with:
      - Flash attention
      - Gradient checkpointing
      - Updated LoRA target modules
      - Conversation template handling
    """

    _shared = None  # singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._shared is None:
            cls._shared = super().__new__(cls)
        return cls._shared

    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL2_5-1B-MPO",
        device: str = "cuda",
        dtype: torch.dtype = None,
        trust_remote_code: bool = True,
        # LoRA shared across all users of the singleton
        use_llm_lora: bool = False,
        use_vision_lora: bool = False,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        max_num_patches: int = 1,
        # QLoRA 4-bit quantization
        load_4bit: bool = False,
        # Temperature for softmax over target tokens (>1 = softer, prevents gradient vanishing)
        softmax_temperature: float = 1.0,
        # Additive bias on the Yes logit (col 1 after _score_batch swap) to escape
        # the extreme cold-start saturation where P(Yes)≈0.001 on CLEVR atoms.
        yes_bias: float = 0.0,
    ):
        if getattr(self, "_initialized", False):
            return

        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device = device
        self.max_num_patches = max_num_patches
        self.load_4bit = load_4bit
        self.softmax_temperature = softmax_temperature
        self.yes_bias = yes_bias

        # Determine compute dtype based on GPU support
        if device.type == "cuda":
            self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.compute_dtype = torch.float32

        if dtype is not None:
            self.compute_dtype = dtype
        self.dtype = self.compute_dtype

        logging.info(f"Initializing InternVL on {device} (dtype={self.compute_dtype}, 4bit={load_4bit}).")

        # Updated: model_max_length=8192 from max_seq_length 8192
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=8192,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<PAD>"

        # Check flash attention availability
        try:
            import flash_attn
            _use_flash_attn = True
        except ImportError:
            _use_flash_attn = False
            logging.info("flash_attn not available, falling back to eager attention.")

        load_kwargs = dict(
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            use_flash_attn=_use_flash_attn,
            torch_dtype=torch.bfloat16,
        )

        # ------------------------------------------------------------------
        # Multi-GPU sharding for the base (frozen) model.
        #
        # Opt-in controls:
        #   PEFT_DEVICE_MAP=auto    → enable HF Accelerate auto-sharding
        #   PEFT_DEVICE_MAP=single  → force single-device placement (default)
        #   (unset)                 → single-device placement
        #
        # 4-bit quantized loading has its own device_map handling below and
        # is not affected by this logic.
        # ------------------------------------------------------------------
        _visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        _dm_env = os.environ.get("PEFT_DEVICE_MAP", "").strip().lower()
        if _dm_env == "auto":
            self._auto_sharded = True
        else:
            # Default (including ``single`` or unset): keep everything on one
            # device. Users on multi-GPU boxes with oversized models can
            # explicitly set PEFT_DEVICE_MAP=auto.
            self._auto_sharded = False

        if load_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = {"": device}
            # 4-bit always uses a single device — bitsandbytes does its own
            # placement and we don't pipeline-shard a quantized backbone here.
            self._auto_sharded = False
            logging.info("Loading model with QLoRA 4-bit quantization.")
        elif self._auto_sharded:
            # Build a *custom* device map that keeps known-problematic
            # Sequentials (notably InternVL's ``mlp1`` projection between
            # vision_model and language_model) pinned atomically to one GPU.
            custom_map = self._build_peft_device_map(
                model_path=model_path,
                n_gpus=_visible_gpus,
                dtype=self.compute_dtype,
                trust_remote_code=trust_remote_code,
            )
            if custom_map is not None:
                load_kwargs["device_map"] = custom_map
                logging.info(
                    "Loading model with custom device_map across %d GPUs "
                    "(PEFT_DEVICE_MAP=%s). mlp1 pinned atomically.",
                    _visible_gpus, _dm_env or "default",
                )
            else:
                # Fallback: accelerate's default auto. Pass ``no_split``
                # hints via the model's own ``_no_split_modules`` attribute
                # (set by transformers when the config declares it).
                load_kwargs["device_map"] = "auto"
                logging.info(
                    "Loading model with device_map='auto' across %d GPUs "
                    "(custom map unavailable; PEFT_DEVICE_MAP=%s).",
                    _visible_gpus, _dm_env or "default",
                )

        base_model = AutoModel.from_pretrained(model_path, **load_kwargs)

        # Defensive consolidation: if the loaded base model ended up with
        # ``mlp1`` parameters split across devices (e.g. the fallback path
        # above or a stale cached device_map), move the whole projection
        # onto the device of its first parameter.
        if self._auto_sharded and hasattr(base_model, "mlp1"):
            try:
                self._consolidate_submodule_if_split(base_model, "mlp1")
            except Exception as e:
                logging.warning("mlp1 consolidation failed, continuing: %s", e)

        _any_lora = use_llm_lora or use_vision_lora

        if _any_lora:
            # Training-related setup: only needed when LoRA is applied
            if load_4bit:
                # prepare_model_for_kbit_training handles freezing + gradient checkpointing
                # use_reentrant=False is critical: with True (default), gradient checkpointing
                # requires inputs to have requires_grad=True, but with QLoRA only internal LoRA
                # weights need grad, not the layer inputs — so gradients die silently.
                base_model = prepare_model_for_kbit_training(
                    base_model,
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                # Fix: prepare_model_for_kbit_training casts norm layers to float32,
                # causing dtype mismatch in vision model's Conv2d (float32 bias vs bfloat16 input).
                # Cast vision_model and mlp1 back to compute_dtype since they aren't quantized.
                base_model.vision_model = base_model.vision_model.to(self.compute_dtype)
                base_model.mlp1 = base_model.mlp1.to(self.compute_dtype)
            else:
                # Enable gradient checkpointing (use_reentrant=False for LoRA compatibility)
                base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                # Freeze base model — LoRA will add its own trainable adapters on top.
                # Setting requires_grad=True on the base (previous behaviour) did not
                # actually train base weights — PEFT still reports only 2.5M trainable
                # params — but it forced autograd to record saved tensors through every
                # base-model op, which blew up activation memory on long sequences.
                for block in [base_model.language_model, base_model.mlp1, base_model.vision_model]:
                    freeze(block, False)

            # Enable input require grads and disable cache
            base_model.language_model.enable_input_require_grads()
            base_model.language_model.config.use_cache = False
            base_model.vision_model.gradient_checkpointing = True
            if hasattr(base_model.vision_model, 'encoder'):
                base_model.vision_model.encoder.gradient_checkpointing = True
        else:
            # Eval-only: no LoRA, no training setup — clean base model
            logging.info("No LoRA requested — loading base model for eval only.")
            base_model.eval()

        # Apply LoRA with updated target modules
        if use_llm_lora:
            self._apply_llm_lora(base_model, lora_r, lora_alpha, lora_dropout)

        if use_vision_lora:
            self._apply_vision_lora(base_model, lora_r, lora_alpha, lora_dropout)

        if load_4bit:
            self.model = base_model  # Already on device from device_map
        elif self._auto_sharded:
            # Model is pipeline-sharded across multiple GPUs by HF Accelerate.
            self.model = base_model.to(dtype=self.compute_dtype)
        else:
            self.model = base_model.to(dtype=self.compute_dtype, device=self.device)

        # Set up image tokens
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id

        # Initialize conversation template
        self.template, self.eos_token_id = self._init_template()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.eos_token_id

        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("No")

        self._initialized = True

    # ------------------------------------------------------------------
    # Multi-GPU device-map helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_peft_device_map(
        model_path: str,
        n_gpus: int,
        dtype: torch.dtype,
        trust_remote_code: bool,
    ) -> Optional[Dict[str, Any]]:
        """Compute a device map that keeps InternVL's ``mlp1`` projection whole.

        Strategy
        --------
        1. Construct the model on the ``meta`` device from its config alone
           (no weight download) so we can inspect its structure cheaply.
        2. Ask ``accelerate.infer_auto_device_map`` for a balanced split,
           passing every likely transformer-block class as
           ``no_split_module_classes`` so accelerate won't partition them.
        3. Post-process: find any key prefixed ``mlp1`` and collapse them all
           onto a single device (the device holding the *last* vision layer
           — i.e. where ``mlp1``'s input naturally arrives).

        Returns ``None`` if this analysis fails for any reason, signalling
        the caller to fall back to vanilla ``device_map="auto"``.
        """
        if n_gpus < 2:
            return None
        try:
            from accelerate import init_empty_weights, infer_auto_device_map
            from transformers import AutoConfig
        except ImportError:
            return None

        try:
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )
            with init_empty_weights():
                meta_model = AutoModel.from_config(
                    config, trust_remote_code=trust_remote_code
                )
        except Exception as e:
            logging.warning("meta-model construction failed: %s", e)
            return None

        # Start from the model's own no-split hints, then add likely block
        # class names. ``no_split_module_classes`` matches by class __name__.
        no_split = list(getattr(meta_model, "_no_split_modules", []) or [])
        for extra in (
            "InternVisionEncoderLayer",
            "InternVisionTransformerBlock",
            "Qwen2DecoderLayer",
            "LlamaDecoderLayer",
            "Phi3DecoderLayer",
        ):
            if extra not in no_split:
                no_split.append(extra)

        # Leave headroom so activations + LoRA adapters + optimiser state
        # fit. On 10.75 GiB 2080 Tis we cap weights at 8 GiB per GPU.
        try:
            per_gpu_gib = max(
                4,
                int(
                    min(
                        torch.cuda.get_device_properties(i).total_memory
                        for i in range(n_gpus)
                    )
                    / (1024 ** 3)
                )
                - 2,  # leave 2 GiB headroom
            )
        except Exception:
            per_gpu_gib = 8
        max_memory = {i: f"{per_gpu_gib}GiB" for i in range(n_gpus)}
        max_memory["cpu"] = "16GiB"

        try:
            device_map = infer_auto_device_map(
                meta_model,
                no_split_module_classes=no_split,
                max_memory=max_memory,
                dtype=dtype,
            )
        except Exception as e:
            logging.warning("infer_auto_device_map failed: %s", e)
            return None

        # Pin ``mlp1`` whole. Prefer the device holding the final vision
        # layer (natural data-flow placement), falling back to GPU 0.
        vision_last_device: Optional[Union[int, str]] = None
        for k, v in device_map.items():
            if k.startswith("vision_model"):
                vision_last_device = v
        pin_device = vision_last_device if vision_last_device is not None else 0
        # Strip any partial ``mlp1.*`` entries, assign ``mlp1`` whole.
        for k in list(device_map.keys()):
            if k == "mlp1" or k.startswith("mlp1."):
                del device_map[k]
        device_map["mlp1"] = pin_device

        logging.info(
            "Built custom device map: %d modules across GPUs, mlp1 -> %s, "
            "no_split=%s",
            len(device_map), pin_device, no_split,
        )
        return device_map

    @staticmethod
    def _consolidate_submodule_if_split(base_model: torch.nn.Module, attr: str) -> bool:
        """Force an attribute submodule onto a single device if its params
        currently live on more than one.

        Used as a safety net after ``from_pretrained`` in case the device
        map still split the target module (e.g. because the model class
        name didn't match ``no_split_module_classes``). Returns ``True``
        if a move was performed.
        """
        sub = getattr(base_model, attr, None)
        if sub is None:
            return False
        devices = {p.device for p in sub.parameters() if p.device.type != "meta"}
        if len(devices) <= 1:
            return False
        # Pick the device holding the first parameter as the target — it's
        # typically where this module's input arrives from the preceding
        # pipeline stage.
        target = next(sub.parameters()).device
        # Removing accelerate's AlignDevicesHook is necessary so our ``.to``
        # actually moves the weights (hooks otherwise re-offload them).
        try:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(sub, recurse=True)
        except ImportError:
            pass
        setattr(base_model, attr, sub.to(target))
        logging.info(
            "Consolidated split submodule %r onto %s (was on %s)",
            attr, target, sorted(str(d) for d in devices),
        )
        return True

    def _apply_llm_lora(self, base_model: torch.nn.Module, lora_r: int, lora_alpha: int, lora_dropout: float) -> None:
        """Apply LoRA to language model with updated target modules."""
        # Updated target modules from reference
        llm_targets = [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj",
        ]
        llm_lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=llm_targets,
            task_type="CAUSAL_LM",
        )
        base_model.language_model = get_peft_model(base_model.language_model, llm_lora)
        base_model.language_model.enable_input_require_grads()
        base_model.language_model.print_trainable_parameters()
        logging.info("LoRA applied to language_model.")

    def _apply_vision_lora(self, base_model: torch.nn.Module, lora_r: int, lora_alpha: int, lora_dropout: float) -> None:
        """Apply LoRA to vision model with r // 2."""
        if not hasattr(base_model, "vision_model"):
            return
        # Updated: r // 2 and alpha // 2 for vision model
        vision_lora_config = LoraConfig(
            r=lora_r // 2,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha // 2,
            lora_dropout=lora_dropout,
        )
        base_model.vision_model = get_peft_model(base_model.vision_model, vision_lora_config)
        base_model.vision_model.print_trainable_parameters()
        logging.info("LoRA applied to vision_model.")

    def _init_template(self) -> Tuple[Any, int]:
        """Initialize conversation template for internvl2_5."""
        config_to_check = getattr(self.model, "config", None)
        template_name = getattr(config_to_check, "template", "internvl2_5") if config_to_check else "internvl2_5"
        system_message = getattr(config_to_check, "system_message", None) if config_to_check else None

        if template_name:
            try:
                get_conv_template = get_conv_template_from_model(self.model)
                template = get_conv_template(template_name)
                if system_message:
                    template.system_message = system_message
                sep = getattr(template, "sep", None)
                eos_id = self.tokenizer.convert_tokens_to_ids(sep.strip()) if sep else self.tokenizer.eos_token_id
                logging.info(f"Using conversation template: {template_name}")
            except Exception as e:
                logging.warning(f"Failed to get template '{template_name}': {e}. Using tokenizer EOS.")
                template = None
                eos_id = self.tokenizer.eos_token_id
        else:
            template = None
            eos_id = self.tokenizer.eos_token_id

        if eos_id is None:
            eos_id = self.tokenizer.pad_token_id
        if eos_id is None:
            raise ValueError("Cannot resolve EOS token id.")
        return template, eos_id

    def _build_prompt(
        self, question: str, history: Optional[List[Tuple[str, str]]] = None,
        pixel_values: Optional[torch.Tensor] = None, num_patches_list: Optional[List[int]] = None,
    ) -> str:
        """Build prompt using conversation template."""
        if self.template is None:
            raise ValueError("Conversation template not found.")
        self.template.messages = []
        if history:
            for q, a in history:
                self.template.append_message(self.template.roles[0], q)
                self.template.append_message(self.template.roles[1], a)
        if pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question
        self.template.append_message(self.template.roles[0], question)
        self.template.append_message(self.template.roles[1], None)
        prompt = self.template.get_prompt()
        if pixel_values is not None:
            if num_patches_list is None:
                num_patches_list = [pixel_values.shape[0]]
            for num_patches in num_patches_list:
                image_tokens = (
                    self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + self.IMG_END_TOKEN
                )
                prompt = prompt.replace("<image>", image_tokens, 1)
        return prompt

    # --------------------------
    # internals
    # --------------------------
    def _prep_batch(
        self,
        image_paths,
        questions,
        input_size=448,
        max_num=1,
        use_thumbnail=False,
    ):
        """
        Returns:
          pixel_values: [sum_tiles, 3, H, W]
          image_flags:  [sum_tiles, 1]
          input_ids:    [B, T]
          attention_mask:[B, T]
        """
        assert len(image_paths) == len(questions)
        B = len(image_paths)

        # images -> tiles
        pvs = []
        num_patches_per_image = []
        for idx, img in enumerate(image_paths):
            if isinstance(img, Image.Image):
                _dbg_sz = img.size
            else:
                _dbg_sz = "not-pil"
            pv = load_image_to_tiles(img, input_size=input_size, max_num=max_num, use_thumbnail=use_thumbnail)
            if idx == 0 and not getattr(self, '_tile_logged', False):
                print(f"[TILE DEBUG] img_size={_dbg_sz}, max_num={max_num}, use_thumbnail={use_thumbnail}, "
                      f"tiles={pv.shape[0]}, tile_shape={pv.shape[1:]}")
                self._tile_logged = True
            pvs.append(pv)
            num_patches_per_image.append(pv.shape[0])
        pixel_values = torch.cat(pvs, dim=0).to(self.device, dtype=self.dtype)
        image_flags = torch.ones(pixel_values.shape[0], 1, device=self.device, dtype=torch.long)

        # prompts — pass num_patches so prompt has correct number of IMG_CONTEXT tokens
        queries = [build_internvl_query(self.model, self.tokenizer, q, num_patches=np)
                   for q, np in zip(questions, num_patches_per_image)]
        enc = self.tokenizer(queries, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        return B, pixel_values, image_flags, input_ids, attention_mask

    def _next_token_logits(
        self,
        image_paths,
        questions,
        input_size=448,
        max_num=1,
        use_thumbnail=False,
    ):
        B, pixel_values, image_flags, input_ids, attention_mask = self._prep_batch(
            image_paths=image_paths,
            questions=questions,
            input_size=input_size,
            max_num=max_num,
            use_thumbnail=use_thumbnail,
        )

        out = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            return_dict=True,
        )
        logits = out.logits  # [B, T, V]

        last_pos = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(B, device=self.device)
        next_logits = logits[batch_idx, last_pos, :]  # [B, V]
        return next_logits

    # --------------------------
    # public API (like vLLM version)
    # --------------------------
    def _score(
        self,
        image_paths,
        question: str,
        candidates: list = None,
        temperature=0.0,   # kept for signature compatibility; not used here
        target_tokens=None,
        input_size=448,
        max_num=1,
    ):
        """
        Returns tensor probs on device.
        Default target_tokens=["Yes","No"] -> shape [2]
        If target_tokens=[...] -> shape [len(target_tokens)]
        """
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        if candidates is not None:
            question = f"{question}\n If you have to classify among one of these objects {candidates}, {question}"

        if target_tokens is None:
            target_tokens = ["Yes", "No"]

        token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in target_tokens]
        next_logits = self._next_token_logits(
            image_paths=image_paths,
            questions=[question],
            input_size=input_size,
            max_num=max_num,
            use_thumbnail=False,
        )  # [1,V]

        # Select Yes/No logits, scale by temperature, then subtract
        # C = log(exp(yes) + exp(no)) to match vLLM's normalization.
        # No softmax — inferLocal() applies the only softmax later.
        sel = next_logits[0, token_ids] / self.softmax_temperature  # [K]
        if getattr(self, "yes_bias", 0.0) != 0.0:
            for _i, _tok in enumerate(target_tokens):
                if _tok == "Yes":
                    sel = sel.clone()
                    sel[_i] = sel[_i] + self.yes_bias
        C = torch.logsumexp(sel, dim=-1)  # scalar
        return (sel - C).to(self.device)

    def _score_batch(
        self,
        image_paths,
        questions,
        candidates=None,
        target_tokens=None,
        target_token="Yes",  # kept for signature compatibility; not used here
        temperature=0.0,     # kept for signature compatibility; not used here
        input_size=448,
        max_num=1,
        max_batch_size=None,
        call_metadata=None,
        topk=5,
    ):
        """
        Returns tensor probs [B,K] on device.
        Default target_tokens=["Yes","No"] -> [B,2]
        Chunks forward passes to limit peak VRAM from intermediate activations.

        ``max_batch_size`` defaults to ``INTERNVL_SCORE_CHUNK`` env var if set,
        else 16. Benchmarked on InternVL3-1B + CLEVR + H100: chunk=16 runs
        ~3× faster than chunk=2 with virtually identical peak alloc (~3.5 GB)
        and reserved ~7 GB. Going to chunk=64 only shaves ~6% wall-clock but
        quadruples reserved VRAM (~29 GB). Tests on smaller GPUs should
        override to 8–12 for safety headroom.
        """
        if max_batch_size is None:
            max_batch_size = int(os.environ.get("INTERNVL_SCORE_CHUNK", "16"))
        if candidates is not None:
            questions = [
                f"{q}\n If you have to classify among one of these objects {candidates}, {q}"
                for q in questions
            ]

        if target_tokens is None:
            target_tokens = ["Yes", "No"]

        token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in target_tokens]

        # Build a per-column additive bias vector so `yes_bias` shifts the
        # "Yes" logit regardless of target_tokens order ([Yes,No] vs [No,Yes]).
        # This is applied BEFORE logsumexp so the normalized log-prob reflects
        # the intended Yes-prior — counters the extreme cold-start where
        # baseline P(Yes) ≈ 0.001 for every CLEVR atom.
        if self.yes_bias != 0.0:
            bias_vec = torch.zeros(len(target_tokens), device=self.device, dtype=self.compute_dtype)
            for _i, _tok in enumerate(target_tokens):
                if _tok == "Yes":
                    bias_vec[_i] = self.yes_bias
        else:
            bias_vec = None

        # Only import the step-notebook hook lazily so this module stays
        # importable when domiknows isn't on sys.path (e.g. standalone eval).
        try:
            from domiknows.step_notebook import StepNotebook, record_vlm_call
            _notebook_active = StepNotebook.active() is not None
        except Exception:
            _notebook_active = False
            record_vlm_call = None

        # Gradient checkpoint each chunk's forward when training with grad enabled.
        # A CLEVR scene with N objects produces N² QA pairs, chunked into ceil(N²/K)
        # sub-forwards. Without checkpointing, every chunk's VLM activations stay in
        # GPU memory until the single loss.backward() at end of step — peak scales
        # linearly with the number of chunks. With checkpointing, only chunk boundary
        # tensors are retained; activations are recomputed during backward.
        use_checkpoint = (
            torch.is_grad_enabled()
            and getattr(self.model, 'training', False)
        )

        def _chunk_forward(img_paths_chunk, qs_chunk):
            cl = self._next_token_logits(
                image_paths=img_paths_chunk,
                questions=qs_chunk,
                input_size=input_size,
                max_num=max_num,
                use_thumbnail=False,
            )  # [chunk, V]
            sel_local = cl[:, token_ids] / self.softmax_temperature  # [chunk, K]
            if bias_vec is not None:
                sel_local = sel_local + bias_vec
            C_local = torch.logsumexp(sel_local, dim=-1, keepdim=True)  # [chunk, 1]
            return sel_local - C_local

        all_probs = []
        B = len(image_paths)
        for start in range(0, B, max_batch_size):
            end = min(start + max_batch_size, B)
            if use_checkpoint:
                chunk_norm = torch.utils.checkpoint.checkpoint(
                    _chunk_forward,
                    image_paths[start:end],
                    questions[start:end],
                    use_reentrant=False,
                )  # [chunk, K]
                # One-shot grad-flow diagnostic. Set DOMIKNOWS_GRAD_DEBUG=1 to
                # verify checkpoint() is producing a live autograd edge back
                # into the LoRA params. If this prints requires_grad=False or
                # grad_fn=None during training, the graph is severed and LoRA
                # never updates — i.e. root cause of the 50%/answer=False
                # regression.
                if os.environ.get("DOMIKNOWS_GRAD_DEBUG") == "1":
                    # One-shot fixed-connectivity dump on the very first
                    # chunk so we can confirm checkpoint is producing a live
                    # autograd edge and PEFT is wiring both A and B halves
                    # of the LoRA adapter as trainable.
                    if not getattr(self, "_grad_debug_printed", False):
                        lora_trainable = [
                            (n, p) for n, p in self.model.named_parameters()
                            if p.requires_grad
                        ]
                        n_lora = len(lora_trainable)
                        a_params = [(n, p) for n, p in lora_trainable if "lora_A" in n]
                        b_params = [(n, p) for n, p in lora_trainable if "lora_B" in n]
                        print(
                            f"[grad_debug] chunk_norm.requires_grad={chunk_norm.requires_grad} "
                            f"grad_fn={type(chunk_norm.grad_fn).__name__ if chunk_norm.grad_fn else None} | "
                            f"trainable_params={n_lora} | "
                            f"lora_A={len(a_params)} lora_B={len(b_params)}",
                            flush=True,
                        )
                        # Stash a reference to one A and one B weight so we
                        # can track their norms across training — LoRA init
                        # is A~Kaiming, B=0, so if B.norm stays at 0 after
                        # many steps, backward is never reaching B and the
                        # adapter delta (A@B) stays at 0 → model output
                        # never changes → exact same accuracy per epoch.
                        self._debug_a = a_params[0] if a_params else None
                        self._debug_b = b_params[0] if b_params else None

                        # Register hooks on the stashed A / B tensors so we
                        # see whether autograd is delivering non-zero grads.
                        # The A-update-is-stuck symptom from the earlier run
                        # has three candidate explanations: grad never fires
                        # on A, grad fires with zero norm, or grad fires
                        # but the optimizer is not stepping. This hook
                        # distinguishes (1) / (2) from (3).
                        self._a_grad_snapshots = []
                        self._b_grad_snapshots = []

                        def _make_grad_hook(bucket, tag):
                            def _hook(grad):
                                if len(bucket) < 4:
                                    bucket.append(grad.detach().float().norm().item())
                                    print(
                                        f"[grad_norm] {tag}.grad.norm={bucket[-1]:.6e} "
                                        f"(seen={len(bucket)})",
                                        flush=True,
                                    )
                            return _hook

                        if self._debug_a is not None:
                            self._debug_a[1].register_hook(
                                _make_grad_hook(self._a_grad_snapshots, "A")
                            )
                        if self._debug_b is not None:
                            self._debug_b[1].register_hook(
                                _make_grad_hook(self._b_grad_snapshots, "B")
                            )
                        self._grad_debug_printed = True

                    # Per-call weight-norm trace at sparse checkpoints so we
                    # can tell if LoRA weights are actually drifting over
                    # time. Checked at calls 1/25/100/300 — enough resolution
                    # to see motion without spamming stdout.
                    self._debug_counter = getattr(self, "_debug_counter", 0) + 1
                    if self._debug_counter in (1, 25, 100, 300):
                        def _wstats(entry):
                            if entry is None:
                                return "<none>"
                            _, p = entry
                            return f"norm={p.detach().float().norm().item():.6e} std={p.detach().float().std().item():.6e}"
                        print(
                            f"[weight_debug] call={self._debug_counter} "
                            f"A[{self._debug_a[0] if self._debug_a else '-'}]={_wstats(self._debug_a)} | "
                            f"B[{self._debug_b[0] if self._debug_b else '-'}]={_wstats(self._debug_b)}",
                            flush=True,
                        )
                all_probs.append(chunk_norm)
                continue

            # Non-checkpointed path: keeps chunk_logits live for notebook dump or eval.
            chunk_logits = self._next_token_logits(
                image_paths=image_paths[start:end],
                questions=questions[start:end],
                input_size=input_size,
                max_num=max_num,
                use_thumbnail=False,
            )  # [chunk, V]
            # Select Yes/No logits, scale by temperature, then subtract
            # C = log(exp(yes) + exp(no)) to match vLLM's normalization.
            # No softmax — inferLocal() applies the only softmax later.
            sel = chunk_logits[:, token_ids] / self.softmax_temperature  # [chunk, K]
            if bias_vec is not None:
                sel = sel + bias_vec
            C = torch.logsumexp(sel, dim=-1, keepdim=True)  # [chunk, 1]
            all_probs.append(sel - C)

            # Notebook instrumentation disabled — the topk/prob dump keeps a full
            # [chunk, V] logits tensor per chunk alive on the GPU, defeating the
            # memory benefit of chunking. Re-enable only when actively debugging.
            # if _notebook_active and record_vlm_call is not None:
            #     try:
            #         with torch.no_grad():
            #             raw_target = chunk_logits[:, token_ids].detach().float().cpu()
            #             full_probs = F.softmax(chunk_logits.detach().float(), dim=-1).cpu()
            #             norm_logp = (sel - C).detach().float().cpu()  # log P(target_i | target)
            #             k = min(topk, chunk_logits.shape[-1])
            #             topk_ids = chunk_logits.detach().float().topk(k, dim=-1).indices.cpu()
            #             topk_p = full_probs.gather(-1, topk_ids)
            #         for i_in_chunk in range(end - start):
            #             q_idx = start + i_in_chunk
            #             meta = call_metadata[q_idx] if call_metadata else {}
            #             topk_tokens = [
            #                 self.tokenizer.decode([int(t)])
            #                 for t in topk_ids[i_in_chunk].tolist()
            #             ]
            #             record_vlm_call(
            #                 concept=meta.get('concept'),
            #                 relation=meta.get('relation'),
            #                 obj_i=meta.get('obj_i'),
            #                 obj_j=meta.get('obj_j'),
            #                 prompt=questions[q_idx],
            #                 target_tokens=list(target_tokens),
            #                 target_raw_logits=raw_target[i_in_chunk].tolist(),
            #                 target_log_probs=norm_logp[i_in_chunk].tolist(),
            #                 topk_tokens=topk_tokens,
            #                 topk_probs=topk_p[i_in_chunk].tolist(),
            #             )
            #     except Exception:
            #         # Notebook instrumentation must never break training.
            #         pass

        logits_out = torch.cat(all_probs, dim=0)  # [B, K]

        # Sample a handful of rows on the very first and roughly the Nth
        # training call so we can see whether VLM output is actually
        # informative (spread across log-prob range) or collapsed toward a
        # single class. The scorer returns log-softmax values — i.e. values
        # in (-inf, 0] where 0 = certain and very negative = near-zero
        # probability. If ``Yes_logprob`` rows are uniformly ≪ 0, every
        # atom reads as "No" downstream and the solver returns False.
        if os.environ.get("DOMIKNOWS_GRAD_DEBUG") == "1":
            batch_call = getattr(self, "_debug_counter", 0)
            if batch_call in (1, 300):
                rows = logits_out.detach().float().cpu()
                # K columns should line up with target_tokens; for binary
                # Yes/No, col 0 = Yes, col 1 = No after this scorer's mask.
                head = rows[: min(6, rows.shape[0])]
                print(
                    f"[prob_debug] call={batch_call} shape={tuple(rows.shape)} "
                    f"log_softmax_head={head.tolist()} "
                    f"mean={rows.mean().item():.4f} "
                    f"col_mean={rows.mean(dim=0).tolist() if rows.ndim == 2 else '-'}",
                    flush=True,
                )

        return logits_out.to(self.device)


# =============================================================================
# Drop-in shared nn.Module like your InternVLShared (relation + attr + bbox questions)
# =============================================================================
class InternVLSharedHF(nn.Module):
    """
    Drop-in replacement for your vLLM InternVLShared:

      InternVLSharedHF(model_path, device, relation=..., attr=...)
      forward(image, bounding_boxes) -> probs

    Additional optional training:
      forward(image, bounding_boxes, labels=...) -> (probs, loss)

    labels:
      - if relation!=2: shape [num_boxes] with 0=Yes,1=No
      - if relation==2: shape [num_boxes*num_boxes]
    """

    model = None  # singleton InternVLHF

    def __init__(
        self,
        model_path="OpenGVLab/InternVL2_5-1B-MPO",
        device="cuda",
        dtype=None,
        relation=1,
        attr="no name",
        # tiling/batch stability knobs
        input_size=448,
        max_num=1,
        # LoRA (shared)
        use_llm_lora=False,
        use_vision_lora=False,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        # QLoRA
        load_4bit=False,
        # Temperature for softmax (>1 = softer outputs, prevents gradient vanishing)
        softmax_temperature=1.0,
        # Additive bias on the Yes logit to counter CLEVR cold-start Yes≈0 saturation.
        yes_bias=0.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.relation = relation
        self.attr = attr
        self.device = device
        self.softmax_temperature = softmax_temperature
        self.yes_bias = yes_bias

        if dtype is None:
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        self.dtype = dtype

        self.input_size = input_size
        self.max_num = max_num

        if InternVLSharedHF.model is None:
            InternVLSharedHF.model = InternVLHF(
                model_path=model_path,
                device=device,
                dtype=dtype,
                use_llm_lora=use_llm_lora,
                use_vision_lora=use_vision_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                max_num_patches=max_num,
                load_4bit=load_4bit,
                softmax_temperature=softmax_temperature,
                yes_bias=yes_bias,
            )
            # Register the actual nn.Module only in first instance
            # so DomiKnowS can discover parameters for training
            self._hf_model = InternVLSharedHF.model.model
        self.model = InternVLSharedHF.model

    def _to_pil(self, image):
        """
        Accept:
          - PIL.Image
          - path str
          - torch.Tensor (C,H,W) or (1,C,H,W)
        """
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        # torch.Tensor
        if isinstance(image, torch.Tensor):
            img = image
            if img.is_cuda:
                img = img.cpu()
            if img.dim() == 4:
                img = img.squeeze(0)
            # img expected [C,H,W] in [0,1] or [0,255]
            if img.dtype != torch.uint8:
                img = img.clamp(0, 1)
            # convert manually to PIL
            img = img.mul(255).byte() if img.dtype != torch.uint8 else img
            img = img.permute(1, 2, 0).numpy()
            return Image.fromarray(img).convert("RGB")

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _draw_and_resize(self, base_pil: Image.Image, boxes, colors):
        """
        Draw bounding boxes on image.  Keep original aspect ratio and let
        load_image_to_tiles / dynamic_preprocess handle resizing — this
        matches vLLM behavior which passes raw PIL images to the processor.
        """
        img = base_pil.copy()
        draw = ImageDraw.Draw(img)
        for box, color in zip(boxes, colors):
            if isinstance(box, torch.Tensor):
                box = box.detach().cpu().tolist()
            draw.rectangle(box, outline=color, width=3)
        return img

    def _build_call_metadata(self, n_boxes):
        """Per-question metadata matching the order produced by
        ``_prepare_images_questions``. Consumed by the step-notebook to
        tag each VLM call with its concept and object indices.
        """
        if self.relation == 2:
            return [
                {'concept': self.attr, 'relation': 2, 'obj_i': i, 'obj_j': j}
                for i in range(n_boxes) for j in range(n_boxes)
            ]
        return [
            {'concept': self.attr, 'relation': 1, 'obj_i': i, 'obj_j': None}
            for i in range(n_boxes)
        ]

    def _prepare_images_questions(self, image, image_filename, bounding_boxes):
        """Prepare image-question pairs for scoring."""
        if isinstance(image, (list, tuple)) and len(image) == 1:
            image = image[0]
        if isinstance(image_filename, (list, tuple)) and len(image_filename) == 1:
            image_filename = image_filename[0]
        # Fallback: if pil_image is None (stale cache built before images were
        # downloaded), try to load directly from the images directory on disk.
        if image is None and image_filename is not None:
            _path = os.path.join("train", "images", image_filename)
            try:
                image = Image.open(_path).convert("RGB")
            except (FileNotFoundError, OSError):
                pass
        if image is None:
            raise TypeError(
                f"pil_image is None for '{image_filename}' and the file was not found on disk. "
                f"Ensure CLEVR images are downloaded to train/images/."
            )
        base = self._to_pil(image)

        images, questions = [], []
        if self.relation == 2:
            for box1 in bounding_boxes:
                for box2 in bounding_boxes:
                    img = self._draw_and_resize(base, [box1, box2], ["red", "green"])
                    q = f"Is the object in the red bounding box {self.attr} of the object in the green bounding box? answer with only Yes or No."
                    images.append(img)
                    questions.append(q)
        else:
            for box in bounding_boxes:
                img = self._draw_and_resize(base, [box], ["red"])
                q = f"Is the object in the red bounding box {self.attr}? answer with only Yes or No."
                images.append(img)
                questions.append(q)
        return images, questions

    def forward(self, image, image_filename, bounding_boxes, label=None):
        """
        Forward pass for DomiKnowS.
        Returns probs tensor [N,2] matching vLLM output order: [P(No), P(Yes)].
        vLLM's _score_batch has a variable-name swap (yes_logit gets No logprob
        and vice-versa), so its output is effectively [P(No), P(Yes)].
        DomiKnowS was calibrated with that order, so we match it here.
        """
        images, questions = self._prepare_images_questions(image, image_filename, bounding_boxes)
        call_metadata = self._build_call_metadata(len(bounding_boxes))

        probs = self.model._score_batch(
            image_paths=images,
            questions=questions,
            target_tokens=["No", "Yes"],
            input_size=self.input_size,
            max_num=self.max_num,
            call_metadata=call_metadata,
        )
        return probs.float()

    def train_step(self, image, image_filename, bounding_boxes, gt_labels, optimizer, grad_accum_steps=1):
        """
        Memory-efficient training: processes one pair at a time with gradient accumulation.

        Args:
            image: PIL image or tensor (may be None if cache is stale; image_filename is the fallback)
            image_filename: image filename for on-demand disk loading when image is None
            bounding_boxes: list of bounding boxes
            gt_labels: ground truth labels [N] where 0=Yes, 1=No
            optimizer: torch optimizer for LoRA parameters
            grad_accum_steps: number of steps to accumulate before optimizer step
        Returns:
            avg_loss: average loss over all pairs
        """
        images, questions = self._prepare_images_questions(image, image_filename, bounding_boxes)
        N = len(images)

        yes_id = self.model.yes_token_id
        no_id = self.model.no_token_id
        total_loss = 0.0

        for i in range(N):
            logits = self.model._next_token_logits(
                image_paths=[images[i]],
                questions=[questions[i]],
                input_size=self.input_size,
                max_num=self.max_num,
                use_thumbnail=False,
            )  # [1, V]
            yn_logits = torch.stack([logits[0, yes_id], logits[0, no_id]]).float()  # [2]
            target = gt_labels[i].to(self.device).long()
            loss = F.cross_entropy(yn_logits.unsqueeze(0), target.unsqueeze(0)) / N
            loss.backward()
            total_loss += loss.item()

        return total_loss


# =============================================================================
# 5. Dataset for Training
# =============================================================================

class ReferralTrainingDataset(Dataset):
    """Dataset for referral training with metadata JSON file."""
    
    def __init__(self, metadata_file: str, image_base_dir: str):
        self.metadata_file = metadata_file
        self.image_base_dir = image_base_dir
        logging.info(f"Loading metadata: {metadata_file}")
        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)
            self.metadata = list(data.values()) if isinstance(data, dict) else data
        except Exception as e:
            logging.error(f"Failed to load metadata: {e}")
            self.metadata = []
        logging.info(f"Loaded {len(self.metadata)} samples.")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.metadata[idx]
        image_rel = item.get("image_path")
        if not image_rel:
            raise ValueError(f"Missing image_path at index {idx}")
        image_path = os.path.join(self.image_base_dir, image_rel)
        expression = item.get("expression")
        boxes = item.get("candidate_bboxes")
        gt_index = item.get("gt_index")
        code_snippet = item.get("code_snippet")

        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        if not expression:
            raise ValueError(f"Missing expression at {idx}")
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(f"Invalid candidate_bboxes at {idx}")
        if not isinstance(gt_index, int) or not (0 <= gt_index < len(boxes)):
            raise ValueError(f"Invalid gt_index at {idx}")
        if not code_snippet:
            raise ValueError(f"Missing code_snippet at {idx}")

        return {
            "image_path": image_path,
            "expression": expression,
            "candidate_bboxes": boxes,
            "gt_index": gt_index,
            "code_snippet": code_snippet,
            "metadata_idx": idx,
        }


# =============================================================================
# 6. Training Helpers
# =============================================================================

def forward_once(vl_model, batch, exec_cfg, criterion=None, grad: bool = False):
    """Single forward pass with optional loss computation."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        image_path = batch["image_path"][0]
        expression = batch["expression"][0]
        candidate_bboxes = batch["candidate_bboxes"]
        candidate_bboxes = [list(map(float, box)) for box in candidate_bboxes]
        gt_index = batch["gt_index"][0].item()
        code_snippet = batch["code_snippet"][0]
        if isinstance(code_snippet, tuple):
            code_snippet = code_snippet[0]
        image = Image.open(image_path).convert("RGB")
        code_to_exec = exec_cfg.code_template.format(code=code_snippet.replace("\n", "\n    "))
        code_to_exec = code_to_exec.encode("utf-8").decode("utf-8")
        
        # Calculate spatial relations if probabilistic tensor is available
        if PROB_TENSOR_AVAILABLE:
            spatial_relations = vl_model.calculate_all_spatial_metrics(image, candidate_bboxes, wrapper=ProbabilisticTensor)
            spatial_relations = {k.encode("utf-8").decode("utf-8"): v for k, v in spatial_relations.items()}
            ns = {"ProbabilisticTensor": ProbabilisticTensor, "and_op": and_op, "or_op": or_op, **spatial_relations}
            exec(code_to_exec, ns, ns)
            logic_executor = ns["logic_executor"]
            ProbabilisticTensor.start_cache()
            out = logic_executor(
                query=expression, score_fn=vl_model.score, query_fn=vl_model.query,
                all_bounding_boxes=candidate_bboxes, image=image, history=None,
            )
            ProbabilisticTensor.end_cache()
            probs = out.tensor if isinstance(out, ProbabilisticTensor) else out
        else:
            # Fallback: simple scoring without probabilistic tensor
            probs = torch.zeros(len(candidate_bboxes), device=vl_model.device)
            for i, box in enumerate(candidate_bboxes):
                score = vl_model._score(image, f"Is this object matching: {expression}?")
                probs[i] = score[0] if len(score) > 0 else 0.5
        
        loss = None
        if criterion is not None:
            target = torch.zeros_like(probs)
            target[gt_index] = 1.0
            loss = criterion(probs, target)
    return probs, gt_index, loss


def evaluate(vl_model, loader, criterion, exec_cfg):
    """Validation loop with accuracy/loss tracking."""
    if PROB_TENSOR_AVAILABLE:
        ProbabilisticTensor.set_and_op("mul")
    vl_model.model.eval()
    total = correct = 0
    cum_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            try:
                probs, gt_idx, loss = forward_once(vl_model, batch, exec_cfg, criterion, grad=False)
            except Exception as e:
                logging.error(f"Error during evaluation: {e}\n{traceback.format_exc()}")
                continue
            pred = probs.argmax().item()
            correct += int(pred == gt_idx)
            cum_loss += float(loss.item()) if loss is not None else 0.0
            total += 1
    acc = correct / max(1, total)
    avg_loss = cum_loss / max(1, total)
    return acc, avg_loss


def has_any_grad(model: torch.nn.Module) -> bool:
    """Debug helper to check if any gradients exist."""
    for p in model.parameters():
        if p.requires_grad and p.grad is not None and torch.any(p.grad != 0):
            return True
    return False


# =============================================================================
# 7. Training Loop
# =============================================================================

def run_training(
    paths: PathConfig,
    model_cfg: ModelConfig,
    lora_cfg: LoraConfigWrap,
    train_cfg: TrainConfig,
    exec_cfg: ExecConfig,
) -> None:
    """Main training loop with TensorBoard logging, LR scheduling, and early stopping."""
    if PROB_TENSOR_AVAILABLE:
        ProbabilisticTensor.set_and_op("min")
    setup_logging()
    logging.info("=== Starting VLM LoRA fine-tuning with new hyperparameters ===")
    
    if train_cfg.num_samples is not None:
        logging.info(f"Using only {train_cfg.num_samples} samples for training as specified.")
        paths.output_dir = paths.output_dir + f"_{train_cfg.num_samples}_samples"
    os.makedirs(paths.output_dir, exist_ok=True)

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S-refgta")
    log_dir = os.path.join(paths.output_dir, "runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"TensorBoard logs will be saved to: {log_dir}")

    # Initialize model with LoRA
    vl_model = InternVLHF(
        model_path=model_cfg.model_path,
        device=model_cfg.device,
        use_llm_lora=True,
        use_vision_lora=True,
        lora_r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
    )
    logging.info(f"Trainable params: {count_trainable_params(vl_model.model):,}")

    # Load dataset
    full_dataset = ReferralTrainingDataset(paths.metadata_file, paths.image_dir)
    if train_cfg.num_samples is not None:
        full_dataset = Subset(full_dataset, range(min(train_cfg.num_samples, len(full_dataset))))

    # Validate data items
    valid_idx: List[int] = []
    for i in tqdm(range(len(full_dataset)), desc="Validating data"):
        try:
            _ = full_dataset[i]
            valid_idx.append(i)
        except Exception as e:
            logging.warning(f"Skipping idx {i}: {e}")
    dataset = Subset(full_dataset, valid_idx)
    logging.info(f"Using {len(dataset)} valid items.")

    # Train/validation split
    if not (0 < train_cfg.validation_split < 1):
        raise ValueError("validation_split must be between 0 and 1")
    val_size = int(len(dataset) * train_cfg.validation_split)
    train_size = len(dataset) - val_size
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    logging.info(f"Split data into {len(train_dataset)} training and {len(val_dataset)} validation samples.")

    # DataLoaders with num_workers=4 from reference
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        vl_model.model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    num_training_steps = len(train_loader) // train_cfg.grad_accum_steps * train_cfg.epochs
    num_warmup_steps = int(num_training_steps * train_cfg.warmup_ratio)
    scheduler = get_scheduler(
        name=train_cfg.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
    )
    criterion = nn.BCELoss()

    best_val_accuracy = -1.0
    patience_counter = 0
    total_steps = (len(train_loader) // train_cfg.grad_accum_steps) * train_cfg.epochs
    step_bar = tqdm(total=total_steps, desc="Global steps")
    global_step = 0

    for epoch in range(train_cfg.epochs):
        logging.info(f"Epoch {epoch + 1}/{train_cfg.epochs}")
        vl_model.model.train()
        epoch_loss = 0.0
        acc_counter, acc_correct = 0, 0.0
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        optimizer.zero_grad()

        for i, batch in enumerate(batch_bar):
            try:
                try:
                    probs, gt_index, loss = forward_once(vl_model, batch, exec_cfg, criterion, grad=True)
                except Exception as e:
                    logging.error(f"Error during forward pass: {e}\n{traceback.format_exc()}")
                    continue
                if loss is None:
                    continue
                loss = loss / train_cfg.grad_accum_steps
                loss.backward()
                pred = torch.argmax(probs, dim=0)
                acc_correct += float(pred == gt_index)
                acc_counter += 1
                epoch_loss += loss.item() * train_cfg.grad_accum_steps
            except Exception as e:
                logging.error(f"Error during training step: {e}\n{traceback.format_exc()}")
                optimizer.zero_grad()
                continue

            if ((i + 1) % train_cfg.grad_accum_steps == 0) or (i + 1 == len(train_loader)):
                if PROB_TENSOR_AVAILABLE:
                    ProbabilisticTensor.set_and_op("min")
                torch.nn.utils.clip_grad_norm_(
                    [p for p in vl_model.model.parameters() if p.requires_grad],
                    train_cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                step_bar.update(1)
                writer.add_scalar("Loss/train", loss.item() * train_cfg.grad_accum_steps, global_step)
                writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], global_step)

                if global_step % train_cfg.report_every_steps == 0 and acc_counter > 0:
                    current_accuracy = acc_correct / acc_counter
                    writer.add_scalar("Accuracy/train_running", current_accuracy, global_step)
                    acc_correct, acc_counter = 0.0, 0

                if paths.output_dir and global_step > 0 and global_step % train_cfg.save_interval_steps == 0:
                    logging.info(f"\n--- Running Validation at Step {global_step} ---")
                    val_acc, val_loss = evaluate(vl_model, val_loader, criterion, exec_cfg)
                    writer.add_scalar("Loss/validation", val_loss, global_step)
                    writer.add_scalar("Accuracy/validation", val_acc, global_step)
                    logging.info(f"Validation Results: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}")
                    vl_model.model.train()

                    if val_acc > best_val_accuracy:
                        logging.info(f"New best validation accuracy! {val_acc:.4f} (previously {best_val_accuracy:.4f})")
                        best_val_accuracy = val_acc
                        patience_counter = 0
                        best_dir = os.path.join(paths.output_dir, "best_model")
                        os.makedirs(best_dir, exist_ok=True)
                        vl_model.model.save_pretrained(best_dir)
                        vl_model.tokenizer.save_pretrained(best_dir)
                    else:
                        patience_counter += 1
                        logging.info(f"Validation accuracy did not improve. Patience: {patience_counter}/{train_cfg.early_stopping_patience}")

                    if patience_counter >= train_cfg.early_stopping_patience:
                        logging.warning(f"Early stopping triggered after {patience_counter} checks with no improvement.")
                        writer.close()
                        return

        avg_epoch_loss = epoch_loss / max(1, len(train_dataset))
        logging.info(f"Epoch {epoch + 1} average training loss: {avg_epoch_loss:.6f}")

    writer.close()
    logging.info(f"Training finished. Best model saved to {os.path.join(paths.output_dir, 'best_model')}")
    last_dir = os.path.join(paths.output_dir, "last_model")
    os.makedirs(last_dir, exist_ok=True)
    vl_model.model.save_pretrained(last_dir)
    vl_model.tokenizer.save_pretrained(last_dir)


# =============================================================================
# 8. CLI Entrypoint
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM LoRA Fine-tuning")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of training samples to use.")
    parser.add_argument("--train", action="store_true",
                        help="Run training mode with the training loop.")
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL2_5-1B-MPO",
                        help="Model path for InternVL.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for saving models.")
    parser.add_argument("--metadata_file", type=str, default=None,
                        help="Path to training metadata JSON file.")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Base directory for images.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=4e-5,
                        help="Learning rate.")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)

    if args.train:
        # Training mode with run_training
        logging.info("--- TRAINING MODE ---")
        
        paths = PathConfig()
        if args.output_dir:
            paths.output_dir = args.output_dir
        if args.metadata_file:
            paths.metadata_file = args.metadata_file
        if args.image_dir:
            paths.image_dir = args.image_dir
            
        model_cfg = ModelConfig(model_path=args.model_path)
        lora_cfg = LoraConfigWrap(r=args.lora_r)
        train_cfg = TrainConfig(num_samples=args.num_samples, epochs=args.epochs, lr=args.lr)
        exec_cfg = ExecConfig()

        logging.info("--- CONFIGURATIONS ---")
        logging.info(json.dumps({
            "paths": asdict(paths),
            "model": asdict(model_cfg),
            "lora": asdict(lora_cfg),
            "train": asdict(train_cfg),
        }, indent=2))

        run_training(paths, model_cfg, lora_cfg, train_cfg, exec_cfg)
    else:
        # Sanity test mode (original behavior)
        logging.info("--- SANITY TEST MODE ---")
        logging.info("Use --train flag to run training mode.")
        
        MODEL_PATH = args.model_path
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        # Shared instance with relation/attr just like your module
        m = InternVLSharedHF(
            model_path=MODEL_PATH,
            device=device,
            dtype=dtype,
            relation=1,
            attr="a cat",
            input_size=448,
            max_num=1,
            use_llm_lora=False,  # turn on if you want training adapters shared
        )

        dummy = Image.new("RGB", (800, 600), color="black")
        boxes = [
            torch.tensor([50, 50, 200, 200]),
            torch.tensor([300, 100, 500, 400]),
        ]

        probs = m(dummy, boxes)  # [2,2]
        print("probs:", probs)

        # Training-style call (optional)
        labels = torch.tensor([1, 1], dtype=torch.long, device=device)  # pretend both are "No"
        probs2, loss = m(dummy, boxes, label=labels)
        print("loss:", loss.item())
        loss.backward()
        print("backward ok")
