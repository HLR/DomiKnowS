"""Qwen-VL adapter for CLEVR DomiKnowS concept scoring.

This module mirrors the public ``InternVLSharedHF`` interface used by
``main.py`` so CLEVR concepts can be backed by Qwen3-VL without changing the
graph/program declaration.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover - PEFT is optional for eval-only smoke.
    LoraConfig = None
    get_peft_model = None

from qwen_vl_utils import process_vision_info
from peftvllm import InternVLSharedHF


class QwenVLHF:
    """Small scoring wrapper around Qwen3-VL next-token logits."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype=None,
        use_llm_lora: bool = False,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        softmax_temperature: float = 1.0,
        yes_bias: float = 0.0,
        **_: object,
    ):
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.device = device
        self.dtype = dtype
        self.softmax_temperature = softmax_temperature
        self.yes_bias = yes_bias

        self.processor = AutoProcessor.from_pretrained(model_path)
        model_kwargs = dict(torch_dtype=dtype, device_map=None)
        attn_impl = os.environ.get("QWENVL_ATTN_IMPL")
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs,
        ).to(device)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        if use_llm_lora:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if get_peft_model is None or LoraConfig is None:
                raise ImportError("PEFT is required for --peft Qwen-VL LoRA training")
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            )
            self.model = get_peft_model(self.model, config)
            self.model.print_trainable_parameters()

        self.model.train()

    def _messages(self, image: Image.Image, question: str):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "resized_height": int(os.environ.get("QWENVL_RESIZED_HEIGHT", "224")),
                        "resized_width": int(os.environ.get("QWENVL_RESIZED_WIDTH", "224")),
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

    def _next_token_logits(self, images: List[Image.Image], questions: List[str]) -> torch.Tensor:
        messages = [self._messages(img, q) for img, q in zip(images, questions)]
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**inputs, return_dict=True)
        logits = out.logits
        last_pos = inputs["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(logits.shape[0], device=self.device)
        return logits[batch_idx, last_pos, :]

    def _score_batch(
        self,
        image_paths,
        questions,
        candidates=None,
        target_tokens: Optional[List[str]] = None,
        max_batch_size=None,
        **_: object,
    ) -> torch.Tensor:
        if candidates is not None:
            questions = [
                f"{q}\n If you have to classify among one of these objects {candidates}, {q}"
                for q in questions
            ]
        if target_tokens is None:
            target_tokens = ["No", "Yes"]
        if max_batch_size is None:
            max_batch_size = int(os.environ.get("QWENVL_SCORE_CHUNK", "1"))

        token_ids = [self.processor.tokenizer.convert_tokens_to_ids(tok) for tok in target_tokens]
        if any(tid is None or tid < 0 for tid in token_ids):
            raise ValueError(f"Could not map target tokens {target_tokens} to tokenizer ids")

        chunks = []
        for start in range(0, len(questions), max_batch_size):
            imgs = image_paths[start : start + max_batch_size]
            qs = questions[start : start + max_batch_size]
            next_logits = self._next_token_logits(imgs, qs)
            sel = next_logits[:, token_ids] / self.softmax_temperature
            if self.yes_bias:
                sel = sel.clone()
                for idx, tok in enumerate(target_tokens):
                    if tok == "Yes":
                        sel[:, idx] = sel[:, idx] + self.yes_bias
            chunks.append(sel - torch.logsumexp(sel, dim=-1, keepdim=True))
        return torch.cat(chunks, dim=0).to(self.device)


class QwenVLSharedHF(InternVLSharedHF):
    """Drop-in CLEVR ModuleLearner wrapper backed by Qwen3-VL."""

    model = None

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype=None,
        relation: int = 1,
        attr: str = "no name",
        use_llm_lora: bool = False,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.05,
        softmax_temperature: float = 1.0,
        yes_bias: float = 0.0,
        *args,
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.relation = relation
        self.attr = attr
        self.device = device
        self.softmax_temperature = softmax_temperature
        self.yes_bias = yes_bias
        self.input_size = 448
        self.max_num = 1

        if QwenVLSharedHF.model is None:
            QwenVLSharedHF.model = QwenVLHF(
                model_path=model_path,
                device=device,
                dtype=dtype,
                use_llm_lora=use_llm_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                softmax_temperature=softmax_temperature,
                yes_bias=yes_bias,
            )
            self._hf_model = QwenVLSharedHF.model.model
        self.model = QwenVLSharedHF.model
