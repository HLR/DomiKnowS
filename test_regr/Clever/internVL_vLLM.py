import torch
import os

from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    from vllm.assets.image import ImageAsset
except ImportError:
    raise ImportError("Please install vllm: pip install vllm")
from typing import Optional, List, Tuple
import logging

logging.getLogger().setLevel(logging.ERROR)
"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you wish to use it.
If you have changes in mind, please contribute back so the community can benefit collectively and continue to maintain these valuable templates.

Modified from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""

import dataclasses
from enum import IntEnum, auto
from typing import Dict, List, Tuple, Union
import re
import json
import numpy as np
from PIL import Image
import torch  # Import torch only if you might pass a tensor mask
from PIL import ImageDraw

BBOX_PATTERN_VLM = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')

import torch
from vllm import LLM

import torch
from typing import Optional, List

from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,  # Wrapper base-class
    RequestLogitsProcessor,  # Request-level logitsproc type annotation
    LogitsProcessor
)


class DummyPerReqLogitsProcessor:
    """The request-level logits processor masks out all logits except the
    token id identified by `target_token`"""

    def __init__(self, target_tokens: List[int]) -> None:
        """Specify `target_token`"""
        self.target_tokens = target_tokens
        # print("USING PER REQUEST LOGITS PROCESSOR FOR TARGET TOKEN", target_tokens)

    def __call__(
            self,
            output_ids: list[int],
            logits: torch.Tensor,
    ) -> torch.Tensor:
        # val_to_keep = logits[self.target_tokens].item()
        # logits[:] = float("-inf")
        # for target_token in self.target_tokens:
        #     logits[target_token] = val_to_keep

        # New approach: set everything to a very low value except target tokens
        mask = torch.ones_like(logits) * -1e9  # Very low value
        for target_token in self.target_tokens:
            # print("Keeping token", target_token, "with logit", logits[target_token].item())
            mask[target_token] = 0  # Keep the logit for target tokens
        logits = logits + mask

        return logits


from typing import Any, Optional
import logging
import numpy as np


class WrappedPerReqLogitsProcessor(AdapterLogitsProcessor):
    """Example of wrapping a fake request-level logit processor to create a
    batch-level logits processor"""

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
            self,
            params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        """This method returns a new request-level logits processor, customized
        to the `target_token` value associated with a particular request.

        Returns None if the logits processor should not be applied to the
        particular request. To use the logits processor the request must have
        a "target_token" custom argument with an integer value.

        Args:
        params: per-request sampling params

        Returns:
        `Callable` request logits processor, or None
        """
        target_tokens: Optional[Any] = params.extra_args and params.extra_args.get(
            "target_tokens"
        )
        if target_tokens is None:
            return None
        return DummyPerReqLogitsProcessor(target_tokens)


def make_llm(model_path,
             yes_token_id=None,  # "Yes" token in InternVL tokenizer
             no_token_id=None,  # "No" token in InternVL tokenizer
             # raise/lower based on VRAM & your output length
             ):
    # 1) Pick the best dtype for your GPU (BF16 on A100/H100, else FP16)
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = "bfloat16" if bf16_ok else "float16"
    want_len = 4096 * 4  # tighten this if you can — it’s a big speed lever
    target_util = 0.85
    target_concurrency = 8  # bump up until you hit VRAM or no longer see speedup
    # 2) Try FP8 KV cache for speed+capacity; fall back safely if unsupported
    if "1b" in model_path.lower():
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="auto",
            enable_prefix_caching=True,
            #max_seq_len_to_capture=want_len,  # or the real upper bound of your shared prefix
            max_model_len=want_len,
            max_num_batched_tokens=4096 * target_concurrency,
            mm_processor_cache_gb=5,
            gpu_memory_utilization=0.9,
            logprobs_mode='processed_logits',
            logits_processors=[WrappedPerReqLogitsProcessor],
        )
    else:

        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="auto",
            enable_prefix_caching=True,
            #max_seq_len_to_capture=want_len,  # or the real upper bound of your shared prefix
            max_model_len=want_len,
            max_num_batched_tokens=4096 * target_concurrency,
            gpu_memory_utilization=0.85,
            logprobs_mode='processed_logits',
            logits_processors=[WrappedPerReqLogitsProcessor],

        )

    # llm = LLM(
    #     model=model_path,
    #     trust_remote_code=True,
    #     dtype=dtype,
    #     enable_prefix_caching=True,
    #     max_num_seqs=target_concurrency,
    #     max_seq_len_to_capture=want_len*4,
    #     max_num_batched_tokens=want_len*4,
    #     max_model_len=want_len,
    #     gpu_memory_utilization=target_util,
    # )
    return llm


# Example: tight context, short answers, heavy batching for max throughput


################################################################################
# Example: InternVL class
################################################################################

class InternVL():
    """
    An example agent that mirrors your QwenVL style but uses
    InternVL for scoring yes/no and generating text answers.
    """

    def __init__(
            self,
            # model_path: str = "OpenGVLab/InternVL3-8B-Instruct",
            # model_path: str = "OpenGVLab/InternVL3-1B",
            # model_path: str = "OpenGVLab/InternVL2_5-1B-MPO",
            # model_path: str = "OpenGVLab/InternVL2_5-2B-MPO",
            # model_path: str = "/egr/research-hlr2/kamalida/model_weights/refcocog_lora_adapters_1b_full_model/epoch_2",
            # model_path: str = "OpenGVLab/InternVL2_5-8B-MPO-AWQ",
            # model_path: str = "/egr/research-hlr2/kamalida/model_weights/refcocog_lora_adapters_refgta/merged3",
            # model_path: str = "OpenGVLab/InternVL2_5-8B-MPO",
            model_path: str = "OpenGVLab/InternVL2_5-8B-MPO",
            # model_path: str = "OpenGVLab/InternVL2_5-26B-MPO",
            # model_path: str = "OpenGVLab/InternVL2-8B",
            # model_path: str = "OpenGVLab/InternVL2_5-4B-MPO",
            # model_path: str = "OpenGVLab/InternVL2_5-4B-MPO",
            # model_path: str = "OpenGVLab/InternVL2_5-26B-MPO-AWQ",
            device: str = "cuda",
            trust_remote_code: bool = True,
            llm=None,
            *args,
            **kwargs,
    ):
        """
        :param model_path: The HF Hub repo or local path for the InternVL model
        :param device: "cuda" or "cpu"
        :param load_4bit: Whether to load in 4-bit precision
        :param max_num_patches: Maximum number of patches/tiles for an image
        :param trust_remote_code: Whether to trust the remote code
        """
        super().__init__(*args, **kwargs)
        # model_path = "/egr/research-hlr2/kamalida/model_weights/refcocog_lora_adapters_refgta_2000_samples/merged1"
        self.device = device
        # 1) Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=1,
            trust_remote_code=trust_remote_code
        )
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("No")
        # 2) Load model
        if llm is not None:
            self.model = llm
        else:
            # self.model  = LLM(
            #     model=model_path,
            #     trust_remote_code=True,
            #     max_model_len=12_000,
            #     gpu_memory_utilization=0.8,
            # )
            # self.model = make_llm(
            #     model_path,
            #     want_len=12_000,          # tighten this if you can — it’s a big speed lever
            #     target_util=0.85,
            #     target_concurrency=64,  # bump up until you hit VRAM or no longer see speedup
            # )
            self.model = make_llm(
                model_path,
                yes_token_id=self.yes_token_id,
                no_token_id=self.no_token_id,
            )

    

    def _score(self, image_paths: Union[str, 'PIL.Image', list], question: str, target_token="Yes",
               candidates: list = None, temperature=0.5, target_tokens=None):
        """
        1) Loads the image,
        2) builds a prompt with a single turn,
        3) asks for 1 new token,
        4) gets its logits for "Yes" vs "no"
        5) returns probability of "Yes" or "no".
        """
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        if candidates is not None:
            question = f"{question}\n If you have to classify among one of these objects {candidates}, {question}"
        else:
            question = f"{question}"
        # Build the prompt with a single user question
        messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
        prompt = self.tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)
        batch_inputs = [{
            "prompt": prompt,
            "multi_modal_data": {"image": image_paths},
        }]

        # We only have 1 step of generation, so the relevant logits are generation_output.scores[0]
        # This is shape: [batch_size, vocab_size]
        if target_tokens is not None:
            token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                extra_args={"target_tokens": token_ids},
            )
            outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
            for o in outputs:
                # from pprint import pprint
                # pprint(o.outputs)
                # Convert from logprob (log-space) to logits
                logprobs = np.array(
                    [o.outputs[0].logprobs[-1][x].logprob for x in token_ids if x in o.outputs[0].logprobs[-1]])

                # Apply softmax
                exp_logits = np.exp(logprobs - np.max(logprobs))  # numerical stability
                probs = exp_logits / exp_logits.sum()

                # return torch.tensor(probs, device=self.device)
                return torch.tensor(probs, device=self.device)
        else:
            target_tokens = ["Yes", "No"]
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                extra_args={"target_tokens": [self.yes_token_id, self.no_token_id]},
            )
            outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
            token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            for o in outputs:
                # Extract logits for Yes and No tokens [the logprob mode is raw_logits]
                try:
                    yes_logit = o.outputs[0].logprobs[-1][self.yes_token_id].logprob
                except:
                    yes_logit = -100.0  # very low logprob if "Yes" is not in the top-k
                try:
                    no_logit = o.outputs[0].logprobs[-1][self.no_token_id].logprob
                except:
                    no_logit = -100.0  # very low logprob if "No" is not in the top-k

                # Convert from logprob (log-space) to logits
                logits = np.array([yes_logit, no_logit])

                # Apply softmax
                exp_logits = np.exp(logits - np.max(logits))  # numerical stability
                probs = exp_logits / exp_logits.sum()
                # return torch.tensor(probs[0], device=self.device)
                return torch.tensor(probs, device=self.device)

    def _score_batch(self, image_paths: List[str], questions: List[str], candidates: List[str] = None,
                     target_tokens: Optional[List[str]] = None, target_token="Yes", temperature=0.5):
        """
        1) Loads the image,
        2) builds a prompt with a single turn,
        3) asks for 1 new token,
        4) gets its logits for "Yes" vs "no"
        5) returns probability of "Yes" or "no".
        """
        batch_inputs = []
        for i, (image_path, question) in enumerate(zip(image_paths, questions)):
            if not isinstance(image_path, list):
                image_path = [image_path]
            if candidates is not None:
                question = f"{question}\n If you have to classify among one of these objects {candidates}, {question}"
            else:
                question = f"{question}"
            # Build the prompt with a single user question
            messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
            prompt = self.tokenizer.apply_chat_template(messages,
                                                        tokenize=False,
                                                        add_generation_prompt=True)
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image_path},
            })

        # We only have 1 step of generation, so the relevant logits are generation_output.scores[0]
        # This is shape: [batch_size, vocab_size]
        if target_tokens is not None:
            token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                extra_args={"target_tokens": token_ids},
            )
            outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)

            probs = []
            for o in outputs:
                # Convert from logprob (log-space) to logits
                logits = np.array(
                    [o.outputs[0].logprobs[-1][x].logprob for x in token_ids if x in o.outputs[0].logprobs[-1]])

                # Apply softmax
                exp_logits = np.exp(logits - np.max(logits))  # numerical stability
                probs = exp_logits / exp_logits.sum()

                # return torch.tensor(probs, device=self.device)
                probs.append(torch.tensor(probs))
            return torch.stack(probs).to(self.device)
        else:
            probs = []
            target_tokens = ["Yes", "No"]
            token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                extra_args={"target_tokens": token_ids},
            )
            outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)

            for o in outputs:
                # Extract logits for Yes and No tokens
                yes_logit = o.outputs[0].logprobs[-1][self.yes_token_id].logprob
                no_logit = o.outputs[0].logprobs[-1][self.no_token_id].logprob
                # Convert from logit (log-space) to logits
                logits = np.array([yes_logit, no_logit])

                # Apply softmax
                exp_logits = np.exp(logits - np.max(logits))  # numerical stability
                temp_probs = exp_logits / exp_logits.sum()
                probs.append(torch.tensor(temp_probs[0]))
            return probs
            # return torch.tensor(probs[0], device=self.device)
            # return torch.stack(probs).to(self.device)

class InternVLShared(torch.nn.Module):
    model = None

    def __init__(self,model_path="OpenGVLab/InternVL3_5-8B",device = "cuda", relation=1, attr="no name", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relation = relation
        self.attr = attr
        if InternVLShared.model is None:
            InternVLShared.model = InternVL(model_path=model_path, device=device)
        self.model = InternVLShared.model

    def forward(self, image, bounding_boxes):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        images, questions = [], []
        if self.relation == 2:
            for box1 in bounding_boxes:
                for box2 in bounding_boxes:
                    img_copy = image.copy()
                    draw = ImageDraw.Draw(img_copy)
                    draw.rectangle(box1, outline="green", width=3)
                    draw.rectangle(box2, outline="green", width=3)
                    images.append(img_copy)
                    questions.append("Are the two objects in the bounding boxes red?")
        else:
            for box in bounding_boxes:
                img_copy = image.copy()
                ImageDraw.Draw(img_copy).rectangle(box, outline="green", width=3)
                images.append(img_copy)
                questions.append("Is the object in the bounding box red?")
            
        return self.model._score_batch(images, questions)


if __name__ == "__main__":
    import sys

    # 1. Configuration
    # Ensure this path points to a model you have downloaded or access to on HuggingFace
    # e.g., "OpenGVLab/InternVL2_5-8B-MPO" or "OpenGVLab/InternVL2-8B"
    MODEL_PATH = "OpenGVLab/InternVL3_5-8B"

    print(f"--- Initializing InternVL with model: {MODEL_PATH} ---")

    try:
        # Initialize the model
        # We pass trust_remote_code=True as usually required by InternVL
        model = InternVL(model_path=MODEL_PATH, device="cuda")

        # 2. Prepare a Dummy Image
        # Your _score method hardcodes "<image>\n{question}" in the prompt.
        # For a text-only fact check, we feed it a black square.
        dummy_image = Image.new('RGB', (224, 224), color='black')

        # 3. Define the Test Questions
        test_cases = [
            "Is Tehran the capital of Iran?",
            "Is Paris the capital of France?"
        ]

        print("\n--- Starting Probability Tests ---")

        for question in test_cases:
            print(f"\nQuerying: '{question}'")

            # 4. Call the scoring function
            # By default, _score compares "Yes" vs "No"
            probs_tensor = model._score(
                image_paths=[dummy_image],
                question=question
            )

            # 5. Interpret Output
            # _score returns a tensor. Based on your code logic:
            # It normalizes [Yes_logit, No_logit] -> Softmax -> [Prob_Yes, Prob_No]

            # Move to CPU for printing
            probs = probs_tensor.cpu().numpy()

            prob_yes = probs[0]
            prob_no = probs[1]

            print(f"Raw Output Tensor: {probs_tensor}")
            print(f"Probability 'Yes': {prob_yes:.6f}")
            print(f"Probability 'No' : {prob_no:.6f}")

            # Simple assertion logic
            if prob_yes > prob_no:
                print(">> Model Result: TRUE (Yes)")
            else:
                print(">> Model Result: FALSE (No)")

    except ImportError as e:
        print(f"Error: Missing dependency. {e}")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"An error occurred: {e}")