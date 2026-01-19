import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# =============================================================================
# Image preprocessing (InternVL-ish)
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
            candidates.append((abs(target_ratio - ratio), gw, gh))
    candidates.sort(key=lambda x: x[0])
    _, gw, gh = candidates[0]
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

def build_internvl_query(model, tokenizer, question: str):
    """
    Matches InternVL chat-style prompt:
      <img> <IMG_CONTEXT>*num_image_token </img>\n{question}
    """
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    get_conv_template = get_conv_template_from_model(model)
    template = get_conv_template(model.template)

    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * model.num_image_token) + IMG_END_TOKEN
    template.append_message(template.roles[0], image_tokens + "\n" + question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return query


# =============================================================================
# Shared singleton: InternVL HF (trainable) with InternVL-like API
# =============================================================================
class InternVLHF:
    """
    HF/PyTorch implementation that mirrors your vLLM InternVL API:
      - _score(image_paths, question, target_tokens=...)
      - _score_batch(image_paths, questions, ...)
    Returns probabilities for target tokens (default ["Yes","No"]).
    Supports backprop if you use it under a module that computes a loss.
    """

    _shared = None  # singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._shared is None:
            cls._shared = super().__new__(cls)
        return cls._shared

    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL2_5-8B-MPO",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        # LoRA shared across all users of the singleton
        use_llm_lora: bool = False,
        use_vision_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        if getattr(self, "_initialized", False):
            return

        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        if use_llm_lora:
            llm_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            llm_lora = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                task_type="CAUSAL_LM",
                target_modules=llm_targets,
            )
            if hasattr(self.model, "language_model"):
                self.model.language_model = get_peft_model(self.model.language_model, llm_lora)
            else:
                self.model = get_peft_model(self.model, llm_lora)

        if use_vision_lora and hasattr(self.model, "vision_model"):
            vit_targets = ["qkv", "proj", "fc1", "fc2"]
            vit_lora = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=vit_targets,
            )
            self.model.vision_model = get_peft_model(self.model.vision_model, vit_lora)

        self.model.to(device)
        self.model.train()

        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_token_id  = self.tokenizer.convert_tokens_to_ids("No")

        self._initialized = True

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
        for img in image_paths:
            pv = load_image_to_tiles(img, input_size=input_size, max_num=max_num, use_thumbnail=use_thumbnail)
            pvs.append(pv)
        pixel_values = torch.cat(pvs, dim=0).to(self.device, dtype=self.dtype)
        image_flags = torch.ones(pixel_values.shape[0], 1, device=self.device, dtype=torch.long)

        # prompts
        queries = [build_internvl_query(self.model, self.tokenizer, q) for q in questions]
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

        sel = next_logits[0, token_ids]  # [K]
        probs = torch.softmax(sel, dim=-1)
        return probs.to(self.device)

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
    ):
        """
        Returns tensor probs [B,K] on device.
        Default target_tokens=["Yes","No"] -> [B,2]
        """
        if candidates is not None:
            questions = [
                f"{q}\n If you have to classify among one of these objects {candidates}, {q}"
                for q in questions
            ]

        if target_tokens is None:
            target_tokens = ["Yes", "No"]

        token_ids = [self.tokenizer.convert_tokens_to_ids(tok) for tok in target_tokens]
        next_logits = self._next_token_logits(
            image_paths=image_paths,
            questions=questions,
            input_size=input_size,
            max_num=max_num,
            use_thumbnail=False,
        )  # [B,V]

        sel = next_logits[:, token_ids]  # [B,K]
        probs = torch.softmax(sel, dim=-1)
        return probs.to(self.device)


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
        model_path="OpenGVLab/InternVL2_5-8B-MPO",
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
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.relation = relation
        self.attr = attr
        self.device = device

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
            )
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
        Draw one or more boxes on a copy of base image, then resize to fixed input_size
        to keep tile count stable in batched InternVL forward.
        """
        img = base_pil.copy()
        draw = ImageDraw.Draw(img)
        for box, color in zip(boxes, colors):
            if isinstance(box, torch.Tensor):
                box = box.detach().cpu().tolist()
            draw.rectangle(box, outline=color, width=3)
        # resize to fixed size (stabilizes tiling)
        img = img.resize((self.input_size, self.input_size), resample=Image.BICUBIC)
        return img

    def forward(self, image, bounding_boxes, labels=None):
        """
        Exactly like your vLLM InternVLShared:
          image: [something] (you used image=image[0]) — we support both.
          bounding_boxes: iterable of boxes (Tensor/list)
          returns: probs tensor [N,2] (Yes,No) on device

        If labels provided, returns (probs, loss).
        """
        # match your old behavior: image may come in as [image]
        if isinstance(image, (list, tuple)) and len(image) == 1:
            image = image[0]

        base = self._to_pil(image)

        images, questions = [], []

        if self.relation == 2:
            # all ordered pairs (box1, box2)
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

        # probs: [N,2] for ["Yes","No"]
        probs = self.model._score_batch(
            image_paths=images,
            questions=questions,
            target_tokens=["Yes", "No"],
            input_size=self.input_size,
            max_num=self.max_num,
        )

        if labels is None:
            return probs

        # training option: labels 0=Yes,1=No
        labels = labels.to(self.device).long()
        # convert probs back to logits safely: use model next-token logits instead (better gradients)
        # But easiest: recompute logits directly here for correct CE gradients.
        # We'll do it properly: compute next-token logits and pick yes/no.
        next_logits = self.model._next_token_logits(
            image_paths=images,
            questions=questions,
            input_size=self.input_size,
            max_num=self.max_num,
            use_thumbnail=False,
        )  # [N,V]
        yes_id = self.model.yes_token_id
        no_id  = self.model.no_token_id
        yn_logits = torch.stack([next_logits[:, yes_id], next_logits[:, no_id]], dim=-1)  # [N,2]
        loss = F.cross_entropy(yn_logits, labels)
        probs = torch.softmax(yn_logits, dim=-1)
        return probs, loss


# =============================================================================
# Quick sanity test (mirrors your vLLM main)
# =============================================================================
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    MODEL_PATH = "OpenGVLab/InternVL2_5-1B-MPO"
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
        max_num=1,          # keep batching stable
        use_llm_lora=False, # turn on if you want training adapters shared
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
    probs2, loss = m(dummy, boxes, labels=labels)
    print("loss:", loss.item())
    loss.backward()
    print("✅ backward ok")