from dataclasses import dataclass

import torch
from torch.nn import functional as F

try:
    from .dataset import ACTION_VOCAB, EOS_TOKEN
except ImportError:  # Direct execution through EmbodiedAgentInterface/main.py.
    from dataset import ACTION_VOCAB, EOS_TOKEN
from domiknows.generation.prompting import (
    encode_label_prefix_prompt,
    label_prefix_token_ids,
)


CAUSAL_PROMPT_FORMAT = "qwen-chat-label-prefix-v1"


@dataclass
class CausalLMRolloutState:
    """Batched no-gradient Qwen state shared by parallel RL rollouts."""

    base_input_ids: tuple[int, ...]
    assistant_token_rows: list[list[int]]
    attention_mask: torch.Tensor | None = None
    past_key_values: object | None = None
    next_logits: torch.Tensor | None = None
    cache_advances: int = 0
    cache_rebuilds: int = 0


class EOSMaskedCrossEntropyLoss(torch.nn.Module):
    """Cross entropy through the first EOS, excluding EOS padding afterward."""

    def __init__(self, eos_label):
        super().__init__()
        self.eos_label = int(eos_label)

    def forward(self, input, target, *args, **kwargs):
        del args, kwargs
        logits = input.reshape(-1, input.shape[-1])
        labels = torch.as_tensor(target, dtype=torch.long, device=input.device)
        if labels.dim() == 0:
            labels = labels.reshape(1)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        label_shape = tuple(labels.shape)

        # Keep all tokens up to and including the first EOS in each sequence.
        eos_count = (labels == self.eos_label).cumsum(dim=-1)
        keep = eos_count <= 1
        keep = keep.reshape(-1)
        labels = labels.reshape(-1)
        if keep.numel() != logits.shape[0]:
            raise ValueError(
                f"Logit/label shape mismatch: {tuple(input.shape)} vs {label_shape}"
            )
        return F.cross_entropy(logits[keep], labels[keep])


def _prepare_transformers_imports():
    """Avoid hard failures when torchvision is installed but broken for the current torch build."""
    import transformers.utils.import_utils as hf_import_utils

    if not hf_import_utils.is_torchvision_available():
        return

    try:
        import torchvision  # noqa: F401
    except Exception:
        hf_import_utils._torchvision_available = False


class TextBERTEncoder(torch.nn.Module):
    def __init__(self, model_path="bert-base-uncased", device="cpu", max_length=256, freeze=True):
        super().__init__()
        _prepare_transformers_imports()
        from transformers import AutoModel, AutoTokenizer

        self.model_path = model_path
        self.device_name = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.hidden_size = self.model.config.hidden_size
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.eval() if freeze else self.model.train()

    def forward(self, text):
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, (list, tuple)):
            texts = [str(item) for item in text]
        else:
            texts = [str(text)]

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device_name)

        with torch.set_grad_enabled(any(param.requires_grad for param in self.model.parameters())):
            outputs = self.model(**tokens)
            pooled = outputs.last_hidden_state[:, 0, :]
        return pooled.float()


class AutoregressiveActionObjectGenerator(torch.nn.Module):
    def __init__(
        self,
        model_path="bert-base-uncased",
        label_count=2,
        eos_label=0,
        device="cpu",
        max_length=256,
        freeze=True,
        hidden_dim=128,
    ):
        super().__init__()
        self.encoder = TextBERTEncoder(
            model_path=model_path,
            device=device,
            max_length=max_length,
            freeze=freeze,
        )
        self.label_count = label_count
        self.eos_label = int(eos_label)
        self.device_name = device
        self.hidden_dim = hidden_dim
        self.token_embedding = torch.nn.Embedding(label_count, hidden_dim).to(device)
        self.context_projection = torch.nn.Linear(self.encoder.hidden_size, hidden_dim).to(device)
        self.gru = torch.nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True).to(device)
        self.output = torch.nn.Linear(hidden_dim, label_count).to(device)

    @property
    def hidden_size(self):
        return self.encoder.hidden_size

    def token_id_for_label(self, label):
        return int(label)

    def _context_hidden(self, text):
        context = self.encoder(text).float()
        if context.dim() == 1:
            context = context.unsqueeze(0)
        hidden = torch.tanh(self.context_projection(context)).unsqueeze(0)
        return hidden

    def _shift_right(self, target_labels):
        labels = torch.as_tensor(target_labels, dtype=torch.long, device=self.device_name)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        start = torch.full(
            (labels.shape[0], 1),
            self.eos_label,
            dtype=torch.long,
            device=self.device_name,
        )
        return torch.cat([start, labels[:, :-1]], dim=1)

    def sequence_logits(self, text, prefix_labels):
        prefix = torch.as_tensor(prefix_labels, dtype=torch.long, device=self.device_name)
        if prefix.dim() == 1:
            prefix = prefix.unsqueeze(0)
        prefix = prefix.clamp(min=0, max=self.label_count - 1)
        embeddings = self.token_embedding(prefix)
        outputs, _hidden = self.gru(embeddings, self._context_hidden(text))
        return self.output(outputs)

    def forward(self, _contains, text, target_labels):
        prefix = self._shift_right(target_labels)
        logits = self.sequence_logits(text, prefix)
        return logits.squeeze(0) if logits.shape[0] == 1 else logits

    def next_label_logits(self, input_ids, text=""):
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device_name)
        if ids.dim() == 2:
            ids = ids[0]
        if ids.numel() == 0:
            ids = torch.tensor([self.eos_label], dtype=torch.long, device=self.device_name)
        logits = self.sequence_logits(text, ids.unsqueeze(0))[0, -1, :]
        return logits


class ByteTextEncoder(torch.nn.Module):
    def __init__(self, hidden_dim=128, device="cpu", max_length=512):
        super().__init__()
        self.device_name = device
        self.max_length = max_length
        self.embedding = torch.nn.Embedding(256, hidden_dim).to(device)
        self.projection = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
        ).to(device)

    def _encode_one(self, text):
        raw = str(text).encode("utf-8", errors="ignore")[: self.max_length]
        if not raw:
            raw = b" "
        return torch.tensor(list(raw), dtype=torch.long, device=self.device_name)

    def forward(self, text):
        texts = [text] if isinstance(text, str) else list(text) if isinstance(text, (list, tuple)) else [str(text)]
        encoded = [self._encode_one(item) for item in texts]
        max_len = max(item.numel() for item in encoded)
        padded = torch.zeros((len(encoded), max_len), dtype=torch.long, device=self.device_name)
        mask = torch.zeros((len(encoded), max_len), dtype=torch.float32, device=self.device_name)
        for row, item in enumerate(encoded):
            padded[row, : item.numel()] = item
            mask[row, : item.numel()] = 1.0
        embeddings = self.embedding(padded)
        pooled = (embeddings * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.projection(pooled.float())


class TinyTransformerActionObjectGenerator(torch.nn.Module):
    supports_batched_prefixes = True

    def __init__(
        self,
        label_count=2,
        eos_label=0,
        device="cpu",
        max_length=512,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.label_count = label_count
        self.eos_label = int(eos_label)
        self.device_name = device
        self.hidden_dim = hidden_dim
        self.text_encoder = ByteTextEncoder(hidden_dim=hidden_dim, device=device, max_length=max_length)
        self.token_embedding = torch.nn.Embedding(label_count, hidden_dim).to(device)
        self.position_embedding = torch.nn.Embedding(512, hidden_dim).to(device)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=num_layers).to(device)
        self.output = torch.nn.Linear(hidden_dim, label_count).to(device)

    def token_id_for_label(self, label):
        return int(label)

    def _shift_right(self, target_labels):
        labels = torch.as_tensor(target_labels, dtype=torch.long, device=self.device_name)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        start = torch.full((labels.shape[0], 1), self.eos_label, dtype=torch.long, device=self.device_name)
        return torch.cat([start, labels[:, :-1]], dim=1)

    def sequence_logits(self, text, prefix_labels):
        prefix = torch.as_tensor(prefix_labels, dtype=torch.long, device=self.device_name)
        if prefix.dim() == 1:
            prefix = prefix.unsqueeze(0)
        prefix = prefix.clamp(min=0, max=self.label_count - 1)
        seq_len = prefix.shape[1]
        positions = torch.arange(seq_len, dtype=torch.long, device=self.device_name).unsqueeze(0)
        positions = positions.clamp(max=self.position_embedding.num_embeddings - 1)
        context = self.text_encoder(text).unsqueeze(1)
        hidden = self.token_embedding(prefix) + self.position_embedding(positions) + context
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=self.device_name),
            diagonal=1,
        )
        outputs = self.transformer(hidden, mask=causal_mask)
        return self.output(outputs)

    def forward(self, _contains, text, target_labels):
        logits = self.sequence_logits(text, self._shift_right(target_labels))
        return logits.squeeze(0) if logits.shape[0] == 1 else logits

    def next_label_logits(self, input_ids, text=""):
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device_name)
        if ids.dim() == 2:
            ids = ids[0]
        if ids.numel() == 0:
            ids = torch.tensor([self.eos_label], dtype=torch.long, device=self.device_name)
        return self.sequence_logits(text, ids.unsqueeze(0))[0, -1, :]


class PretrainedLabelAdapter(torch.nn.Module):
    """Score EAI labels with Qwen's pretrained output geometry plus a low-rank residual."""

    def __init__(self, model, tokenizer, vocabulary, eos_label, hidden_size, rank=64, device=None):
        super().__init__()
        if vocabulary is None:
            raise ValueError("pretrained-adapter requires the EAI TokenVocabulary")
        if int(rank) <= 0:
            raise ValueError("label adapter rank must be positive")
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise ValueError("causal LM does not expose pretrained output embeddings")
        native_weight = output_embeddings.weight.detach()
        label_token_ids = []
        for label in range(vocabulary.label_count):
            if label == int(eos_label):
                token_ids = [tokenizer.eos_token_id]
            else:
                surface = vocabulary.token_for_label(label)
                token_ids = tokenizer(
                    " " + surface, add_special_tokens=False
                )["input_ids"]
                if not token_ids:
                    token_ids = tokenizer(surface, add_special_tokens=False)["input_ids"]
            token_ids = [int(token_id) for token_id in token_ids if token_id is not None]
            if not token_ids:
                raise ValueError(f"label {label} has no native tokenizer representation")
            label_token_ids.append(token_ids)

        unique_ids = sorted({token_id for ids in label_token_ids for token_id in ids})
        native_index = torch.tensor(
            unique_ids, dtype=torch.long, device=native_weight.device
        )
        selected = native_weight.index_select(0, native_index).float().cpu()
        selected_row = {token_id: index for index, token_id in enumerate(unique_ids)}
        vectors = [
            selected[
                torch.tensor([selected_row[token_id] for token_id in token_ids])
            ].mean(dim=0)
            for token_ids in label_token_ids
        ]

        target_device = device or native_weight.device
        self.register_buffer("base_label_vectors", torch.stack(vectors).to(target_device))
        self.residual_down = torch.nn.Linear(hidden_size, int(rank), bias=False).to(target_device)
        self.residual_up = torch.nn.Linear(int(rank), vocabulary.label_count, bias=False).to(target_device)
        self.bias = torch.nn.Parameter(torch.zeros(vocabulary.label_count, device=target_device))
        self.log_temperature = torch.nn.Parameter(torch.zeros((), device=target_device))
        torch.nn.init.zeros_(self.residual_up.weight)

    @property
    def weight(self):
        """Compatibility with code that uses a linear head's device and dtype."""
        return self.base_label_vectors

    def forward(self, hidden):
        hidden = hidden.float()
        base = F.linear(hidden, self.base_label_vectors)
        residual = self.residual_up(F.gelu(self.residual_down(hidden)))
        temperature = self.log_temperature.exp().clamp(max=100.0)
        return temperature * base + residual + self.bias


class CausalLMActionObjectGenerator(torch.nn.Module):
    supports_batched_prefixes = True
    supports_incremental_rollout = True

    def __init__(
        self,
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        label_count=2,
        eos_label=0,
        device="cpu",
        max_length=512,
        freeze=True,
        hidden_dim=None,
        vocabulary=None,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=None,
        device_map=None,
        gradient_checkpointing=False,
        low_cpu_mem_usage=True,
        shared_model=None,
        shared_tokenizer=None,
        label_head="pretrained-adapter",
        label_adapter_rank=64,
        prompt_builder=None,
        prompt_key="causal_prompt_text",
    ):
        super().__init__()
        _prepare_transformers_imports()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.label_count = label_count
        self.eos_label = int(eos_label)
        self.device_name = device
        self.max_length = max_length
        self.vocabulary = vocabulary
        self.use_lora = bool(use_lora)
        self.prompt_builder = prompt_builder
        self.prompt_key = str(prompt_key)
        self.prompt_format = CAUSAL_PROMPT_FORMAT
        if label_head not in {"pretrained-adapter", "linear"}:
            raise ValueError(f"Unsupported causal label head {label_head!r}")
        self.label_head_type = label_head
        self.label_adapter_rank = int(label_adapter_rank)
        if self.use_lora and shared_model is not None:
            raise ValueError("shared_model cannot be used with --use-lora because LoRA mutates the backbone")
        self.tokenizer = shared_tokenizer or AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        if shared_model is None:
            model_kwargs = {
                "dtype": dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": low_cpu_mem_usage,
            }
            if device_map and str(device_map).lower() != "none":
                model_kwargs["device_map"] = device_map
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        else:
            self.model = shared_model
        if shared_model is None and not (
            device_map and str(device_map).lower() != "none"
        ):
            self.model = self.model.to(device)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        if self.use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:
                raise ImportError(
                    "--use-lora requires the 'peft' package in this environment."
                ) from exc
            target_modules = lora_target_modules or (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            )
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=list(target_modules),
            )
            self.model = get_peft_model(self.model, config)
            self.model.train()
        else:
            if freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model.eval() if freeze else self.model.train()

        model_hidden = getattr(self.model.config, "hidden_size", hidden_dim or 768)
        if label_head == "pretrained-adapter":
            self.output = PretrainedLabelAdapter(
                self.model,
                self.tokenizer,
                vocabulary,
                self.eos_label,
                model_hidden,
                rank=self.label_adapter_rank,
                device=self._model_input_device(),
            )
        else:
            self.output = torch.nn.Linear(model_hidden, label_count).to(self._model_input_device())

    def _model_input_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device(self.device_name)

    def trainable_parameter_count(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def token_id_for_label(self, label):
        return int(label)

    def _label_to_token(self, label):
        if self.vocabulary is None:
            return str(int(label))
        try:
            return self.vocabulary.token_for_label(int(label))
        except Exception:
            return str(int(label))

    def _prompt_user_content(self, text):
        prompt_builder = getattr(self, "prompt_builder", None)
        if prompt_builder is not None:
            return str(prompt_builder(text)).strip()
        return (
            "Predict an embodied-agent action plan one label at a time.\n"
            f"{str(text).strip()}\n"
            "Return only action or entity labels; do not explain."
        )

    def _prefix_tokens(self, prefix_labels):
        return tuple(
            self._label_to_token(label)
            for label in prefix_labels
            if int(label) != self.eos_label
        )

    def _prompt_encoding(self, text, prefix_labels=()):
        return encode_label_prefix_prompt(
            self.tokenizer,
            self._prompt_user_content(text),
            self._prefix_tokens(prefix_labels),
            max_length=self.max_length,
            enable_thinking=False,
        )

    def _prompt_base(self, text):
        return self._prompt_encoding(text).rendered_text

    def _prompt(self, text, prefix_labels):
        return self._prompt_encoding(text, prefix_labels).rendered_text

    def _rollout_pad_token_id(self):
        for name in ("pad_token_id", "eos_token_id"):
            token_id = getattr(self.tokenizer, name, None)
            if token_id is not None:
                return int(token_id)
        return 0

    def _rollout_forward(
        self,
        input_ids,
        attention_mask,
        *,
        past_key_values=None,
        position_ids=None,
    ):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": True,
            "return_dict": True,
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        with torch.no_grad():
            outputs = self.model(**kwargs)
        hidden = outputs.hidden_states[-1][:, -1, :].float().to(
            self.output.weight.device
        )
        return self.output(hidden), getattr(outputs, "past_key_values", None)

    def _rebuild_rollout_state(self, state):
        rows = [
            list(state.base_input_ids) + list(assistant_ids)
            for assistant_ids in state.assistant_token_rows
        ]
        rows = [row[-self.max_length :] for row in rows]
        width = max(len(row) for row in rows)
        device = self._model_input_device()
        input_ids = torch.full(
            (len(rows), width),
            self._rollout_pad_token_id(),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(rows), width), dtype=torch.long, device=device
        )
        for index, row in enumerate(rows):
            offset = width - len(row)
            input_ids[index, offset:] = torch.tensor(
                row, dtype=torch.long, device=device
            )
            attention_mask[index, offset:] = 1
        position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)
        logits, cache = self._rollout_forward(
            input_ids,
            attention_mask,
            position_ids=position_ids,
        )
        state.attention_mask = attention_mask
        state.past_key_values = cache
        state.next_logits = logits
        state.cache_rebuilds += 1
        return state

    def start_incremental_rollout(self, text, batch_size):
        """Encode one prompt batch and return its first-label logits and cache."""
        if int(batch_size) < 1:
            raise ValueError("incremental rollout batch_size must be positive")
        base = self._prompt_encoding(text)
        state = CausalLMRolloutState(
            base_input_ids=base.input_ids,
            assistant_token_rows=[[] for _ in range(int(batch_size))],
        )
        return self._rebuild_rollout_state(state)

    def advance_incremental_rollout(self, state, labels, continuing):
        """Advance parallel rollouts, reusing KV cache when chunk widths agree."""
        label_values = torch.as_tensor(labels).detach().cpu().reshape(-1).tolist()
        continuing_values = (
            torch.as_tensor(continuing).detach().cpu().bool().reshape(-1).tolist()
        )
        if len(label_values) != len(state.assistant_token_rows):
            raise ValueError("incremental rollout label batch changed size")
        if len(continuing_values) != len(label_values):
            raise ValueError("incremental rollout continuation mask changed size")

        chunks: list[tuple[int, ...]] = []
        for index, (label, is_continuing) in enumerate(
            zip(label_values, continuing_values)
        ):
            if is_continuing:
                chunk = label_prefix_token_ids(
                    self.tokenizer, self._label_to_token(label)
                )
                state.assistant_token_rows[index].extend(chunk)
            else:
                chunk = ()
            chunks.append(chunk)

        active_lengths = {len(chunk) for chunk in chunks if chunk}
        if not active_lengths:
            return state
        chunk_width = next(iter(active_lengths))
        can_advance_cache = (
            state.past_key_values is not None
            and len(active_lengths) == 1
            and state.attention_mask is not None
            and state.attention_mask.shape[1] + chunk_width <= self.max_length
        )
        if not can_advance_cache:
            return self._rebuild_rollout_state(state)

        device = self._model_input_device()
        batch_size = len(chunks)
        input_ids = torch.full(
            (batch_size, chunk_width),
            self._rollout_pad_token_id(),
            dtype=torch.long,
            device=device,
        )
        chunk_mask = torch.zeros(
            (batch_size, chunk_width), dtype=torch.long, device=device
        )
        for index, chunk in enumerate(chunks):
            if not chunk:
                continue
            input_ids[index] = torch.tensor(chunk, dtype=torch.long, device=device)
            chunk_mask[index] = 1
        attention_mask = torch.cat([state.attention_mask, chunk_mask], dim=-1)
        position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)
        logits, cache = self._rollout_forward(
            input_ids,
            attention_mask,
            past_key_values=state.past_key_values,
            position_ids=position_ids[:, -chunk_width:],
        )
        state.attention_mask = attention_mask
        state.past_key_values = cache
        state.next_logits = logits
        state.cache_advances += 1
        return state

    def _model_hidden_states(self, input_ids, attention_mask=None):
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        grad_enabled = torch.is_grad_enabled() and any(
            param.requires_grad for param in self.model.parameters()
        )
        with torch.set_grad_enabled(grad_enabled):
            outputs = self.model(**kwargs, output_hidden_states=True, use_cache=False)
        return outputs.hidden_states[-1].float().to(self.output.weight.device)

    def _next_logits_for_prefix(self, text, prefix_labels):
        encoding = self._prompt_encoding(text, prefix_labels)
        input_ids = torch.tensor(
            encoding.input_ids,
            dtype=torch.long,
            device=self._model_input_device(),
        ).unsqueeze(0)
        hidden = self._model_hidden_states(input_ids)[:, -1, :]
        return self.output(hidden)[0]

    def _shift_right(self, target_labels):
        labels = torch.as_tensor(target_labels, dtype=torch.long, device=self.device_name)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        start = torch.full((labels.shape[0], 1), self.eos_label, dtype=torch.long, device=self.device_name)
        return torch.cat([start, labels[:, :-1]], dim=1)

    def _constant_eos_logits(self):
        logits = torch.full(
            (self.label_count,),
            -8.0,
            dtype=self.output.weight.dtype,
            device=self.output.weight.device,
        )
        logits[self.eos_label] = 8.0
        return logits

    def _padding_start_from_shifted_prefix(self, row):
        # ``row`` is shifted-right target labels: [BOS/EOS, y0, ..., y_{n-1}].
        # The first EOS after index 0 corresponds to already-finished padding
        # positions, so those later predictions do not need another LLM forward.
        eos_positions = (row[1:] == self.eos_label).nonzero(as_tuple=False)
        if eos_positions.numel() == 0:
            return row.numel()
        return int(eos_positions[0].item()) + 1

    def _teacher_forced_input(self, text, row, padding_start):
        encoding = self._prompt_encoding(
            text,
            row[1:padding_start].detach().cpu().tolist(),
        )

        return (
            torch.tensor(
                encoding.input_ids,
                dtype=torch.long,
                device=self._model_input_device(),
            ).unsqueeze(0),
            torch.tensor(
                encoding.boundary_positions,
                dtype=torch.long,
                device=self.output.weight.device,
            ),
        )

    def sequence_logits(self, text, prefix_labels):
        prefix = torch.as_tensor(prefix_labels, dtype=torch.long, device=self.device_name)
        if prefix.dim() == 1:
            prefix = prefix.unsqueeze(0)
        rows = []
        eos_logits = None
        for row in prefix:
            padding_start = self._padding_start_from_shifted_prefix(row)
            input_ids, boundary_positions = self._teacher_forced_input(text, row, padding_start)
            hidden_states = self._model_hidden_states(input_ids)[0]
            active_logits = self.output(hidden_states.index_select(0, boundary_positions))
            logits = [active_logits[index] for index in range(active_logits.shape[0])]
            while len(logits) < row.numel():
                if eos_logits is None:
                    eos_logits = self._constant_eos_logits()
                logits.append(eos_logits)
            rows.append(torch.stack(logits[: row.numel()], dim=0))
        return torch.stack(rows, dim=0)

    def forward(self, _contains, text, target_labels):
        logits = self.sequence_logits(text, self._shift_right(target_labels))
        return logits.squeeze(0) if logits.shape[0] == 1 else logits

    def next_label_logits(self, input_ids, text=""):
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device_name)
        if ids.dim() == 2:
            ids = ids[0]
        if ids.numel() == 0:
            ids = torch.tensor([self.eos_label], dtype=torch.long, device=self.device_name)
        return self._next_logits_for_prefix(text, ids.tolist())


class TextBERTTokenEncoder(TextBERTEncoder):
    def __init__(
        self,
        model_path="bert-base-uncased",
        device="cpu",
        max_length=256,
        freeze=True,
        max_steps=8,
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            max_length=max_length,
            freeze=freeze,
        )
        self.position_embedding = torch.nn.Embedding(max_steps, self.hidden_size).to(device)

    def forward(self, _contains, text, positions):
        pooled = super().forward(text)
        if pooled.dim() == 2 and pooled.shape[0] == 1:
            pooled = pooled[0]
        positions = torch.as_tensor(positions, dtype=torch.long, device=self.device_name).reshape(-1)
        positions = positions.clamp(min=0, max=self.position_embedding.num_embeddings - 1)
        context = pooled.float().reshape(1, -1).expand(positions.shape[0], -1)
        return context + self.position_embedding(positions)


class TokenActionClassifier(torch.nn.Module):
    def __init__(self, feature_dim, label_count, hidden_dim=128, device="cpu"):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, label_count),
        )
        self.to(device)

    def forward(self, features):
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return self.net(features.float())


class BERTActionSequenceGenerator(torch.nn.Module):
    def __init__(self, feature_dim, label_count, max_steps=8, hidden_dim=128, device="cpu"):
        super().__init__()
        self.max_steps = max_steps
        self.label_count = label_count
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, max_steps * label_count),
        )
        self.to(device)

    def forward(self, features):
        if features.dim() == 1:
            features = features.unsqueeze(0)
        logits = self.net(features.float()).view(-1, self.max_steps, self.label_count)
        return logits.squeeze(0) if logits.shape[0] == 1 else logits

    def decode_labels(self, features):
        logits = self.forward(features)
        if logits.dim() == 3:
            logits = logits[0]
        return torch.argmax(logits, dim=-1)


class SmallLLMPlanGenerator(torch.nn.Module):
    def __init__(
        self,
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        max_new_tokens=128,
        max_steps=8,
        vocabulary=None,
        policy_dfa=None,
    ):
        super().__init__()
        _prepare_transformers_imports()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.device_name = device
        self.max_new_tokens = max_new_tokens
        self.max_steps = max_steps
        if vocabulary is None:
            from domiknows.generation.dfa.vocabulary import TokenVocabulary

            vocabulary = TokenVocabulary(ACTION_VOCAB, eos_token=EOS_TOKEN)
        self.vocabulary = vocabulary
        self.policy_dfa = policy_dfa
        self.allowed_actions = tuple(
            token for token in self.vocabulary.tokens if token != self.vocabulary.eos_token
        )
        self.fallback_token = "other" if "other" in self.allowed_actions else self.vocabulary.other_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

    def build_prompt(self, instruction, goal=""):
        action_names = ", ".join(token.upper() for token in self.allowed_actions)
        return (
            "Generate a short embodied-agent action plan. Use one action per line. "
            f"Use only these action names: {action_names}.\n"
            f"Instruction: {instruction}\n"
            f"Goal: {goal}\n"
            "Plan:"
        )

    @torch.no_grad()
    def generate_from_text(self, instruction, goal=""):
        prompt = self.build_prompt(instruction, goal)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device_name)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def normalize_generated_text(self, text):
        actions = []
        for line in str(text).splitlines():
            item = line.strip().strip("-0123456789. )").lower()
            if not item:
                continue
            action = item.split()[0]
            if action not in self.allowed_actions:
                action = self.fallback_token
            actions.append(action)
        if self.vocabulary.eos_token not in actions:
            actions.append(self.vocabulary.eos_token)
        if self.policy_dfa is not None:
            constrained = []
            state = self.policy_dfa.start_state
            for step in range(self.max_steps):
                allowed = {
                    int(label)
                    for label in self.policy_dfa.allowed_tokens(
                        state, remaining_steps=self.max_steps - step
                    )
                }
                if not allowed:
                    raise RuntimeError(
                        f"graph policy DFA has no productive label at step {step}"
                    )
                candidate = actions[step] if step < len(actions) else self.vocabulary.eos_token
                try:
                    candidate_label = self.vocabulary.label_for_token(candidate)
                except KeyError:
                    candidate_label = self.vocabulary.other_label
                label = candidate_label if candidate_label in allowed else min(allowed)
                constrained.append(self.vocabulary.token_for_label(label))
                next_state = self.policy_dfa.step(state, label)
                if next_state is None:
                    raise RuntimeError("graph policy DFA rejected an allowed label")
                state = next_state
                if label == self.vocabulary.eos_label:
                    break
            actions = constrained
        actions = actions[: self.max_steps]
        while len(actions) < self.max_steps:
            actions.append(self.vocabulary.eos_token)
        return actions

    def sequence_to_logits(self, actions):
        logits = torch.full(
            (self.max_steps, self.vocabulary.label_count),
            -8.0,
            dtype=torch.float32,
            device=self.device_name,
        )
        for step, action in enumerate(actions[: self.max_steps]):
            try:
                label_id = self.vocabulary.label_for_token(action)
            except KeyError:
                label_id = self.vocabulary.other_label
            logits[step, label_id] = 8.0
        return logits

    def generate_action_sequence(self, sample):
        return self.normalize_generated_text(self.generate(sample))

    def generate(self, sample):
        instruction = sample.get("natural_language_description") or sample.get("text") or ""
        goal = sample.get("tl_goal") or ""
        return self.generate_from_text(instruction, goal)

    def forward(self, _contains, text, goal=""):
        if isinstance(text, (list, tuple)):
            if not isinstance(goal, (list, tuple)):
                goal = [goal] * len(text)
            return torch.stack(
                [self.sequence_to_logits(self.normalize_generated_text(self.generate_from_text(item, g)))
                 for item, g in zip(text, goal)],
                dim=0,
            )
        actions = self.normalize_generated_text(self.generate_from_text(text, goal))
        return self.sequence_to_logits(actions)
