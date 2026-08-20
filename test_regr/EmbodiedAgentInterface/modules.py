import torch
from torch.nn import functional as F

from dataset import ACTION_VOCAB, EOS_TOKEN


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


class CausalLMActionObjectGenerator(torch.nn.Module):
    supports_batched_prefixes = True

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
        if not (device_map and str(device_map).lower() != "none"):
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

    def _prompt_base(self, text):
        return f"Instruction: {text}\nGenerated action tokens:"

    def _prompt(self, text, prefix_labels):
        prefix = " ".join(self._label_to_token(label) for label in prefix_labels if int(label) != self.eos_label)
        if prefix:
            return f"{self._prompt_base(text)} {prefix}"
        return self._prompt_base(text)

    def _model_hidden_states(self, input_ids, attention_mask=None):
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        with torch.set_grad_enabled(any(param.requires_grad for param in self.model.parameters())):
            outputs = self.model(**kwargs, output_hidden_states=True, use_cache=False)
        return outputs.hidden_states[-1].float().to(self.output.weight.device)

    def _next_logits_for_prefix(self, text, prefix_labels):
        prompt = self._prompt(text, prefix_labels)
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._model_input_device())
        hidden = self._model_hidden_states(
            inputs["input_ids"],
            inputs.get("attention_mask"),
        )[:, -1, :]
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
        ids = self.tokenizer(
            self._prompt_base(text),
            add_special_tokens=True,
        )["input_ids"]
        if not ids:
            ids = [self.tokenizer.eos_token_id or 0]
        boundary_positions = [len(ids) - 1]
        for label in row[1:padding_start].detach().cpu().tolist():
            label_ids = self.tokenizer(
                " " + self._label_to_token(label),
                add_special_tokens=False,
            )["input_ids"]
            if not label_ids:
                continue
            ids.extend(label_ids)
            boundary_positions.append(len(ids) - 1)

        if len(ids) > self.max_length:
            offset = len(ids) - self.max_length
            ids = ids[offset:]
            boundary_positions = [max(0, pos - offset) for pos in boundary_positions]

        return (
            torch.tensor(ids, dtype=torch.long, device=self._model_input_device()).unsqueeze(0),
            torch.tensor(boundary_positions, dtype=torch.long, device=self.output.weight.device),
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
