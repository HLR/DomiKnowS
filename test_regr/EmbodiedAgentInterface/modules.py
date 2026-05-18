import torch

from dataset import ACTION_VOCAB, EOS_TOKEN


class TextBERTEncoder(torch.nn.Module):
    def __init__(self, model_path="bert-base-uncased", device="cpu", max_length=256, freeze=True):
        super().__init__()
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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.device_name = device
        self.max_new_tokens = max_new_tokens
        self.max_steps = max_steps
        if vocabulary is None:
            from domiknows.generation.vocabulary import TokenVocabulary

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
            torch_dtype=dtype,
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
