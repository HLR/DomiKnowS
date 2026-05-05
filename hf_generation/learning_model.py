"""Frozen-backbone generation learner for the hf_generation task."""
from __future__ import annotations

import torch


class FrozenBackboneGenerationHead(torch.nn.Module):
    """Predict compact generation labels from a frozen HuggingFace-style backbone.

    The module returns log-probabilities shaped ``[pad_size, label_count]`` so
    it can be consumed by DomiKnowS' enum-concept loss machinery, mirroring the
    Collie task while keeping only a small trainable head.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        label_count: int,
        hidden_size: int | None = None,
        pad_size: int = 4,
        label_to_token_id: list[int | None] | tuple[int | None, ...] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.pad_size = int(pad_size)
        self.label_count = int(label_count)
        if label_to_token_id is None:
            label_to_token_id = tuple(range(self.label_count))
        if len(label_to_token_id) != self.label_count:
            raise ValueError("label_to_token_id must contain one entry per compact label")
        self.label_to_token_id = tuple(None if token_id is None else int(token_id) for token_id in label_to_token_id)

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        if hidden_size is None:
            hidden_size = self._infer_hidden_size()
        self.head = torch.nn.Linear(hidden_size, self.label_count)

    def _infer_hidden_size(self) -> int:
        if hasattr(self.backbone, "embedding"):
            return int(self.backbone.embedding.embedding_dim)
        config = getattr(self.backbone, "config", None)
        if config is not None:
            for name in ("hidden_size", "n_embd"):
                value = getattr(config, name, None)
                if value is not None:
                    return int(value)
        raise ValueError("hidden_size is required when it cannot be inferred from the backbone")

    def _backbone_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.backbone(input_ids)
            if isinstance(output, torch.Tensor):
                features = output
            elif hasattr(output, "last_hidden_state"):
                features = output.last_hidden_state
            elif hasattr(output, "hidden_states") and output.hidden_states:
                features = output.hidden_states[-1]
            elif hasattr(output, "logits"):
                features = output.logits
            else:
                raise ValueError("backbone output must expose tensor features, hidden states, or logits")
        return features.detach()

    def token_id_for_label(self, label: int) -> int:
        """Return the raw tokenizer id used when feeding a compact label back."""
        label = int(label)
        if label < 0 or label >= self.label_count:
            raise ValueError(f"label {label} is out of range for {self.label_count} labels")
        token_id = self.label_to_token_id[label]
        if token_id is None:
            raise ValueError(f"label {label} does not map to a single tokenizer id")
        return token_id

    def next_label_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return next-step logits over compact generation labels."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        features = self._backbone_features(input_ids.long())
        return self.head(features[:, -1, :])[0]

    def forward(self, _contains, instruction_tokens: torch.Tensor, target_labels: torch.Tensor):
        if instruction_tokens.dim() == 1:
            instruction_tokens = instruction_tokens.unsqueeze(0)
        target_labels = target_labels.long()
        generated = []
        current = instruction_tokens.long()

        for step in range(self.pad_size):
            features = self._backbone_features(current)
            logits = self.head(features[:, -1, :])[0]
            generated.append(logits)

            if step < target_labels.numel():
                next_label = int(target_labels[step].item())
            else:
                next_label = 0
            next_token = torch.tensor([[self.token_id_for_label(next_label)]], dtype=torch.long, device=current.device)
            current = torch.cat([current, next_token], dim=1)

        logits = torch.stack(generated, dim=0)
        return torch.log_softmax(logits, dim=-1)

    def trainable_parameter_names(self) -> list[str]:
        """Return names of parameters that should be optimized."""
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]
