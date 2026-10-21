"""Learned truth for the ``task__<name>`` concepts: a classifier over the instruction text.

At ``--knowledge learned`` the task concepts are observations predicted from the instruction instead of read from
the episode. The model is a hashed word and bigram bag with a linear softmax head. It is fitted on the training
split only; the episode ids it saw are stored with the weights so evaluation can refuse an overlapping split.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

BUCKETS = 8192
_WORD = re.compile(r"[a-z0-9_']+")


def features(text: str, buckets: int = BUCKETS) -> torch.Tensor:
    words = _WORD.findall(str(text).lower())
    grams = words + [f"{a} {b}" for a, b in zip(words, words[1:])]
    vector = torch.zeros(buckets)
    for gram in grams:
        vector[zlib.crc32(gram.encode("utf-8")) % buckets] += 1.0
    vector = torch.log1p(vector)
    return vector / vector.norm().clamp_min(1e-6)


@dataclass
class TaskClassifier:
    classes: tuple[str, ...]
    weight: torch.Tensor
    bias: torch.Tensor
    train_episodes: frozenset[str]
    buckets: int = BUCKETS

    def probabilities(self, instruction: str) -> torch.Tensor:
        logits = features(instruction, self.buckets) @ self.weight + self.bias
        return torch.softmax(logits, dim=-1)

    def predict(self, instruction: str, *, min_confidence: float = 0.0) -> str | None:
        """The most likely task, or ``None`` when its probability is below ``min_confidence``."""
        probabilities = self.probabilities(instruction)
        confidence, index = probabilities.max(dim=-1)
        return self.classes[int(index)] if float(confidence) >= min_confidence else None

    def save(self, path: str | Path) -> None:
        torch.save({"classes": list(self.classes), "weight": self.weight, "bias": self.bias,
                    "train_episodes": sorted(self.train_episodes), "buckets": self.buckets}, path)

    @classmethod
    def load(cls, path: str | Path) -> "TaskClassifier":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(tuple(payload["classes"]), payload["weight"], payload["bias"],
                   frozenset(payload["train_episodes"]), int(payload["buckets"]))

    def assert_disjoint(self, examples: Sequence) -> None:
        overlap = self.train_episodes & {example.episode_id for example in examples}
        if overlap:
            raise ValueError(f"task classifier was fitted on {len(overlap)} evaluation episodes, e.g. {sorted(overlap)[:3]}")


def fit_task_classifier(examples: Sequence, *, epochs: int = 300, learning_rate: float = 0.5,
                        weight_decay: float = 1e-4, seed: int = 0, buckets: int = BUCKETS) -> TaskClassifier:
    """Fit on ``examples`` (instruction and task name per episode); pass the training split only."""
    examples = [e for e in examples if e.task]
    if not examples:
        raise ValueError("no labelled examples to fit the task classifier")
    classes = tuple(sorted({e.task for e in examples}))
    index = {name: i for i, name in enumerate(classes)}
    inputs = torch.stack([features(e.instruction, buckets) for e in examples])
    targets = torch.tensor([index[e.task] for e in examples])
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.randn(buckets, len(classes), generator=generator) * 0.01).requires_grad_()
    bias = torch.zeros(len(classes), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=learning_rate * 0.1, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(inputs @ weight + bias, targets)
        loss.backward()
        optimizer.step()
    return TaskClassifier(classes, weight.detach(), bias.detach(), frozenset(e.episode_id for e in examples), buckets)


def accuracy(classifier: TaskClassifier, examples: Sequence, *, min_confidence: float = 0.0) -> dict:
    """Task-name accuracy on ``examples``, plus accuracy restricted to the tasks the classifier was fitted on."""
    seen = set(classifier.classes)
    rows = [(e.task, classifier.predict(e.instruction, min_confidence=min_confidence)) for e in examples if e.task]
    known = [(t, p) for t, p in rows if t in seen]
    return {"examples": len(rows), "accuracy": sum(t == p for t, p in rows) / max(1, len(rows)),
            "known_task_examples": len(known), "known_task_accuracy": sum(t == p for t, p in known) / max(1, len(known))}


def predict_tasks(examples: Sequence, path: str | Path | None, *, observation_mode: str = "visual",
                  ground_complex: bool = True, min_confidence: float = 0.0) -> list:
    """Examples whose task-derived inputs come from the classifier's prediction (``--knowledge learned``)."""
    if path is None:
        raise ValueError("--knowledge learned needs --task-classifier (fit with the fit-task-classifier command)")
    try:
        from ..dataset import with_predicted_task
    except ImportError:
        from dataset import with_predicted_task
    classifier = TaskClassifier.load(path)
    classifier.assert_disjoint(examples)
    return [with_predicted_task(e, classifier.predict(e.instruction, min_confidence=min_confidence),
                                observation_mode=observation_mode, ground_complex=ground_complex)
            for e in examples]
