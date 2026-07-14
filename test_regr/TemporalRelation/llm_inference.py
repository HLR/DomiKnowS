"""Small-LLM inference helpers for the TemporalRelation adapter.

The adapter keeps the DomiKnowS graph generic. This module provides the learned
front-end that predicts query-event marker predicates and temporal-relation
labels from multiple-choice prompts, similar in spirit to CLEVR concept
prediction from compositional prompts.
"""

import re
from dataclasses import dataclass

from .execution import create_candidate_event_pairs, mark_text_for_pair
from .graph import TEMPORAL_LABELS, unpack_pair


@dataclass
class ChoiceExample:
    task: str
    prompt: str
    choices: list
    answer: str | None = None
    metadata: dict | None = None


class ChoiceBackend:
    """Minimal backend interface used by tests and real small-LLM inference."""

    def choose(self, prompt, choices):
        raise NotImplementedError


class SmallCausalLMChoiceBackend(ChoiceBackend):
    """Multiple-choice wrapper around a small Hugging Face causal LM.

    The model is asked to output exactly one candidate answer. We then parse the
    generated text back into one of the provided choices. This intentionally does
    not introduce new DomiKnowS APIs; it only supplies predicate labels/logits at
    the adapter boundary.
    """

    def __init__(self, model_path="Qwen/Qwen2.5-0.5B-Instruct", device="cpu", max_new_tokens=16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model.to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def choose(self, prompt, choices):
        full_prompt = format_choice_prompt(prompt, choices)
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            full_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": full_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return parse_choice(text, choices)


class StaticChoiceBackend(ChoiceBackend):
    """Test backend that mimics an LLM returning free-form answer text."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def choose(self, prompt, choices):
        if not self.answers:
            raise RuntimeError("StaticChoiceBackend has no remaining answers")
        raw_answer = self.answers.pop(0)
        self.prompts.append((prompt, tuple(choices), raw_answer))
        return parse_choice(raw_answer, choices)


def format_choice_prompt(prompt, choices):
    lines = [prompt.strip(), "", "Candidate answers:"]
    for index, choice in enumerate(choices):
        lines.append(f"{choice_letter(index)}. {choice}")
    lines.append("")
    lines.append("Return exactly one candidate answer. You may return the letter or the answer text.")
    return "\n".join(lines)


def parse_choice(text, choices):
    """Parse LLM text into one of the allowed choices."""
    if not choices:
        raise ValueError("Expected at least one choice")
    raw = str(text).strip()
    normalized = _normalize(raw)

    for choice in choices:
        if normalized == _normalize(choice):
            return choice

    letter_match = re.search(r"\b([A-Z])\b", raw.upper())
    if letter_match:
        index = ord(letter_match.group(1)) - ord("A")
        if 0 <= index < len(choices):
            return choices[index]

    number_match = re.search(r"\b(\d+)\b", raw)
    if number_match:
        index = int(number_match.group(1)) - 1
        if 0 <= index < len(choices):
            return choices[index]

    matches = []
    for choice in choices:
        choice_norm = _normalize(choice)
        if choice_norm in normalized or normalized in choice_norm:
            matches.append(choice)
    if len(matches) == 1:
        return matches[0]

    # Event choices use "event_id: text (...)"; accept a returned event id.
    id_matches = [
        choice
        for choice in choices
        if _normalize(str(choice).split(":", 1)[0]) == normalized
        or _normalize(str(choice).split(":", 1)[0]) in normalized
    ]
    if len(id_matches) == 1:
        return id_matches[0]

    raise ValueError(f"Could not parse LLM answer {text!r}; choices={choices!r}")


def build_query_event_choice_examples(instance):
    events = instance.get("events", [])
    choices = [_event_choice(event) for event in events]
    event_choice_by_id = {_event_id(event): _event_choice(event) for event in events}
    query_pair = instance.get("query_pair") or (instance.get("event_pairs") or [{}])[0]
    e1, e2, _label = unpack_pair(query_pair)
    return [
        ChoiceExample(
            task="query_event1",
            prompt=_query_event_prompt(instance, role="first"),
            choices=choices,
            answer=event_choice_by_id.get(e1),
            metadata={"role": "query_event1", "event_id": e1},
        ),
        ChoiceExample(
            task="query_event2",
            prompt=_query_event_prompt(instance, role="second"),
            choices=choices,
            answer=event_choice_by_id.get(e2),
            metadata={"role": "query_event2", "event_id": e2},
        ),
    ]


def predict_query_events_with_llm(instance, backend):
    examples = build_query_event_choice_examples(instance)
    predictions = {}
    for example in examples:
        selected = backend.choose(example.prompt, example.choices)
        event_id = _choice_event_id(selected)
        predictions[example.task] = event_id
    return [
        {
            "event_id": _event_id(event),
            "query_event1": _event_id(event) == predictions.get("query_event1"),
            "query_event2": _event_id(event) == predictions.get("query_event2"),
        }
        for event in instance.get("events", [])
    ]


def build_temporal_relation_choice_examples(instance):
    examples = []
    for pair in create_candidate_event_pairs(instance):
        e1, e2, label = unpack_pair(pair)
        examples.append(
            ChoiceExample(
                task="temporal_relation",
                prompt=_temporal_relation_prompt(instance, e1, e2),
                choices=list(TEMPORAL_LABELS),
                answer=label,
                metadata={"e1": e1, "e2": e2},
            )
        )
    return examples


def predict_temporal_relations_with_llm(instance, backend):
    predictions = []
    for example in build_temporal_relation_choice_examples(instance):
        label = backend.choose(example.prompt, example.choices)
        predictions.append({**example.metadata, "label": label})
    return predictions


def run_llm_inference(instance, backend):
    """Predict both learned TemporalRelation front-end outputs for an instance."""
    return {
        "query_event_groundings": predict_query_events_with_llm(instance, backend),
        "event_pair_predictions": predict_temporal_relations_with_llm(instance, backend),
    }


def choice_letter(index):
    return chr(ord("A") + index)


def _query_event_prompt(instance, role):
    text = instance.get("text") or instance.get("doc_id") or ""
    return chr(10).join(
        [
            f"Select the event that should be used as the {role} event in the temporal-relation query.",
            "",
            f"Text: {text}",
        ]
    )


def _temporal_relation_prompt(instance, e1, e2):
    marked = mark_text_for_pair(instance, e1, e2)
    return chr(10).join(
        [
            "Classify the temporal relation of E1 relative to E2.",
            "Answer Before only when E1 happened earlier than E2.",
            "Answer After only when E1 happened later than E2.",
            "Answer Equal only when E1 and E2 are simultaneous.",
            "Answer Vague when the order is unclear or not stated.",
            "Example: Text='[E1]packed[/E1] before [E2]left[/E2]' => Before.",
            "Example: Text='[E1]left[/E1] after [E2]packed[/E2]' => After.",
            "",
            f"Text: {marked}",
        ]
    )


def _event_choice(event):
    event_id = _event_id(event)
    if isinstance(event, dict):
        text = event.get("text") or event_id
        token_id = event.get("token_id") or event.get("token") or event_id
    else:
        text = event_id
        token_id = event_id
    return f"{event_id}: {text} (token={token_id})"


def _choice_event_id(choice):
    return str(choice).split(":", 1)[0].strip()


def _event_id(event):
    return event.get("id") if isinstance(event, dict) else event


def _normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
