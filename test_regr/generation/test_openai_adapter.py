import pytest

from domiknows.generation import GenerationResult, TokenVocabulary
from domiknows.generation.learners import DFA
from domiknows.generation.adapters import OpenAIResponsesAdapter


class FakeResponses:
    def __init__(self, text="AB"):
        self.text = text
        self.request = None

    def create(self, **kwargs):
        text = self.text

        class Response:
            output_text = text

        self.request = kwargs
        return Response()


class FakeClient:
    def __init__(self, text="AB"):
        self.responses = FakeResponses(text)


class FakeTokenizer:
    def encode(self, text):
        mapping = {"<eos>": 0, "A": 1, "B": 2}
        if text in mapping:
            return [mapping[text]]
        return [mapping[ch] for ch in text]


def test_openai_adapter_generates_and_encodes_with_mock_client():
    tokenizer = FakeTokenizer()
    client = FakeClient()
    adapter = OpenAIResponsesAdapter(client=client, model="test-model", tokenizer=tokenizer)
    result = adapter.generate("prompt", max_output_tokens=5)
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)

    assert result.text == "AB"
    assert client.responses.request["model"] == "test-model"
    assert adapter.encode_output(result.text, vocab) == [1, 2]


def test_openai_adapter_verify_result_accepts_output():
    tokenizer = FakeTokenizer()
    adapter = OpenAIResponsesAdapter(client=FakeClient(), model="test-model", tokenizer=tokenizer)
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = _exact_ab_dfa()
    raw = object()

    verified = adapter.verify_result(GenerationResult(text="AB", raw=raw), vocab, dfa)

    assert verified.text == "AB"
    assert verified.token_ids == [1, 2]
    assert verified.labels == [1, 2]
    assert verified.accepted is True
    assert verified.raw is raw
    assert verified.rejection is None


def test_openai_adapter_verify_result_rejects_and_explains_output():
    tokenizer = FakeTokenizer()
    adapter = OpenAIResponsesAdapter(client=FakeClient(), model="test-model", tokenizer=tokenizer)
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = _exact_ab_dfa()

    verified = adapter.verify_result(GenerationResult(text="A"), vocab, dfa, explain=True)

    assert verified.labels == [1]
    assert verified.accepted is False
    assert verified.rejection


def test_openai_adapter_generate_and_verify_forwards_request_kwargs():
    tokenizer = FakeTokenizer()
    client = FakeClient("AB")
    adapter = OpenAIResponsesAdapter(client=client, model="test-model", tokenizer=tokenizer)
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=tokenizer)
    dfa = _exact_ab_dfa()

    verified = adapter.generate_and_verify(
        "prompt",
        vocab,
        dfa,
        max_output_tokens=7,
        temperature=0.2,
    )

    assert verified.accepted is True
    assert verified.labels == [1, 2]
    assert client.responses.request["max_output_tokens"] == 7
    assert client.responses.request["temperature"] == 0.2


def test_openai_adapter_verify_result_requires_tokenizer():
    adapter = OpenAIResponsesAdapter(client=FakeClient(), model="test-model", tokenizer=None)
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    dfa = _exact_ab_dfa()

    with pytest.raises(ValueError, match="tokenizer is required"):
        adapter.verify_result(GenerationResult(text="AB"), vocab, dfa)


def _exact_ab_dfa():
    return DFA(
        states=frozenset({"start", "saw_a", "done"}),
        alphabet=frozenset({0, 1, 2}),
        transitions={
            ("start", 1): "saw_a",
            ("saw_a", 2): "done",
        },
        start_state="start",
        accepting_states=frozenset({"done"}),
    )
