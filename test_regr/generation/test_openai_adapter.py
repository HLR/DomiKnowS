from domiknows.generation import TokenVocabulary
from domiknows.generation.adapters import OpenAIResponsesAdapter


class FakeResponses:
    def create(self, **kwargs):
        class Response:
            output_text = "AB"

        self.request = kwargs
        return Response()


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


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
