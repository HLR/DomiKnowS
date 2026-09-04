from domiknows.generation import encode_label_prefix_prompt


class _ChatTokenizer:
    chat_template = "fake-template"
    eos_token_id = 99

    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, dict(kwargs)))
        if kwargs["tokenize"]:
            return [10, 11, 12]
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": {" find": [20], " cup_1": [21, 22]}.get(text, [30])}


def test_chat_label_prefix_encoding_uses_one_generation_marker_and_boundaries():
    tokenizer = _ChatTokenizer()
    encoded = encode_label_prefix_prompt(
        tokenizer,
        "Instruction: find a cup",
        ("find", "cup_1"),
        enable_thinking=False,
    )

    assert encoded.used_chat_template
    assert encoded.input_ids == (10, 11, 12, 20, 21, 22)
    assert encoded.boundary_positions == (2, 3, 5)
    assert encoded.rendered_text == "<user>Instruction: find a cup<assistant> find cup_1"
    assert len(tokenizer.template_calls) == 2
    assert all(call[1]["add_generation_prompt"] for call in tokenizer.template_calls)
    assert all(call[1]["enable_thinking"] is False for call in tokenizer.template_calls)


def test_chat_label_prefix_encoding_left_truncates_ids_and_boundaries_together():
    encoded = encode_label_prefix_prompt(
        _ChatTokenizer(),
        "task",
        ("find", "cup_1"),
        max_length=4,
    )
    assert encoded.input_ids == (12, 20, 21, 22)
    assert encoded.boundary_positions == (0, 1, 3)

