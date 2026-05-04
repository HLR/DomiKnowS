from domiknows.generation import GenerationEncoder, max_non_eos, no_token_after_eos


class FakeTokenizer:
    def encode(self, token):
        return {"<eos>": [0], "A": [1]}[token]


def test_generation_encoder_builds_domiknows_graph():
    encoder = GenerationEncoder(
        ["<eos>", "A"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
    )
    graph, bundle = encoder.build_graph([no_token_after_eos(), max_non_eos(1)])

    assert graph is not None
    assert bundle.vocabulary.label_count == 3
    assert bundle.generated_token.name == "generated_token"
    assert len(graph.logicalConstrains) >= 2
