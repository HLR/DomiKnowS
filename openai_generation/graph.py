"""DomiKnowS graph for the OpenAI-compatible generation demo."""
from __future__ import annotations

from domiknows.generation import GenerationEncoder
from domiknows.graph.logicalConstrain import andL, atLeastAL, atMostAL, ifL, notL, orL


EOS_TOKEN = "<eos>"
VOCAB = [EOS_TOKEN, " The", " cat", " mat", " dog"]


def apply_graph_constraints(bundle) -> None:
    """Write raw DomiKnowS constraints discoverable by generation tooling."""
    ctx = bundle.context
    eos_token = bundle.vocabulary.eos_token

    ifL(
        ctx.is_before_rel("before"),
        ifL(
            ctx.token_value(eos_token, "x", path=("before", ctx.first_token)),
            ctx.token_value(eos_token, "y", path=("before", ctx.second_token)),
        ),
    )

    atMostAL(notL(ctx.token_value(eos_token, "x")), 3)
    atLeastAL(ctx.token_value(" cat", "x"), 1)
    atMostAL(ctx.token_value(" dog", "x"), 0)

    andL(
        orL(
            atLeastAL(ctx.token_value(" The", "x"), 1),
            atLeastAL(ctx.token_value(" mat", "y"), 1),
        ),
        atMostAL(notL(ctx.token_value(eos_token, "z")), 3),
    )


def build_generation_graph(tokenizer, vocab=None, eos_token: str = EOS_TOKEN):
    """Build the demo graph and return ``(graph, bundle)``."""
    encoder = GenerationEncoder(
        vocab=list(vocab or VOCAB),
        eos_token=eos_token,
        tokenizer=tokenizer,
        graph_name="openai_generation",
    )
    graph, bundle = encoder.build_graph()
    with graph:
        apply_graph_constraints(bundle)
    return graph, bundle
