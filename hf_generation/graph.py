"""DomiKnowS generation graph for the HuggingFace constrained decoding demo."""
from __future__ import annotations

from domiknows.generation import GenerationEncoder
from domiknows.graph.logicalConstrain import andL, atLeastAL, atMostAL, ifL, notL, orL


EOS_TOKEN = "<eos>"
VOCAB = [EOS_TOKEN, " The", " cat", " mat", " dog"]


def apply_graph_constraints(bundle) -> None:
    """Write raw DomiKnowS constraints that generation can discover."""
    ctx = bundle.context
    eos_token = bundle.vocabulary.eos_token

    # EOS closure: once EOS appears, every later token must also be EOS.
    ifL(
        ctx.is_before_rel("before"),
        ifL(
            ctx.token_value(eos_token, "x", path=("before", ctx.first_token)),
            ctx.token_value(eos_token, "y", path=("before", ctx.second_token)),
        ),
    )

    # Keep generated outputs short for a readable demo.
    atMostAL(notL(ctx.token_value(eos_token, "x")), 3)

    # The output must contain " cat" at least once.
    atLeastAL(ctx.token_value(" cat", "x"), 1)

    # Forbid " dog", even though the mock LM strongly prefers it.
    atMostAL(ctx.token_value(" dog", "x"), 0)

    # Boolean graph discovery example:
    # require (" The" OR " mat") while also staying inside the short length cap.
    andL(
        orL(
            atLeastAL(ctx.token_value(" The", "x"), 1),
            atLeastAL(ctx.token_value(" mat", "y"), 1),
        ),
        atMostAL(notL(ctx.token_value(eos_token, "z")), 3),
    )


def build_generation_graph(tokenizer, vocab=None, eos_token: str = EOS_TOKEN):
    """Build the demo graph and return ``(graph, bundle)``."""
    vocab = list(vocab or VOCAB)
    if eos_token not in vocab:
        vocab = [eos_token] + [token for token in vocab if token != EOS_TOKEN]
    encoder = GenerationEncoder(
        vocab=vocab,
        eos_token=eos_token,
        tokenizer=tokenizer,
        graph_name="hf_generation",
    )
    graph, bundle = encoder.build_graph()
    with graph:
        apply_graph_constraints(bundle)
    return graph, bundle
