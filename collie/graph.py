from domiknows.generation import GenerationEncoder, LatentWindowSpec, mark_for_latent
from domiknows.graph.logicalConstrain import atLeastAL, atMostAL, existsAL, ifL, notL

from tokens import TokenMap


EOS_TOKEN = "<|endoftext|>"


def build_default_constraints():
    return ()


def apply_graph_written_constraints(bundle):
    """Task-local raw DomiKnowS constraints discovered by generation enforcement."""

    is_before_rel = bundle.is_before_rel
    first_token = bundle.first_token
    second_token = bundle.second_token

    def get_token_concept(token):
        return lambda variable, path=None: bundle.context.token_value(token, variable, path=path)

    # ensures that a valid sequence is generated: no non-EOS tokens can follow an EOS token
    # this also ensures that we can check values in our sequence by only looking at non-EOS tokens
    ifL(
        # for each pair of tokens `first_token`, `second_token`
        # such that `first_token` is before `second_token` in the sequence
        is_before_rel("before"),

        # if `first_token` is EOS, then `second_token` must be EOS
        ifL(
            get_token_concept(EOS_TOKEN)("x", path=("before", first_token)),
            get_token_concept(EOS_TOKEN)("y", path=("before", second_token)),
        ),
    )

    # at most 4 tokens are generated
    atMostAL(
        notL(get_token_concept(EOS_TOKEN)("x")),
        4,
    )

    # at most 32 tokens are generated
    # atMostAL(
    #     notL(get_token_concept(EOS_TOKEN)("x")),
    #     32,
    # )

    # at least one of the " The" token is generated
    # existsAL(get_token_concept(" The")("x"))
    atLeastAL(
        get_token_concept(" The")("x"),
        1,
    )

    # at least one of the " slide" token is generated
    # existsAL(get_token_concept(" slide")("x"))
    atLeastAL(
        get_token_concept(" slide")("x"),
        1,
    )

    # if there is a token " The", then there are at most 16 tokens generated total
    ifL(
        existsAL(get_token_concept(" The")("x")),
        atMostAL(
            notL(get_token_concept(EOS_TOKEN)("y")),
            16,
        ),
    )

    # latent-only soft preference: if " The" appears, prefer " slide" soon after it
    mark_for_latent(
        ifL(
            existsAL(get_token_concept(" The")("x")),
            existsAL(get_token_concept(" slide")("y")),
        ),
        LatentWindowSpec(
            if_label=bundle.vocabulary.label_for_token(" The"),
            formula=bundle.vocabulary.label_for_token(" slide"),
            window=2,
            weight=0.5,
        ),
    )


def build_generation_bundle(tokenizer, vocab: list[str]):
    encoder = GenerationEncoder(
        vocab=vocab,
        eos_token=EOS_TOKEN,
        tokenizer=tokenizer,
        graph_name="main",
    )
    graph, bundle = encoder.build_graph(build_default_constraints())
    with graph:
        apply_graph_written_constraints(bundle)
    return graph, bundle


def build_graph(lm: TokenMap, tokenizer, vocab: list[str]):
    graph, bundle = build_generation_bundle(tokenizer, vocab)
    return graph, (
        bundle.text,
        bundle.token,
        bundle.contains,
        bundle.generated_token,
        bundle.is_before_rel,
        bundle.first_token,
        bundle.second_token,
    )
