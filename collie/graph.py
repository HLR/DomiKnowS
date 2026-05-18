from domiknows.generation import LatentWindowSpec, mark_for_latent
from domiknows.graph import Concept, EnumConcept, Graph, Relation
from domiknows.graph.logicalConstrain import atLeastAL, atMostAL, existsAL, ifL, notL

from tokens import TokenMap


EOS_TOKEN = "<|endoftext|>"


def build_graph(lm: TokenMap, tokenizer, vocab: list[str]):
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("main") as graph:
        text = Concept(name="text")
        token = Concept(name="token")

        contains, = text.contains(token)

        # Relation for whether `first_token` is before `second_token` in the sequence.
        is_before_rel = Concept(name="is_before_rel")
        first_token, second_token = is_before_rel.has_a(arg1=token, arg2=token)

        vocab_all = vocab + ["_other"]

        # All tokens predicted by the model, including the reserved `_other` compact label.
        generated_token = token(
            name="generated_token",
            ConceptClass=EnumConcept,
            values=[str(v) for v in range(len(vocab_all))],
        )

        def get_token_concept(token: str):
            """
            Convert string to EnumConcept.
            """
            # encoded = tokenizer.encode(token)
            # assert len(encoded) == 1
            # return getattr(generated_token, str(lm.label_map[encoded[0]]))
            assert token in vocab_all, f"token {token} not in vocab"
            return getattr(generated_token, str(vocab_all.index(token)))

        # Ensures that a valid sequence is generated: no non-EOS tokens can follow an EOS token.
        # This also ensures that we can check values in our sequence by only looking at non-EOS tokens.
        ifL(
            # For each pair of tokens `first_token`, `second_token`
            # such that `first_token` is before `second_token` in the sequence.
            is_before_rel("before"),

            # If `first_token` is EOS, then `second_token` must be EOS.
            ifL(
                get_token_concept(EOS_TOKEN)("x", path=("before", first_token)),
                get_token_concept(EOS_TOKEN)("y", path=("before", second_token)),
            ),
        )

        # At most 4 non-EOS tokens are generated.
        atMostAL(
            notL(get_token_concept(EOS_TOKEN)("x")),
            4,
        )

        # At least one of the " The" token is generated.
        atLeastAL(
            get_token_concept(" The")("x"),
            1,
        )

        # At least one of the " slide" token is generated.
        atLeastAL(
            get_token_concept(" slide")("x"),
            1,
        )

        # If there is a token " The", then there are at most 16 non-EOS tokens generated total.
        ifL(
            existsAL(get_token_concept(" The")("x")),
            atMostAL(
                notL(get_token_concept(EOS_TOKEN)("y")),
                16,
            ),
        )

        # Latent-only soft preference: if " The" appears, prefer " slide" soon after it.
        mark_for_latent(
            ifL(
                existsAL(get_token_concept(" The")("x")),
                existsAL(get_token_concept(" slide")("y")),
            ),
            LatentWindowSpec(
                if_label=vocab_all.index(" The"),
                formula=vocab_all.index(" slide"),
                window=2,
                weight=0.5,
            ),
        )

    return graph, (text, token, contains, generated_token, is_before_rel, first_token, second_token)
