from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import ifL, atMostL, atMostAL, atLeastAL, eqL, notL, existsAL

from transformers import PreTrainedTokenizer

from tokens import TokenMap


def build_graph(lm: TokenMap, tokenizer: PreTrainedTokenizer, vocab: list[str]):
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('main') as graph:
        text = Concept(name='text')
        token = Concept(name='token')

        contains, = text.contains(token)

        # relation for whether `first_token` is before `second_token` in the sequence
        is_before_rel = Concept(name='is_before_rel')
        first_token, second_token = is_before_rel.has_a(arg1=token, arg2=token)

        vocab_all = vocab + ['_other']

        # all tokens predicted by the model
        generated_token = token(
            name="generated_token",
            ConceptClass=EnumConcept,
            values=[str(v) for v in range(len(vocab))]
        )

        def get_token_concept(token: str):
            '''
            Convert string to EnumConcept
            '''
            # encoded = tokenizer.encode(token)
            # assert len(encoded) == 1
            # return getattr(generated_token, str(lm.label_map[encoded[0]]))
            assert token in vocab_all, f"token {token} not in vocab"
            return getattr(generated_token, str(vocab_all.index(token)))

        # ensures that a valid sequence is generated: no non-EOS tokens can follow an EOS token
        # this also ensures that we can check values in our sequence by only looking at non-EOS tokens
        ifL(
            # for each pair of tokens `first_token`, `second_token`
            # such that `first_token` is before `second_token` in the sequence
            is_before_rel('before'),

            # if `first_token` is EOS, then `second_token` must be EOS
            ifL(
                get_token_concept('<|endoftext|>')("x", path=("before", first_token)),
                get_token_concept('<|endoftext|>')("y", path=("before", second_token))
            )
        )

        # at most 4 tokens are generated
        atMostAL(
            notL(get_token_concept('<|endoftext|>')("x")),
            4
        )

        # at most 32 tokens are generated
        # atMostAL(
        #     notL(get_token_concept('<|endoftext|>')("x")),
        #     32
        # )

        # at least one of the " The" token is generated
        # existsAL(get_token_concept(' The')("x"))
        # atLeastAL(
        #     get_token_concept(' The')("x"),
        #     1
        # )

        # at least one of the " slide" token is generated
        # existsAL(get_token_concept(' slide')("x"))
        # atLeastAL(
        #     get_token_concept(' slide')("x"),
        #     1
        # )

        # if there is a token " The", then there are at most 16 tokens generated total
        # ifL(
        #     existsAL(get_token_concept(' The')("x")),
        #     atMostAL(
        #         notL(get_token_concept('<|endoftext|>')("y")),
        #         16
        #     )
        # )

    return graph, (text, token, contains, generated_token, is_before_rel, first_token, second_token)
