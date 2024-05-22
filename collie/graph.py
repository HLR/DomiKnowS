from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import ifL, atMostL, atMostAL, atLeastAL, eqL

from transformers import PreTrainedTokenizer

from tokens import TokenMap


def build_graph(lm: TokenMap, tokenizer: PreTrainedTokenizer):
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('main') as graph:
        text = Concept(name='text')
        token = Concept(name='token')

        contains, = text.contains(token)

        # all tokens predicted by the model
        generated_token = token(
            name="generated_token",
            ConceptClass=EnumConcept,
            values=[str(v) for v in range(len(lm))]
        )

        def get_token_concept(token: str):
            return getattr(generated_token, str(lm.label_map[tokenizer.encode(token)[0]]))

        # at most 16 tokens are generated
        atMostAL(
            generated_token("x", eqL(token, "is_before_eos", {True})),
            16,
            p=80
        )

        # at most 32 tokens are generated
        # atMostAL(
        #     generated_token("x", eqL(token, "is_before_eos", {True})),
        #     32
        # )

        # # at least one of the " The" token is generated
        # atLeastAL(
        #     get_token_concept(' The')("x", eqL(token, "is_before_eos", {True})),
        #     1
        # )

        # # at least one of the " slide" token is generated
        # atLeastAL(
        #     get_token_concept(' slide')("x", eqL(token, "is_before_eos", {True})),
        #     1
        # )

        # # if there is a token " The", then there are at most 16 tokens generated total
        # ifL(
        #     atLeastAL(
        #         get_token_concept(' The')("x", eqL(token, "is_before_eos", {True})),
        #         1
        #     ),
        #     atMostAL(
        #         generated_token("y", eqL(token, "is_before_eos", {True})),
        #         16
        #     )
        # )

    return graph, (text, token, contains, generated_token)
