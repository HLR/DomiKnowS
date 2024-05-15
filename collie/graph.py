from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import ifL, atMostAL, atLeastAL

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

        generated_token = token(
            name="generated_token",
            ConceptClass=EnumConcept,
            values=[str(v) for v in range(len(lm))]
        )

        def get_token_concept(token: str) -> EnumConcept:
            return generated_token.attributes[lm.label_map[tokenizer.encode(token)[0]]]

        # at most three tokens are generated
        atMostAL(token, 3)

        # at most 20 tokens are generated
        atMostAL(token, 20)

        # at least one of the " The" token is generated
        atLeastAL(get_token_concept(' The'), 1)

        # at least one of the " girl" token is generated
        atLeastAL(get_token_concept(' girl'), 1)

        # if there is a token 'The', then there are at most three tokens generated total
        ifL(
            atLeastAL(get_token_concept(' The'), 1),
            atMostAL(token, 3)
        )

    return graph, (text, token, contains, generated_token)
