"""Small Collie example for graph-marked latent generation constraints.

The graph in ``graph.py`` marks one DomiKnowS logical constraint as latent-only:
if the generated token is ``" The"``, prefer ``" slide"`` soon after it. This
file shows how the generation module discovers that metadata and turns it into
a differentiable loss over token-label probabilities.
"""

from __future__ import annotations

import torch

from domiknows.generation import discover_generation_enforcement

from graph import EOS_TOKEN, build_generation_bundle


EXAMPLE_VOCAB = [EOS_TOKEN, " The", " slide"]


class ExampleTokenizer:
    def encode(self, token):
        return {token_value: [index] for index, token_value in enumerate(EXAMPLE_VOCAB)}[token]


def build_example_probabilities(bundle):
    """Return toy probabilities shaped [seq_len, label_count]."""

    eos = bundle.vocabulary.label_for_token(EOS_TOKEN)
    the = bundle.vocabulary.label_for_token(" The")
    slide = bundle.vocabulary.label_for_token(" slide")

    probs = torch.zeros((4, bundle.vocabulary.label_count), dtype=torch.float32)
    probs[:, eos] = 1.0
    probs[0, eos] = 0.2
    probs[0, the] = 0.8
    probs[1, eos] = 0.4
    probs[1, slide] = 0.6
    return probs


def build_latent_example():
    graph, bundle = build_generation_bundle(ExampleTokenizer(), EXAMPLE_VOCAB)
    enforcement = discover_generation_enforcement(graph, bundle)
    probs = build_example_probabilities(bundle)
    loss = enforcement.latent_loss(probs)
    return graph, bundle, enforcement, probs, loss


def main():
    _, _, enforcement, _, loss = build_latent_example()
    print(f"latent specs: {enforcement.latent_specs}")
    print(f"latent loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
