This task involves imposing constraints on generated tokens, loosely based off of [Yao et al., 2023](https://arxiv.org/abs/2307.08689).
For the package-level view of soft latent constraint encoding, hard DFA decoding,
and how they fit together, see `../../domiknows/generation/README.md`.

For example, a constraint could be: if the generated sentence contains the token " The", then there must be at least 16 tokens generated total. We specify this as:
```python
ifL(
    existsAL(get_token_concept(' The')("x")),
    atMostAL(
        notL(get_token_concept('<|endoftext|>')("y")), # only consider non-EOS tokens
        16
    )
)
```

For efficiency, this example currently uses models trained on the [TinyStories dataset](https://arxiv.org/pdf/2305.07759), which have a limited vocabulary. We also restrict the vocabulary to the top-k most used tokens in the corpus.

Run `python build_vocab.py` to find and output the most used tokens in the corpus.

Then, run `python program.py --vocab_file vocab_val.pkl` where the `vocab_file` argument specifies the path to the output from `build_vocab.py`. Run `python program.py --help` for descriptions of all the arguments.

Collie currently demonstrates DomiKnowS soft/loss constraints and DFA-style hard
decoding. Latent constraint encoding is the next soft internal-bias concept: it
would reshape hidden-state or transition probabilities before decoding, while
the DFA layer still provides exact constraint enforcement when needed.

## Latent Constraint Example

`graph.py` also includes one simple latent-only generation constraint:

```python
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
```

This rule is not a DFA guarantee. It is a soft auxiliary loss over latent/token
probabilities: when `" The"` is likely at position `t`, the model is encouraged
to put probability mass on `" slide"` within the next two generated positions.

Run `uv run python Tasks/collie/latent_example.py` from the repository root to
see the generation module discover the latent spec from the graph and compute a
toy loss.
