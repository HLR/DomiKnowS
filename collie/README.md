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

## Learning vs Enforcement Paths

The same graph constraints are used in two different ways:

```text
graph logical constraints
    |-> PrimalDualProgram / cmodel: soft DomiKnowS constraint loss
    |-> DFA compiler: hard token masking when --constrained_decoding is enabled
```

The DFA is not a differentiable loss in Collie. During PMD training,
`program.cmodel(...)` evaluates the DomiKnowS logical constraints over the
populated token DataNodes. During constrained generation, the supported graph
constraints are compiled to a DFA and used to mask invalid next-token logits.

Collie's current `generate` mode returns detached generated logits, so the
constraint loss can be measured but does not cleanly backpropagate through the
autoregressive generation decisions. Use this task as a bridge/diagnostic
example; see `Tasks/hf_generation/learn_demo.py` for a smaller learning path
where a compact generation head is trained with supervised plus PMD constraint
loss and then decoded with DFA enforcement.

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

## Graph-HMM PMD Learner Path

Collie can also use `domiknows.generation.graph_hmm` as the trainable
`ModuleLearner` attached to `token[generated_token]`:

```bash
uv run --project Tasks/collie python Tasks/collie/program.py --vocab_file data/vocab_val.pkl --graph_hmm_learner hmm
uv run --project Tasks/collie python Tasks/collie/program.py --vocab_file data/vocab_val.pkl --graph_hmm_learner spectral
```

With `--graph_hmm_learner hmm`, Collie uses `GraphHMMGenerationHead`: a
trainable compact-label HMM whose transition and emission probabilities are
projected through graph-aware masks before PMD sees them.

With `--graph_hmm_learner spectral`, Collie uses
`GraphSpectralGenerationHead`: a trainable signed WFA-style head whose scores
are exposed as compact-label logits.

Both heads populate DomiKnowS token DataNodes, so `PrimalDualProgram.cmodel(...)`
receives ordinary `generated_token` probabilities and computes the same graph
constraint loss. DFA decoding remains the hard enforcement path; graph-HMM
heads provide trainable graph-aware probabilities for PMD.

By default the graph-HMM heads learn from compact labels derived from the target
sequence. To make the head learn from real TinyStories-generated tokens instead,
use the hybrid source:

```bash
uv run --project Tasks/collie python Tasks/collie/program.py --vocab_file data/vocab_val.pkl --graph_hmm_learner hmm --graph_hmm_source generated
```

In that mode, a frozen `TinyModel`/TinyStories backbone generates raw tokenizer
ids, Collie maps them into the compact `TokenVocabulary` labels, and the
graph-HMM head is trained with an imitation loss on those generated labels while
PMD still receives graph-HMM probabilities for the DomiKnowS constraint loss.
