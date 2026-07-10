# HMM Distillation Decoder Baseline

`hmm_distillation_decoder_baseline.py` is an offline baseline for comparing
constraint-aware HuggingFace generation decoders. It uses the task's tiny
HuggingFace-shaped `MockCausalLM`, distills its next-token behavior into a
compact `HMMGenerationHead`, then evaluates how well different decoders satisfy
the DFA constraints.

The demo is intentionally small and deterministic enough for regression tests.
It does not download a real model.

## What It Trains

The student model is a compact HMM generation head:

```text
MockCausalLM token logits
    -> projected compact-label distribution
    -> KL distillation target for HMMGenerationHead.next_label_logits(...)
```

Training prefixes are short compact-label sequences from the task vocabulary,
including partial valid paths, forbidden-token examples, and EOS examples. The
distilled HMM is then used as the compact scorer in `HybridController`.

## Constraints

The demo builds an equivalent DFA directly from primitive DFA helpers, avoiding
the heavier graph/program import path while enforcing the same constraints used
by the main HuggingFace generation example:

- after `<eos>`, only `<eos>` may follow;
- at most three non-EOS tokens;
- `" cat"` is required;
- `" dog"` is forbidden;
- at least one of `" The"` or `" mat"` must appear.

The mock LM prefers the forbidden `" dog"` path, so raw greedy decoding should
fail while DFA-guided decoders should avoid it.

## Decoder Baselines

The report compares:

- `raw_lm_greedy`: unconstrained greedy decoding from the small mock LM.
- `dfa_greedy`: hard DFA greedy decoding over mock-LM logits.
- `dfa_beam`: hard DFA beam search over mock-LM logits.
- `dfa_sample`: hard DFA sampling over mock-LM logits.
- `product_compact_learner_dfa`: generic compact-head plus DFA product decoding
  through `generate_verify_rerank(...)`.
- `product_hmm_dfa`: Ctrl-G-compatible HMM+DFA decoding through the direct
  `HybridController.decode_hmm_dfa(...)` API.

The strict HMM+DFA path tracks:

```text
(HMM belief h_t, DFA state q_t, generated prefix)
```

and uses beam search by default with Ctrl-G-compatible lookahead scoring:

```text
base_weight * base_model_label_score
+ lookahead_weight * log P_HMM(DFA success | h_{t+1}, q_{t+1})
```

The default `hmm_dfa_base="auto"` uses projected backend label logits when they
are available and falls back to HMM next-label logits otherwise. The log-linear
product-style objective is available as `hmm_dfa_log_linear` or
`hmm_dfa_objective="log_linear_blend"`:

```text
hf_weight * HF_label_logit
+ hmm_weight * log P_HMM(label | h_t)
+ lookahead_weight * log P_HMM(DFA success | h_{t+1}, q_{t+1})
```

## Run

From the repository root:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/hmm_distillation_decoder_baseline.py --steps 40
```

For a faster smoke run:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/hmm_distillation_decoder_baseline.py --steps 1 --prompts Once --max-new-tokens 4
```

Useful arguments:

- `--steps`: number of HMM distillation optimization steps.
- `--lr`: Adam learning rate for the HMM head.
- `--state-count`: number of HMM hidden states.
- `--max-new-tokens`: decode cap used by all baselines.
- `--prompts`: one or more prompt strings to evaluate.

## Output

The CLI prints:

- first and last distillation loss;
- an accuracy table with accepted-sequence accuracy, required-cat accuracy, and
  forbidden-dog avoidance;
- one example output per decoder mode;
- the compact vocabulary.

In the offline mock setup, the expected qualitative result is:

- `raw_lm_greedy` follows the mock LM into the forbidden `" dog"` branch;
- DFA-only decoders satisfy the hard graph constraints;
- `product_hmm_dfa` uses the direct HMM+DFA product decoder and should also
  return accepted, dog-free sequences.

## Programmatic Use

```python
from Tasks.hf_generation.hmm_distillation_decoder_baseline import run_hmm_distillation_decoder_baseline

summary = run_hmm_distillation_decoder_baseline(
    prompts=("Once", "Story"),
    steps=2,
    max_new_tokens=4,
)

print(summary["accuracy"]["product_hmm_dfa"])
```

The returned `summary` contains:

- `losses`: distillation loss values;
- `hmm_head`: trained `HMMGenerationHead`;
- `reports`: per-decoder `DecoderReport` rows;
- `accuracy`: aggregate metrics per decoder;
- `vocabulary`: compact-label vocabulary.

## Regression Test

The focused regression test is:

```powershell
.\.venv\Scripts\python.exe -m pytest test_regr\generation\test_hf_generation_hmm_distillation_decoder_baseline_task.py -q
```

That test also spies on `HybridController.decode_hmm_dfa` to ensure the
`product_hmm_dfa` baseline uses the direct API rather than the rerank wrapper.
