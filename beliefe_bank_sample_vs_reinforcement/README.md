# BeliefBank Sample Loss vs ReinforcementProgram

This task compares BeliefBank constraint learning with:

- `SampleLossProgram`, using sampled DomiKnowS graph-constraint loss.
- `ReinforcementProgram`, using a generated-output reward over sampled fact
  assignments.

It reuses the BeliefBank graph structure and RoBERTa utilities from
`../beliefe_bank`, but keeps the comparison in a separate `uv` project.

Run a small smoke test:

```bash
uv run --project Tasks/beliefe_bank_sample_vs_reinforcement python Tasks/beliefe_bank_sample_vs_reinforcement/main.py --epochs 1 --train-items 2 --eval-items 2 --batch-size 8 --num-samples 8
```

Run the default comparison:

```bash
uv run --project Tasks/beliefe_bank_sample_vs_reinforcement python Tasks/beliefe_bank_sample_vs_reinforcement/main.py
```

The first run may download `roberta-base`.

## Reward

`reward.py` implements a dense partial-credit reward:

```text
0.40 * label_accuracy
+ 0.25 * positive_implication_satisfaction
+ 0.25 * negative_implication_satisfaction
+ 0.10 * yes_count_calibration
```

Positive implications reward sampled states where `source=yes` implies
`target=yes`. Negative implications reward states where `source=yes` implies
`target=no`. The yes-count term discourages degenerate all-yes or all-no
policies by matching the number of sampled yes labels to the gold yes count.

## Output

The script reports before/after:

- generated reward from the shared BeliefBank reward function
- gold-label argmax accuracy
- graph constraint satisfaction
- predicted yes count vs. gold yes count
- gradient diagnostics for both programs
- which program performed better on generated reward

`SampleLossProgram` uses sampled graph-constraint loss directly; it should not
fall back to supervised-only training.
