# Sample-Loss Task Examples

This file lists task examples that actively use DomiKnowS sample-loss training
through `SampleLossProgram` or a subclass of it. In this codebase, this is the
sampling-based variant of the primal-dual constraint-loss family.

## Active Examples

| Task | Location | Notes |
| --- | --- | --- |
| Simple logic | `simple-logic/main_pd_sample.py` | Small binary toy constraint with `sample=True`, `sampleSize=100`, `sampleGlobalLoss=True`. |
| Sample loss vs reinforcement | `primaldual_sample_vs_reinforcement/main.py` | Purpose-built exact-count comparison against `ReinforcementProgram`. |
| BeliefBank | `beliefe_bank/main.py` | Rule/fact implication task using ILP and local argmax inference. |
| RuleTaker | `Ruletaker/main.py` | Question-label implication task using sampled constraint loss. |
| WIQA | `WIQA/WIQA_PD.py`, `WIQA/WIQA_aug.py` | Multiple answer/relation constraints with sampled loss variants. |
| CIFAR100 | `CIFAR100/main.py` | Image hierarchy/classification sample-loss variants with and without ILP inference. |
| MNIST binary | `MNIST_binary/main.py` | Digit-label constraints with both global and per-constraint sample loss. |
| NLI | `NLI/program_declaration.py` | Two sample-loss programs over NLI POIs. |
| Event-to-event relation | `EventToEventRelation/program_declaration.py` | Relation constraints with `sampleGlobalLoss=True`. |
| Spatial QA rules | `SpatialQARules/program_declaration.py` | Multiple task declarations with sampled spatial-rule constraints. |
| Sequence tagging, RNN BIO | `sequence-tagging/domi_rnn_bio/train_domi_rnn.py` | BIO tagging with local and ILP inference sample-loss variants. |
| Sequence tagging, BERT BIO | `sequence-tagging/domi_bert_bio/train_domi_bert_no_batch.py` | BERT BIO tagging with local and ILP inference sample-loss variants. |
| Sudoku | `sudoku/main.py`, `sudoku/main_simple.py`, `sudoku/main_6by6.py` | Large combinatorial counting/consistency constraints. |
| CoNLL04 callback sampling | `conll04/CallBackModel.py` | `CallbackSamplingProgram(SampleLossProgram)` subclass. |
| MNIST arithmetic callback sampling | `mnist-arithmetic-2/train.py`, `mnist-arithmetic-2/train_samplingloss_10k.py` | `CallbackProgram(SampleLossProgram)` callback-based sample-loss training. |

Commented or template-only sample-loss references also exist in a few files
such as `VQA/main.py`, `mnist-arithmetic-2/train_baseline_10k.py`, and
`mnist-arithmetic-2/train_digitlbl_10k.py`; those are not counted above as
active examples.

## Best Candidates for ReinforcementProgram

`ReinforcementProgram` has the best chance when the task can be expressed as:

- sample a small set of discrete decisions,
- decode the sampled decisions into a compact generated output,
- score that output with a clear reward, and
- get enough nonzero reward samples for a usable policy-gradient signal.

Best candidates from the current sample-loss examples:

1. `primaldual_sample_vs_reinforcement/main.py`
   - Best fit because it was designed around an explicit generated-output
     reward: sampled labels are decoded to `"zero"`/`"one"` and scored by an
     exact-count reward.
   - It has a small decision space, cheap evaluation, and direct before/after
     reward reporting.

2. `simple-logic/main_pd_sample.py`
   - Strong fit because it is a tiny binary decision problem with an obvious
     reward: return `1` when exactly one of `y0` or `y1` is true.
   - It should be the easiest existing non-RL sample-loss example to port to
     `ReinforcementProgram`.

3. `Ruletaker/main.py` and `beliefe_bank/main.py`
   - Good conceptual fit because rule/fact decisions can often be decoded to a
     yes/no answer and compared to a label.
   - Expected to need a task-specific decoder so the reward is not too sparse.

4. `MNIST_binary/main.py` and `mnist-arithmetic-2/*`
   - Plausible if the reward is defined over generated digit assignments or
     arithmetic answers.
   - Harder than simple logic because there are more decisions and the visual
     classifier must still learn useful local evidence.

Lower-probability candidates without additional reward shaping:

- `sudoku/*`: the reward is natural, but exact board satisfaction is extremely
  sparse over a large combinatorial space.
- `sequence-tagging/*`, `conll04/*`, `EventToEventRelation/*`,
  `SpatialQARules/*`, `WIQA/*`, and `NLI/*`: feasible, but likely need careful
  decoders and partial-credit rewards to avoid sparse or high-variance learning.
- `CIFAR100/main.py`: least direct fit because the task is mostly supervised
  visual classification with hierarchy constraints; RL would add variance
  unless the reward is carefully shaped around hierarchy consistency and label
  correctness.

Overall, the best expected `ReinforcementProgram` result is the purpose-built
`primaldual_sample_vs_reinforcement` example, followed by `simple-logic`.
