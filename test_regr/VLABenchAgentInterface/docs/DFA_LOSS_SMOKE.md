# Smoke test: does the Stage 1 DFA loss (allowed mass + primal-dual) help?

Code: commit e9450533 (HEAD at the time), run on gpu2 in a fresh directory, nothing pushed.

## Setup

Three Stage 1 runs on the real Qwen3-VL-8B planner, same seed (7), same data (the first 1,000 sorted planning
episodes, no held-out tasks), no EAI warm-up, EAI validation limited to 8 items:

- `init`: 1 round (an essentially untrained baseline).
- `dfa_off`: 100 rounds, cross entropy only (`--stage1-dfa-loss off`).
- `dfa_on`: 100 rounds, cross entropy + allowed-mass loss on EAI and VLABench + primal-dual constraint loss on
  VLABench (the defaults).

Measure (`dfa_mass`, logged each epoch): the geometric-mean probability the unmasked planner puts on labels the
DFA allows at the teacher-forced step, over 20 validation items, up to 12 scored steps each, one-step validity.
1.0 would mean no mask is needed.

## Result

| Run | EAI | VLABench | Time |
| --- | --- | --- | --- |
| init | 0.214 | 0.035 | 0.1 min |
| dfa_off (100 rounds) | 0.833 | 0.159 | 9.3 min |
| dfa_on (100 rounds) | 0.853 | 0.138 | 94.1 min |

## Reading

- Both paths run end to end on the real model, on EAI and on VLABench, with gradients reaching the planner and the
  multipliers.
- **No benefit is shown.** Cross entropy alone raises the valid-label mass from 0.21 to 0.83 on EAI and from 0.035
  to 0.159 on VLABench in 100 rounds. Adding the DFA loss moves EAI by +0.02 and VLABench by -0.02. One seed and 20
  items per domain; I did not measure run-to-run variation, so these differences are not distinguishable from noise.
- It is about 10 times slower (94 vs 9 minutes), because of the compiled primal-dual constraint evaluation on
  VLABench (the stack dump pointed at `constructCompiled`); I did not find out why this data is slower than the
  earlier 14 s/round run.
- Not tested: more rounds, a larger mass weight, scoring every step, or the lookahead allowed set on EAI.

## Follow-up: cross entropy under the DFA mask (commit 10117c42)

The allowed-mass term is redundant with cross entropy on teacher-forced steps (the target is always allowed, so
`-log P(allowed) <= -log p(target)`), which explains the null result above. Option A was chosen instead: accept that
decoding needs the mask and train the distribution decoding samples from, i.e. cross entropy with the softmax
restricted to the DFA-allowed labels at each step (`--stage1-masked-ce on`, the default; the mass term is off).
Primal-dual constraint loss off in this run (`--stage1-pmd-domains none`).

Same seed, same data, 100 rounds, validated through the DFA-masked decoder (100 VLABench validation episodes):

| Run | Overall | SR | PR | SPR | PM | Exact graph | Reward | Unmasked mass EAI / VLABench |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cross entropy | 66.0 | 73.5 | 71.0 | 52.0 | 30.0 | 0.30 | 0.660 | 0.83 / 0.16 |
| masked cross entropy | **70.8** | 74.0 | 79.5 | 57.5 | 36.0 | 0.36 | 0.708 | 0.44 / 0.035 |

Training took about the same time (about 10 minutes). EAI validation (8 items) is unchanged (goal success 0.25, goal
recall 0.47). As expected the unmasked probability on valid labels is lower, since the mask now does that work.

Caveats: one seed, 100 validation episodes, 100 rounds; the gap (+4.8 overall) is consistent across metrics but its
significance is not established. The share of steps whose gold label the DFA disallows (`masked_ce_misses`) is
reported by the loss but I did not aggregate it over the run.

## EAI: plain versus masked cross entropy (commit 844014b9)

Question: should the masked loss also apply to EAI? Four runs, 100 rounds each, VLABench masked in all of them,
primal-dual off, EAI validated through the DFA on 60 items (the earlier 8-item check could not show a difference).
Only the EAI setting differs within a seed (`--stage1-masked-ce-domains vlabench` against `vlabench,eai`).

| | Seed 7 plain | Seed 7 masked | Seed 11 plain | Seed 11 masked |
| --- | --- | --- | --- | --- |
| EAI goal success | 0.333 | 0.367 | 0.317 | 0.367 |
| EAI goal recall | 0.446 | 0.496 | 0.438 | 0.496 |
| EAI positive reward rate | 0.550 | 0.617 | 0.550 | 0.617 |
| EAI exact sequence | 0.150 | 0.150 | 0.133 | 0.133 |
| EAI unmasked mass on valid labels | 0.84 | 0.44 | 0.85 | 0.45 |
| VLABench overall score | 69.6 | 70.8 | 69.8 | 69.3 |

Reading: masked is higher on three EAI metrics in both seeds (success +2 episodes out of 60, recall +0.05 to +0.06,
reward rate +0.07) and equal on exact sequence; the VLABench score is unaffected within about one point. The
direction is consistent but the effect is small and rests on two seeds and 60 items. The two masked seeds give
identical success, recall and reward-rate values although their models differ (training losses and exact-sequence
rates differ), which is probably coincidence at this sample size but is worth keeping in mind. The default stays
`vlabench` only; widening it to EAI is reasonable but not yet well supported.
