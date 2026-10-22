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
