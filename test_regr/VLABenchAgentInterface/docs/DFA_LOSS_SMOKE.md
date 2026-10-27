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

## Replicating the primal-dual results

The three runs of the table above (`init`, `dfa_off`, `dfa_on`) on gpu2. The general procedure (ssh through `cracow`,
container `vigorous_easley`, GPU etiquette) is in the `domiknows-gpu2` skill and `DomiknowsRelated/research/GPU2_Server_Guide.md`.

**1. Pick free GPUs** (three runs, one GPU each; others use this server):

```bash
ssh gpu2.ihmc.us 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

**2. Ship the code.** The numbers were measured at commit `e9450533`. Ship that commit, not a later one, to reproduce
them exactly (the server checkout is old and dirty; do not `git pull` there). Run from the repository root:

```bash
git archive --format=tar.gz -o dfa_smoke_src.tar.gz e9450533 domiknows test_regr/__init__.py test_regr/common_backbone.py test_regr/VLABenchAgentInterface test_regr/EmbodiedAgentInterface test_regr/JointEmbodiedAgentInterface
```

```bash
scp dfa_smoke_src.tar.gz gpu2.ihmc.us:~/
```

```bash
ssh gpu2.ihmc.us 'rm -rf ~/dfa-smoke && mkdir -p ~/dfa-smoke/out && tar xzf ~/dfa_smoke_src.tar.gz -C ~/dfa-smoke'
```

**3. Write the launcher** `~/dfa-smoke/smoke_dfa_run.sh` (usage: `bash smoke_dfa_run.sh <gpu> <tag> <rounds> [extra flags]`).
Planner `Qwen3-VL-8B-Instruct` in 4-bit, seed 7, the first 1,000 planning episodes, Stage 1 only, one epoch:

```bash
#!/bin/bash
gpu=$1; tag=$2; rounds=$3; shift 3
cd /home/auszok/dfa-smoke && mkdir -p out checkpoints/$tag
CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl PYTORCH_ALLOC_CONF=expandable_segments:True \
exec /opt/dominows/venv/bin/python -u -m test_regr.JointEmbodiedAgentInterface.main train-agent \
  --two-stage --seed 7 \
  --planner-model /home/auszok/models/Qwen/Qwen3-VL-8B-Instruct \
  --eai-dataset all --eai-split train \
  --vlabench-planning-dir /workspace/test_regr/VLABenchAgentInterface/data/planning --vlabench-limit 1000 \
  --control-source /workspace/test_regr/VLABenchAgentInterface/data/control \
  --task all --device cuda:0 --load-in-4bit --batch-size 8 --workers 0 \
  --stage1-epochs 1 --stage1-rounds-per-epoch $rounds --stage1-eai-warmup-steps 0 --stage2-epochs 0 \
  --controller-warmup-steps 0 --validation-limit 8 --output checkpoints/$tag "$@"
```

Copy it to the server (for example `scp smoke_dfa_run.sh gpu2.ihmc.us:/home/auszok/dfa-smoke/`).

**4. Launch the three runs**, detached, replacing `0`, `5` and `6` with the free GPUs. At `e9450533` the defaults are
the primal-dual configuration of that table (allowed-mass weight 1.0 on EAI and VLABench, primal-dual constraint loss on
VLABench, plain cross entropy), so `dfa_on` needs no extra flag:

```bash
ssh gpu2.ihmc.us 'docker exec -d vigorous_easley bash -lc "cd /home/auszok/dfa-smoke && bash smoke_dfa_run.sh 0 dfa_on 100 > out/smoke_dfa_on.log 2>&1"'
```

(On a later commit, the defaults changed to masked cross entropy on VLABench and mass weight 0, so the same
configuration needs `--stage1-dfa-mass-weight 1.0 --stage1-masked-ce-domains none` appended after `100`.)

```bash
ssh gpu2.ihmc.us 'docker exec -d vigorous_easley bash -lc "cd /home/auszok/dfa-smoke && bash smoke_dfa_run.sh 5 dfa_off 100 --stage1-dfa-loss off > out/smoke_dfa_off.log 2>&1"'
```

```bash
ssh gpu2.ihmc.us 'docker exec -d vigorous_easley bash -lc "cd /home/auszok/dfa-smoke && bash smoke_dfa_run.sh 6 init 1 --stage1-dfa-loss off > out/smoke_init.log 2>&1"'
```

Create `out/` first (step 2 does); a redirect into a missing directory fails silently.

**5. Wait and read the result.** `dfa_off` and `init` take about 10 minutes, `dfa_on` about 94 minutes. Each run logs one
line at the end of the epoch with the measured quantity:

```bash
ssh gpu2.ihmc.us 'grep -a "DFA mass\|Traceback" /home/auszok/dfa-smoke/out/smoke_dfa_on.log /home/auszok/dfa-smoke/out/smoke_dfa_off.log /home/auszok/dfa-smoke/out/smoke_init.log | cut -c1-300'
```

The line reads `Stage 1 epoch 1/1 DFA mass (unmasked probability on valid labels) {...}` with the EAI and VLABench values
of the table (also stored under `metrics.dfa_mass` in `checkpoints/<tag>/joint_stage1_epoch_000.pt`). The wall time of
`dfa_on` against `dfa_off` is the other number in the table.

**Variants.** With `--stage1-pmd-domains none` the primal-dual term is off and only the allowed-mass term remains;
`--stage1-pmd-weight` and `--stage1-dual-lr` (default 0.05) scale and step the primal-dual term;
`--stage1-dfa-mass-steps 0` scores every step instead of 12 random ones; `--stage1-dfa-lookahead-domains eai,vlabench`
adds the reachable-completion test to EAI (slow); `--stage1-pmd-domains eai,vlabench` also puts the primal-dual term on
EAI (minutes per round).

These commands were reassembled from the launch scripts of the original runs; the `e9450533` flag names were checked
against that commit's `main.py`, but the full sequence was not re-run after writing this section.

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
