# Research constraints and matched experiments

`research_v1` is optional. The normal eleven-skill domain, vocabulary, continuous
controller loss and explicitly selected official nine-skill languages are unchanged.
The default profile remains `baseline`; the research grounding default is `robust`.
The optional [image perception adapter](IMAGE_PERCEPTION.md) can now generate visual
evidence from the raw image, numbered image and instruction for both main pipelines.
These changes do not establish an improvement in benchmark performance. Historical
evaluation-exposed weights must not initialize the matched experiments.

## Main commands

Both `test_regr.VLABenchAgentInterface.main` and
`test_regr.JointEmbodiedAgentInterface.main` accept:

```bash
--constraint-profile research_v1 --grounding-policy robust \
--research-supervised-mode full --motion-geometry off \
--max-grounding-hypotheses 128 --research-max-state-expansions 10000
```

Use these with the existing `train-agent`, standalone `train-planner`, and planner
evaluation commands. Joint runs apply the profile to VLABench. EAI retains its own
graph and losses. Explicit `--ce-only`, `--ce-dfa-only`, or
`--stage1-dfa-loss off` controls remain effective. The research-only `ce` and
`masked-ce` modes affect VLABench without changing EAI loss configuration.

For canonical simulator training, add `--research-rollout-manifest FILE.json`.
The file contains `train`, `validation`, `calibration`, and `test` lists of
descriptors with `task`, `scene_group`, explicit `env_kwargs`, observation-only
`planning_facts`, and, where required, the observed entity inventory. Training uses
`train`; model-selection rollouts use `validation`. All variants of a scene must
share a group. Do not put references, future observations or winning bindings in
these descriptors. `--research-rollout-split` selects held-out scenes for manifest
rollouts. Official evaluation and research perturbations require separate reports.
Research and matched-baseline contexts discard legacy `task_target_entity` and
`task_target_container` annotations. Observed candidate bindings belong in the
grounding evidence block; evaluator intent cannot preselect one candidate.

## Observation contract

Prepared planner examples carry `planning_facts.research` through the standalone
and joint training forward passes, validation, decoding and RL replay. Dataset
episodes may supply this block in `input/research_observations.json`. Store scene
group and split in `provenance/research.json`, for example
`{"scene_group":"scene-001","split":"train"}`. All research scene groups use
70/10/10/10 train/validation/calibration/test assignments.

An abbreviated observation block:

```json
{
  "evidence": {
    "episode": "scene-001", "step": 0,
    "revisions": {"object:apple": 0},
    "records": [
      {"id":"pose:apple","value":[0.1,0.2,0.8],"step":0,
       "source":"perception","kind":"observed","frame":"world","units":"m",
       "dependencies":{"object:apple":0}}
    ]
  },
  "grounding": {
    "complete": true,
    "hypotheses":[{"id":"h0","bindings":{"object":"apple"},"weight":1.0}],
    "requests":[{"action":"pick","role":"object","argument":"subject"}]
  },
  "freshness":[{"action":"pick","evidence":"pose:apple"}]
}
```

Hypotheses preserve correlations across roles and operations. `roles` is an
alternative Cartesian-product input. Uniform scores apply when scores are absent.
Joint selects the best scored feasible generated witness; this is a bounded
comparison, not a proof of global optimality or user intent. Robust admits only
bindings consistent with every supported interpretation in the same concrete
plan. Abstain requires unique bindings for the roles used by the next decision.
Incomplete candidate universes cannot establish robust or unique grounding.
Limits raise `budget_exhausted`; candidates are never silently truncated.

Every evidence revision is immutable. Observed, predicted and derived records
remain distinct. Derived records need fresh supports. Dynamic records expire at
their captured step by default. Motion invalidates object, joint, tracking and
camera dependencies conservatively. Reset clears event/progress state. Unknown
values, lost tracking and stale dependencies cannot establish a sensor fact.
Simulator reset also clears captured observations and candidate bindings, while
retaining policy requests. A fresh perception callback can return `grounding`
with observed `roles` or `hypotheses` and `complete`; it cannot redefine requests
or provide an intended binding. Missing refreshed grounding remains unavailable.

`read_research_observations(step, episode, revisions, observation)` is the sensor
adapter contract. It returns current records, explicitly measured events and
optional measured `initial_state`. The supplied adapter reads declared observation
fields; privileged entity/joint readers require
`sensor_profile="simulator_diagnostic"` and must be reported separately.
Evaluator targets, task-success annotations and reference actions are excluded
from observation prompts. A commit gate refreshes once and returns
`needs_observation` or `needs_clarification` if unresolved. Neither is a skill.

Measured sensor events can advance `completed_skill_prefix` only when an event
matches the executing action and its entity arguments. Stable event IDs prevent
double counting. Generating an action, predicting an effect, elapsed time, and
completing a controller chunk do not establish a physical event. The canonical
RL/evaluation loop and the lightweight agent share these gates.

## Spatial and event policies

Requirements have `phase="commit"`, `operation`, or `completion`; commit checks
the next action, while completion checks the terminal state. Spatial requirements
name evidence IDs in `left`/`right`, an explicit `frame`, and a relation:

| Relation | Required primitive evidence |
|---|---|
| left/right, front/behind, above/below | Same-frame point/bounds in metres; axes default to x/y/z respectively, or explicit `axis` |
| near/far | Same-frame bounds and finite `threshold_m` |
| inside | Actual box bounds, `geometry="boxes"`, fresh Boolean interior-observation evidence named by `interior` |
| on | Box bounds, measured `contact` evidence and `tolerance_m` |
| disjoint | Actual box bounds and `geometry="boxes"` |
| ordered | Ordered `entities` evidence IDs and explicit axis; optional descending order |

Uncertain comparisons remain unknown. Frames are never implicitly transformed.
Use calibrated upstream transforms before supplying measurements. Explicit
`spatial_effects` may project a pose transfer for planning; those projections never
enter the measured ledger or establish execution success.

Temporal requirements support `count` with min/max, `before` with a predecessor,
`within` with a step bound, and `until` with start step and retained sample prefix.
Events have unique IDs, episode, simulator step, entity arguments and provenance.
Missing interval samples are unknown; interval samples include both endpoints.
Until completion requires its measured end
event; an ongoing interval can be checked only through the current step.
Planned traces are isolated and explicitly selected with `trace="planned"`.
Deadline planning also requires per-action `duration_upper_steps`. Physical
execution always checks measured events and observations.

## Declarations, losses and ablations

`knowledge/research_lcs.py` declares fifteen formulas through `RelayedKnowledge`.
They enter the existing task-tier grounded goal/affordance families. The original
formulas determine admission in DataNodes, compiled contextual token masks,
bounded action reachability and runtime evaluation.

| Formula family | Formulas | Training connection |
|---|---|---|
| Freshness | `fresh_observed`, `fresh_current`, `fresh_dependencies` | Admissible action/binding probabilities |
| Spatial, temporal, geometry | Each has `_known` and `_holds` | Admissible continuation probabilities when applicable |
| Grounding | `grounding_available`, `_complete`, `_robust`, `_joint`, `_unique`, `_abstain_match` | Feasible binding probabilities and consistent hypotheses |
| Model projection | `research__model_choice_admitted` | DomiKnowS implication over model probabilities and compiled admission |

Names have prefix `research__`. Full planner loss is supervised CE plus weighted
allowed-mass loss and projected primal-dual LC loss, using the shared generation
loss helpers and DomiKnowS `LossModel` dual variables. Constant sensor predicates
are not advertised as model gradients. The model-dependent projection penalizes
probability assigned to choices rejected by the compiled original formulas.
Existing defaults are mass weight 0, PMD weight 1, dual Adam LR .05, warm-up 10,
update every 10. They are configurable explicitly. Empty admissible sets are gated;
no infinite allowed-mass loss or invented teacher action is used.

Controller behavior cloning keeps the continuous pose/gripper loss. Its skill
labels come only from actual expert-call boundaries on measured-success traces.
The LCs affect demonstration admission, execution admission and RL feedback.
There is no assumed gradient through collision checks.

Use `--research-disable-formula research__NAME` to remove one formula and rebuild
the runtime. Evaluate policy comparisons and formula removals on fixed checkpoints
and fixed scenes. Evaluation overrides are disclosed; training resume rejects a
different policy, loss mode or active formula set. Checkpoints record profile,
formula activation, loss settings, schema/vocabulary checksums and dual state.
Eleven-skill weights remain compatible when the profile is off. Nine-skill
checkpoints retain the explicit incompatibility error.

## Optional sampled geometry

`--motion-geometry sampled` needs an environment `research_geometry_backend` with
an explicit proposed joint path, isolated clone and allowed contact pairs. The
backend must attest complete robot/self/scene collision pairs **and** held-object
geometry, including attachment updates on the clone. `MujocoPathClone` supplies an
isolated MjData adapter; the caller supplies audited pairs and any attachment
transform. Missing paths, held geometry, pair coverage or sampling budget return
unavailable. Certificates are bound to path digest, episode, step and dependency
revisions. Sampled clearance does not establish continuous collision safety.

## GPU2 data and experiment gates

Collection code is versioned; datasets and run artifacts belong only on GPU2.

```bash
python -m test_regr.VLABenchAgentInterface.research.collection \
  --output /GPU2/PATH/pilot_control --minimum-per-skill 20 --scenes-per-task 30
python -m test_regr.VLABenchAgentInterface.research.experiments \
  --manifest /GPU2/PATH/experiment.json --pilot
```

The collector records real macro-call boundaries and independent simulator task
success. Each counted segment also needs stage success and a measured
postcondition. Helpers are not divided into fake equal-duration skill labels.
The version-2 trace recorder captures actual `env.step` calls, including terminal
controls that upstream waypoint-return helpers can omit. It converts commanded
joint targets to EE training targets with forward kinematics on a physics clone;
observations remain measured before each control. Simulator steps and times are
stored with the trace. Native helper success flags are diagnostic; independent
physical postconditions determine macro success. Older returned-waypoint traces
do not count toward this measured-trace gate.
Each completed episode keeps a hashed `episode.json` with all macro outcomes,
including failed-task segments excluded from the training manifest.
The new cool-drink expert is a candidate collection implementation; it does not
count as validated until measured checks pass. Missing or unsupported skill
postconditions and failed experts keep the collection gate closed.

The control corpus has `segments.jsonl`, `episodes/**/control.npz`, instruction
metadata, `collection_gate.json`, hashes and a run record. Research control config
must additionally name `official_source` and explicit `official_episode_splits`
for matched experiments. Official warm-up has no invented skill conditioning.
Windows from segmented traces clamp at their real macro boundaries.
The version-2 segmented contract uses the same sorted entity-pointer vocabulary
as live numbered views. Place/insert/pour condition on their destination; carry
skills without an explicit pointer use padding. Macro-call position is retained
as provenance, but its compact-plan position is unknown and uses padding in both
research BC and execution. It is never guessed from normalized episode time.
Snapshots outside the primary GPU2 checkout must set
`VLABENCH_OFFICIAL_EVALUATION_ROOT` to the reserved official planning directory.

The experiment manifest names `planning_dir`, `control_source`, `rollout_manifest`,
`output`, optional `planner_model`, and seeds `[42,43,44]`. Its gate validates
independent planning roles and measured segment coverage before producing runnable
commands. Add `--execute` to execute a validated matrix on GPU2. Full requires
500 successful segments per each of eleven skills; pilot requires 20.
Full execution additionally requires `pilot_output` pointing to an accepted pilot.
The runner resolves the original Hub backbone once to a local snapshot, records
its revision and weight hashes, and uses it for every seed/condition. Optional
`backbone_revision` pins the requested Hub revision.
The matrix selects `--use-compiled-planner-loss` for both conditions so their loss
machinery agrees. Baseline keeps research rules off. The default profile-off
standalone builder remains unchanged outside this explicit matched-run setting.
Both matched conditions receive the identical observation text. Observation-input
mode is fingerprinted; the comparison does not hide evidence from the baseline.
Set `research_family` on rollout descriptors to group perturbations independently
of native task names. `--research-eval-episodes-per-family` selects exactly two
pilot or ten full held-out scene groups per family, with one rollout each.
Manifest `rollout` evaluation uses the canonical RL rollout implementation. The
full runner schedules policy comparisons and individual formula removals on the
same fixed full-research checkpoint and scenes.

Pilot uses one planner epoch, 200 BC updates, and two validation episodes per
research family. Full uses three SFT epochs, 20,000 BC updates split 10,000 official
and 10,000 freshly segmented, then three RL epochs. The research condition clones
the same seed's baseline BC controller via `--warm-controller-from`, before RL.
The standalone matrix selects `--controller-warmup-units updates`; accumulation
increases microbatches rather than reducing the optimizer update count. Warm-up
saves optimizer/RNG checkpoints every 1,000 updates. Legacy standalone warm-up
continues to count batches unless this option is explicitly selected.
Planning uses the original pretrained backbone independently in both conditions.
Run records include physical GPU inventory/visibility, PID, stage, source revision,
manifest hash and logs. Stage and epoch checkpoints permit explicit verified
resume; a failed experiment must be inspected before restarting it.

Measured end-to-end success is reported separately from LC-conforming success.
Reports also include LC violations, refresh/abstention, evidence coverage, search
exhaustion, episode latency and assistance. Wrong-object counts remain unavailable
without an independent measured identity/intent audit. Static evaluation explicitly
marks measured freshness-over-execution and interval success unavailable.

At implementation time there are no trained-checkpoint simulator results for
this profile. Collection, pilot/full coverage, held-out ten-episode evaluation per
family/seed, and performance claims remain gated by actual measured run artifacts.
