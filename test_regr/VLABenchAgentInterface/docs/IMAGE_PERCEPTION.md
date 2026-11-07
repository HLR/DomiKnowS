# Image-only research perception

The image perception adapter runs a frozen pretrained Qwen3-VL backbone on exactly
one original RGB image, its numbered version, and the instruction. It emits strict
structured JSON, validates it, and converts accepted estimates into the existing
DomiKnowS observation contracts. It does not receive entity metadata names,
reference plans, intended-target annotations, evaluator conditions, depth, poses,
joint readings, or simulator success. Generic skill names and the JSON schema are
declarative knowledge in its prompt.
The network proposes action/role requests; the declared skill schema supplies
their `subject`/`destination` argument fields. Canonical numeric strings are
normalized to integer pointers without consulting metadata identities.

This is a pretrained perception implementation, not a newly trained or calibrated
perception checkpoint. Confidence is model-reported. The default 0.85 admission
threshold and 0.01 normalized box margin are configurable heuristics. No benchmark
accuracy, physical success, or calibrated confidence is claimed.

## Shared constraint inputs

| Existing constraint | Image perception contribution |
|---|---|
| Visibility | Confident, visibly numbered object IDs populate `visible_entities`; metadata membership alone supplies no visibility. |
| Object kind | Each confidently visible object gets a predicted `category`, a short lowercase noun read from its appearance ("tomato", "flask"), recorded as `visual_category:<id>` and in the audit's `visible_categories`. It is free text from pixels, not a scene-table name, and `null` when the kind cannot be read. Under `--protocol official` it grounds the instruction's candidates (below). A readable playing card also reports its `card` face (rank and suit), recorded as `visual_card:<id>`. |
| Ambiguous grounding/selection | Candidate role bindings and relative weights inferred from pixels and instruction; multiple interpretations remain explicit. |
| Freshness | Each capture is stamped with episode, runtime step and dependency revisions. A failed refresh removes same-step visual evidence. |
| Open/closed affordances | Confident explicit visual states populate initial positive/negative facts; they remain noisy visual estimates. |
| Spatial direction/order | Normalized image bounds support explicit screen-space left/right/above/below, ordering and projected separation. |
| Temporal/event policies | Successive known captures produce named visual state-change events and retained interval samples. Occlusion or reset breaks the transition chain. |
| Syntax and skill schemas | Continue to use declarative DomiKnowS rules. The perception model does not define or disable them. |
| World-metric distance, physical containment/support/contact, joint intervals, sampled collision checks | Unavailable from these RGB inputs. Image boxes cannot satisfy these contracts. |
| Poker rank/suit selection and physical-property selection | Their existing primitive observation contracts still apply. This adapter does not turn descriptions into certified card faces, mass, friction or material measurements. |

Visual change events use names such as `visual_closed:0:false`. They are not
automatically relabeled as completed `open_door`, `pick`, or other physical skills.
Completing a controller chunk cannot manufacture an observed grasp. Measured
simulator task success remains an independent evaluation signal.

Unknown values, low confidence, malformed outputs, unreadable IDs and missing
candidates remain unavailable. Invalid JSON is not partially recovered. Hypothesis
overflow retains the existing distinct search-budget exception and never truncates
the candidate list. An empty/incompletely interpreted instruction does not bypass
the observation gate.

## Main pipeline

Both existing main entry points accept these additional options:

```bash
--constraint-profile research_v1 --research-perception image-vlm \
--perception-model /home/auszok/models/Qwen/Qwen3-VL-8B-Instruct \
--perception-confidence 0.85 --perception-bbox-margin 0.01 \
--perception-camera 0
```

Use them with planner training, agent training, graph planner evaluation and
simulator rollouts. The perception model is separate from the trained planner;
it stays frozen, runs before dataset preparation and during observation refresh,
and shares its resulting facts with training masks, losses, decoding and checks.
Inference responses are cached by model configuration, image pixels and instruction;
the observation ledger is freshly stamped at each actual capture.

The default `--research-perception auto` enables image perception for `research_v1`
and leaves it off for the ordinary baseline. `--research-perception off` explicitly
selects supplied-evidence runs, including resume of older research checkpoints
without image perception. Matched experiment commands explicitly enable it in both
conditions, preserving identical observation inputs. `--perception-model` defaults
to the original `--planner-model` backbone, loaded as a separate frozen model.
An explicit matched baseline may use
`--constraint-profile baseline --use-compiled-planner-loss --research-perception image-vlm`.
It receives the same visual evidence. Enabled research formulas still depend on
the selected profile. EAI inputs and losses remain unchanged.
The perception model/configuration is saved in research checkpoint metadata;
training resume rejects a changed perception configuration. Evaluation overrides
are disclosed. Use an immutable local model snapshot for reproducible experiments.
`--perception-load-in-4bit` enables optional CUDA quantization; weights are otherwise
loaded in the existing CUDA/CPU precision. Loading defaults to local files only.

For static examples, the loader uses the sole raw and sole numbered image when
that pairing is unambiguous. For multiple views, add `input/perception_images.json`:

```json
{"raw":"input.png","numbered":"input_mask.png"}
```

Paths are relative to that input directory and cannot escape it. Images must have
the same pixel coordinates and dimensions. The adapter cannot verify that two
arbitrary files depict the same capture; this pairing is a dataset contract.
Existing reference labels remain supervised targets, never perception inputs.
Metadata-grounded observations are replaced in image mode. Planner display names
are reduced to numbered objects; opaque names remain internal pointer handles.

Canonical rollouts reuse the benchmark's existing numeric-label rendering and one
chosen RGB camera. Only the two resulting images enter the perception network.
An external environment can instead provide `research_image_pair` (raw, numbered)
and its `entity_table`. Selecting image perception replaces privileged research
sensor callbacks, and the lightweight helper's direct simulator-joint refresh is
disabled. Controller proprioception and independent evaluator diagnostics retain
their existing roles; they are not inputs to this perception network.

## Official protocol

`--protocol official` accepts the perception with `--knowledge grammar` or `causal`:

```bash
python -m test_regr.VLABenchAgentInterface.main evaluate-planner --protocol official \
  --knowledge causal --constraint-profile research_v1 --research-perception image-vlm \
  --planner-model /home/auszok/models/Qwen/Qwen3-VL-8B-Instruct --load-in-4bit
```

The run stays within the protocol's inputs and scoring:

| Requirement | How the official path meets it |
|---|---|
| Inputs | Perception reads the episode's `input.png`, `input_mask.png` and instruction, the same items the official prompt shows (`official/perception_observations.py`, `episode_inputs`). |
| No entity table | `perceive(..., entity_table=None)`: pointers stay tag ids, any printed tag is accepted, and `complete` is the model's own claim about the image instead of a comparison with the scene inventory. |
| No task identity | No task-level rule runs; `--task-identity` (oracle or predicted) is refused. |
| Frozen model | When `--perception-model` is the planner model (the default), the loaded planner perceives; no adapter or vocabulary is accepted. |
| Threshold | The fixed default 0.85, or a value fit on a dataset named by `--perception-calibration-data` that `dataset_protocol.assert_training_dataset` proves independent of the official episodes. |
| Scoring | Unchanged upstream scorer over all 4,500 episodes; `run.json` records the perception model, fingerprint, policy and calibration. |

The facts reach the plan only through compiled logical constraints:

- the largest visible tag grounds `scene_ids_bound_targets` (`official/decoding_graph.py`): an id above it is
  refused. A lower id whose tag was not seen stays allowed, because correct targets can lack a visible tag: on the
  validation split, requiring each id to be seen refused 39 of 450 reference plans, 36 of them presses of untagged
  buttons, while the bound refused none;
- the predicted kinds ground `instruction_candidates_bound_targets`: when the instruction's source clause names a
  predicted kind ("put the **apple** in the bowl"), `pick` takes only tags of that kind, and a kind named in the
  destination clause bounds `place`, `insert`, `pour` and `push` the same way
  (`object_grounding.perceived_object_constraints`). A kind matches by its whole phrase or head noun, singular or
  plural ("red apple" matches "apples"). Tags whose kind could not be read stay candidates, so only tags seen as
  another kind are excluded. Negation, spatial or ordinal wording and source relations ("from the plate") abstain,
  as in the scene-name filter;
- ordinal and property requests ground the selection formulas of `official/selection_graph.py`, and the tags they
  admit join the same candidate LC for `pick` (see below);
- with `--grounding-policy` (opt-in; none runs by default, see the ablation below), the role hypotheses ground the
  research grounding formulas (`knowledge/research_lcs.py`), evaluated by the same `ResearchMonitor` as the internal
  runtime and compiled into `KnowledgePlanDFA`. Under `joint` each operation must keep at least one hypothesis
  consistent with every id the plan has written (`research__grounding_joint`). `robust` or `abstain` selects the
  stricter formulas (`_robust`, `_unique`, `_abstain_match`, `_complete`); they refuse to act when the hypotheses
  disagree or the grounding is incomplete, and such an episode ends as a format error. The policy is part of the
  run's decoding label.

When several sources bound the same skill and role, a tag must satisfy all of them; if they leave no tag in common,
that role is left unrestricted.

### Ordinal and property selection

Each confidently visible object may also report a `card` face (`rank` 2..10, jack, queen, king or ace; `suit`
spades, clubs, diamonds or hearts), recorded as `visual_card:<id>`. `official/selection_observations.py` parses
these requests from the instruction:

| Request | Example | Group | Order |
|---|---|---|---|
| Spatial ordinal | "Please pick the 2nd poker from left to right", "Add 3rd condiment from bottom to the dish" | Tags whose predicted kind the noun names; for poker or card, also tags showing a face | Image boxes, strictly separated along the axis |
| Spatial ordinal on a shelf layer | "fetch the second book from the left on the bottom layer" | The group's lowest (or highest) cluster of vertically overlapping boxes | As above |
| Card value | "Please pick the 2nd largest poker", "the smallest poker" | Tags showing a face | Card rank 2..14, ace high |
| Named card | "Please pick the poker jack of spades" | Tags showing a face | — |

`official/selection_graph.py` declares nine formulas: `ordinal_rank_1` .. `ordinal_rank_8` (the chosen tag is a
group member with exactly k − 1 members before it, an `exactL` count) and `chosen_has_requested_property`, which names no kind of object: card faces are the one reading source in use (`official/card_observations.py`). The compiled program is
grounded for every visible tag, and the admitted tags become `pick` candidates.

"Left" and "bottom" are the viewer's directions in the image the model sees. This convention is declared, not
fitted; upstream numbers some groups by world x, and a camera that mirrored world x would make every left/right
ordinal wrong. A request abstains, leaving `pick` unrestricted, when:

- any reported object fell below the confidence threshold or was not visible (it might belong to the group);
- the group is empty, larger than eight, or smaller than the ordinal;
- two members overlap along the axis, a card value is unread, or two values are equal;
- no tag shows a named face, or the instruction negates ("not", "except").

Not read: grid rows and columns ("the solution on the 2nd row and 4th column"), since an empty grid slot has no tag
and the image cannot fix a column number; "on the top layer" without a direction; depth words such as "farthest".

Unusable perception (invalid JSON, no confident object, no bound role) supplies no facts, and the episode
decodes under grammar or causal alone. Every reply records the result in
`<!-- perception: status=... visible=[...] categories={...} cards={...} candidates={...} hypotheses=n complete=... -->`.

Not used under this protocol:
- open/closed states, since the official decoder keeps no world state;
- metric spatial, temporal and freshness policies: one capture, no execution;
- every rule written against scene names (physics, complex, scene and task groups), since a predicted kind is not a
  scene name;
- Texas hold'em selection (`knowledge/poker_lcs.py`). Its formulas check at plan completion that the required cards
  ended on the destination, which needs plan state the official decoder does not keep, and it needs five to seven
  read card faces.

Report such a run as a perception-augmented constrained system, apart from the zero-shot rows: the research and
selection formulas are priors written for VLABench, and a second call to the model precedes the plan. Neither
perception accuracy nor the left/right convention has been checked.

### Constraint ablation on the official episodes

Measured 30 October to 1 November 2026 on all 4,500 episodes, 4-bit Qwen3-VL-8B-Instruct, `--knowledge causal`
unless noted (code at commit 1b0393a7 plus the switches below). Δ is the official score (mean of the six dimension
total scores) against the reference, with a paired bootstrap 95% interval (episodes resampled within each dimension).

| Added to the reference | Reference | Δ | 95% interval | Plans changed (better / worse) |
|---|---|---:|---|---|
| Visibility only | causal, 37.39 | +0.00 | [−0.05, +0.06] | 219 (20 / 19) |
| Object-kind filter | visibility | −0.02 | [−0.06, +0.02] | 35 (11 / 16) |
| Ordinal / named-card selection | visibility + kinds | −0.02 | [−0.06, +0.00] | 5 (0 / 2) |
| Visibility, kinds and selection | causal | −0.04 | [−0.12, +0.04] | 256 (31 / 37) |
| Joint grounding | the three above | −1.05 | [−1.27, −0.82] | 780 (147 / 347) |
| Joint grounding alone | causal | −1.09 | [−1.34, −0.85] | 858 (156 / 358) |
| All, robust (or abstain) policy | causal | −5.34 | [−6.05, −4.62] | 1,966 (393 / 1,374) |
| Visibility, kinds and selection | grammar, 37.44 | −0.06 | [−0.13, +0.00] | 195 (16 / 33) |

The 450-episode held-out split moves the same way in every row. Perception reported no facts for 856 episodes (19%),
which decode as the reference. Robust and abstain gave identical plans on every episode; they refuse to act on 708
(16%), which score as format errors. They raise PhysLaw from 0 to 35.2 only because refusing every pick leaves
`press`, the skill those question tasks need. Conclusion: the observation formulas (visibility, kinds, selection) are
neutral and stay available; the grounding policies lower the score and are opt-in. These are decisions made on the
official episodes, so report them as a diagnosis, not as tuned settings.

Reproduce one configuration with the switches below; `--perception-cache DIR` replays the raw perception replies
across runs (keyed by perception fingerprint, instruction and image hashes), so only the first run calls the model
twice per episode:

```bash
python -m test_regr.VLABenchAgentInterface.main evaluate-planner --protocol official \
  --planning-dir /path/to/vlm_evaluation_v1.0 --planner-model /path/to/Qwen3-VL-8B-Instruct --load-in-4bit \
  --knowledge causal --constraint-profile research_v1 --research-perception image-vlm \
  --perception-cache results/perception_cache \
  --official-disable-formula instruction_candidates_bound_targets \
  --official-output-dir results/ablation_visibility_only
```

`--official-disable-formula` takes any formula of `official/decoding_graph.py` or `official/selection_graph.py`
(`scene_ids_bound_targets`, `instruction_candidates_bound_targets`, `ordinal_rank_1` .. `_8`,
`chosen_has_requested_property`) and may be repeated; `--grounding-policy joint|robust|abstain` adds a grounding policy. Both are
recorded in the run's decoding label.

### The official knowledge level for training and evaluation

`--knowledge official` brings the same knowledge to the trained planners. It activates the world graph's causal group
and its `official` group, `official__scene_ids_bound_targets` and
`official__instruction_candidates_bound_targets`. The level also turns image perception on, so every example is read
from its image pair and instruction by `research/perception.py:enrich_examples`. Each example then stores its visible
tags and the candidates of `official/perception_observations.perceived_observations`: perceived kinds named by the
clauses, and tags admitted by the selection formulas. The graph DFA, the DataNode verifier and the Stage 1 DFA loss
all read these, and they are recorded in the checkpoint's perception configuration. The level is available on
`train-agent`, on `evaluate-planning` and standalone `eval_planning`, and on `evaluate-planner` (where under
`--protocol official` it means `causal` with image perception):

```bash
python -u -m test_regr.JointEmbodiedAgentInterface.main train-agent --two-stage \
  --vlabench-planning-dir test_regr/VLABenchAgentInterface/data/planning_train \
  --control-source test_regr/VLABenchAgentInterface/data/control \
  --knowledge official --device cuda:0 --load-in-4bit --output checkpoints/official_level
python -u -m test_regr.JointEmbodiedAgentInterface.eval_planning \
  --checkpoint checkpoints/official_level/joint_stage2_best.pt \
  --planning-dir test_regr/VLABenchAgentInterface/data/planning --split test --device cuda:0 \
  --knowledge official --output results/official_level.json
```

`--research-perception off` is refused at this level. No score has been measured at this level yet.

## GPU2 single-pair preparation

Run inside the existing GPU2 environment, keeping generated observations there:

```bash
DOMINOWS_COLLECTION_HOST=gpu2 CUDA_VISIBLE_DEVICES=4 \
python -m test_regr.VLABenchAgentInterface.research.perception \
  --raw /path/to/input.png --numbered /path/to/input_mask.png \
  --instruction 'Pick the red apple and place it in the bowl' \
  --model /home/auszok/models/Qwen/Qwen3-VL-8B-Instruct \
  --episode independent-scene-001 --output /path/to/research_observations.json
```

`--entity-count` optionally validates IDs against an existing numeric inventory;
semantic metadata names are never supplied to the model. Outputs include image
hashes, instruction hash, model configuration/revision, confidence provenance, and
explicit unavailable metric geometry. The command refuses to overwrite a file.

An image spatial requirement must explicitly opt into the image frame:

```json
{"id":"screen_order","relation":"left","left":"image_bbox:0",
 "right":"image_bbox:1","frame":"image:0","geometry":"image_boxes"}
```

The default metre-based spatial API is unchanged. An image relation never silently
becomes a world-frame relation. Calibration, depth, extra viewpoints or physical
sensors would be needed to extend the unavailable physical measurements.

See [IMAGE_PERCEPTION_VALIDATION.md](IMAGE_PERCEPTION_VALIDATION.md) for the
regression results and the separate pretrained-model development check.
