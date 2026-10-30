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
| Object kind | Each confidently visible object gets a predicted `category`, a short lowercase noun read from its appearance ("tomato", "flask"), recorded as `visual_category:<id>` and in the audit's `visible_categories`. It is free text from pixels, not a scene-table name, and `null` when the kind cannot be read. Under `--protocol official` it grounds the instruction's candidates (below). |
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
python -m test_regr.VLABenchAgentInterface.main evaluate-planner --protocol official   --knowledge causal --constraint-profile research_v1 --research-perception image-vlm   --planner-model /home/auszok/models/Qwen/Qwen3-VL-8B-Instruct --load-in-4bit
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

- the visible tags ground `observation_supported_references` (`official/decoding_graph.py`): an id the
  perception did not confidently see is refused;
- the predicted kinds ground `instruction_candidates_bound_targets`: when the instruction's source clause names a
  predicted kind ("put the **apple** in the bowl"), `pick` takes only tags of that kind, and a kind named in the
  destination clause bounds `place`, `insert`, `pour` and `push` the same way
  (`object_grounding.perceived_object_constraints`). A kind matches by its whole phrase or head noun, singular or
  plural ("red apple" matches "apples"). Tags whose kind could not be read stay candidates, so only tags seen as
  another kind are excluded. Negation, spatial or ordinal wording and source relations ("from the plate") abstain,
  as in the scene-name filter;
- the role hypotheses ground the research grounding formulas (`knowledge/research_lcs.py`), evaluated by the same
  `ResearchMonitor` as the internal runtime and compiled into `KnowledgePlanDFA`. Each operation must keep at least one
  hypothesis consistent with every id the plan has written (`research__grounding_joint`). Only the joint policy is used:
  robust and abstain refuse to act under ambiguity, which in a planning-only benchmark is a format error.

Unusable perception (invalid JSON, no confident object, no bound role) supplies no facts, and the episode
decodes under grammar or causal alone. Every reply records the result in
`<!-- perception: status=... visible=[...] categories={...} candidates={...} hypotheses=n complete=... -->`.

Not used under this protocol: open/closed states (the official decoder keeps no world state), spatial, temporal and
freshness policies (one capture, no execution), and every rule written against scene names (physics, complex, scene and task groups): a predicted kind is not a
scene name, so those tables do not match it.
Report such a run as a perception-augmented constrained system, apart from the zero-shot rows: the research formulas are
priors written for VLABench, and a second call to the model precedes the plan. No official score has been measured yet.

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
