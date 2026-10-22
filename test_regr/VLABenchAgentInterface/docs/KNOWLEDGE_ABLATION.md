# Knowledge ablation of the champion Joint planner

What each knowledge tier of the world graph contributes when the same checkpoint is evaluated on the same
450 test episodes. Task knowledge is expressed as graph concepts and logical constraints
(`knowledge/`, see the README source code map); `--knowledge` switches which constraint groups are active.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 10%;">
    <col style="width: 30%;">
    <col style="width: 18%;">
    <col style="width: 42%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="10%" style="width: 10%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Level</th>
      <th scope="col" width="30%" style="width: 30%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Active knowledge</th>
      <th scope="col" width="18%" style="width: 18%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Source of the task</th>
      <th scope="col" width="42%" style="width: 42%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">How new knowledge changes enforcement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">no DFA</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">nothing: every label is allowed at every step until the end label or the 61-label budget (<code style="white-space: normal; overflow-wrap: anywhere;">--no-dfa</code>)</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">none</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">No grammar or world constraints are enforced.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">no DFA, end withheld</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">as <code style="white-space: normal; overflow-wrap: anywhere;">no DFA</code>, but the end label is blocked at the first step only (<code style="white-space: normal; overflow-wrap: anywhere;">--no-dfa --no-dfa-min-steps 1</code>)</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">none</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">The decoder blocks immediate termination; no logical constraint is added.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">grammar</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">plan grammar DFA only: the token syntax of a plan, the nine skills, any of the 64 entity ids, the end token, the operation budget and the language of allowed skill patterns; no constraint group</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">none</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Enable grammar formulas; an incomplete subtask cannot terminate.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">generic</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">+ skill schema, subtask patterns, pointer, anti-self-placement and repeated-pick rules</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">none</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Enable generic formulas; self-placement, for example, is rejected.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">scene</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">+ scene-grounded rules (buttons, fixtures, containers, open-before-place)</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">none</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Enable scene formulas and evaluate scene facts; a static fixture cannot be picked.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">learned</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">+ the per-task LCs of <code style="white-space: normal; overflow-wrap: anywhere;">task</code> (skill-sequence, completion, obligations) and the grounded physics/complex families</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">predicted from the instruction by a classifier fitted on the training split</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Enable task/grounded formulas. A predicted <code style="white-space: normal; overflow-wrap: anywhere;">select_fruit</code> task makes its task concept true and requires <code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code>; the classifier supplies identity, not new formulas.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">the same rules as <code style="white-space: normal; overflow-wrap: anywhere;">learned</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">the episode&#x27;s task name</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Keep the same formulas; episode identity can select different task-guarded requirements. Correcting a prediction from <code style="white-space: normal; overflow-wrap: anywhere;">select_chemistry_tube</code> to <code style="white-space: normal; overflow-wrap: anywhere;">select_fruit</code> changes the required sequence from <code style="white-space: normal; overflow-wrap: anywhere;">pick, lift</code> to <code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code>.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">metadata</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">+ QA property observations (<code style="white-space: normal; overflow-wrap: anywhere;">--observation-mode metadata-assisted</code>)</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">the episode&#x27;s task name</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Keep the same formulas; added property facts can change which candidates satisfy existing grounded selection requirements.</td>
    </tr>
  </tbody>
</table>

Logical constraints express the rules; concept values express the task identity and observed facts to which those rules apply. Thus `learned` enables additional constraints, while `task` and `metadata` can change accepted plans through different facts without adding constraints. The current declaration counts and concrete examples are documented in [VLABenchLogicaConstraints.md](VLABenchLogicaConstraints.md#ablation-levels).

EAI has its own activation settings and inventory: see [EAI logical constraints](../../EmbodiedAgentInterface/docs/EAILogicalConstraints.md). Its declarative action effects and goal evaluation do not change the VLABench levels or counts below.

Joint checkpoints now record logical-knowledge provenance separately from vocabulary/schema compatibility. Activation and task-identity-source changes remain visible without changing these ablation counts; explicit held-out knowledge evaluation is reported as an ablation. Historical checkpoints without a manifest are marked unrecorded. See [Joint logical constraints](../../JointEmbodiedAgentInterface/docs/JointLogicalConstraints.md#knowledge-provenance-and-checkpoint-behavior).

Actual DomiKnowS formula examples for every rule family, including each task/interface variant and grounded family, are provided in the [constraint inventory](VLABenchLogicaConstraints.md#reading-the-formula-examples).

The `no DFA` rows decode the same checkpoint with the grammar removed. Free decoding makes the champion emit the end label
as its first label on almost every episode, so its plan is empty (Macro 0.2). The second row blocks that one step to get
a non-empty output; it is still ungrammatical and is the more informative of the two. Neither is a fair baseline for a
model trained without the grammar. The frozen VLM in the standard protocol is a different system and is not comparable
with these numbers.

## Executable knowledge and source map

The declaration refactor keeps the levels and the historical results below.
It does not constitute a new evaluation of the checkpoint. The results remain
measurements of the code revisions identified in Setup; rerun the ablation
before attributing scores to this refactor.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 17%;">
    <col style="width: 43%;">
    <col style="width: 40%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="17%" style="width: 17%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Knowledge</th>
      <th scope="col" width="43%" style="width: 43%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Logical declaration</th>
      <th scope="col" width="40%" style="width: 40%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Observations supplied to it</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">graph.py</code>: EOS existence/closure, token/operation limits, one-hot progress concepts, legal starts, skill patterns and ordered/optional arguments</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">token labels and request-specific continuation roots</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Generic</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">world_graph.py</code>, <code style="white-space: normal; overflow-wrap: anywhere;">_declare_generic_rules</code>: exactly one skill, each skill&#x27;s required/forbidden roles, usable argument values, declared subtask membership, distinct subject/destination, controlled object reuse</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">skill labels, role presence, pointer resolution, subject equality and pick history</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Scene</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">world_graph.py</code>, <code style="white-space: normal; overflow-wrap: anywhere;">_declare_scene_rules</code>: pointers resolve, button targets require press and vice versa, picks exclude fixtures, destinations match supplied requirements, enclosed destinations require opening when access is known</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">scene classifications, destination equality, opened entities and explicit open/closed observations</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Task sequences</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">knowledge/task_lcs.py</code>, <code style="white-space: normal; overflow-wrap: anywhere;">_declare_task_sequence</code>: task/interface and operation position imply the declared skill; the final position completes the sequence</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">task identity, interface, position, skill-choice/final phase; explicit requests override defaults</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Task obligations</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">knowledge/task_lcs.py</code>, <code style="white-space: normal; overflow-wrap: anywhere;">_declare_obligations</code> and <code style="white-space: normal; overflow-wrap: anywhere;">_declare_requested_rules</code>: minimum length, all required history skills, explicit sequence/cycle choices, initial target and request completion</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">operation counts/history and explicitly supplied request arrays/targets</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grounded families</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">domiknows/generation/grounded.py</code>, <code style="white-space: normal; overflow-wrap: anywhere;">declare_predicates</code>: each known required fact must hold; holding/capacity formulas expose their Boolean conditions</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">separate requirements for capabilities, preconditions, compatibility, destinations, milestone progress, goals and preservation; raw hand/occupancy/capacity facts</td>
    </tr>
  </tbody>
</table>

The generic schema already checks the validity of supplied argument values,
including scene membership, as it did in the measured implementation. The scene
pointer rule is therefore an additional explicit check, not the first source
of pointer validity. The scene open-before-placement rule retains the legacy
closed-world fallback when no planning facts are supplied; with explicit facts,
unknown open/closed status is not fabricated.

### Declarations and observation machinery

- `world_graph.py` contains the ontology and the named generic/scene rules.
  `world_domain.py` holds stable vocabulary/task contracts and wire-plan parsing;
  `world_runtime.py` constructs DataNodes and invokes verification. Existing
  imports through `world_graph` remain supported.
- `knowledge/tables.py` contains the declared task sequences and obligations;
  `knowledge/task_lcs.py` turns them into task concepts and LCs. Its observation
  adapter supplies identity, position and history, rather than reimplementing
  the declared sequence rule in Python.
- `domiknows/generation/grounded.py` contains the grounded graph declarations.
  `grounded_facts.py` parses supplied facts, compares measured values, applies
  declared symbolic effects and retains individual requirement observations.
  These computations are input grounding, not an alternate rule switchboard.
- `semantics.py` supplies primitive comparisons and state observations;
  `constraint_compiler.py` binds their concept identities to the generic LC
  compiler. Default world rules no longer read composite flags such as
  `is_self_placement` or `unopened_container_placement`; those flags remain
  available to existing custom rules.

VLABench materializes a separate requirement node for each required fact and
emits the same facts as decoding events. For example, a known false capability
violates `grounded__affordance` even when another capability is unknown. An
unknown requirement has `known=False`; it is not asserted satisfied, and the
existing grounding report records incomplete evidence. The old aggregate
`applicable/satisfied` observations remain available to custom rules and the
EAI adapter, but do not determine VLABench's default detailed enforcement.

The current default world graph has **86 logical constraints**: 24 generic, 6 scene,
and 56 task constraints (including the eight grounded families). With the five
grammar constraints, the total is **91**. Counts refer to complete logical
constraints, not operators within their formulas. Grounded per-rule satisfaction percentages now average requirement
nodes rather than operation nodes, so partial diagnostic percentages can differ.
The reward and validity gate remain unchanged: every active rule must be fully
satisfied; a partial result still fails the gate.

### Subtask progress and diagnostic changes

The transition rule now uses prefix progress projected from the same generation formulas, rather than a whole-plan validation flag. Each operation advances the compiled state; earlier valid transitions are not marked invalid because of a later bad transition or an unrelated pointer error. Decoding supplies the same progress observations instead of setting `allowed_transition=True` unconditionally.

Two additional generic constraints require a legal first operation and a complete final subtask, including explicit empty-plan permission. Thus the current cumulative counts are grammar **5**, generic **29**, scene **35**, and learned/task/metadata **91**. This implementation change has not rerun the historical checkpoint scores below. The added checks improve direct world verification of invalid starts and incomplete/single-operation/empty plans; a fully satisfied active constraint set is still required for hard validity. Partial diagnostic scores can differ because unrelated argument errors no longer invalidate the pattern constraint.

`pattern_runtime.py` caches the structural projection. Bound-policy verification reuses that policy's progress DFA, preserving its continuation language and declared limits. The projection reads the actual graph formulas and relation paths; it does not maintain a separate table of allowed skill pairs.

### Tier activation contract

`knowledge/__init__.py` remains the single source for level ordering and group
activation: `grammar` has no world group; `generic` adds generic rules; `scene`
adds scene rules; `learned`, `task`, and `metadata` activate all three groups.
`learned` obtains task identity from the classifier, while `task` and `metadata`
use episode identity. Metadata assistance changes observations, not LC groups.
The no-DFA modes bypass grammar and world enforcement entirely.

All declared groups, including the eight grounded families, exist even when
inactive. `set_knowledge_level` switches their `active` flags in place; recompile
the DFA afterward. Building at a low level and enabling task knowledge must
activate the same rules as building directly at task level. Request/task facts
are not consulted below `learned`.

The declaration refactor is validated with independently specified semantic
counterexamples, edits to actual formulas, tier-toggle equivalence, preservation
of unknown observations, and agreement between DataNode verification and
incremental decoding. It does not add observations inferred from gold plans or
rerun the empirical score tables below.

## Setup

- Checkpoint `joint_stage2_best.pt` (Stage 2, epoch 4), 450 test episodes, seed 42, original stored labels,
  no generation failures at any level. Code: commit `4c04f1e1` (the `learned` and `grammar` rows were run on later commits that only add those levels), run on gpu2 (one GPU per level).
- Command (per level; `metadata` uses `--observation-mode metadata-assisted`):

```bash
python -u -m test_regr.JointEmbodiedAgentInterface.eval_planning \
  --checkpoint test_regr/JointEmbodiedAgentInterface/checkpoints/joint_stage2_best.pt \
  --planning-dir test_regr/VLABenchAgentInterface/data/planning \
  --split test --seed 42 --device cuda:0 --allow-planning-schema-migration \
  --observation-mode visual --knowledge <grammar|generic|scene|task> \
  --output test_regr/JointEmbodiedAgentInterface/results/knowledge_ablation/<level>.json
```

The `learned` run adds `--knowledge learned --task-classifier task_classifier.pt` (see below). Raw reports: `test_regr/JointEmbodiedAgentInterface/results/knowledge_ablation/<level>.json`.

## Results (planning progress score, 0-100)

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 24%;">
    <col style="width: 13%;">
    <col style="width: 13%;">
    <col style="width: 13%;">
    <col style="width: 13%;">
    <col style="width: 12%;">
    <col style="width: 12%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="24%" style="width: 24%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Level</th>
      <th scope="col" width="13%" style="width: 13%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Macro</th>
      <th scope="col" width="13%" style="width: 13%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Overall</th>
      <th scope="col" width="13%" style="width: 13%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">SR</th>
      <th scope="col" width="13%" style="width: 13%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">PR</th>
      <th scope="col" width="12%" style="width: 12%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">SPR</th>
      <th scope="col" width="12%" style="width: 12%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">PM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">no DFA</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0.3</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0.0</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">no DFA, end withheld</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">26.6</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">28.4</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">57.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">8.6</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">16.5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1.8</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">grammar</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">60.9</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">64.3</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">89.4</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">51.1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">55.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">25.5</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">generic</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">66.9</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">70.8</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">89.4</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">62.4</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">63.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">37.7</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">scene</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">67.9</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">71.8</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">89.4</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">63.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">65.5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">39.6</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">learned</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">77.1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">78.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">98.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">66.6</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">70.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">50.8</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">task</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">77.1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">78.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">98.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">66.6</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">70.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">50.8</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">metadata</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">81.5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">81.5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">98.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">72.4</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">75.9</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">56.6</td>
    </tr>
  </tbody>
</table>

Per-dimension score:

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 22%;">
    <col style="width: 13%;">
    <col style="width: 13%;">
    <col style="width: 13%;">
    <col style="width: 13%;">
    <col style="width: 13%;">
    <col style="width: 13%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="22%" style="width: 22%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Level</th>
      <th scope="col" width="13%" style="width: 13%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">M&amp;T</th>
      <th scope="col" width="13%" style="width: 13%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Spatial</th>
      <th scope="col" width="13%" style="width: 13%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">ComSense</th>
      <th scope="col" width="13%" style="width: 13%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Semantic</th>
      <th scope="col" width="13%" style="width: 13%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">PhysLaw</th>
      <th scope="col" width="13%" style="width: 13%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Complex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">no DFA</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1.1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0.0</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">no DFA, end withheld</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">37.5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">41.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">37.5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">27.1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">16.3</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">grammar</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">74.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">63.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">75.8</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">75.8</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">37.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">37.9</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">generic</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">85.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">71.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">81.6</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">82.3</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">37.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">43.6</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">scene</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">85.4</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">72.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">81.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">85.5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">37.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">44.6</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">learned</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">78.9</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">72.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">81.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">85.5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">57.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">87.4</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">task</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">78.9</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">72.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">81.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">85.5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">57.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">87.4</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">metadata</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">78.9</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">72.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">81.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">85.5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">83.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">87.4</td>
    </tr>
  </tbody>
</table>

## Learned task concept

`learned` replaces the episode's task name by a classifier over the instruction (hashed word and bigram bag,
linear softmax head, `knowledge/task_classifier.py`). It is fitted on the training split only (3,600 episodes,
45 task names, the episode ids are stored with the weights and evaluation refuses an overlapping split).
Task-name accuracy is 95.1% on the 450 test episodes (94.2% on validation). At this level the declared
sequence, task targets and the facts grounded from the task name are rebuilt from the prediction.

```bash
python -m test_regr.VLABenchAgentInterface.main fit-task-classifier   --planning-dir test_regr/VLABenchAgentInterface/data/planning --out task_classifier.pt
python -u -m test_regr.JointEmbodiedAgentInterface.eval_planning ... --knowledge learned   --task-classifier task_classifier.pt
```

The `learned` scores equal the `task` scores to every digit. Of the 212 test episodes whose task has declared
knowledge, the rebuilt requirements differ from the task-level ones in 66: 60 are QA episodes where the same
sequence is now supplied by the declared LCs instead of the dataset (same constraint), and 6 are
`insert_bloom_flower` episodes predicted as `insert_flower`. Those 6 changed no score; I did not check
whether their generated plans are identical. The misclassifications otherwise fall on task variants
(`*_semantic`, `*_common_sense`) that have no declared rules. So on the seen tasks the task name adds
nothing the instruction does not already give; the unseen-task test (Phase D) is what separates the two.

## Reading

- **The grammar is what makes the champion usable:** without it the plan is empty (Macro 0.2), and with the end label
  withheld for one step it reaches 26.6 against 60.9 with the grammar. Skill recall falls from 89.4 to 57.7 and graph
  match from 25.5 to 1.8. This measures dependence of this checkpoint on its decoding mask (it was trained and
  selected under the mask), not what an unconstrained model could do.
- **The generic rules are worth +6.0 Macro over the grammar alone** (60.9 to 66.9): parameter recall 51 to 62 and
  graph match 26 to 38, with gains in every dimension except PhysLaw. Skill recall is unchanged (89.4), so the gain is in
  arguments and graph structure, not in which skills are chosen.
- **Scene knowledge is worth about 1 point of Macro** (66.9 to 67.9); only Semantic moves (+3.2).
- **The task tier is the large step, +9.2 Macro**, and it comes from two dimensions: Complex 44.6 to 87.4
  (the obligations and sequence LCs) and PhysLaw 37.7 to 57.0 (grounded QA families). Skill recall rises
  from 89 to 98.
- **Metadata adds +4.3 Macro**, all of it PhysLaw (57.0 to 83.0), the dimension whose answer needs measured
  property values.
- **Task knowledge hurts M&T** (85.4 to 78.9): skill recall reaches 100 but parameter recall falls from 83 to 65.
  The cause was not investigated; the declared M&T sequence may force a skill order that the model's
  parameters were not trained to follow.
- Spatial, ComSense and Semantic are unchanged by the task tier, so for those dimensions the model's own
  predictions plus generic rules already match what the task LCs enforce.

## Caveats

- The checkpoint was trained with task-tier inputs, so the `generic` and `scene` rows mix the removal of
  knowledge with a train/test mismatch. They are a lower bound for a model trained at that level; Phase D
  (retrain per level, hold out whole tasks) removes the confound.
- Single seed, one checkpoint; differences of about 1 point (generic vs scene) are not established.
- `task` gives Macro 77.1 here against the 75.1 recorded in `README_checkpoint.md`; that earlier figure came
  from an older code revision. This run is the reference for the knowledge comparison.
- The `metadata` level uses a different observation mode, so it is not a pure LC-group toggle.
