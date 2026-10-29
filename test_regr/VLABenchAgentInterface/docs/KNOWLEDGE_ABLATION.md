# Knowledge ablation of the champion Joint planner

**Default-domain expansion, 29 October 2026:** the internal domain now has eleven
skills, articulated-object terminal goals and benchmark-compatible poker rules.
The `world` native profile is the general default; explicitly selected official
`benchmark`/`legacy` profiles remain nine-skill languages. Old nine-skill
checkpoints require retraining for the expanded domain. See
[Practical constraint additions](PRACTICAL_CONSTRAINTS.md). No recorded score below
measures these additions.

**Implementation update, 29 October 2026 (`2dab4db3`):** syntax validation is stricter,
explicit initial-observation visibility is a scene-level logical constraint, and
action-budget goal reachability is available as a separate search/decoding option.
The measured results below predate these additions and have not been rerun. See
[Syntax, visibility and reachability](#syntax-visibility-and-reachability) for the
current behavior and the controls needed for a new ablation.

What each knowledge tier of the world graph contributes when the same checkpoint is evaluated on the same
450 test episodes. Task knowledge is expressed as graph concepts and logical constraints
(`knowledge/`, see the README source code map); `--knowledge` switches which constraint groups are active; `--task-identity` (`oracle` or `predicted`) and `--observation-mode` (`visual` or `metadata-assisted`) are separate settings of the facts those rules read. The former levels `learned` and `metadata` remain aliases (`task` with a predicted or an oracle identity).
`causal` is a `--knowledge` level on every path: it enables only **3 DomiKnowS logical constraints**, the world graph's `causal` group for the Joint and internal graph decoders and the equivalent official causal graph for `--protocol official` (where `--constraints causal` remains an alias). It is the first cumulative tier above `grammar`, so `generic` and every higher level also keep the causal group active, and its result-table rows come from `eval_planning --knowledge causal`. Rows from `generic` up were measured before causal joined them (see Caveats).

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
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar only: the token syntax of a plan, the eleven internal skills (nine in the explicit official/legacy native profiles), any of the 64 entity ids, the end token, the operation budget and the language of allowed skill patterns; no constraint group</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">none</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Enable grammar formulas; an incomplete subtask cannot terminate.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">causal</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">+ Causal: three id-only DomiKnowS ifL constraints over the ids the plan writes: a destination is never the object picked in its subtask (anti-self-placement), a subtask never repeats a destination, an object that has served as a destination is never picked later; no entity table, so any of the 64 ids remains allowed</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">none</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Enable only the three causal formulas (<code style="white-space: normal; overflow-wrap: anywhere;">--knowledge causal</code>): the world graph&#x27;s <code style="white-space: normal; overflow-wrap: anywhere;">causal</code> group in the graph DFA, or the official causal graph under <code style="white-space: normal; overflow-wrap: anywhere;">--protocol official</code>. Placing the picked object on itself is rejected without any scene table; repeated picks stay legal.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">generic</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">+ Generic: skill schema, subtask patterns, pointer, anti-self-placement and repeated-pick rules</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">none</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Enable generic formulas; self-placement, for example, is rejected.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">scene</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">+ Scene: scene-grounded rules (buttons, fixtures, containers, open-before-place), plus explicit initial-observation visibility when supplied</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">none</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Enable scene formulas and evaluate scene facts; a static fixture cannot be picked. With explicit visibility evidence, an inventory entity unsupported by the initial observation cannot be referenced.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task</code>, <code style="white-space: normal; overflow-wrap: anywhere;">--task-identity predicted</code> (alias <code style="white-space: normal; overflow-wrap: anywhere;">learned</code>)</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">+ Task sequences, Task obligations, Complex tasks, Physics and Grounded families (rows of the source map below), with the task identity predicted</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">predicted from the instruction by a classifier fitted on the training split</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Enable task/grounded formulas. A predicted <code style="white-space: normal; overflow-wrap: anywhere;">select_fruit</code> task makes its task concept true and requires <code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code>; the classifier supplies identity, not new formulas.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task</code>, <code style="white-space: normal; overflow-wrap: anywhere;">--task-identity oracle</code> (default)</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">the same rules; only the task identity differs</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">the episode&#x27;s task name</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Keep the same formulas; episode identity can select different task-guarded requirements. Correcting a prediction from <code style="white-space: normal; overflow-wrap: anywhere;">select_chemistry_tube</code> to <code style="white-space: normal; overflow-wrap: anywhere;">select_fruit</code> changes the required sequence from <code style="white-space: normal; overflow-wrap: anywhere;">pick, lift</code> to <code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code>.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task</code>, oracle, <code style="white-space: normal; overflow-wrap: anywhere;">--observation-mode metadata-assisted</code> (alias <code style="white-space: normal; overflow-wrap: anywhere;">metadata</code>)</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">the same rules, + QA property observations (<code style="white-space: normal; overflow-wrap: anywhere;">--observation-mode metadata-assisted</code>): the Physics rules then see each option&#x27;s material, shape or bound reflectance</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">the episode&#x27;s task name</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Keep the same formulas; added property facts can change which candidates satisfy existing grounded selection requirements.</td>
    </tr>
  </tbody>
</table>

Logical constraints express the rules; concept values express the task identity and observed facts to which those rules apply. Thus `--knowledge task` enables additional constraints, while `--task-identity` and `--observation-mode` can change accepted plans through different facts without adding constraints. The current declaration counts and concrete examples are documented in [VLABenchLogicaConstraints.md](VLABenchLogicaConstraints.md#ablation-levels).

EAI has its own activation settings and inventory: see [EAI logical constraints](../../EmbodiedAgentInterface/docs/EAILogicalConstraints.md). Its declarative action effects and goal evaluation do not change the VLABench levels or counts below.

Joint checkpoints now record logical-knowledge provenance separately from vocabulary/schema compatibility. Activation and task-identity-source changes remain visible without changing these ablation counts; explicit held-out knowledge evaluation is reported as an ablation. Historical checkpoints without a manifest are marked unrecorded. See [Joint logical constraints](../../JointEmbodiedAgentInterface/docs/JointLogicalConstraints.md#knowledge-provenance-and-checkpoint-behavior).

Actual DomiKnowS formula examples for every rule family, including each task/interface variant and grounded family, are provided in the [constraint inventory](VLABenchLogicaConstraints.md#reading-the-formula-examples).

The `no DFA` rows decode the same checkpoint with the grammar removed. Free decoding makes the champion emit the end label
as its first label on almost every episode, so its plan is empty (Macro 0.2). The second row blocks that one step to get
a non-empty output; it is still ungrammatical and is the more informative of the two. Neither is a fair baseline for a
model trained without the grammar. The frozen VLM in the standard protocol is a different system and is not comparable
with these numbers.

The causal declaration refactor expresses the three existing admission rules as logical constraints and preserves the recorded historical scores; it does not rerun those evaluations. Candidate skill concepts and prefix equality/history observations ground the formulas; changing or removing a formula changes acceptance after rebuilding the DFA. It adds no scene facts or task identity. See the [three complete causal formulas](VLABenchLogicaConstraints.md#causal-id-only-dfa-rules-3-logical-constraints).

## Executable knowledge and source map

The declaration refactor keeps the levels and the historical results below.
It does not constitute a new evaluation of the checkpoint. The results remain
measurements of the code revisions identified in Setup; rerun the ablation
before attributing scores to this refactor.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 14%;">
    <col style="width: 11%;">
    <col style="width: 42%;">
    <col style="width: 33%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="14%" style="width: 14%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Knowledge</th>
      <th scope="col" width="11%" style="width: 11%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">First active level</th>
      <th scope="col" width="42%" style="width: 42%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Logical declaration</th>
      <th scope="col" width="33%" style="width: 33%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Observations supplied to it</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">grammar</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">graph.py</code>: EOS existence/closure, token/operation limits, one-hot progress concepts, legal starts, skill patterns and ordered/optional arguments</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">token labels and request-specific continuation roots</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Causal</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">causal</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">world_graph.py</code>, <code style="white-space: normal; overflow-wrap: anywhere;">_declare_causal_rules</code>: the world graph&#x27;s <code style="white-space: normal; overflow-wrap: anywhere;">causal</code> group of three <code style="white-space: normal; overflow-wrap: anywhere;">ifL</code> rules for the Joint and internal graph decoders (anti-self-placement, distinct destinations within a subtask, no pick of a former destination). Under <code style="white-space: normal; overflow-wrap: anywhere;">--protocol official</code> the same three rules are <code style="white-space: normal; overflow-wrap: anywhere;">official/causal_graph.py</code>, compiled into <code style="white-space: normal; overflow-wrap: anywhere;">CausalPlanDFA</code>.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">the subject of the current subtask, its destinations so far and every earlier destination (<code style="white-space: normal; overflow-wrap: anywhere;">subject_equals_destination</code>, <code style="white-space: normal; overflow-wrap: anywhere;">destination_used_in_subtask</code>, <code style="white-space: normal; overflow-wrap: anywhere;">target_was_destination</code>; plan ids under the official protocol)</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Generic</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">generic</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">world_graph.py</code>, <code style="white-space: normal; overflow-wrap: anywhere;">_declare_generic_rules</code>: exactly one skill, each skill&#x27;s required/forbidden roles, usable argument values, no undeclared argument keys, declared subtask membership, distinct subject/destination, controlled object reuse</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">skill labels, role presence, pointer resolution, subject equality and pick history</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Scene</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">scene</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">world_graph.py</code>, <code style="white-space: normal; overflow-wrap: anywhere;">_declare_scene_rules</code>: pointers resolve, supplied visibility supports all referenced subjects/destinations, button targets require press and vice versa, picks exclude fixtures, destinations match supplied requirements, enclosed destinations require opening when access is known</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">scene classifications, explicit initial-observation support ids, destination equality, opened entities and explicit open/closed observations</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Task sequences</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">knowledge/task_lcs.py</code>, <code style="white-space: normal; overflow-wrap: anywhere;">_declare_task_sequence</code>: task/interface and operation position imply the declared skill; the final position completes the sequence</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">task identity, interface, position, skill-choice/final phase; explicit requests override defaults</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Task obligations</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">knowledge/task_lcs.py</code>, <code style="white-space: normal; overflow-wrap: anywhere;">_declare_obligations</code> and <code style="white-space: normal; overflow-wrap: anywhere;">_declare_requested_rules</code>: minimum length, all required history skills, explicit sequence/cycle choices, initial target and request completion</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">operation counts/history and explicitly supplied request arrays/targets</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grounded families</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">domiknows/generation/grounded.py</code>, <code style="white-space: normal; overflow-wrap: anywhere;">declare_predicates</code>: each known required fact must hold; holding/capacity formulas expose their Boolean conditions</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">separate requirements for capabilities, preconditions, compatibility, destinations, milestone progress, goals and preservation; raw hand/occupancy/capacity facts</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Complex tasks</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">knowledge/complex_lcs.py</code>, <code style="white-space: normal; overflow-wrap: anywhere;">declare_complex</code>: 85 <code style="white-space: normal; overflow-wrap: anywhere;">ifL</code> rules generated from <code style="white-space: normal; overflow-wrap: anywhere;">knowledge/tables.py</code> (recipes, scene roles, hammer chain): allowed pick subjects, destinations, step order from performed operations, and plan-end steps and goals, relayed by <code style="white-space: normal; overflow-wrap: anywhere;">ComplexMonitor</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">task, named recipe, unique role bindings, foods present, book years and direction, named reagents, performed operations, final facts</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Physics</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">knowledge/physics_lcs.py</code>, <code style="white-space: normal; overflow-wrap: anywhere;">declare_physics</code>: 82 <code style="white-space: normal; overflow-wrap: anywhere;">ifL</code> rules generated from <code style="white-space: normal; overflow-wrap: anywhere;">knowledge/tables.py</code> (property tiers, instruction phrases): press-option membership, max/min, below-water and rank rules per QA task, relayed into <code style="white-space: normal; overflow-wrap: anywhere;">grounded__affordance</code> by <code style="white-space: normal; overflow-wrap: anywhere;">PhysicsMonitor</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">task, instruction phrases, option count, each option&#x27;s material, shape or reflectance, the pressed option. These attributes come from scene metadata, so they exist only with metadata-assisted observations (<code style="white-space: normal; overflow-wrap: anywhere;">metadata</code>); in visual mode the VLM chooses and no physics rule fires</td>
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
- `articulation.py` declares four terminal task formulas and binds explicit joint
  targets and measured state. `knowledge/poker_lcs.py` declares eleven poker
  selection/completion formulas, fed by ten auxiliary compiled classification
  queries. Both relay into the existing grounded goal family at `task` level.

VLABench materializes a separate requirement node for each required fact and
emits the same facts as decoding events. For example, a known false capability
violates `grounded__affordance` even when another capability is unknown. An
unknown requirement has `known=False`; it is not asserted satisfied, and the
existing grounding report records incomplete evidence. The old aggregate
`applicable/satisfied` observations remain available to custom rules and the
EAI adapter, but do not determine VLABench's default detailed enforcement.

The current default world graph has **91 logical constraints**: 3 causal, 28 generic, 7 scene,
and 53 task constraints (including the eight grounded families). With the five
grammar constraints, the total is **96**; the causal group is active from `causal` upwards. Counts refer to complete logical
constraints, not operators within their formulas. Grounded per-rule satisfaction percentages now average requirement
nodes rather than operation nodes, so partial diagnostic percentages can differ.
The reward and validity gate remain unchanged: every active rule must be fully
satisfied; a partial result still fails the gate.

### Causal logical constraints (official protocol)

`official/causal_graph.py` declares **3 constraints**, all using `ifL` as the outermost operator. `CausalPlanDFA` compiles the actual formulas with `compile_predicate_constraints` and evaluates them while pruning partial ids and when closing each operation. Names do not supply policy semantics. Removing, disabling, or editing a formula changes acceptance after rebuilding the DFA; existing compiled DFAs keep their previous snapshot.

`pick`, `place`, `insert`, `pour`, and `push` denote skill concepts under the operation concept. The string `"operation"` binds the candidate operation consistently throughout each formula. The candidate skill and three Boolean observations ground the formulas before the candidate executes: `target_is_subject` compares the target id with the current picked subject; `target_used_in_subtask` checks the current subtask's destination history; `target_was_destination` checks the entire earlier plan's destination history. Pick/press start a new subtask and reset local history; earlier destinations remain in plan-wide history. These observations require no entity names, scene table, task identity, or gold plan.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup><col style="width: 25%;"><col style="width: 10%;"><col style="width: 38%;"><col style="width: 27%;"></colgroup>
<thead><tr><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Rule family</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Count at causal</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Example and effect</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Definition</th></tr></thead>
<tbody>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>anti_self_placement</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>pick 2, place 2</code>: Reject a destination equal to the current picked subject. The guard covers place, insert, pour and push.</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../official/causal_graph.py#L52">official/causal_graph.py:52</a></td></tr>
<tr data-constraint-example="true"><td colspan="4" style="padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
    orL(place(&quot;operation&quot;), insert(&quot;operation&quot;), pour(&quot;operation&quot;), push(&quot;operation&quot;)),
    notL(target_is_subject(&quot;operation&quot;)),
    name=&quot;anti_self_placement&quot;,
)</code></pre></td></tr>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>distinct_subtask_destinations</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>pick 2, pour 3, place 3</code>: Reject reuse of destination 3 within one subtask. A later subtask may use 3 again.</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../official/causal_graph.py#L58">official/causal_graph.py:58</a></td></tr>
<tr data-constraint-example="true"><td colspan="4" style="padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
    orL(place(&quot;operation&quot;), insert(&quot;operation&quot;), pour(&quot;operation&quot;), push(&quot;operation&quot;)),
    notL(target_used_in_subtask(&quot;operation&quot;)),
    name=&quot;distinct_subtask_destinations&quot;,
)</code></pre></td></tr>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>destinations_stay_put</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>pick 2, place 3, pick 3, place 4</code>: Reject picking 3 after it served as a destination in an earlier operation.</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../official/causal_graph.py#L64">official/causal_graph.py:64</a></td></tr>
<tr data-constraint-example="true"><td colspan="4" style="padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
    pick(&quot;operation&quot;),
    notL(target_was_destination(&quot;operation&quot;)),
    name=&quot;destinations_stay_put&quot;,
)</code></pre></td></tr>
</tbody>
</table>

The official causal setting therefore enables **3 logical constraints**; its native JSON grammar adds no DomiKnowS declarations. It does not use the five grammar formulas from `graph.py` or activate the generic/scene/task world groups. The grammar/world path declares the same three rules as its `causal` group (96 declarations, all enabled at `task`); with the official causal graph and the six episode-decoding formulas the inventory is **105 declarations**, excluding the separate knowledge graphs. Including physics (82), complex (85), articulation (4) and poker (11) gives **287 declarations across paths**, not 287 constraints active together. The sixth episode-decoding formula enforces explicit initial-observation visibility. Each complete formula counts once; nested operators are excluded. Full formulas and observation definitions are in the [causal constraint inventory](VLABenchLogicaConstraints.md#causal-id-only-dfa-rules-3-logical-constraints).

The DFA still accumulates prefix observations and handles JSON bytes, pointer bounds, EOS, and operation limits. The compiled formulas decide the three causal admission rules. This refactor changes where the rules are expressed, and adds no new task or scene knowledge. Historical result tables below are unchanged.

### Subtask progress and diagnostic changes

The transition rule now uses prefix progress projected from the same generation formulas, rather than a whole-plan validation flag. Each operation advances the compiled state; earlier valid transitions are not marked invalid because of a later bad transition or an unrelated pointer error. Decoding supplies the same progress observations instead of setting `allowed_transition=True` unconditionally.

Two additional generic constraints require a legal first operation and a complete final subtask, including explicit empty-plan permission. With the new scene visibility formula, the current cumulative counts are grammar **5**, causal **8**, generic **36**, scene **43**, and task **96**, plus the **82** physics constraints of `knowledge/physics_lcs.py` and the **85** complex-task constraints of `knowledge/complex_lcs.py` that the task group enables, plus four articulation and eleven poker formulas (**278** total). These implementation changes have not rerun the historical checkpoint scores below. The progress checks improve direct world verification of invalid starts and incomplete/single-operation/empty plans; a fully satisfied active constraint set is still required for hard validity. Partial diagnostic scores can differ because unrelated argument errors no longer invalidate the pattern constraint.

`pattern_runtime.py` caches the structural projection. Bound-policy verification reuses that policy's progress DFA, preserving its continuation language and declared limits. The projection reads the actual graph formulas and relation paths; it does not maintain a separate table of allowed skill pairs.

### Tier activation contract

`knowledge/__init__.py` remains the single source for level ordering and group
activation: `grammar` has no world group; `causal` adds the causal rules; `generic`
adds generic rules; `scene` adds scene rules; `task` activates the task group as well.
The task identity is a separate setting (`--task-identity`, stored with the level in
`KnowledgeState`): `predicted` reads the classifier's task, `oracle` the episode's.
Metadata assistance (`--observation-mode`) changes observations, not LC groups.
The no-DFA modes bypass grammar and world enforcement entirely.

All declared groups, including the eight grounded families, exist even when
inactive. `set_knowledge_level` switches their `active` flags in place; recompile
the DFA afterward. Building at a low level and enabling task knowledge must
activate the same rules as building directly at task level. Request/task facts
are not consulted below `task`, except for scene observations such as explicit
`visible_entities`, which are independent of task identity and enforced from `scene`.

### Syntax, visibility and reachability

Commit `2dab4db3` adds the following behavior without introducing a new knowledge
tier or changing the recorded result tables:

- **Syntax:** the generic `operation_schema_is_valid` formula now rejects
  undeclared argument keys. Canonical parsing rejects duplicate JSON members and
  non-string skill names, and argument validation rejects unsupported value
  types. Symbolic token encoding checks syntax before serialization so invalid
  arguments cannot disappear. Parser/encoder validation is shared preprocessing,
  separate from the knowledge-tier switch. Native decoding retains its exact JSON
  grammar. These fixes add no logical constraint declarations.
- **Visibility:** `observation_supported_references` adds one world scene formula
  and one separate native episode-decoding formula. Every explicitly referenced
  subject/destination, including optional roles, must belong to `visible_entities`
  when evidence is supplied. An entity inventory, instruction candidate or
  `--scene-ids` largest-tag bound does not establish visibility. `None` means
  unavailable evidence; an empty collection establishes an empty support set.
  The support set stays fixed across the plan. Initial-view segmentation masks
  can supply it through explicit label-to-entity bindings; no gold plans or
  target annotations supply support.
- **Reachability:** `policy.action_reachability()` performs cached bounded search
  over complete actions using the existing compiled grammar, world LCs and state
  effects. Native generation can opt in through
  `planner.generate(..., reachability_policy=policy)`. This is a separate decoding
  control, not an additional LC or an automatic change to the Joint ablation
  command below. Completion must satisfy the bound policy's active goals and
  obligations; no supplied goal means no independent claim of task success.
  Strict mode distinguishes missing required grounding (`None`) from proven
  unreachability (`False`); search-limit exhaustion raises an error. Existing HMM
  guidance still ranks the feasible set and does not add CLAMP's separate
  zero-HMM-mass exclusion.

For a new ablation, hold the checkpoint, episodes, seed, action/token budgets,
entity inventory, task identity and property observations fixed. To isolate the
visibility formula, supply the same initial-observation support in both runs and
toggle only that formula before rebuilding the policy. To isolate reachability,
keep the same active LCs and episode facts and vary only the search/decoding
option; record strictness, the expansion cap, missing-grounding outcomes and
search-limit failures separately. Neither feature has a new score row in this
document. Existing official runs receive no new support set automatically, and
`--observation-mode visual` alone does not populate `visible_entities`.

See [Syntax and initial-observation visibility](SYNTAX_AND_VISIBILITY.md) and
[Action-budget goal reachability](GOAL_REACHABILITY.md) for APIs and examples.

`causal` is the second level of `knowledge/__init__.py` (`grammar < causal < generic < scene < task`); `group_active` keeps the `causal` group active at every level from `causal` up. `--knowledge causal` selects it in
`evaluate-planner` (graph and official protocols), `eval_planning` and `train-agent`.

The declaration refactor is validated with independently specified semantic
counterexamples, edits to actual formulas, tier-toggle equivalence, preservation
of unknown observations, and agreement between DataNode verification and
incremental decoding. It does not add observations inferred from gold plans or
rerun the empirical score tables below.

## Setup

- Checkpoint `joint_stage2_best.pt` (Stage 2, epoch 4), 450 test episodes, seed 42, original stored labels,
  no generation failures at any level. Code: commit `4c04f1e1` (the predicted-identity and `grammar` rows were run on later commits that only add those settings; the `causal` row on `54da7bba`, 28 October 2026, where a rerun of `grammar` and the metadata-assisted row reproduced them), run on gpu2 (one GPU per level).
- Command (per level; the metadata-assisted row uses `--observation-mode metadata-assisted`):

```bash
python -u -m test_regr.JointEmbodiedAgentInterface.eval_planning \
  --checkpoint test_regr/JointEmbodiedAgentInterface/checkpoints/joint_stage2_best.pt \
  --planning-dir test_regr/VLABenchAgentInterface/data/planning \
  --split test --seed 42 --device cuda:0 --allow-planning-schema-migration \
  --observation-mode visual --knowledge <grammar|causal|generic|scene|task> \
  --output test_regr/JointEmbodiedAgentInterface/results/knowledge_ablation/<level>.json
```

The predicted-identity run adds `--task-identity predicted --task-classifier task_classifier.pt` (formerly `--knowledge learned`; see below). Raw reports: `test_regr/JointEmbodiedAgentInterface/results/knowledge_ablation/<level>.json`.

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
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">causal</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">62.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">65.5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">89.4</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">51.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">55.8</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">37.8</td>
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
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">task, predicted identity</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">77.1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">78.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">98.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">66.6</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">70.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">50.8</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">task, oracle identity</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">77.1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">78.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">98.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">66.6</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">70.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">50.8</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">task, oracle, metadata-assisted</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">81.4</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">81.1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">98.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">71.6</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">75.6</td>
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
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">causal</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">76.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">64.3</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">76.6</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">77.9</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">37.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">38.5</td>
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
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">task, predicted identity</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">78.9</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">72.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">81.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">85.5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">57.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">87.4</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">task, oracle identity</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">78.9</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">72.2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">81.7</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">85.5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">57.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">87.4</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">task, oracle, metadata-assisted</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">77.8</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">75.1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">80.4</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">84.6</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">83.0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">87.4</td>
    </tr>
  </tbody>
</table>

## Predicted task identity

`--task-identity predicted` replaces the episode's task name by a classifier over the instruction (hashed word and bigram bag,
linear softmax head, `knowledge/task_classifier.py`). It is fitted on the training split only (3,600 episodes,
45 task names, the episode ids are stored with the weights and evaluation refuses an overlapping split).
Task-name accuracy is 95.1% on the 450 test episodes (94.2% on validation). With this identity the declared
sequence, task targets and the facts grounded from the task name are rebuilt from the prediction.

```bash
python -m test_regr.VLABenchAgentInterface.main fit-task-classifier   --planning-dir test_regr/VLABenchAgentInterface/data/planning --out task_classifier.pt
python -u -m test_regr.JointEmbodiedAgentInterface.eval_planning ... --knowledge task --task-identity predicted --task-classifier task_classifier.pt
```

The predicted-identity scores equal the oracle-identity scores to every digit. Of the 212 test episodes whose task has declared
knowledge, the rebuilt requirements differ from the oracle ones in 66: 60 are QA episodes where the same
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
- **Causal knowledge alone is worth +1.1 Macro over the grammar** (60.9 to 62.0), about a sixth of what the generic
  rules add. Almost all of it is graph match: PM rises from 25.5 to 37.8 while skill recall is unchanged and parameter
  recall flat (51.1 to 51.0), so the three rules fix how subtasks connect rather than which skills or objects are chosen.
  M&T, Spatial, ComSense, Semantic and Complex gain 0.6 to 2.1; PhysLaw is unchanged.
- **Scene knowledge is worth about 1 point of Macro** (66.9 to 67.9); only Semantic moves (+3.2).
- **In the standard protocol causal knowledge does not help the frozen model**, measured with Qwen3-VL-8B and upstream
  total_score (another scale from the tables above): 38.20 against 38.24 for the grammar DFA alone over the 3,700
  mapped episodes. It changes 134 of 4,500 plans (16 worse, 1 better). The rules catch real errors (91 self-placements,
  34 repeated destinations, 17 picks of a former destination), but a refused id is completed to a two-digit id that is
  not in the scene (34, 44), so the repair does not reach a real object. With `--scene-ids` (the model reads the largest
  tag number, 6.7 on average, and ids are bounded by it) causal scores 38.26 against 38.24 for grammar with the same
  bound: 134 plans change, 22 better and 18 worse. The bound alone changes no grammar plan. The rules are about neutral.
  With the label-free HMM lookahead (weight 0.5) on the LC-compiled DFA, causal scores 37.96 against 38.02 for grammar
  (4 better, 22 worse) and 38.01 against 38.02 with `--scene-ids`; grammar + `--scene-ids` + HMM reproduces the earlier
  grammar + HMM plans exactly.
- **The task tier is the large step, +9.2 Macro**, and it comes from two dimensions: Complex 44.6 to 87.4
  (the obligations and sequence LCs) and PhysLaw 37.7 to 57.0 (grounded QA families). Skill recall rises
  from 89 to 98.
- **Metadata adds about +4.3 Macro**, almost all of it PhysLaw (57.0 to 83.0), the dimension whose answer needs
  measured property values. The metadata-assisted row was rerun on `1bd53605`, where the task level also enforces the causal
  rules and the physics and complex knowledge are logical constraints: 81.4 against 81.5 before causal joined the
  levels (parameter recall 72.4 to 71.6; Spatial +2.9, ComSense, M&T and Semantic about -1). The same run on
  `e358e881`, before the physics/complex conversion, gives identical metrics to full precision.
- **Task knowledge hurts M&T** (85.4 to 78.9): skill recall reaches 100 but parameter recall falls from 83 to 65.
  The cause was not investigated; the declared M&T sequence may force a skill order that the model's
  parameters were not trained to follow.
- Spatial, ComSense and Semantic are unchanged by the task tier, so for those dimensions the model's own
  predictions plus generic rules already match what the task LCs enforce.

## Caveats

- The eleven-skill default and new poker/articulation rules have not been evaluated
  with retrained checkpoints. Current counts exclude the ten auxiliary poker
  classification queries and relay applies/decided views. Benchmark poker behavior
  follows the pinned upstream helper, including its nonstandard ties and ace-low
  flush classification; no standard-poker rules are substituted.

- The recorded scores have not been rerun on `2dab4db3`. They do not measure the
  syntax fixes, explicit visibility constraint or action-budget reachability.
  The historical `--scene-ids` results use an ID bound, not the new support-set
  constraint. Current declaration counts describe the updated implementation.

- The checkpoint was trained with task-tier inputs, so the `generic` and `scene` rows mix the removal of
  knowledge with a train/test mismatch. They are a lower bound for a model trained at that level; Phase D
  (retrain per level, hold out whole tasks) removes the confound.
- Single seed, one checkpoint; differences of about 1 point (generic vs scene) are not established.
- `task` gives Macro 77.1 here against the 75.1 recorded in `README_checkpoint.md`; that earlier figure came
  from an older code revision. This run is the reference for the knowledge comparison.
- The metadata-assisted row changes the observation mode, not the active rules.
- The `generic`, `scene` and both `task` identity rows were measured while the causal group was not part of those
  levels; since causal became cumulative they also enforce its three rules, and they have not been rerun. The
  metadata-assisted row is the rerun on `1bd53605` (with causal). The `causal` row ran on `54da7bba`, where `grammar` and the
  pre-fold metadata-assisted run reproduced their earlier rows. None of the 4,500 reference plans
  breaks its three rules, but the two rules that `generic` lacks were chosen after that check, so like the press
  question they encode benchmark knowledge.
