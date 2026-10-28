# VLABench logical constraints

Inventory of the current local declarations, checked on **2026-10-27**. This document covers both meanings of a constraint class: the DomiKnowS operator class and the named knowledge/rule family built from those operators. It complements [KNOWLEDGE_ABLATION.md](KNOWLEDGE_ABLATION.md); its counts describe declarations, not new ablation scores.

## Scope and counting

The baseline is one standalone `build_vlabench_world_graph(knowledge_level="task")` and one `create_planner_generation_graph(world)` with the default task tables, no excluded tasks or custom constraint builders, 64 entity pointers, `max_operations=12`, and no request-specific language override. Grammar/world counts refer to that build. The official-protocol causal graph is counted separately: one default `build_causal_graph()` declares three additional formulas and one default `build_decoding_graph()` five more. Together they contain 102 declarations, while the enabled count depends on the selected protocol and level.

A **logical constraint** is one complete declared formula enforced as a rule. For example, `ifL(A, andL(B, C))` counts as **one logical constraint**. Operators within that formula are not counted separately. Concept calls, relation declarations, DFA states, auxiliary progress concepts, input facts, and decoder transitions do not add logical constraints.

A **grounded instance** evaluates a logical constraint for a token, operation, transition, or requirement node. Its count depends on the plan and observations; a logical constraint declared once can be evaluated many times.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 48%;">
    <col style="width: 16%;">
    <col style="width: 36%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="48%" style="width: 48%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Declaration group</th>
      <th scope="col" width="16%" style="width: 16%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Logical constraints</th>
      <th scope="col" width="36%" style="width: 36%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">First active level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Generation grammar</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">grammar</code></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Causal (id-only) DFA rules</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">3 (separate official-protocol graph)</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">causal</code>, standard protocol only</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Episode decoding knowledge</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">5 (separate decoding graph)</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">--press-question</code>, <code style="white-space: normal; overflow-wrap: anywhere;">--scene-ids</code>, <code style="white-space: normal; overflow-wrap: anywhere;">--pattern-first</code>; instruction object candidates of the internal frozen-VLM DFA</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Causal world knowledge</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">3</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">causal</code> only (a branch beside <code style="white-space: normal; overflow-wrap: anywhere;">generic</code>, not below it)</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Generic world knowledge</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">24</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">generic</code></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Scene world knowledge</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">6</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">scene</code></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Task world knowledge</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">56</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">learned</code></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>World subtotal</strong></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>89</strong></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Declared; <code style="white-space: normal; overflow-wrap: anywhere;">learned</code> and above enable 86 (all but the causal group)</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>Grammar + world total</strong></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>94</strong></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Declared; <code style="white-space: normal; overflow-wrap: anywhere;">learned</code> and above enable 91</td>
    </tr>
  <tr><td style="padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>Inventory across both paths</strong></td><td style="padding: 0.55em 0.65em;"><strong>102</strong></td><td style="padding: 0.55em 0.65em; overflow-wrap: anywhere;">94 grammar/world + 3 official causal + 5 episode decoding declarations; not one active setting</td></tr>
</tbody>
</table>

The task subtotal is **4 request constraints + 36 sequence/completion constraints + 8 complex-task obligation constraints + 8 grounded-family constraints = 56**. The separate `causal` level declares **3 id-only logical constraints**, compiled into `CausalPlanDFA` and described below. The world graph also declares the same three rules as its `causal` group, for the graph DFA of the Joint and internal planners; the VLABench grammar/world baseline is therefore 94 declarations (91 enabled at `learned` and above), and the inventory across that baseline, the separate causal graph and the separate decoding graph (5 formulas, below) contains **102 declarations**. These are counts of declarations, so the causal and generic anti-self-placement formulas count separately even though their rule names overlap. All 89 world constraints use `ifL` as their outermost operator. The five grammar constraints use one `atLeastAL`, one `ifL`, two `atMostAL`, and one `andL`.

## Ablation levels

The grammar/world levels are cumulative. The official-protocol `causal` row is a separate decoding path. Inactive world constraints still exist in the graph; the table counts only enabled logical constraints. The official native grammar is procedural and contributes no DomiKnowS declarations, so `causal` enables 3, not 5 + 3.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 12%;">
    <col style="width: 13%;">
    <col style="width: 9%;">
    <col style="width: 32%;">
    <col style="width: 34%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="12%" style="width: 12%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Evaluation setting</th>
      <th scope="col" width="13%" style="width: 13%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Active groups</th>
      <th scope="col" width="9%" style="width: 9%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Active logical constraints</th>
      <th scope="col" width="32%" style="width: 32%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">New knowledge and change from the preceding setting</th>
      <th scope="col" width="34%" style="width: 34%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Example of its effect on constraint enforcement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">no DFA</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">None</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar and world enforcement bypassed.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">No logical constraint prevents an incomplete subtask or an invalid argument sequence.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">no DFA, end withheld</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">None</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A decoder policy blocks the first end label; no logical constraint is added.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">An immediate empty output is blocked by the decoder policy; later labels still have no grammar or world enforcement.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">grammar</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">5</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Enable five grammar constraints: boundaries, budgets, skill patterns, and arguments. No task identity is supplied.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code> may complete a subtask; EOS after an incomplete <code style="white-space: normal; overflow-wrap: anywhere;">pick</code> subtask is rejected.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">causal</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar + causal</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">8 (graph DFA) / 3 (official DFA)</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">--knowledge causal</code> enables only the three id-level causal formulas: the world graph&#x27;s <code style="white-space: normal; overflow-wrap: anywhere;">causal</code> group (with the five grammar constraints) for the Joint and internal graph decoders, or the official causal graph for <code style="white-space: normal; overflow-wrap: anywhere;">--protocol official</code>. No scene facts and no task identity are used.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">In a subtask that picks 2, placing into 2 is refused, as by generic&#x27;s anti_self_placement; a later pick of an object that was already a destination is refused. Any id 0–63 remains allowed.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">generic</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar + generic</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">29</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Enable 24 additional schema and domain constraints, including explicit subtask-start and completion checks.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A required subject must be supplied to <code style="white-space: normal; overflow-wrap: anywhere;">pick</code>; placing an object into itself violates <code style="white-space: normal; overflow-wrap: anywhere;">anti_self_placement</code>.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">scene</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar + generic + scene</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">35</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Enable six additional constraints using scene observations.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">If a target is classified as a static fixture, the existing <code style="white-space: normal; overflow-wrap: anywhere;">pick_excludes_static_fixtures</code> formula rejects picking it.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">learned</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar + all world groups</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">91</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Enable 56 additional task/grounded constraints. A classifier&#x27;s predicted task sets task-concept values; supplied requirements provide further facts. The formulas are predefined, not learned by this classifier.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Predicting <code style="white-space: normal; overflow-wrap: anywhere;">select_fruit</code> makes its task concept true. Its sequence constraint requires <code style="white-space: normal; overflow-wrap: anywhere;">place</code> at operation index 1, and its completion constraint prevents stopping after index 0.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar + all world groups</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">91</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">No additional constraints. Use the episode task identity instead of the predicted identity; this can change which task-guarded formulas apply.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">If the classifier predicted <code style="white-space: normal; overflow-wrap: anywhere;">select_chemistry_tube</code> but the episode task is <code style="white-space: normal; overflow-wrap: anywhere;">select_fruit</code>, the applicable sequence changes from <code style="white-space: normal; overflow-wrap: anywhere;">pick, lift</code> to <code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code>. If the prediction was correct, this identity change has no effect.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">metadata</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar + all world groups</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">91</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">No additional constraints. Supply metadata-assisted property observations to the existing formulas.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">For a requested highest-density choice, supplied density comparisons can make candidate A satisfy the selection requirement and candidate B fail it. The existing <code style="white-space: normal; overflow-wrap: anywhere;">grounded__affordance</code> constraint then rejects B.</td>
    </tr>
  </tbody>
</table>

`causal` is not cumulative with the levels around it in this table: it belongs to the standard protocol, whose grammar DFA (`NativePlanDFA`) is not the five-constraint grammar of `graph.py`, and it enforces three separate id-only logical constraints. It does not enable the generic/scene/task world groups. `metadata` introduces observations, not a fourth world rule group. A task rule is guarded by task/interface/phase facts, so an active rule need not apply to every operation. Explicit requests are consulted starting at `learned`.

The distinction is between **enabling a logical constraint** and **changing the facts evaluated by that constraint**. Moving from `scene` to `learned` enables more constraints. Moving from `learned` to `task` or `metadata` retains the same formulas but can change their applicable cases and accepted plans through different concept values. More supplied knowledge therefore need not increase the constraint count.

For example, the declared sequence formula includes `task_select_fruit(x) AND choosing_skill(x) AND position_1(x) -> place(x)` (with the interface and explicit-request guards described below). The predicted or episode task identity supplies `task_select_fruit(x)`; it does not rewrite the implication. Task identity selection is defined in [task_policy.py:59](../task_policy.py#L59), task-concept values in [knowledge/task_lcs.py:160](../knowledge/task_lcs.py#L160), and the sequence formulas in [knowledge/task_lcs.py:89](../knowledge/task_lcs.py#L89).

Activation is defined at [test_regr/VLABenchAgentInterface/knowledge/__init__.py:29](../knowledge/__init__.py#L29) and [test_regr/VLABenchAgentInterface/knowledge/__init__.py:48](../knowledge/__init__.py#L48). World logical constraint activation can be changed at [test_regr/VLABenchAgentInterface/world_graph.py:341](../world_graph.py#L341); recompile after changing it.

## DomiKnowS operator classes

Four operator classes are used as the outermost operator of complete logical constraints. Each declaration is classified once. “Task” means the world group first enabled at `learned`. The causal column counts the world graph&#x27;s causal group (3) and the separate official-protocol causal graph (3); the episode column is the decoding graph of episode knowledge; the inventory total includes all of them.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup><col style="width: 12%;"><col style="width: 7%;"><col style="width: 7%;"><col style="width: 7%;"><col style="width: 7%;"><col style="width: 7%;"><col style="width: 7%;"><col style="width: 13%;"><col style="width: 33%;"></colgroup>
<thead><tr><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Class</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Generic</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Scene</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Task</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Causal</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Episode</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Inventory total</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Class definition</th></tr></thead>
<tbody>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>ifL</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">24</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">6</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">56</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">6</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">5</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">98</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/graph/logicalConstrain.py#L1331">logicalConstrain.py:1331</a></td></tr>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>andL</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/graph/logicalConstrain.py#L1307">logicalConstrain.py:1307</a></td></tr>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>atLeastAL</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/graph/logicalConstrain.py#L1528">logicalConstrain.py:1528</a></td></tr>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>atMostAL</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">0</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/graph/logicalConstrain.py#L1527">logicalConstrain.py:1527</a></td></tr>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>Total</strong></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">5</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">24</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">6</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">56</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">6</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">5</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>102</strong></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Across both decoding paths; not simultaneously enabled.</td></tr>
</tbody>
</table>

The grammar/world subtotal is **90 ifL + 1 andL + 1 atLeastAL + 2 atMostAL = 94** (3 of the ifL are the world causal group). The official causal graph adds **3 ifL** declarations and the decoding graph **5 ifL**, giving **98 ifL + 1 andL + 1 atLeastAL + 2 atMostAL = 102**. Nested `orL` and `notL` do not add declarations.

`orL`, `notL`, and `exactL` are also used to express parts of these formulas. They do not create separately counted logical constraints in this inventory. The examples below explain all seven operators used in the formulas.

Examples below use the descriptive aliases explained under “Reading the formula examples.” They preserve the source concepts and use consistent logical variables.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 10%;">
    <col style="width: 44%;">
    <col style="width: 30%;">
    <col style="width: 16%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="10%" style="width: 10%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Class</th>
      <th scope="col" width="44%" style="width: 44%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Meaning and example</th>
      <th scope="col" width="30%" style="width: 30%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Where instantiated in VLABench</th>
      <th scope="col" width="16%" style="width: 16%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Ablation membership</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">ifL</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Implication: <code style="white-space: normal; overflow-wrap: anywhere;">ifL(pick(&#x27;x&#x27;), notL(entity_is_static_fixture(&#x27;x&#x27;)))</code>. A pick may not target a static fixture.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L170">test_regr/VLABenchAgentInterface/world_graph.py:170</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar, generic, scene, task</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">andL</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">All terms must hold: <code style="white-space: normal; overflow-wrap: anywhere;">andL(hand_capacity_is_positive(&#x27;r&#x27;), notL(andL(hand_state_known(&#x27;r&#x27;), object_is_held(&#x27;r&#x27;), notL(held_object_matches_subject(&#x27;r&#x27;)))))</code> is the capacity conclusion.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L101">domiknows/generation/grounded.py:101</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar, generic, scene, task</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">orL</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">At least one term holds: <code style="white-space: normal; overflow-wrap: anywhere;">orL(entity_is_button(&#x27;x&#x27;), container_is_button(&#x27;x&#x27;))</code> recognizes a button in either role.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L167">test_regr/VLABenchAgentInterface/world_graph.py:167</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar, generic, scene, task</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">notL</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Negation: <code style="white-space: normal; overflow-wrap: anywhere;">notL(target_previously_picked(&#x27;t&#x27;))</code> allows a new object in the repeated-pick formula.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L145">test_regr/VLABenchAgentInterface/world_graph.py:145</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar, generic, scene, task</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">exactL</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Local exact count: <code style="white-space: normal; overflow-wrap: anywhere;">exactL(*(skill_concept(&#x27;x&#x27;) for skill_concept in skill_concepts.values()), 1)</code> requires exactly one skill on this operation. The grammar also requires exactly one progress state per token.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L129">test_regr/VLABenchAgentInterface/world_graph.py:129</a>; <a href="../graph.py#L79">test_regr/VLABenchAgentInterface/graph.py:79</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar, generic</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">atLeastAL</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Accumulated lower bound: <code style="white-space: normal; overflow-wrap: anywhere;">atLeastAL(eos(&#x27;p&#x27;), 1)</code> requires at least one EOS across the sequence.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L129">test_regr/VLABenchAgentInterface/graph.py:129</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">atMostAL</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Accumulated upper bound: <code style="white-space: normal; overflow-wrap: anywhere;">atMostAL(notL(eos(&#x27;p&#x27;)), operation_limit * 5)</code> caps non-EOS tokens at 60 in the baseline.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L134">test_regr/VLABenchAgentInterface/graph.py:134</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar</td>
    </tr>
  </tbody>
</table>

The separate `causal` graph uses three `ifL` formulas, with nested `orL` and `notL`; the decoding graph uses five, with nested `andL`. `exactL` counts the alternatives at a bound position/operation; the `AL` operators accumulate across token positions. The grammar/world subtotal is 94, the separate causal graph contributes 3 and the decoding graph 5, giving an inventory of 102 declarations. Operators inside a formula do not increase its count.

### Reading the formula examples

Every complete formula is displayed on multiple lines with descriptive concept names. These names are readable aliases for the existing graph concepts; the source declarations and their meaning are unchanged. Formula snippets assume those aliases are bound to their source concepts and the graph context is open. They do not create additional logical constraints.

**Names outside quotes are Python references to concepts. Names inside quotes are logical variables.** For example, `operation('x')` means that logical variable `'x'` refers to an operation. Repeating `'x'` in `pick('x')` or `has_target_entity('x')` refers to that same operation. The examples use `'x'` for an operation, `'t'` for a transition or token-adjacency relation, `'p'` for a token position, `'s'` for a sequence-start relation, `'r'` for a grounded requirement, and `'episode'` for a plan. Each formula has its own variable scope.

The aliases correspond to source concepts as follows:

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup><col style="width: 28%;"><col style="width: 44%;"><col style="width: 28%;"></colgroup>
  <thead><tr><th scope="col" width="28%" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em;">Readable name</th><th scope="col" width="44%" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em;">Source binding</th><th scope="col" width="28%" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em;">Meaning</th></tr></thead>
  <tbody>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">operation; transition</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.operation; world.transition</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Operation and operation-transition concepts.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">pick; place; press; other skill names</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.skills[&quot;pick&quot;]; world.skills[&quot;place&quot;]; world.skills[&quot;press&quot;]; likewise for each skill</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">The skill predicates of an operation.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">skill_concepts</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.skills</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">All nine skill concepts. A loop variable named skill_concept refers to one of them.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">has_target_entity; has_target_container; allowed_transition</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.has_target_entity; world.has_target_container; world.allowed_transition</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Role presence and subtask-language membership.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">entity_resolves; subject_equals_destination; other primitive fact names</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.facts[&quot;entity_resolves&quot;]; world.facts[&quot;subject_equals_destination&quot;]; likewise for each fact name</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Observed facts, rather than extra policy rules.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">task_select_fruit; interface_compact; interface_full</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.task_concepts.tasks[&quot;select_fruit&quot;]; world.task_concepts.interfaces[&quot;compact&quot;]; world.task_concepts.interfaces[&quot;full&quot;]</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Task identity and plan interface; other task names follow the same convention.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">choosing_skill; final_operation; explicit_sequence_supplied</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.task_concepts.at_skill_choice; world.task_concepts.is_final; world.task_concepts.explicit_sequence</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">The phase and explicit-request guards.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">position_at_least_1; position_at_least_2; other thresholds</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.task_concepts.positions[1]; world.task_concepts.positions[2]; likewise for each threshold</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Zero-based operation index comparisons. Exact index 1 is AND(&gt;=1, NOT &gt;=2).</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">history_includes_pick; skill_history</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.task_concepts.history[&quot;pick&quot;]; world.task_concepts.history</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A particular skill in completed history, and the dictionary of all such concepts.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">requested_sequence_allows; requested_cycle_allows; requested_skill_requirements</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">For each skill name: world.task_concepts.requests[&quot;sequence_skill:&quot; + skill_name], [&quot;cycle_skill:&quot; + skill_name], or [&quot;required_skill:&quot; + skill_name]</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Dictionaries of request predicates, indexed by skill name.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">request_initial_target_known; request_target_matches; other request names</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.task_concepts.requests[&quot;initial_target_known&quot;]; world.task_concepts.requests[&quot;target_matches&quot;]; likewise for each request key</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Facts supplied by the explicit request.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">destination_required; holding_required; other requirement-family names</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.grounded_constraints.families[&quot;destination&quot;]; world.grounded_constraints.families[&quot;holding&quot;]; likewise for each family</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">The type of a grounded requirement.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">observation_known; required_fact_holds</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.grounded_constraints.known; world.grounded_constraints.holds</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Whether the requirement is known and whether it is satisfied.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">operation_is_idempotent; hand_state_known; object_is_held; held_object_matches_subject</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.grounded_constraints.inputs[&quot;idempotent&quot;]; [&quot;hand_known&quot;]; [&quot;held_exists&quot;]; [&quot;held_matches_subject&quot;]</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Primitive observations used in the holding and capacity formulas.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">subject_unspecified; hand_capacity_known; hand_capacity_is_zero; hand_capacity_is_positive</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">world.grounded_constraints.inputs[&quot;subject_unspecified&quot;]; [&quot;capacity_known&quot;]; [&quot;capacity_zero&quot;]; [&quot;capacity_positive&quot;]</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">The other primitive requirement observations.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">generation_context; eos; adjacent_tokens; sequence_start</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">generation_bundle.context; its token_value(plan_vocabulary.eos_token, variable); is_before_rel; sequence_start</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Generation concepts. eos is the predicate for the EOS token label.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">first_token_relation; second_token_relation; start_token_relation</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">generation_bundle.first_token; generation_bundle.second_token; generation_bundle.sequence_start_token</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Relation paths that bind the token positions used by the formula.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">progress_states; token_labels_by_progress_state; successors_by_progress_state; initial_progress_states</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">The declared progress enum values; graph.py positions; following; starts</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">The progress concepts and their declared token labels, successors, and initial states.</td></tr>
    <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">operation_limit; skill_language; plan_vocabulary</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">graph.py max_operations; language; vocabulary</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Generation configuration and token vocabulary.</td></tr>
  <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>plan; first_operation; pattern_start_allowed; pattern_complete</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>world.plan; world.pattern_first_operation; world.pattern_start_allowed; world.pattern_complete</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Plan identity, the first-operation flag, and start/completion observations from compiled progress.</td></tr>
</tbody>
</table>

The examples expand the skill-schema and exact-position helpers so their conditions are visible. The grammar examples spell out disjunctions with `orL`; a disjunction containing one alternative has that alternative's meaning. Names such as `required_skills_supported` and `entity_value_usable` remain observed concepts, with their grounding described in the source map. The examples cover **91 grammar/world constraints and 3 separate causal constraints**, or **94 declarations** across the two paths.

## Grammar rule families: five logical constraints

All rows in this section belong to the grammar and remain enforced at every DFA-enabled level. The first four logical constraints have automatically assigned names (`LC0`, etc. in a fresh standalone graph); the last has the explicit name shown below. Use the source/formula to identify these rules rather than depending on auto-generated names.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 23%;">
    <col style="width: 9%;">
    <col style="width: 40%;">
    <col style="width: 28%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="23%" style="width: 23%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Family</th>
      <th scope="col" width="9%" style="width: 9%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Logical constraints</th>
      <th scope="col" width="40%" style="width: 40%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Example / effect</th>
      <th scope="col" width="28%" style="width: 28%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">EOS existence</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A sequence without EOS cannot accept. At least one end token is required.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L129">test_regr/VLABenchAgentInterface/graph.py:129</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>atLeastAL(
    eos(&#x27;p&#x27;),
    1,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">EOS closure</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">After EOS, the only legal labels are further EOS labels. Repeated EOS is permitted; <code style="white-space: normal; overflow-wrap: anywhere;">EOS, skill:pick</code> is rejected.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L130">test_regr/VLABenchAgentInterface/graph.py:130</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    adjacent_tokens(&#x27;t&#x27;),
    ifL(
        eos(&#x27;first_position&#x27;, path=(&#x27;t&#x27;, first_token_relation)),
        eos(&#x27;next_position&#x27;, path=(&#x27;t&#x27;, second_token_relation)),
    ),
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Non-EOS token budget</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">At most <code style="white-space: normal; overflow-wrap: anywhere;">12 * 5 = 60</code> non-EOS tokens in the baseline. EOS padding does not consume this budget.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L134">test_regr/VLABenchAgentInterface/graph.py:134</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>atMostAL(
    notL(eos(&#x27;p&#x27;)),
    operation_limit * 5,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Operation budget</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">At most 12 skill tokens; a custom language may tighten the limit through <code style="white-space: normal; overflow-wrap: anywhere;">language.maximum</code>.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L135">test_regr/VLABenchAgentInterface/graph.py:135</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>atMostAL(
    orL(
        *(
            generation_context.token_value(f&#x27;skill:{skill_name}&#x27;, &#x27;p&#x27;)
            for skill_name in plan_vocabulary.skills
        ),
    ),
    min(operation_limit, skill_language.maximum),
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">plan_skill_patterns_and_arguments</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A complete <code style="white-space: normal; overflow-wrap: anywhere;">pick(obj:0), place(obj:1)</code> subtask may terminate or concatenate with another complete subtask. A lone incomplete <code style="white-space: normal; overflow-wrap: anywhere;">pick</code> cannot terminate.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L100">test_regr/VLABenchAgentInterface/graph.py:100</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>progress_constraints = [
    exactL(
        *(
            progress_states[progress_state_name](&#x27;p&#x27;)
            for progress_state_name in token_labels_by_progress_state
        ),
        1,
    )
]
for progress_state_name, allowed_token_labels in token_labels_by_progress_state.items():
    progress_constraints.append(
        ifL(
            progress_states[progress_state_name](&#x27;p&#x27;),
            orL(
                *(
                    generation_context.token_value(token_label, &#x27;p&#x27;)
                    for token_label in allowed_token_labels
                )
            ),
        )
    )
first_position_path = (&#x27;s&#x27;, start_token_relation)
allowed_initial_states = [
    progress_states[progress_state_name](&#x27;first_position&#x27;, path=first_position_path)
    for progress_state_name in sorted(initial_progress_states)
]
if not allowed_initial_states:
    first_position = progress_states[&#x27;finished&#x27;](
        &#x27;first_position&#x27;, path=first_position_path
    )
    allowed_initial_states = [andL(first_position, notL(first_position))]
progress_constraints.append(
    ifL(sequence_start(&#x27;s&#x27;), orL(*allowed_initial_states)),
)
progress_constraints.append(
    ifL(
        adjacent_tokens(&#x27;t&#x27;),
        orL(
            *(
                andL(
                    progress_states[current_state_name](
                        &#x27;current_position&#x27;, path=(&#x27;t&#x27;, first_token_relation)
                    ),
                    orL(
                        *(
                            progress_states[next_state_name](
                                &#x27;next_position&#x27;, path=(&#x27;t&#x27;, second_token_relation)
                            )
                            for next_state_name in sorted(next_state_names)
                        )
                    ),
                )
                for current_state_name, next_state_names in successors_by_progress_state.items()
                if next_state_names
            ),
        ),
    ),
)
andL(
    *progress_constraints,
    name=&#x27;plan_skill_patterns_and_arguments&#x27;,
)</code></pre></td>
    </tr>
  </tbody>
</table>

The `plan_skill_patterns_and_arguments` logical constraint combines progress-state consistency, legal token values, legal starts, and legal transitions in one formula. The following components explain its meaning and are not counted as separate logical constraints.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 23%;">
    <col style="width: 47%;">
    <col style="width: 30%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="23%" style="width: 23%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grammar component</th>
      <th scope="col" width="47%" style="width: 47%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Example</th>
      <th scope="col" width="30%" style="width: 30%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Exactly one progress state</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A token cannot simultaneously represent a skill position and an argument-name position.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L79">test_regr/VLABenchAgentInterface/graph.py:79</a></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">State determines token alternatives</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">An entity position allows <code style="white-space: normal; overflow-wrap: anywhere;">obj:0</code> through <code style="white-space: normal; overflow-wrap: anywhere;">obj:63</code>; an argument-name position requires its particular <code style="white-space: normal; overflow-wrap: anywhere;">arg:...</code> label.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L81">test_regr/VLABenchAgentInterface/graph.py:81</a></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Legal starts</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">The unrestricted language starts at a declared subtask root, not an argument or EOS. A continuation can explicitly permit an empty plan.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L89">test_regr/VLABenchAgentInterface/graph.py:89</a></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Legal adjacent progress states</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A skill is followed by its first required argument; an argument name is followed by an entity pointer.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L93">test_regr/VLABenchAgentInterface/graph.py:93</a></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Required argument order</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick</code> requires <code style="white-space: normal; overflow-wrap: anywhere;">arg:target_entity_name, obj:i</code>; an entity pointer before the argument name fails.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L51">test_regr/VLABenchAgentInterface/graph.py:51</a></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Optional argument suffix</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">push</code> may omit its destination or append <code style="white-space: normal; overflow-wrap: anywhere;">arg:target_container_name, obj:i</code>; it cannot supply an unrelated role.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L52">test_regr/VLABenchAgentInterface/graph.py:52</a></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Subtask completion and repetition</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, pour</code> and <code style="white-space: normal; overflow-wrap: anywhere;">pick, pour, place</code> share a prefix; both complete interpretations remain possible. Another subtask may follow a completed one.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../graph.py#L68">test_regr/VLABenchAgentInterface/graph.py:68</a></td>
    </tr>
  </tbody>
</table>

The 13 declared subtask skill patterns are defined at [test_regr/VLABenchAgentInterface/world_domain.py:20](../world_domain.py#L20):

- `pick, lift, pull`
- `pick, lift, place`
- `pick, pour, place`
- `pick, push, place`
- `pick, push, pull`
- `pick, place`
- `pick, insert`
- `pick, pour`
- `pick, pull`
- `pick, lift`
- `pick, push`
- `pick, open_door`
- `press`

Skill role data are defined at [test_regr/VLABenchAgentInterface/world_domain.py:123](../world_domain.py#L123). A pointer within the 64-label vocabulary is syntactically legal; whether it resolves to a supplied scene entity is checked by world knowledge. Labels outside the generation vocabulary cannot match the declared token alternatives. The baseline does not allow an empty plan. Task-specific continuation declarations can alter roots, patterns, repetition, bounds, and empty-plan permission, so the resulting formulas can differ from this baseline.

## Causal (id-only) DFA rules: 3 logical constraints

These rules are active only at `causal`, in the standard VLABench protocol. `official/causal_graph.py` declares three formulas and compiles them with the generic DomiKnowS predicate compiler. `CausalPlanDFA.admits` evaluates that compiled program at both partial-id reachability checks and operation closure. Rule names are labels: changing a formula, removing it, or making it inactive changes the language after creating a new DFA. An already compiled DFA retains its snapshot.

Each formula binds `operation` to the candidate operation. `pick`, `place`, `insert`, `pour`, and `push` are skill concepts under the graph's operation concept; the grammar already determines the skill. `target_is_subject` observes equality with the object picked in the current subtask; `target_used_in_subtask` observes membership in its earlier destinations; `target_was_destination` observes membership in all earlier destinations. These are comparisons of plan ids, computed before the candidate executes, without scene metadata or task identity. They are not precomputed legality flags.

The prefix state stores `(subject, local_destinations, former_destinations)`. Pick/press start a subtask, reset local destinations, and set the subject to the picked id or to none for press. A destination operation appends its id to local and plan-wide history. This state update accumulates observations; it does not refuse actions. Native JSON syntax, operation limits, EOS, byte-token projection, and this observation lifecycle remain runtime responsibilities. No additional state-update logical constraints are counted.

Ids are pruned as soon as no grammatical operation satisfies the compiled formulas; whitespace before a refused closing brace is also refused. Existing checks found no violations in the 4,500 reference plans; this declaration refactor was validated locally without rerunning that dataset evaluation.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup><col style="width: 25%;"><col style="width: 9%;"><col style="width: 39%;"><col style="width: 27%;"></colgroup>
<thead><tr><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Rule family</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Logical constraints</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Example and meaning</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Definition</th></tr></thead>
<tbody>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>anti_self_placement</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Reject a destination equal to the current picked subject. The guard covers place, insert, pour and push. Example: <code>pick 2, place 2</code>.</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../official/causal_graph.py#L52">official/causal_graph.py:52</a></td></tr>
<tr data-constraint-example="true"><td colspan="4" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
    orL(place(&quot;operation&quot;), insert(&quot;operation&quot;), pour(&quot;operation&quot;), push(&quot;operation&quot;)),
    notL(target_is_subject(&quot;operation&quot;)),
    name=&quot;anti_self_placement&quot;,
)</code></pre></td></tr>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>distinct_subtask_destinations</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Reject reuse of destination 3 within one subtask. A later subtask may use 3 again. Example: <code>pick 2, pour 3, place 3</code>.</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../official/causal_graph.py#L58">official/causal_graph.py:58</a></td></tr>
<tr data-constraint-example="true"><td colspan="4" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
    orL(place(&quot;operation&quot;), insert(&quot;operation&quot;), pour(&quot;operation&quot;), push(&quot;operation&quot;)),
    notL(target_used_in_subtask(&quot;operation&quot;)),
    name=&quot;distinct_subtask_destinations&quot;,
)</code></pre></td></tr>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>destinations_stay_put</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Reject picking 3 after it served as a destination in an earlier operation. Example: <code>pick 2, place 3, pick 3, place 4</code>.</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../official/causal_graph.py#L64">official/causal_graph.py:64</a></td></tr>
<tr data-constraint-example="true"><td colspan="4" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
    pick(&quot;operation&quot;),
    notL(target_was_destination(&quot;operation&quot;)),
    name=&quot;destinations_stay_put&quot;,
)</code></pre></td></tr>
</tbody>
</table>

The official protocol still does not enforce scene-based pointer classification or the generic repeated-pick rule. Repeated picks of a non-destination stay permitted, including chemistry-experiment patterns. Pointer bounds remain the native grammar's configured `entity_count` (64 in the official setup) unless `--scene-ids` is set; then the `scene_ids_bound_targets` formula below bounds them. The three declarations are independent of the generic/scene/task world groups; their counts are not cumulative with those groups.

The world graph declares the same three rules as its `causal` group (`_declare_causal_rules`, [world_graph.py:158](../world_graph.py#L158)), so `--knowledge causal` also constrains the Joint and internal graph decoders. They read the world facts `subject_equals_destination`, `destination_used_in_subtask` and `target_was_destination`, which `semantics.py` grounds from the prefix (`PlanSemanticState.subtask_destinations` and `former_destinations`); the graph DFA and DataNode verification agree on them in the bounded-equivalence test.

```text
ifL(operation('op'), notL(andL(has_target_container('op'), orL(place('op'), insert('op'), pour('op'), push('op')),
                               subject_equals_destination('op'))), name='causal__anti_self_placement')
ifL(operation('op'), notL(andL(has_target_container('op'), orL(place('op'), insert('op'), pour('op'), push('op')),
                               destination_used_in_subtask('op'))), name='causal__distinct_subtask_destinations')
ifL(operation('op'), notL(andL(pick('op'), has_target_entity('op'), target_was_destination('op'))),
    name='causal__destinations_stay_put')
```

## Episode decoding knowledge: 5 logical constraints

The model's answers to the official protocol's stage-one questions, and the internal frozen-VLM path's instruction object candidates, are observations of one episode. `official/decoding_graph.py` declares the formulas that decide what they allow, and `KnowledgePlanDFA` (`official/knowledge_dfa.py`) compiles them with the same predicate compiler as the causal graph, together with the causal formulas when `--constraints causal` is set. The answers never configure the grammar directly: the DFA keeps the full grammar (64 ids, all benchmark patterns, the operation budget) and the compiled program refuses operations and plan endings. Each restriction was checked against the former procedural DFA on every bounded plan and on every prefix of a compact and a pretty-printed plan; removing or deactivating a formula changes acceptance after the DFA is rebuilt.

Operation observations: `press`, `first_operation`, `press_answer_yes` (the model answered Yes to the press question), `has_target`, `scene_bound_read` and `target_within_scene_bound` (the id is at most the largest tag number the model read), `patterns_chosen` and `continues_chosen_pattern` (the operation extends a named pattern that still fits the remaining budget), `instruction_candidates_known` and `target_is_instruction_candidate`. Plan observations, at the closing bracket: `plan_patterns_chosen` and `chosen_patterns_complete`. The DFA tracks the named-pattern prefixes as a fourth state field; like the causal fields, it stores facts and refuses nothing.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup><col style="width: 25%;"><col style="width: 9%;"><col style="width: 39%;"><col style="width: 27%;"></colgroup>
<thead><tr><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Rule family</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Logical constraints</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Example and meaning</th><th scope="col" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Definition</th></tr></thead>
<tbody>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>press_answer_single_press</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A Yes to the press question allows only one operation, a press (<code>--press-question</code>). Example: after Yes, <code>pick 2, place 3</code> and <code>press 0, press 1</code> are refused.</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../official/decoding_graph.py#L58">official/decoding_graph.py:58</a></td></tr>
<tr data-constraint-example="true"><td colspan="4" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
    press_answer_yes(&quot;operation&quot;),
    andL(press(&quot;operation&quot;), first_operation(&quot;operation&quot;)),
    name=&quot;press_answer_single_press&quot;,
)</code></pre></td></tr>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>scene_ids_bound_targets</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Every object id is at most the largest tag number the model read (<code>--scene-ids</code>). Example: with 6 read, <code>press 7</code> is refused and the prefix <code>3</code> cannot grow into <code>34</code>.</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../official/decoding_graph.py#L62">official/decoding_graph.py:62</a></td></tr>
<tr data-constraint-example="true"><td colspan="4" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
    andL(scene_bound_read(&quot;operation&quot;), has_target(&quot;operation&quot;)),
    target_within_scene_bound(&quot;operation&quot;),
    name=&quot;scene_ids_bound_targets&quot;,
)</code></pre></td></tr>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>chosen_patterns_continue</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Each operation extends a subtask pattern the model named and that still fits the operation budget (<code>--pattern-first</code>). Example: with only <code>press</code> named, the prefix <code>pi</code> of <code>pick</code> is refused at once.</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../official/decoding_graph.py#L66">official/decoding_graph.py:66</a></td></tr>
<tr data-constraint-example="true"><td colspan="4" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
    patterns_chosen(&quot;operation&quot;),
    continues_chosen_pattern(&quot;operation&quot;),
    name=&quot;chosen_patterns_continue&quot;,
)</code></pre></td></tr>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>chosen_patterns_complete</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">The plan ends on a complete named pattern. Example: with <code>pick, pour, place</code> named, <code>pick 0, pour 1</code> cannot close. This formula alone fixes the accepted plans; the continuation formula prunes earlier.</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../official/decoding_graph.py#L68">official/decoding_graph.py:68</a></td></tr>
<tr data-constraint-example="true"><td colspan="4" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
    plan_patterns_chosen(&quot;plan&quot;),
    chosen_patterns_complete(&quot;plan&quot;),
    name=&quot;chosen_patterns_complete&quot;,
)</code></pre></td></tr>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>instruction_candidates_bound_targets</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">When the instruction grounds candidates for a skill and role (<code>object_grounding.instruction_object_constraints</code>, internal frozen-VLM <code>dfa</code>/<code>hmm</code> modes), the id must be one of them. Example: for <code>Put the apple into the plate_seen</code>, <code>pick</code> must name the apple.</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../official/decoding_graph.py#L71">official/decoding_graph.py:71</a></td></tr>
<tr data-constraint-example="true"><td colspan="4" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
    instruction_candidates_known(&quot;operation&quot;),
    target_is_instruction_candidate(&quot;operation&quot;),
    name=&quot;instruction_candidates_bound_targets&quot;,
)</code></pre></td></tr>
</tbody>
</table>

The HMM action lookahead (`--hmm`) steers this DFA as well. It ranks only the tokens the compiled formulas leave legal, but its action inventory and pattern automaton are those of the full grammar, so its scores can differ slightly from runs made when the answers restricted the grammar itself.

## Generic world families: 24 logical constraints

Every rule here first becomes active at `generic` and stays active at `scene`, `learned`, `task`, and `metadata`. Each rule is one logical constraint using `ifL` as its outermost operator.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 25%;">
    <col style="width: 9%;">
    <col style="width: 39%;">
    <col style="width: 27%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="25%" style="width: 25%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Rule name / family</th>
      <th scope="col" width="9%" style="width: 9%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Logical constraints</th>
      <th scope="col" width="39%" style="width: 39%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Example and meaning</th>
      <th scope="col" width="27%" style="width: 27%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">operation_exactly_one_skill</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">An operation labeled both <code style="white-space: normal; overflow-wrap: anywhere;">pick</code> and <code style="white-space: normal; overflow-wrap: anywhere;">press</code> fails; exactly one of the nine skill concepts must hold.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L129">test_regr/VLABenchAgentInterface/world_graph.py:129</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    operation(&#x27;x&#x27;),
    exactL(*(skill_concept(&#x27;x&#x27;) for skill_concept in skill_concepts.values()), 1),
    name=&#x27;operation_exactly_one_skill&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">operation_schema_is_valid</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick</code> needs a usable subject and forbids a destination. A present role must have a usable value. This already includes scene membership of supplied references.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L130">test_regr/VLABenchAgentInterface/world_graph.py:130</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    operation(&#x27;x&#x27;),
    orL(
        andL(
            pick(&#x27;x&#x27;),
            has_target_entity(&#x27;x&#x27;),
            ifL(has_target_entity(&#x27;x&#x27;), entity_value_usable(&#x27;x&#x27;)),
            notL(has_target_container(&#x27;x&#x27;)),
            ifL(has_target_container(&#x27;x&#x27;), container_value_usable(&#x27;x&#x27;)),
        ),
        andL(
            place(&#x27;x&#x27;),
            notL(has_target_entity(&#x27;x&#x27;)),
            ifL(has_target_entity(&#x27;x&#x27;), entity_value_usable(&#x27;x&#x27;)),
            has_target_container(&#x27;x&#x27;),
            ifL(has_target_container(&#x27;x&#x27;), container_value_usable(&#x27;x&#x27;)),
        ),
        andL(
            insert(&#x27;x&#x27;),
            notL(has_target_entity(&#x27;x&#x27;)),
            ifL(has_target_entity(&#x27;x&#x27;), entity_value_usable(&#x27;x&#x27;)),
            has_target_container(&#x27;x&#x27;),
            ifL(has_target_container(&#x27;x&#x27;), container_value_usable(&#x27;x&#x27;)),
        ),
        andL(
            pour(&#x27;x&#x27;),
            notL(has_target_entity(&#x27;x&#x27;)),
            ifL(has_target_entity(&#x27;x&#x27;), entity_value_usable(&#x27;x&#x27;)),
            has_target_container(&#x27;x&#x27;),
            ifL(has_target_container(&#x27;x&#x27;), container_value_usable(&#x27;x&#x27;)),
        ),
        andL(
            push(&#x27;x&#x27;),
            notL(has_target_entity(&#x27;x&#x27;)),
            ifL(has_target_entity(&#x27;x&#x27;), entity_value_usable(&#x27;x&#x27;)),
            ifL(has_target_container(&#x27;x&#x27;), container_value_usable(&#x27;x&#x27;)),
        ),
        andL(
            pull(&#x27;x&#x27;),
            notL(has_target_entity(&#x27;x&#x27;)),
            ifL(has_target_entity(&#x27;x&#x27;), entity_value_usable(&#x27;x&#x27;)),
            notL(has_target_container(&#x27;x&#x27;)),
            ifL(has_target_container(&#x27;x&#x27;), container_value_usable(&#x27;x&#x27;)),
        ),
        andL(
            lift(&#x27;x&#x27;),
            notL(has_target_entity(&#x27;x&#x27;)),
            ifL(has_target_entity(&#x27;x&#x27;), entity_value_usable(&#x27;x&#x27;)),
            notL(has_target_container(&#x27;x&#x27;)),
            ifL(has_target_container(&#x27;x&#x27;), container_value_usable(&#x27;x&#x27;)),
        ),
        andL(
            open_door(&#x27;x&#x27;),
            has_target_entity(&#x27;x&#x27;),
            ifL(has_target_entity(&#x27;x&#x27;), entity_value_usable(&#x27;x&#x27;)),
            notL(has_target_container(&#x27;x&#x27;)),
            ifL(has_target_container(&#x27;x&#x27;), container_value_usable(&#x27;x&#x27;)),
        ),
        andL(
            press(&#x27;x&#x27;),
            has_target_entity(&#x27;x&#x27;),
            ifL(has_target_entity(&#x27;x&#x27;), entity_value_usable(&#x27;x&#x27;)),
            notL(has_target_container(&#x27;x&#x27;)),
            ifL(has_target_container(&#x27;x&#x27;), container_value_usable(&#x27;x&#x27;)),
        ),
    ),
    name=&#x27;operation_schema_is_valid&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">transition_matches_subtask_pattern</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Each pair extends the prefix state projected from the declared grammar. In pick, place, pick, press, the first two transitions remain valid and only the last fails. A missing argument does not change these pattern observations.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L133">test_regr/VLABenchAgentInterface/world_graph.py:133</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    transition(&#x27;t&#x27;),
    allowed_transition(&#x27;t&#x27;),
    name=&#x27;transition_matches_subtask_pattern&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">anti_self_placement</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A subject and destination cannot be the same entity when a destination is present and the action is place/pour/insert/push or has a subject role.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L141">test_regr/VLABenchAgentInterface/world_graph.py:141</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    operation(&#x27;x&#x27;),
    notL(
        andL(
            has_target_container(&#x27;x&#x27;),
            subject_equals_destination(&#x27;x&#x27;),
            orL(
                *(
                    skill_concepts[skill_name](&#x27;x&#x27;)
                    for skill_name in (&#x27;place&#x27;, &#x27;pour&#x27;, &#x27;insert&#x27;, &#x27;push&#x27;)
                ),
                has_target_entity(&#x27;x&#x27;),
            ),
        ),
    ),
    name=&#x27;anti_self_placement&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">no_redundant_repeated_picks</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Picking the same entity again fails unless that entity is explicitly reusable.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L144">test_regr/VLABenchAgentInterface/world_graph.py:144</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(transition(&#x27;t&#x27;), next_is_pick(&#x27;t&#x27;), next_has_entity(&#x27;t&#x27;)),
    orL(notL(target_previously_picked(&#x27;t&#x27;)), target_reusable(&#x27;t&#x27;)),
    name=&#x27;no_redundant_repeated_picks&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">skill_argument__&lt;skill&gt;__&lt;role&gt;</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">17</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Each required role must be present; each nonoptional forbidden role must be absent. Expanded below.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L154">test_regr/VLABenchAgentInterface/world_graph.py:154</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    pick(&#x27;x&#x27;),
    has_target_entity(&#x27;x&#x27;),
    name=&#x27;skill_argument__pick__target_entity_name&#x27;,
)</code></pre></td>
    </tr>
  <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>operation_starts_subtask_pattern</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">The first operation must begin a declared subtask or an explicitly permitted continuation. For example, a bare lift cannot start the default language.</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L135">world_graph.py:135</a></td></tr>
<tr data-constraint-example="true"><td colspan="4" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(operation(&#x27;x&#x27;), first_operation(&#x27;x&#x27;)),
    pattern_start_allowed(&#x27;x&#x27;),
    name=&#x27;operation_starts_subtask_pattern&#x27;,
)</code></pre></td></tr>
<tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code>plan_completes_subtask_pattern</code></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">The final progress state must permit EOS. A lone pick is incomplete; an empty plan is accepted only when the context explicitly permits it.</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L136">world_graph.py:136</a></td></tr>
<tr data-constraint-example="true"><td colspan="4" style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    plan(&#x27;episode&#x27;),
    pattern_complete(&#x27;episode&#x27;),
    name=&#x27;plan_completes_subtask_pattern&#x27;,
)</code></pre></td></tr>
</tbody>
</table>

The following table enumerates all 17 role logical constraints. Each listed role expands to the exact name `skill_argument__<skill>__<role>`. Each row also gives a concrete accepted role shape. “No role” means omit that key entirely.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 11%;">
    <col style="width: 19%;">
    <col style="width: 19%;">
    <col style="width: 18%;">
    <col style="width: 8%;">
    <col style="width: 25%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="11%" style="width: 11%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Skill</th>
      <th scope="col" width="19%" style="width: 19%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Required roles (one logical constraint each)</th>
      <th scope="col" width="19%" style="width: 19%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Forbidden roles (one logical constraint each)</th>
      <th scope="col" width="18%" style="width: 18%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Optional role (no individual role logical constraint)</th>
      <th scope="col" width="8%" style="width: 8%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Logical constraints</th>
      <th scope="col" width="25%" style="width: 25%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Example role shape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_entity_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_container_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">None</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">target_entity_name=obj:0</td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="6" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    pick(&#x27;x&#x27;),
    has_target_entity(&#x27;x&#x27;),
    name=&#x27;skill_argument__pick__target_entity_name&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    pick(&#x27;x&#x27;),
    notL(has_target_container(&#x27;x&#x27;)),
    name=&#x27;skill_argument__pick__target_container_name&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">place</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_container_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_entity_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">None</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">target_container_name=obj:0</td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="6" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    place(&#x27;x&#x27;),
    has_target_container(&#x27;x&#x27;),
    name=&#x27;skill_argument__place__target_container_name&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    place(&#x27;x&#x27;),
    notL(has_target_entity(&#x27;x&#x27;)),
    name=&#x27;skill_argument__place__target_entity_name&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">insert</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_container_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_entity_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">None</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">target_container_name=obj:0</td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="6" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    insert(&#x27;x&#x27;),
    has_target_container(&#x27;x&#x27;),
    name=&#x27;skill_argument__insert__target_container_name&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    insert(&#x27;x&#x27;),
    notL(has_target_entity(&#x27;x&#x27;)),
    name=&#x27;skill_argument__insert__target_entity_name&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pour</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_container_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_entity_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">None</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">target_container_name=obj:0</td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="6" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    pour(&#x27;x&#x27;),
    has_target_container(&#x27;x&#x27;),
    name=&#x27;skill_argument__pour__target_container_name&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    pour(&#x27;x&#x27;),
    notL(has_target_entity(&#x27;x&#x27;)),
    name=&#x27;skill_argument__pour__target_entity_name&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">push</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">None</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_entity_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_container_name</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">No role; optionally target_container_name=obj:1</td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="6" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    push(&#x27;x&#x27;),
    notL(has_target_entity(&#x27;x&#x27;)),
    name=&#x27;skill_argument__push__target_entity_name&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pull</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">None</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_entity_name</code>, <code style="white-space: normal; overflow-wrap: anywhere;">target_container_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">None</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">No role</td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="6" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    pull(&#x27;x&#x27;),
    notL(has_target_entity(&#x27;x&#x27;)),
    name=&#x27;skill_argument__pull__target_entity_name&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    pull(&#x27;x&#x27;),
    notL(has_target_container(&#x27;x&#x27;)),
    name=&#x27;skill_argument__pull__target_container_name&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">lift</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">None</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_entity_name</code>, <code style="white-space: normal; overflow-wrap: anywhere;">target_container_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">None</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">No role</td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="6" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    lift(&#x27;x&#x27;),
    notL(has_target_entity(&#x27;x&#x27;)),
    name=&#x27;skill_argument__lift__target_entity_name&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    lift(&#x27;x&#x27;),
    notL(has_target_container(&#x27;x&#x27;)),
    name=&#x27;skill_argument__lift__target_container_name&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">open_door</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_entity_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_container_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">None</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">target_entity_name=obj:0</td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="6" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    open_door(&#x27;x&#x27;),
    has_target_entity(&#x27;x&#x27;),
    name=&#x27;skill_argument__open_door__target_entity_name&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    open_door(&#x27;x&#x27;),
    notL(has_target_container(&#x27;x&#x27;)),
    name=&#x27;skill_argument__open_door__target_container_name&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">press</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_entity_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">target_container_name</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">None</td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">target_entity_name=obj:0</td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="6" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    press(&#x27;x&#x27;),
    has_target_entity(&#x27;x&#x27;),
    name=&#x27;skill_argument__press__target_entity_name&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    press(&#x27;x&#x27;),
    notL(has_target_container(&#x27;x&#x27;)),
    name=&#x27;skill_argument__press__target_container_name&#x27;,
)</code></pre></td>
    </tr>
  </tbody>
</table>

In particular, `place`, `insert`, and `pour` refer to the destination; their manipulated subject is carried by state. `push` has an optional destination and no explicit subject role. Optional does not mean arbitrary: a supplied value must still be usable. The composite schema formula is at [test_regr/VLABenchAgentInterface/world_graph.py:104](../world_graph.py#L104).

### How transition observations are obtained

`transition` is the fixed operation-transition concept, with links to its current and following operations. A tested plan creates a separate instance for each adjacent pair. `allowed_transition('t')` is observed by advancing the compiled progress state through the following operation. It therefore depends on the preceding subtask progress, not only on the two skill names.

`pattern_runtime.py` projects the actual generation formulas onto skill labels by existentially hiding argument/object labels. It retains ambiguous progress alternatives and uses legal EOS boundaries for completion. There is no second hand-written skill-pair grammar. Standalone structural compilations are cached; each subsequent operation advances with a DFA lookup. With a bound policy, verification reuses that policy's projection, including its declared limits and continuation language.

Both DataNode verification and decoding consume these progress observations. Decoding no longer supplies an unconditional true flag. Start and completion constraints cover invalid first operations, incomplete final subtasks, single-operation plans, and empty plans. The completion observation is independent of argument-pointer validity; schema and scene constraints diagnose those errors separately. After an invalid prefix, later transitions remain unreachable rather than silently restarting a subtask.

This change adds **two generic logical constraints**. The current totals are **3 causal, 24 generic, 6 scene, and 56 task constraints: 89 world constraints and 94 including grammar** (the causal group was added later and is active only at `causal`). Historical checkpoint ablation scores have not been rerun. Partial diagnostic scores for invalid plans can change because a pointer error no longer marks every transition invalid; the hard validity gate still requires every active constraint to be fully satisfied.

## Scene world families: six logical constraints

Each rule is one logical constraint using `ifL` as its outermost operator, first active at `scene`; all six remain active at higher levels.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 26%;">
    <col style="width: 9%;">
    <col style="width: 38%;">
    <col style="width: 27%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="26%" style="width: 26%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Rule name</th>
      <th scope="col" width="9%" style="width: 9%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Logical constraints</th>
      <th scope="col" width="38%" style="width: 38%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Example and meaning</th>
      <th scope="col" width="27%" style="width: 27%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">operation_pointer_is_valid</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A supplied <code style="white-space: normal; overflow-wrap: anywhere;">obj:7</code> must resolve in the entity table, even if that label is in the generation vocabulary.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L163">test_regr/VLABenchAgentInterface/world_graph.py:163</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    operation(&#x27;x&#x27;),
    andL(
        ifL(has_target_entity(&#x27;x&#x27;), entity_resolves(&#x27;x&#x27;)),
        ifL(has_target_container(&#x27;x&#x27;), container_resolves(&#x27;x&#x27;)),
    ),
    name=&#x27;operation_pointer_is_valid&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">button_entity_requires_press_skill</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">An operation targeting an entity classified as a button must use <code style="white-space: normal; overflow-wrap: anywhere;">press</code>.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L168">test_regr/VLABenchAgentInterface/world_graph.py:168</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    orL(entity_is_button(&#x27;x&#x27;), container_is_button(&#x27;x&#x27;)),
    press(&#x27;x&#x27;),
    name=&#x27;button_entity_requires_press_skill&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">press_requires_button_target</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">press</code> directed at an ordinary nonbutton object fails.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L169">test_regr/VLABenchAgentInterface/world_graph.py:169</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    press(&#x27;x&#x27;),
    orL(entity_is_button(&#x27;x&#x27;), container_is_button(&#x27;x&#x27;)),
    name=&#x27;press_requires_button_target&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick_excludes_static_fixtures</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Picking a static fixture fails.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L170">test_regr/VLABenchAgentInterface/world_graph.py:170</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    pick(&#x27;x&#x27;),
    notL(entity_is_static_fixture(&#x27;x&#x27;)),
    name=&#x27;pick_excludes_static_fixtures&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">container_destination_consistency</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">If the required destination is container A, placing into B fails.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L171">test_regr/VLABenchAgentInterface/world_graph.py:171</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(transition(&#x27;t&#x27;), next_has_destination(&#x27;t&#x27;), expected_destination_known(&#x27;t&#x27;)),
    destination_matches_expected(&#x27;t&#x27;),
    name=&#x27;container_destination_consistency&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">open_before_enclosed_placement</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A known closed enclosed destination must have been opened before placement. With no explicit planning facts the legacy closed-world fallback applies; unknown open/closed status with explicit facts does not become known closed.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_graph.py#L176">test_regr/VLABenchAgentInterface/world_graph.py:176</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        transition(&#x27;t&#x27;),
        next_is_placement(&#x27;t&#x27;),
        next_has_destination(&#x27;t&#x27;),
        destination_is_enclosed(&#x27;t&#x27;),
        orL(
            notL(planning_facts_supplied(&#x27;t&#x27;)),
            open_observed_false(&#x27;t&#x27;),
            closed_observed_true(&#x27;t&#x27;),
        ),
    ),
    destination_was_opened(&#x27;t&#x27;),
    name=&#x27;open_before_enclosed_placement&#x27;,
)</code></pre></td>
    </tr>
  </tbody>
</table>

## Task world families: 56 logical constraints

All families in this section first become active at **`learned`**, and remain active at `task` and `metadata`. Each named rule is one logical constraint using `ifL` as its outermost operator. Task identity comes from the selected ablation level, not from the rule's name.

### Caller/request rules: four logical constraints

These rules combine explicitly supplied request facts. When a request omits a restriction, its observation adapter supplies the corresponding neutral condition.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 25%;">
    <col style="width: 9%;">
    <col style="width: 39%;">
    <col style="width: 27%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="25%" style="width: 25%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Rule name</th>
      <th scope="col" width="9%" style="width: 9%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Logical constraints</th>
      <th scope="col" width="39%" style="width: 39%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Example and meaning</th>
      <th scope="col" width="27%" style="width: 27%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task_skill_sequence</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">For requested sequence <code style="white-space: normal; overflow-wrap: anywhere;">[pick, place]</code>, <code style="white-space: normal; overflow-wrap: anywhere;">pour</code> at index 1 fails. If a cycle is also supplied, both sequence and cycle must permit the chosen skill.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/task_lcs.py#L73">test_regr/VLABenchAgentInterface/knowledge/task_lcs.py:73</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(operation(&#x27;x&#x27;), choosing_skill(&#x27;x&#x27;)),
    orL(
        *(
            andL(
                skill_concept(&#x27;x&#x27;),
                requested_sequence_allows[skill_name](&#x27;x&#x27;),
                requested_cycle_allows[skill_name](&#x27;x&#x27;),
            )
            for skill_name, skill_concept in skill_concepts.items()
        ),
    ),
    name=&#x27;task_skill_sequence&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task_initial_target</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">When a request specifies entity A as the initial target, starting on entity B fails.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/task_lcs.py#L77">test_regr/VLABenchAgentInterface/knowledge/task_lcs.py:77</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        operation(&#x27;x&#x27;),
        request_initial_target_known(&#x27;x&#x27;),
        request_first_operation(&#x27;x&#x27;),
    ),
    request_target_matches(&#x27;x&#x27;),
    name=&#x27;task_initial_target&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task_sequence_completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">EOS before the requested sequence or cycle is complete fails.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/task_lcs.py#L79">test_regr/VLABenchAgentInterface/knowledge/task_lcs.py:79</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(operation(&#x27;x&#x27;), final_operation(&#x27;x&#x27;)),
    andL(request_sequence_complete(&#x27;x&#x27;), request_cycle_complete(&#x27;x&#x27;)),
    name=&#x27;task_sequence_completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">composite_subgoal_completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A request for at least three operations including <code style="white-space: normal; overflow-wrap: anywhere;">push</code> cannot finish after two operations or without <code style="white-space: normal; overflow-wrap: anywhere;">push</code>. Unsupported required skills also fail.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/task_lcs.py#L86">test_regr/VLABenchAgentInterface/knowledge/task_lcs.py:86</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(operation(&#x27;x&#x27;), final_operation(&#x27;x&#x27;)),
    andL(
        request_minimum_complete(&#x27;x&#x27;),
        request_required_skills_supported(&#x27;x&#x27;),
        *(
            ifL(
                requested_skill_requirements[skill_name](&#x27;x&#x27;),
                skill_history[skill_name](&#x27;x&#x27;),
            )
            for skill_name in skill_concepts
        ),
    ),
    name=&#x27;composite_subgoal_completion&#x27;,
)</code></pre></td>
    </tr>
  </tbody>
</table>

### Declared task sequences: 36 logical constraints

Each of the following **18 task/interface variants creates two logical constraints**: one sequence rule and one completion rule. The sequence rule requires the expected skill at each zero-based operation index; the completion rule requires the final index to equal the last declared index. For example, `select_fruit` requires `pick` at index 0, `place` at index 1, and cannot finish at index 0 or index 2. An explicit sequence request disables the default sequence via its guard; the separate request rules still enforce the request.

Sequence template: [test_regr/VLABenchAgentInterface/knowledge/task_lcs.py:103](../knowledge/task_lcs.py#L103). Completion template: [test_regr/VLABenchAgentInterface/knowledge/task_lcs.py:106](../knowledge/task_lcs.py#L106). Exact-position helper: [test_regr/VLABenchAgentInterface/knowledge/task_lcs.py:58](../knowledge/task_lcs.py#L58). Each row below links the declaration data as well as naming both instantiated rules.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 16%;">
    <col style="width: 20%;">
    <col style="width: 32%;">
    <col style="width: 8%;">
    <col style="width: 24%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="16%" style="width: 16%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Task / interface</th>
      <th scope="col" width="20%" style="width: 20%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Required skill sequence (example)</th>
      <th scope="col" width="32%" style="width: 32%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Exact rule names</th>
      <th scope="col" width="8%" style="width: 8%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Logical constraints</th>
      <th scope="col" width="24%" style="width: 24%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Data definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">add_condiment</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, pour</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__add_condiment__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__add_condiment__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L14">test_regr/VLABenchAgentInterface/knowledge/tables.py:14</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_add_condiment(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    orL(
        andL(notL(position_at_least_1(&#x27;x&#x27;)), pick(&#x27;x&#x27;)),
        andL(
            andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
            pour(&#x27;x&#x27;),
        ),
    ),
    name=&#x27;task__add_condiment__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_add_condiment(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
    name=&#x27;task__add_condiment__completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">density_qa</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">press</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__density_qa__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__density_qa__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L34">test_regr/VLABenchAgentInterface/knowledge/tables.py:34</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_density_qa(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    andL(notL(position_at_least_1(&#x27;x&#x27;)), press(&#x27;x&#x27;)),
    name=&#x27;task__density_qa__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_density_qa(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    notL(position_at_least_1(&#x27;x&#x27;)),
    name=&#x27;task__density_qa__completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">friction_qa</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">press</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__friction_qa__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__friction_qa__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L35">test_regr/VLABenchAgentInterface/knowledge/tables.py:35</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_friction_qa(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    andL(notL(position_at_least_1(&#x27;x&#x27;)), press(&#x27;x&#x27;)),
    name=&#x27;task__friction_qa__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_friction_qa(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    notL(position_at_least_1(&#x27;x&#x27;)),
    name=&#x27;task__friction_qa__completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">insert_flower</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, insert</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__insert_flower__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__insert_flower__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L15">test_regr/VLABenchAgentInterface/knowledge/tables.py:15</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_insert_flower(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    orL(
        andL(notL(position_at_least_1(&#x27;x&#x27;)), pick(&#x27;x&#x27;)),
        andL(
            andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
            insert(&#x27;x&#x27;),
        ),
    ),
    name=&#x27;task__insert_flower__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_insert_flower(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
    name=&#x27;task__insert_flower__completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">magnetism_qa</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">press</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__magnetism_qa__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__magnetism_qa__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L36">test_regr/VLABenchAgentInterface/knowledge/tables.py:36</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_magnetism_qa(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    andL(notL(position_at_least_1(&#x27;x&#x27;)), press(&#x27;x&#x27;)),
    name=&#x27;task__magnetism_qa__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_magnetism_qa(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    notL(position_at_least_1(&#x27;x&#x27;)),
    name=&#x27;task__magnetism_qa__completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">reflection_qa</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">press</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__reflection_qa__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__reflection_qa__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L37">test_regr/VLABenchAgentInterface/knowledge/tables.py:37</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_reflection_qa(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    andL(notL(position_at_least_1(&#x27;x&#x27;)), press(&#x27;x&#x27;)),
    name=&#x27;task__reflection_qa__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_reflection_qa(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    notL(position_at_least_1(&#x27;x&#x27;)),
    name=&#x27;task__reflection_qa__completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">select_book</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, pull</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__select_book__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__select_book__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L17">test_regr/VLABenchAgentInterface/knowledge/tables.py:17</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_book(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    orL(
        andL(notL(position_at_least_1(&#x27;x&#x27;)), pick(&#x27;x&#x27;)),
        andL(
            andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
            pull(&#x27;x&#x27;),
        ),
    ),
    name=&#x27;task__select_book__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_book(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
    name=&#x27;task__select_book__completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">select_chemistry_tube</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, lift</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__select_chemistry_tube__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__select_chemistry_tube__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L18">test_regr/VLABenchAgentInterface/knowledge/tables.py:18</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_chemistry_tube(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    orL(
        andL(notL(position_at_least_1(&#x27;x&#x27;)), pick(&#x27;x&#x27;)),
        andL(
            andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
            lift(&#x27;x&#x27;),
        ),
    ),
    name=&#x27;task__select_chemistry_tube__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_chemistry_tube(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
    name=&#x27;task__select_chemistry_tube__completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">select_drink</code> / compact</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, pull</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__select_drink__skill_sequence__compact</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__select_drink__completion__compact</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L45">test_regr/VLABenchAgentInterface/knowledge/tables.py:45</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_drink(&#x27;x&#x27;),
        interface_compact(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    orL(
        andL(notL(position_at_least_1(&#x27;x&#x27;)), pick(&#x27;x&#x27;)),
        andL(
            andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
            pull(&#x27;x&#x27;),
        ),
    ),
    name=&#x27;task__select_drink__skill_sequence__compact&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_drink(&#x27;x&#x27;),
        interface_compact(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
    name=&#x27;task__select_drink__completion__compact&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">select_drink</code> / full</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, lift, pull</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__select_drink__skill_sequence__full</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__select_drink__completion__full</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L20">test_regr/VLABenchAgentInterface/knowledge/tables.py:20</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_drink(&#x27;x&#x27;),
        interface_full(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    orL(
        andL(notL(position_at_least_1(&#x27;x&#x27;)), pick(&#x27;x&#x27;)),
        andL(
            andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
            lift(&#x27;x&#x27;),
        ),
        andL(
            andL(position_at_least_2(&#x27;x&#x27;), notL(position_at_least_3(&#x27;x&#x27;))),
            pull(&#x27;x&#x27;),
        ),
    ),
    name=&#x27;task__select_drink__skill_sequence__full&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_drink(&#x27;x&#x27;),
        interface_full(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    andL(position_at_least_2(&#x27;x&#x27;), notL(position_at_least_3(&#x27;x&#x27;))),
    name=&#x27;task__select_drink__completion__full&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">select_fruit</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__select_fruit__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__select_fruit__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L21">test_regr/VLABenchAgentInterface/knowledge/tables.py:21</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_fruit(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    orL(
        andL(notL(position_at_least_1(&#x27;x&#x27;)), pick(&#x27;x&#x27;)),
        andL(
            andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
            place(&#x27;x&#x27;),
        ),
    ),
    name=&#x27;task__select_fruit__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_fruit(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
    name=&#x27;task__select_fruit__completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">select_mahjong</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__select_mahjong__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__select_mahjong__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L22">test_regr/VLABenchAgentInterface/knowledge/tables.py:22</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_mahjong(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    orL(
        andL(notL(position_at_least_1(&#x27;x&#x27;)), pick(&#x27;x&#x27;)),
        andL(
            andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
            place(&#x27;x&#x27;),
        ),
    ),
    name=&#x27;task__select_mahjong__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_mahjong(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
    name=&#x27;task__select_mahjong__completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">select_painting</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">press</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__select_painting__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__select_painting__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L25">test_regr/VLABenchAgentInterface/knowledge/tables.py:25</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_painting(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    andL(notL(position_at_least_1(&#x27;x&#x27;)), press(&#x27;x&#x27;)),
    name=&#x27;task__select_painting__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_painting(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    notL(position_at_least_1(&#x27;x&#x27;)),
    name=&#x27;task__select_painting__completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">select_poker</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, lift</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__select_poker__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__select_poker__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L26">test_regr/VLABenchAgentInterface/knowledge/tables.py:26</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_poker(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    orL(
        andL(notL(position_at_least_1(&#x27;x&#x27;)), pick(&#x27;x&#x27;)),
        andL(
            andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
            lift(&#x27;x&#x27;),
        ),
    ),
    name=&#x27;task__select_poker__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_poker(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
    name=&#x27;task__select_poker__completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">select_toy</code> / compact</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__select_toy__skill_sequence__compact</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__select_toy__completion__compact</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L46">test_regr/VLABenchAgentInterface/knowledge/tables.py:46</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_toy(&#x27;x&#x27;),
        interface_compact(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    orL(
        andL(notL(position_at_least_1(&#x27;x&#x27;)), pick(&#x27;x&#x27;)),
        andL(
            andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
            place(&#x27;x&#x27;),
        ),
    ),
    name=&#x27;task__select_toy__skill_sequence__compact&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_toy(&#x27;x&#x27;),
        interface_compact(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
    name=&#x27;task__select_toy__completion__compact&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">select_toy</code> / full</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, lift, place</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__select_toy__skill_sequence__full</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__select_toy__completion__full</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L28">test_regr/VLABenchAgentInterface/knowledge/tables.py:28</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_toy(&#x27;x&#x27;),
        interface_full(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    orL(
        andL(notL(position_at_least_1(&#x27;x&#x27;)), pick(&#x27;x&#x27;)),
        andL(
            andL(position_at_least_1(&#x27;x&#x27;), notL(position_at_least_2(&#x27;x&#x27;))),
            lift(&#x27;x&#x27;),
        ),
        andL(
            andL(position_at_least_2(&#x27;x&#x27;), notL(position_at_least_3(&#x27;x&#x27;))),
            place(&#x27;x&#x27;),
        ),
    ),
    name=&#x27;task__select_toy__skill_sequence__full&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_select_toy(&#x27;x&#x27;),
        interface_full(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    andL(position_at_least_2(&#x27;x&#x27;), notL(position_at_least_3(&#x27;x&#x27;))),
    name=&#x27;task__select_toy__completion__full&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">speed_of_sound_qa</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">press</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__speed_of_sound_qa__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__speed_of_sound_qa__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L38">test_regr/VLABenchAgentInterface/knowledge/tables.py:38</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_speed_of_sound_qa(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    andL(notL(position_at_least_1(&#x27;x&#x27;)), press(&#x27;x&#x27;)),
    name=&#x27;task__speed_of_sound_qa__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_speed_of_sound_qa(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    notL(position_at_least_1(&#x27;x&#x27;)),
    name=&#x27;task__speed_of_sound_qa__completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">thermal_expansion_qa</code> / both</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">press</code></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__thermal_expansion_qa__skill_sequence</code>; <code style="white-space: normal; overflow-wrap: anywhere;">task__thermal_expansion_qa__completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L39">test_regr/VLABenchAgentInterface/knowledge/tables.py:39</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formulas</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_thermal_expansion_qa(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        choosing_skill(&#x27;x&#x27;),
    ),
    andL(notL(position_at_least_1(&#x27;x&#x27;)), press(&#x27;x&#x27;)),
    name=&#x27;task__thermal_expansion_qa__skill_sequence&#x27;,
)</code></pre><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        task_thermal_expansion_qa(&#x27;x&#x27;),
        notL(explicit_sequence_supplied(&#x27;x&#x27;)),
        final_operation(&#x27;x&#x27;),
    ),
    notL(position_at_least_1(&#x27;x&#x27;)),
    name=&#x27;task__thermal_expansion_qa__completion&#x27;,
)</code></pre></td>
    </tr>
  </tbody>
</table>

“Both” shares one pair of rules across compact and full interfaces. `select_drink` and `select_toy` each need separate compact/full pairs. Thus 16 tasks with sequence defaults produce 18 variants, rather than 32 interface-specific variants.

### Complex-task obligations: eight logical constraints

These require a minimum total operation count and the presence of every listed skill somewhere in the completed history. They do not impose a complete order; grammar, supplied milestone requirements, and other active rules further restrict possible orders. The example in each row is its explicit minimum/history obligation, not a claim that any ordering of those skills is an accepted full plan.

Formula template: [test_regr/VLABenchAgentInterface/knowledge/task_lcs.py:123](../knowledge/task_lcs.py#L123).

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 28%;">
    <col style="width: 9%;">
    <col style="width: 29%;">
    <col style="width: 9%;">
    <col style="width: 25%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="28%" style="width: 28%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Exact rule name</th>
      <th scope="col" width="9%" style="width: 9%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Minimum operations</th>
      <th scope="col" width="29%" style="width: 29%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Required history skills (example obligation)</th>
      <th scope="col" width="9%" style="width: 9%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Logical constraints</th>
      <th scope="col" width="25%" style="width: 25%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Data definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__book_rearrange__obligations</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">3</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, push, pull</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L67">test_regr/VLABenchAgentInterface/knowledge/tables.py:67</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(task_book_rearrange(&#x27;x&#x27;), final_operation(&#x27;x&#x27;)),
    andL(
        position_at_least_2(&#x27;x&#x27;),
        history_includes_pick(&#x27;x&#x27;),
        history_includes_push(&#x27;x&#x27;),
        history_includes_pull(&#x27;x&#x27;),
    ),
    name=&#x27;task__book_rearrange__obligations&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__cook_dishes__obligations</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">4</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L60">test_regr/VLABenchAgentInterface/knowledge/tables.py:60</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(task_cook_dishes(&#x27;x&#x27;), final_operation(&#x27;x&#x27;)),
    andL(
        position_at_least_3(&#x27;x&#x27;),
        history_includes_pick(&#x27;x&#x27;),
        history_includes_place(&#x27;x&#x27;),
    ),
    name=&#x27;task__cook_dishes__obligations&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__cool_drink__obligations</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L61">test_regr/VLABenchAgentInterface/knowledge/tables.py:61</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(task_cool_drink(&#x27;x&#x27;), final_operation(&#x27;x&#x27;)),
    andL(
        position_at_least_1(&#x27;x&#x27;),
        history_includes_pick(&#x27;x&#x27;),
        history_includes_place(&#x27;x&#x27;),
    ),
    name=&#x27;task__cool_drink__obligations&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__hammer_nail_and_hang_picture__obligations</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">3</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, place, push</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L63">test_regr/VLABenchAgentInterface/knowledge/tables.py:63</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(task_hammer_nail_and_hang_picture(&#x27;x&#x27;), final_operation(&#x27;x&#x27;)),
    andL(
        position_at_least_2(&#x27;x&#x27;),
        history_includes_pick(&#x27;x&#x27;),
        history_includes_place(&#x27;x&#x27;),
        history_includes_push(&#x27;x&#x27;),
    ),
    name=&#x27;task__hammer_nail_and_hang_picture__obligations&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__make_juice__obligations</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L62">test_regr/VLABenchAgentInterface/knowledge/tables.py:62</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(task_make_juice(&#x27;x&#x27;), final_operation(&#x27;x&#x27;)),
    andL(
        position_at_least_1(&#x27;x&#x27;),
        history_includes_pick(&#x27;x&#x27;),
        history_includes_place(&#x27;x&#x27;),
    ),
    name=&#x27;task__make_juice__obligations&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__manage_study_table__obligations</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">3</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L70">test_regr/VLABenchAgentInterface/knowledge/tables.py:70</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(task_manage_study_table(&#x27;x&#x27;), final_operation(&#x27;x&#x27;)),
    andL(
        position_at_least_2(&#x27;x&#x27;),
        history_includes_pick(&#x27;x&#x27;),
        history_includes_place(&#x27;x&#x27;),
    ),
    name=&#x27;task__manage_study_table__obligations&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__take_chemistry_experiment__obligations</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">3</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, place, pour</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L65">test_regr/VLABenchAgentInterface/knowledge/tables.py:65</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(task_take_chemistry_experiment(&#x27;x&#x27;), final_operation(&#x27;x&#x27;)),
    andL(
        position_at_least_2(&#x27;x&#x27;),
        history_includes_pick(&#x27;x&#x27;),
        history_includes_place(&#x27;x&#x27;),
        history_includes_pour(&#x27;x&#x27;),
    ),
    name=&#x27;task__take_chemistry_experiment__obligations&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">task__texas_holdem__obligations</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">2</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">pick, place</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L69">test_regr/VLABenchAgentInterface/knowledge/tables.py:69</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="5" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(task_texas_holdem(&#x27;x&#x27;), final_operation(&#x27;x&#x27;)),
    andL(
        position_at_least_1(&#x27;x&#x27;),
        history_includes_pick(&#x27;x&#x27;),
        history_includes_place(&#x27;x&#x27;),
    ),
    name=&#x27;task__texas_holdem__obligations&#x27;,
)</code></pre></td>
    </tr>
  </tbody>
</table>

### Grounded requirement families: eight logical constraints

VLABench uses the detailed requirement-node branch. Destination, affordance, compatibility, milestone order, goal completion, and preservation share the same logical structure, shown here for affordance:

```python
ifL(
    andL(affordance_required('r'), observation_known('r')),
    required_fact_holds('r'),
    name='grounded__affordance',
)
```

One logical constraint is declared for each family; each supplied required fact has its own requirement node/event. A known false fact violates the rule even when another fact is unknown. Unknown facts have `known=False`; they are not asserted satisfied and are reported as incomplete evidence. Absence of supplied physical requirements does not invent simulator facts.

The eight logical constraints are instantiated at [domiknows/generation/grounded.py:102](../../../domiknows/generation/grounded.py#L102). Family meanings are declared at [domiknows/generation/grounded.py:16](../../../domiknows/generation/grounded.py#L16). Holding and capacity use the expanded formulas below.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 25%;">
    <col style="width: 9%;">
    <col style="width: 40%;">
    <col style="width: 26%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="25%" style="width: 25%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Exact rule name</th>
      <th scope="col" width="9%" style="width: 9%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Logical constraints</th>
      <th scope="col" width="40%" style="width: 40%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Example of a violation when applicable/known</th>
      <th scope="col" width="26%" style="width: 26%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Formula definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">grounded__destination</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A supplied required destination is A, but the operation targets B.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L102">domiknows/generation/grounded.py:102</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(destination_required(&#x27;r&#x27;), observation_known(&#x27;r&#x27;)),
    required_fact_holds(&#x27;r&#x27;),
    name=&#x27;grounded__destination&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">grounded__holding</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">The observed hand is empty when a nonidempotent manipulation requires holding the subject.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L93">domiknows/generation/grounded.py:93</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        holding_required(&#x27;r&#x27;),
        orL(operation_is_idempotent(&#x27;r&#x27;), hand_state_known(&#x27;r&#x27;)),
    ),
    orL(
        operation_is_idempotent(&#x27;r&#x27;),
        andL(
            object_is_held(&#x27;r&#x27;),
            orL(subject_unspecified(&#x27;r&#x27;), held_object_matches_subject(&#x27;r&#x27;)),
        ),
    ),
    name=&#x27;grounded__holding&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">grounded__capacity</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A grasp is attempted with zero hand capacity, or with a different object already held.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L99">domiknows/generation/grounded.py:99</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(
        capacity_required(&#x27;r&#x27;),
        andL(
            hand_capacity_known(&#x27;r&#x27;),
            orL(hand_capacity_is_zero(&#x27;r&#x27;), hand_state_known(&#x27;r&#x27;)),
        ),
    ),
    andL(
        hand_capacity_is_positive(&#x27;r&#x27;),
        notL(
            andL(
                hand_state_known(&#x27;r&#x27;),
                object_is_held(&#x27;r&#x27;),
                notL(held_object_matches_subject(&#x27;r&#x27;)),
            ),
        ),
    ),
    name=&#x27;grounded__capacity&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">grounded__affordance</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A required grasp capability/precondition is known false, or a measured selection requirement fails.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L102">domiknows/generation/grounded.py:102</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(affordance_required(&#x27;r&#x27;), observation_known(&#x27;r&#x27;)),
    required_fact_holds(&#x27;r&#x27;),
    name=&#x27;grounded__affordance&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">grounded__compatibility</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">The subject/destination pair or container is explicitly incompatible.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L102">domiknows/generation/grounded.py:102</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(compatibility_required(&#x27;r&#x27;), observation_known(&#x27;r&#x27;)),
    required_fact_holds(&#x27;r&#x27;),
    name=&#x27;grounded__compatibility&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">grounded__milestone_order</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">A required predecessor milestone is known incomplete.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L102">domiknows/generation/grounded.py:102</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(milestone_order_required(&#x27;r&#x27;), observation_known(&#x27;r&#x27;)),
    required_fact_holds(&#x27;r&#x27;),
    name=&#x27;grounded__milestone_order&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">grounded__goal_completion</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">EOS is attempted while a supplied goal or required milestone is known incomplete.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L102">domiknows/generation/grounded.py:102</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(goal_completion_required(&#x27;r&#x27;), observation_known(&#x27;r&#x27;)),
    required_fact_holds(&#x27;r&#x27;),
    name=&#x27;grounded__goal_completion&#x27;,
)</code></pre></td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><code style="white-space: normal; overflow-wrap: anywhere;">grounded__preservation</code></td>
      <td style="text-align: right; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">1</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">An operation violates a supplied protected-entity condition or invariant.</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L102">domiknows/generation/grounded.py:102</a></td>
    </tr>
    <tr data-constraint-example="true">
      <td colspan="4" style="text-align: left; vertical-align: top; padding: 0.35em 0.65em 0.85em; overflow-wrap: anywhere;"><strong>DomiKnowS formula</strong><pre style="white-space: pre-wrap; overflow-wrap: anywhere; margin: 0.5em 0; line-height: 1.45;"><code>ifL(
    andL(preservation_required(&#x27;r&#x27;), observation_known(&#x27;r&#x27;)),
    required_fact_holds(&#x27;r&#x27;),
    name=&#x27;grounded__preservation&#x27;,
)</code></pre></td>
    </tr>
  </tbody>
</table>

For a holding requirement, the expanded formula is:

```text
family AND (idempotent OR hand_known)
    -> idempotent OR (held_exists AND (subject_unspecified OR held_matches_subject))
```

For a capacity requirement:

```text
family AND capacity_known AND (capacity_zero OR hand_known)
    -> capacity_positive AND NOT (hand_known AND held_exists AND NOT held_matches_subject)
```

A zero-capacity hand is therefore rejected even if occupancy is unknown. Holding/capacity formulas consume the explicit Boolean observations; other physical/numeric requirements consume separately grounded comparisons. The legacy operation/episode `applicable/satisfied` concepts are retained for custom rules and other adapters. The legacy `detailed=False` branch is not an additional set of VLABench default rules and is excluded from these counts.

## Where declarations end and execution begins

The following modules supply data, compile formulas, or evaluate them; their helper/runtime classes are not additional logical-constraint operator classes.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <colgroup>
    <col style="width: 24%;">
    <col style="width: 36%;">
    <col style="width: 40%;">
  </colgroup>
  <thead>
    <tr>
      <th scope="col" width="24%" style="width: 24%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Responsibility</th>
      <th scope="col" width="36%" style="width: 36%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Source entry point</th>
      <th scope="col" width="40%" style="width: 40%; text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">What it contributes to the inventory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Vocabulary, skill roles, subtask contracts</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_domain.py#L123">test_regr/VLABenchAgentInterface/world_domain.py:123</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Declarative data used by graph construction; no extra logical constraints.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Task sequence and obligation data</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/tables.py#L13">test_regr/VLABenchAgentInterface/knowledge/tables.py:13</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Data expanded into the 36 sequence/completion and eight obligation logical constraints.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">World observation bindings</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../semantics.py#L208">test_regr/VLABenchAgentInterface/semantics.py:208</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Primitive facts and history observations; no extra logical constraints.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Task identity/position/request observations</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../knowledge/task_lcs.py#L160">test_regr/VLABenchAgentInterface/knowledge/task_lcs.py:160</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Groundings for declared task concepts; no extra logical constraints.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">World constraint compilation</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../constraint_compiler.py#L44">test_regr/VLABenchAgentInterface/constraint_compiler.py:44</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Compiles active declared LCs using concept bindings.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Generation compilation and context binding</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../generation_runtime.py#L84">test_regr/VLABenchAgentInterface/generation_runtime.py:84</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Compiles the grammar and world program; continuation grammars use the same declarations.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Requirement DataNodes</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_runtime.py#L26">test_regr/VLABenchAgentInterface/world_runtime.py:26</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Creates per-fact groundings; the number depends on supplied facts.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Plan materialization/verification</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../world_runtime.py#L234">test_regr/VLABenchAgentInterface/world_runtime.py:234</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Evaluates active world formulas on DataNodes.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Grounded numeric/state observations</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded_facts.py#L262">domiknows/generation/grounded_facts.py:262</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Supplies requirements, measurements, and symbolic state; no extra LC declarations.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Auxiliary grammar projection</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/dfa/auxiliary.py#L94">domiknows/generation/dfa/auxiliary.py:94</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Projects valid progress assignments onto tokens by reading formulas/relations; no extra knowledge rule family.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Causal (id-only) DFA</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../official/causal_dfa.py#L37">test_regr/VLABenchAgentInterface/official/causal_dfa.py:37</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Compiled id-only logical constraints for the <code style="white-space: normal; overflow-wrap: anywhere;">causal</code> level; three formulas in official/causal_graph.py and prefix observations in official/causal_dfa.py.</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Episode decoding knowledge DFA</td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../official/decoding_graph.py#L50">test_regr/VLABenchAgentInterface/official/decoding_graph.py:50</a>; <a href="../official/knowledge_dfa.py#L26">official/knowledge_dfa.py:26</a></td>
      <td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Declares the five episode formulas; the DFA grounds the model's answers, the named-pattern prefixes and the instruction candidates as observations and admits operations and plan endings through the compiled programs.</td>
    </tr>
  <tr><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Compiled subtask progress</td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;"><a href="../pattern_runtime.py#L110">pattern_runtime.py:110</a></td><td style="text-align: left; vertical-align: top; padding: 0.55em 0.65em; overflow-wrap: anywhere;">Cached skill projection of the generation formulas; supplies per-transition progress and start/completion observations.</td></tr>
</tbody>
</table>

The generation graph provides syntax and finite language membership. World verification checks operation/transition/requirement facts; it is not a direct token-stream EOS or argument-serialization verifier. The decoder combines both layers. Plain parsing checks, maximum decode-step policies, primitive numeric comparisons, and simulator execution are not counted as DomiKnowS logical constraints.

## Checking and interpreting counts

The inventory was checked against the complete declared formulas in the current graph build: **5 grammar logical constraints and 89 world logical constraints**, with world groups **causal=3, generic=24, scene=6, task=56**. The separate official-protocol `causal` graph declares three additional `ifL` formulas; its grammar remains NativePlanDFA, so its count is 3, not 5 + 3. The separate decoding graph declares five more `ifL` formulas. Across the paths there are **102 declarations**, classified by their outermost operator in the operator-class table; no current level enables all 102 together.

Counts can change with custom task knowledge/excluded tasks, custom constraint builders, disabling default constraints, or changes that add or remove complete declared formulas. Changing entity vocabulary size, `max_operations`, or continuation data can change a formula's alternatives and limits without adding a logical constraint. Actual active groundings also depend on plan length, task guards, and supplied requirements. Source lines are the local snapshot cited here and can move after code edits.

These counts and examples preserve the tier meaning and historical-results caveat in [KNOWLEDGE_ABLATION.md](KNOWLEDGE_ABLATION.md). They do not rerun or revise its checkpoint evaluation results.
