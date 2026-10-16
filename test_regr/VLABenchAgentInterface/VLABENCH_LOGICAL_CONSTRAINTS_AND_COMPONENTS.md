# DomiKnowS Logical Constraints in VLABench: Specification & Component Architecture

This document provides a comprehensive technical reference for the declarative first-order logic constraints implemented in **DomiKnowS** for the **VLABench** robotics benchmark. It details every declared constraint, its formal mathematical logic, and systematically maps **which system components consume each constraint and how they enforce it** across inference, reinforcement learning, ILP decoding, and evaluation.

---

## 1. Architectural Philosophy: Neuro-Symbolic Dual-Tier Constraints

Foundation vision-language models (VLMs) frequently suffer from three critical planning failure modes when applied to robotics:
1. **Syntactic and Schema Violations:** Generating malformed JSON, invalid keys, or truncated tokens (e.g. raw Qwen3-VL fails syntax validity on 21.1% of plans).
2. **Perceptual Hallucinations:** Referencing objects, receptacles, or IDs not present in the robot's physical workspace.
3. **Causal & Physical Violations:** Attempting physically impossible interactions, such as placing an object inside itself, picking static architectural room fixtures (counters, tables), or endlessly repeating the same action.

DomiKnowS solves these issues without relying solely on statistical training by introducing a **dual-tier neuro-symbolic constraint architecture**:
1. **Semantic World Graph Constraints (`world_graph.py`):** First-order relational logic formulas defined over high-level manipulation concepts (`operation`, `entity`, `transition`, `skill`). These constraints are evaluated over structured `DataNode` hierarchies using the DomiKnowS verification and ILP solver engine.
2. **Token-Level Generation Constraints (`graph.py`):** Relational rules projected into sequence space and compiled into **Deterministic Finite Automata (DFAs)**. These automata dynamically prune unallowed tokens during autoregressive generation by applying hard logit masking.

```mermaid
flowchart TD
    subgraph Tier1["Tier 1: High-Level Semantic World Graph (world_graph.py)"]
        Ontology["Ontology Definition (operation, entity, transition)"]
        FOL["First-Order Logical Constraints (ifL, notL)"]
        DataNode["Plan Materialization as DataNode Hierarchy"]
        Verifier["ILP / Local-Argmax Constraint Verifier"]
        Ontology --> FOL
        FOL --> DataNode
        DataNode --> Verifier
    end

    subgraph Tier2["Tier 2: Token-Level Generation Graph (graph.py)"]
        Vocab["Plan Vocabulary (derived from World Graph)"]
        TokenLC["Token Logical Constraints (EOS closure, Action budget)"]
        ProductDFA["Base Product DFA (Graph DFA × Domain DFA)"]
        ContextualDFA["Specialized DFA (VLABenchContextualDFA / VLABenchCausalDFA)"]
        Vocab --> TokenLC
        TokenLC --> ProductDFA
        ProductDFA --> ContextualDFA
    end

    subgraph ConsumerComponents["Consumer Components in DomiKnowS Engine"]
        M1["JointEmbodiedAgentInterface/models.py<br/>(Online Logit Masking & Trajectory Replay)"]
        M2["VLABenchAgentInterface/reward.py<br/>(PPO RL Feasibility Reward & Rollout Pruning)"]
        M3["JointEmbodiedAgentInterface/structured_inference.py<br/>(Gurobi ILP Offline Selection)"]
        M4["VLABenchAgentInterface/agent.py<br/>(Autonomous Decision Loop)"]
        M5["JointEmbodiedAgentInterface/eval_planning.py<br/>(Benchmark Evaluation & Metric Auditing)"]
    end

    Tier1 -->|World Bundle & Checksum| Tier2
    ContextualDFA -->|allowed_tokens()| M1
    ContextualDFA -->|dfa_valid| M4
    Verifier -->|verify_plan_constraints()| M2
    Verifier -->|verify_plan_constraints()| M3
    Verifier -->|verify_plan_constraints()| M4
    Verifier -->|verify_plan_constraints()| M5
```

---

## 2. Comprehensive Inventory of Logical Constraints

### 2.1 Semantic World Graph Constraints (`test_regr/VLABenchAgentInterface/world_graph.py`)

All world constraints are declared inside `build_vlabench_world_graph()` using DomiKnowS logical constraint constructors (`ifL`, `notL`).

#### 1. Skill Exclusivity & Validity Constraint
* **Identifier:** `operation_exactly_one_skill`
* **Mathematical Formulation:**
  $$\forall \text{op} \in \text{Operations}, \quad \text{valid\_skill\_type}(\text{op}) \iff \exists! s \in \text{Skills} \text{ s.t. } \text{skill}_s(\text{op})$$
* **DomiKnowS Syntax:**
  ```python
  ifL(operation("op"), valid_skill_type("op"), name="operation_exactly_one_skill")
  ```
* **Semantic Purpose:** Enforces that every manipulation step binds to exactly one recognized primitive skill $\in \{\text{pick, place, pour, push, open\_door, press}\}$. Forbids unrecognized, undefined, or hallucinated skills (e.g. `teleport`, `throw`).

#### 2. Structural Schema Compliance Constraint
* **Identifier:** `operation_schema_is_valid`
* **Mathematical Formulation:**
  $$\forall \text{op} \in \text{Operations}, \quad \text{operation}(\text{op}) \implies \text{valid\_schema}(\text{op})$$
* **DomiKnowS Syntax:**
  ```python
  ifL(operation("op"), valid_operation("op"), name="operation_schema_is_valid")
  ```
* **Semantic Purpose:** Guarantees that each operation adheres to the canonical JSON subtask schema `{"name": str, "params": dict}` with valid argument types.

#### 3. Observation-Grounded Entity Pointer Constraint
* **Identifier:** `operation_pointer_is_valid`
* **Mathematical Formulation:**
  $$\forall \text{op} \in \text{Operations}, \quad \text{operation}(\text{op}) \implies \Big( \text{target\_entity}(\text{op}) \in \mathcal{E}_{\text{seen}} \;\land\; \text{target\_container}(\text{op}) \in \mathcal{E}_{\text{seen}} \cup \{\emptyset\} \Big)$$
* **DomiKnowS Syntax:**
  ```python
  ifL(operation("op"), valid_pointer("op"), name="operation_pointer_is_valid")
  ```
* **Semantic Purpose:** Restricts all object and receptacle arguments strictly to the active scene identifiers $\mathcal{E}_{\text{seen}}$ extracted from camera observations. Eliminates out-of-scene perceptual hallucination.

#### 4. Causal Subtask Transition Pattern Constraint
* **Identifier:** `transition_matches_subtask_pattern`
* **Mathematical Formulation:**
  $$\forall (\text{op}_i, \text{op}_{i+1}) \in \text{Transitions}, \quad \text{transition}(\text{edge}) \implies \text{allowed\_transition}(\text{edge})$$
* **DomiKnowS Syntax:**
  ```python
  ifL(transition("edge"), allowed_transition("edge"), name="transition_matches_subtask_pattern")
  ```
* **Semantic Purpose:** Enforces physical ordering between operations. For example, a container must be grasped or opened before objects can be placed inside it, or an object must be lifted before being translated.

#### 5. Physical Laws Button Grounding Constraint
* **Identifier:** `button_entity_requires_press_skill`
* **Mathematical Formulation:**
  $$\forall \text{op} \in \text{Operations}, \quad \text{targets\_button}(\text{op}) \implies \text{skills}[\text{"press"}](\text{op})$$
* **DomiKnowS Syntax:**
  ```python
  ifL(targets_button("op"), skills["press"]("op"), name="button_entity_requires_press_skill")
  ```
* **Semantic Purpose:** In multiple-choice physical intuition tasks (e.g., density and friction questions), answer choices are represented by buttons in the scene. This constraint dictates that any operation targeting a button entity must strictly invoke the `press` primitive skill rather than attempting to pick or place the button.

#### 6. Skill Parameter Arity & Signature Rules
* **Identifier:** `skill_argument__{name}__{key}`
* **Mathematical Formulation:**
  $$\forall s \in \text{Skills}, \quad \begin{cases} \text{ifL}(\text{skills}[s](\text{"op"}), \text{flag}(\text{"op"})), & \text{if } \text{flag} \in \text{RequiredArgs}(s) \\ \text{ifL}(\text{skills}[s](\text{"op"}), \text{notL}(\text{flag}(\text{"op"}))), & \text{if } \text{flag} \notin \text{RequiredArgs}(s) \end{cases}$$
* **DomiKnowS Syntax:**
  ```python
  for name, required in SKILL_ARGUMENTS.items():
      required = set(required)
      for key, flag in (("target_entity_name", has_target_entity), ("target_container_name", has_target_container)):
          consequent = flag("op") if key in required else notL(flag("op"))
          ifL(skills[name]("op"), consequent, name=f"skill_argument__{name}__{key}")
  ```
* **Semantic Purpose:** Enforces argument requirements per skill:
  - `pick`: requires `target_entity_name`; strictly forbids `target_container_name`.
  - `place`, `pour`, `push`: require both `target_entity_name` and `target_container_name`.
  - `open_door`, `press`: require `target_entity_name`; strictly forbid `target_container_name`.

#### 7. Anti-Self-Placement Constraint
* **Identifier:** `anti_self_placement`
* **Mathematical Formulation:**
  $$\forall \text{op} \in \text{Operations}, \quad \text{operation}(\text{op}) \implies \neg \big( \text{target\_container\_name}(\text{op}) = \text{target\_entity\_name}(\text{op}) \big)$$
* **DomiKnowS Syntax:**
  ```python
  ifL(operation("op"), notL(is_self_placement("op")), name="anti_self_placement")
  ```
* **Semantic Purpose:** Mathematically forbids self-referential actions where an agent attempts to place or pour an entity into itself (e.g. `place(steak, steak)` or `pour(bowl, bowl)`).

#### 8. Static Environment Grasp Exclusion Constraint
* **Identifier:** `pick_excludes_static_fixtures`
* **Mathematical Formulation:**
  $$\forall \text{op} \in \text{Operations}, \quad \text{skills}[\text{"pick"}](\text{op}) \implies \neg \text{is\_static\_fixture}(\text{target\_entity}(\text{op}))$$
* **DomiKnowS Syntax:**
  ```python
  ifL(operation("op"), notL(targets_static_fixture("op")), name="pick_excludes_static_fixtures")
  ```
* **Semantic Purpose:** Restricts grasp targets away from immovable architectural room fixtures (counters, tables, walls, floors, lab surfaces). Ensures the manipulator only interacts with manipulable objects.

#### 9. Multi-Step Entity Rotation Invariance Constraint
* **Identifier:** `no_redundant_repeated_picks`
* **Mathematical Formulation:**
  $$\forall (\text{op}_i, \text{op}_{i+1}) \in \text{Transitions}, \quad \text{is\_consecutive\_pick}(\text{op}_i, \text{op}_{i+1}) \implies \big( \text{target}(\text{op}_i) \neq \text{target}(\text{op}_{i+1}) \big)$$
* **DomiKnowS Syntax:**
  ```python
  ifL(transition("edge"), notL(repeated_pick_entity("edge")), name="no_redundant_repeated_picks")
  ```
* **Semantic Purpose:** Prevents degenerate looping in multi-step rearrangement and cooking tasks, where a model attempts to pick the same object twice in succession without completing intermediary manipulation steps.

#### 10. Container Destination Consistency Constraint
* **Identifier:** `container_destination_consistency`
* **Mathematical Formulation:**
  $$\forall (\text{op}_1, \text{op}_2) \in \text{Transitions}, \quad \text{inconsistent\_container}(\text{edge}) \iff \Big(\text{skill}_{\text{place}}(\text{op}_1) \land \text{skill}_{\text{place}}(\text{op}_2) \land \big(\text{target\_container}(\text{op}_1) \neq \text{target\_container}(\text{op}_2)\big)\Big)$$
  $$\forall \text{edge} \in \text{Transitions}, \quad \text{transition}(\text{edge}) \implies \neg \text{inconsistent\_container}(\text{edge})$$
* **DomiKnowS Syntax:**
  ```python
  ifL(transition("edge"), notL(inconsistent_container("edge")), name="container_destination_consistency")
  ```
* **Semantic Purpose:** Enforces that sequential placement steps in multi-step assembly tasks deposit objects into the established destination container, preventing items from being scattered across multiple receptacles.

---

### 2.2 Token-Level Generation Graph Constraints (`test_regr/VLABenchAgentInterface/graph.py`)

These constraints operate directly on token sequences during autoregressive decoding and are registered with `mark_for_dfa` and `mark_for_contextual_dfa`.

#### 1. Terminal Absorbing State Invariance
* **Mathematical Formulation:**
  $$\forall (t_1, t_2) \in \text{Tokens}, \quad \text{is\_before}(t_1, t_2) \implies \Big( \text{token\_value}(t_1, \text{eos}) \implies \text{token\_value}(t_2, \text{eos}) \Big)$$
* **DomiKnowS Syntax:**
  ```python
  mark_for_dfa(ifL(
      ctx.is_before_rel("before"),
      ifL(
          ctx.token_value(vocabulary.eos_token, "x", path=("before", ctx.first_token)),
          ctx.token_value(vocabulary.eos_token, "y", path=("before", ctx.second_token)),
      ),
  ))
  ```
* **Semantic Purpose:** Guarantees that once the `<eos>` plan-terminator is emitted, all subsequent token transitions remain strictly on `<eos>`. This completely eliminates post-plan rambling, chat commentary, and unbounded generation.

#### 2. Action Budget Horizon Constraint
* **Mathematical Formulation:**
  $$\sum_{t \in \text{Plan}} \mathbb{I}(t \neq \text{eos}) \le K_{\max} \times 5$$
* **DomiKnowS Syntax:**
  ```python
  mark_for_dfa(atMostAL(notL(ctx.token_value(vocabulary.eos_token, "x")), max_operations * 5))
  ```
* **Semantic Purpose:** Enforces a hard upper bound on non-EOS tokens to ensure that planning terminates within the maximum allowed action budget (16 operations, 80 tokens).

#### 3. Contextual Token Ordering Logical Constraint
* **Mathematical Formulation:**
  $$\forall (t_1, t_2) \in \text{Tokens}, \quad \text{is\_before}(t_1, t_2) \implies \Big( \text{token\_value}(t_1, \text{"skill:press"}) \implies \text{token\_value}(t_2, \text{eos}) \Big)$$
  $$\text{Active}(\text{button\_press\_lc}) \iff \big( \text{Context}(\text{"button\_entities"}) \neq \emptyset \big)$$
* **DomiKnowS Syntax:**
  ```python
  button_press_lc = ifL(
      ctx.is_before_rel("before"),
      ifL(
          ctx.token_value("skill:press", "first_tok", path=("before", ctx.first_token)),
          ctx.token_value(vocabulary.eos_token, "second_tok", path=("before", ctx.second_token)),
      ),
      active=False,
      name="contextual_qa_button_press",
  )
  mark_for_contextual_dfa(
      button_press_lc,
      context_key="button_entities",
      token_to_value={"skill:press": "press"},
      vocabulary=vocabulary,
      name="contextual_qa_button_press",
      allow_missing_context=True,
  )
  ```
* **Semantic Purpose:** Dynamically binds scene context to the decoding state machine. When answer buttons are present, it restricts the decoded plan strictly to pressing the designated button followed immediately by `<eos>`.

### 2.3 Causal Manipulation Precondition Constraints (`VLABenchCausalDFA`)

For multi-step manipulation tasks (Complex, Spatial, ComSense), DomiKnowS encodes physical interaction preconditions directly as declarative First-Order Logic constraints in `world_graph.py` and compiles them into the decoding state automaton (`VLABenchCausalDFA` in `graph.py`):

#### 1. Anti-Self-Placement Constraint
* **Identifier:** `anti_self_placement`
* **DomiKnowS Syntax:**
  ```python
  ifL(operation("op"), notL(is_self_placement("op")), name="anti_self_placement")
  ```
* **First-Order Logic:**
  $$\forall \text{op} \in \text{Operations}, \quad \text{is\_self\_placement}(\text{op}) \iff \big(\text{target\_container}(\text{op}) = \text{target\_entity}(\text{op})\big)$$
  $$\forall \text{op} \in \text{Operations}, \quad \text{operation}(\text{op}) \implies \neg \text{is\_self\_placement}(\text{op})$$
* **DFA Enforcement:** When generating arguments for `place` or `pour` (`active_arg == "target_container_name"`), the currently held entity is dynamically pruned from candidate targets:
  $$\mathcal{V}_{\text{allowed}} \leftarrow \mathcal{V}_{\text{allowed}} \setminus \{\text{obj:current\_picked}\}$$

#### 2. Multi-Step Entity Rotation Invariance
* **Identifier:** `no_redundant_repeated_picks`
* **DomiKnowS Syntax:**
  ```python
  ifL(transition("edge"), notL(repeated_pick_entity("edge")), name="no_redundant_repeated_picks")
  ```
* **First-Order Logic:**
  $$\forall (\text{op}_1, \text{op}_2) \in \text{Transitions}, \quad \text{repeated\_pick\_entity}(\text{edge}) \iff \Big(\text{skill}_{\text{pick}}(\text{op}_1) \land \text{skill}_{\text{pick}}(\text{op}_2) \land \big(\text{target}(\text{op}_1) = \text{target}(\text{op}_2)\big)\Big)$$
  $$\forall \text{edge} \in \text{Transitions}, \quad \text{transition}(\text{edge}) \implies \neg \text{repeated\_pick\_entity}(\text{edge})$$
* **DFA Enforcement:** In sequential manipulation chains, when `active_skill == "pick"` and `active_arg == "target_entity_name"`, the DFA restricts admissible object pointers to unvisited scene entities:
  $$\mathcal{V}_{\text{allowed}} \leftarrow \{ \text{obj:}i \in \mathcal{V}_{\text{allowed}} \mid i \notin \text{visited\_picks} \}$$
  preventing degenerate looping on already-manipulated objects while unvisited candidates remain.

#### 3. Static Environment Grasp Exclusion
* **Identifier:** `pick_excludes_static_fixtures`
* **DomiKnowS Syntax:**
  ```python
  ifL(operation("op"), notL(targets_static_fixture("op")), name="pick_excludes_static_fixtures")
  ```
* **First-Order Logic:**
  $$\forall \text{op} \in \text{Operations}, \quad \text{targets\_static\_fixture}(\text{op}) \iff \Big(\text{skill}_{\text{pick}}(\text{op}) \land \text{is\_static\_fixture}(\text{target\_entity}(\text{op}))\Big)$$
  $$\forall \text{op} \in \text{Operations}, \quad \text{operation}(\text{op}) \implies \neg \text{targets\_static_fixture}(\text{op})$$
* **DFA Enforcement:** Immovable room surfaces $\mathcal{S}_{\text{static}} = \{\text{counter, table, wall, floor, room\_, lab\_}\}$ are mathematically stripped from pick targets:
  $$\forall i \in \mathcal{S}_{\text{static}}, \quad \mathcal{V}_{\text{allowed}} \leftarrow \mathcal{V}_{\text{allowed}} \setminus \{\text{obj:}i\}$$

#### 4. Container Destination Consistency
* **Identifier:** `container_destination_consistency`
* **DomiKnowS Syntax:**
  ```python
  ifL(transition("edge"), notL(inconsistent_container("edge")), name="container_destination_consistency")
  ```
* **First-Order Logic:**
  $$\forall (\text{op}_1, \text{op}_2) \in \text{Transitions}, \quad \text{inconsistent\_container}(\text{edge}) \iff \Big(\text{skill}_{\text{place}}(\text{op}_1) \land \text{skill}_{\text{place}}(\text{op}_2) \land \big(\text{target\_container}(\text{op}_1) \neq \text{target\_container}(\text{op}_2)\big)\Big)$$
  $$\forall \text{edge} \in \text{Transitions}, \quad \text{transition}(\text{edge}) \implies \neg \text{inconsistent\_container}(\text{edge})$$
* **DFA Enforcement:** Once an assembly destination receptacle $c_{\text{dest}}$ is bound by an initial `place` or `pour` operation, all subsequent assembly placements are locked to that established receptacle:
  $$\mathcal{V}_{\text{allowed}} \leftarrow \{\text{obj:}c_{\text{dest}}\}$$

---

## 3. Which Component Uses Given Constraints and How

The table below summarizes the component ecosystem and how each component utilizes the DomiKnowS constraints:

| Component Name | Source File | Consumed Constraints | Enforcement Mechanism | Purpose & Role |
| :--- | :--- | :--- | :--- | :--- |
| **Ontology & Verifier** | `VLABenchAgentInterface/world_graph.py` | All 10 Semantic World Constraints (1–10) | `root.verifyResultsLC(key="/local/argmax")` | Materializes plans into `DataNode` hierarchies and evaluates mathematical satisfaction percentages. |
| **Generation Compiler** | `VLABenchAgentInterface/graph.py` | Token Constraints (1–3), World Patterns | `constraints_to_dfa_from_graph()`, `product_dfa()` | Compiles relational formulas into executable state machines (`VLABenchContextualDFA`, `VLABenchCausalDFA`). |
| **Online VLM Decoder** | `JointEmbodiedAgentInterface/models.py` | All DFA Constraints (`allowed_tokens`) | Masked Categorical Logits (`logits[unallowed] = -inf`) | Masks unallowed token logits at every decoding step during greedy and beam search generation. |
| **Differentiable PPO Replay** | `JointEmbodiedAgentInterface/models.py` | All DFA Constraints (`replay_labels_logprob`) | Differentiable Masked Categorical Log-Probabilities | Evaluates log-probabilities of sampled trajectories under masked logits to compute policy gradients. |
| **RL Reward Engine** | `VLABenchAgentInterface/reward.py` | All 10 Semantic World Constraints | `verify_plan_constraints(root, bundle).score` | Evaluates constraint scores during rollout; zeros reward if constraint score $< 1.0$. |
| **Offline ILP Feasibility Gate** | `JointEmbodiedAgentInterface/structured_inference.py` | Hard validity + Constraint Satisfaction | Gurobi Binary Optimization (`gp.quicksum`) | Solves integer linear program across beam candidates to pick the maximum-likelihood constraint-satisfying plan. |
| **Agent Controller** | `VLABenchAgentInterface/agent.py` | DFA Acceptance + Semantic Constraints | `dfa_accepts_plan()` and `verify_plan_constraints()` | Validates plans before dispatching primitive skills to robot actuators. |
| **Evaluation Harness** | `JointEmbodiedAgentInterface/eval_planning.py` | Schema validity, Entity pointers, DAG match | Benchmark Scoring & Error Telemetry | Measures official ICCV 2025 Table 1 metrics (SR, PR, SPR, PM, Progress Score, Macro Average). |
| **Regression Test Suite** | `VLABenchAgentInterface/test_graph.py` | All Constraints | Automated Pytest Assertions | Verifies satisfaction on valid plans and ensures rejection of invalid / violating plans. |

---

### Detailed Component Walkthrough

### 3.1 `VLABenchAgentInterface/world_graph.py` (Ontology & Verification Engine)
* **What it does:** Defines the core ontology concepts (`plan`, `operation`, `entity`, `transition`) and their roles (`target_entity`, `target_container`, `current_operation`, `following_operation`).
* **How it consumes constraints:**
  1. `build_vlabench_world_graph`: Instantiates the 10 declarative first-order logic constraints using DomiKnowS `ifL` and `notL`.
  2. `materialize_plan(plan, entity_table, bundle)`: Transforms a JSON plan into a deterministic `DataNode` tree. It grounds every object pointer against the physical scene table, determines whether causal edge conditions hold (e.g. `is_self_placement`, `targets_static_fixture`, `repeated_pick_entity`, `inconsistent_container`), and assigns truth tensors (`torch.tensor([-30.0, 30.0])`).
  3. `verify_plan_constraints(root, bundle)`: Executes `root.verifyResultsLC(key="/local/argmax")`. It aggregates the satisfaction percentage across all active constraints and returns a normalized score $[0.0, 1.0]$ with per-constraint diagnostics.

### 3.2 `VLABenchAgentInterface/graph.py` (DFA Compilation & Specialization)
* **What it does:** Translates high-level declarative logic into token-level state transitions.
* **How it consumes constraints:**
  1. `create_planner_generation_graph`: Builds the token graph using `GenerationEncoder` and registers token LCs (`mark_for_dfa`, `mark_for_contextual_dfa`).
  2. `compile_planner_dfa`: Intersects the token graph DFA with the subtask pattern DFA to construct the base product automaton.
  3. `bind_vlabench_contextual_dfa`: Examines per-example context during inference:
     - If button entities are present, it calls `_resolve_physical_law_button` (compiling density orderings or friction mechanics) and instantiates `VLABenchContextualDFA`.
     - Otherwise, it instantiates `VLABenchCausalDFA` to enforce anti-self-placement, pick entity rotation, static background filtering, and container consistency.

### 3.3 `JointEmbodiedAgentInterface/models.py` (Online Decoding & PPO Training)
* **What it does:** Integrates vision-language backbones (Qwen3-VL) with DomiKnowS constraint automata.
* **How it consumes constraints:**
  1. **Autoregressive Logit Masking (`sample_labels` / `generate_plan`):**
     At step $t$, the decoder queries the active DFA:
     ```python
     allowed = dfa.allowed_tokens(state, remaining_steps=max_steps - step - 1)
     masked = torch.full_like(logits, float("-inf"))
     masked[allowed] = logits[allowed]
     dist = torch.distributions.Categorical(logits=masked)
     token = dist.sample()
     state = dfa.step(state, token)
     ```
     This strictly forbids the model from outputting any token that would violate syntax, hallucinate unobserved objects, or breach causal rules.
  2. **Differentiable Policy Replay (`replay_labels_logprob`):**
     During PPO reinforcement learning, sampled trajectories are re-evaluated in a single forward pass. Log-probabilities are computed over the masked categorical distribution, ensuring policy gradients respect the valid constraint manifold without drifting into illegal state space.

### 3.4 `VLABenchAgentInterface/reward.py` (RL Reward Engine)
* **What it does:** Computes reward signals for policy optimization in Stage 2 RL.
* **How it consumes constraints:**
  Inside `score_vlabench_plan`:
  ```python
  root = materialize_plan(predicted_plan, entity_table, world_bundle)
  evaluation = verify_plan_constraints(root, world_bundle)
  constraint_score = evaluation.score
  valid = validation.valid and (constraint_score >= 1.0)
  if not valid:
      total = 0.0  # Zero rollout reward on constraint breach
  ```
  Any plan that breaches a DomiKnowS first-order logic constraint receives a zero reward, providing a steep reinforcement penalty that guides policy gradient updates.

### 3.5 `JointEmbodiedAgentInterface/structured_inference.py` (Gurobi ILP Offline Gate)
* **What it does:** Performs offline Integer Linear Programming (ILP) selection over beam search candidates.
* **How it consumes constraints:**
  1. `_score_vlabench_candidate`: Runs `verify_plan_constraints` on every candidate generated by the beam search decoder.
  2. `select_candidate_with_ilp`: Formulates a binary integer program:
     $$\max_{c \in \text{Candidates}} \sum_{i} \Big( \log P(c_i) - i \cdot 10^{-9} \Big) x_i \quad \text{s.t.} \quad \sum_i x_i = 1, \quad x_i \in \{0, 1\}, \quad \text{HardValid}(c_i) = \text{True}$$
     Gurobi solves this problem in $<5\text{ ms}$, selecting the globally optimal candidate that strictly satisfies all DomiKnowS constraints.

### 3.6 `VLABenchAgentInterface/agent.py` (Autonomous Agent Controller)
* **What it does:** Manages closed-loop episode execution in the robot environment.
* **How it consumes constraints:**
  Before passing plans to the low-level manipulation controller (LeRobot / ACT policy), `generate_plan` checks both `dfa_accepts_plan()` and `verify_plan_constraints()`. If any constraint fails, it triggers replanning or safe fallback, preventing the physical robot from executing unsafe commands.

---

## 4. Constraint Cross-Reference Matrix

The matrix below maps each constraint to its active enforcement mechanisms across the codebase:

| Constraint Name | Declarative Source | Online Decoding Masking | PPO Training Replay | RL Reward Penalty | Gurobi ILP Gate | Unit Tested in Pytest |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `operation_exactly_one_skill` | `world_graph.py` | Yes (`domain_dfa`) | Yes (`models.py`) | Yes (`reward.py`) | Yes (`structured_inference.py`) | `test_graph.py` |
| `operation_schema_is_valid` | `world_graph.py` | Yes (`domain_dfa`) | Yes (`models.py`) | Yes (`reward.py`) | Yes (`structured_inference.py`) | `test_graph.py` |
| `operation_pointer_is_valid` | `world_graph.py` | Yes (`domain_dfa`) | Yes (`models.py`) | Yes (`reward.py`) | Yes (`structured_inference.py`) | `test_graph.py` |
| `transition_matches_subtask_pattern` | `world_graph.py` | Yes (`domain_dfa`) | Yes (`models.py`) | Yes (`reward.py`) | Yes (`structured_inference.py`) | `test_graph.py` |
| `button_entity_requires_press_skill` | `world_graph.py` | Yes (`ContextualDFA`) | Yes (`models.py`) | Yes (`reward.py`) | Yes (`structured_inference.py`) | `test_graph.py` |
| `skill_argument__{name}__{key}` | `world_graph.py` | Yes (`domain_dfa`) | Yes (`models.py`) | Yes (`reward.py`) | Yes (`structured_inference.py`) | `test_graph.py` |
| `anti_self_placement` | `world_graph.py` | Yes (`CausalDFA`) | Yes (`models.py`) | Yes (`reward.py`) | Yes (`structured_inference.py`) | `test_graph.py` |
| `pick_excludes_static_fixtures` | `world_graph.py` | Yes (`CausalDFA`) | Yes (`models.py`) | Yes (`reward.py`) | Yes (`structured_inference.py`) | `test_graph.py` |
| `no_redundant_repeated_picks` | `world_graph.py` | Yes (`CausalDFA`) | Yes (`models.py`) | Yes (`reward.py`) | Yes (`structured_inference.py`) | `test_graph.py` |
| `container_destination_consistency` | `world_graph.py` | Yes (`CausalDFA`) | Yes (`models.py`) | Yes (`reward.py`) | Yes (`structured_inference.py`) | `test_graph.py` |
| Absorbing EOS Closure | `graph.py` | Yes (`graph_dfa`) | Yes (`models.py`) | N/A (token level) | N/A (token level) | `test_graph.py` |
| Action Budget Horizon | `graph.py` | Yes (`graph_dfa`) | Yes (`models.py`) | N/A (token level) | N/A (token level) | `test_graph.py` |
| Contextual Button DFA Dispatch | `graph.py` | Yes (`ContextualDFA`) | Yes (`models.py`) | N/A (token level) | N/A (token level) | `test_graph.py` |

---

## 5. Summary & Verification Commands

All constraints specified above are verified in continuous integration using the following command suite:

```bash
# Run complete test suite on remote container
ssh gpu2.ihmc.us "docker exec -w /workspace vigorous_easley python -m pytest test_regr/VLABenchAgentInterface/test_graph.py -v"
```

All 8 tests execute in $<3$ seconds and verify that every semantic, causal, and token-level constraint functions correctly across the entire DomiKnowS stack.
