# How DomiKnowS Constraints Are Compiled into a DFA

## Overview
In the DomiKnowS-aware generation framework, logical constraints defined in a DomiKnowS graph are converted directly into a deterministic finite automaton (DFA) that acts as a hard decoding controller.

Rather than generating a sequence first and checking constraints afterward, the system constructs an automaton that accepts only sequences satisfying the specified logical constraints. During decoding, tokens that would lead to an invalid DFA state are removed from consideration.

Conceptually:

```
DomiKnowS Graph
      +
Logical Constraints
      ↓
Constraint Discovery
      ↓
Pattern Recognition
      ↓
DFA Fragments
      ↓
DFA Composition
      ↓
Final DFA
      ↓
Constrained Decoding
```

The resulting automaton represents the regular language defined by the supported DomiKnowS logical constraints.

`graph_discovery.py` lists the constraint shapes it can compile into a DFA
without request data. For `eqL`, `fixedL`, `sumL`, `iotaL`, `miotaL`, `queryL`,
`sameL`, and `differentL`, use `compile_generation_constraint_plan` and bind
a `GenerationSemanticContext`. That path evaluates typed filter, number,
selection, and answer results over a bounded sequence, and constrains decoding
when a Boolean condition or an expected value is available. The request-bound
path requires the relevant attributes, bindings, truth values, or scores from
the caller; it does not infer arbitrary graph relationships automatically.

## Incremental predicates supplied by an application

Use `compile_predicate_constraints(graph, bindings)` when an application already
computes Boolean predicates from its own evolving state. This is a separate
adapter to the same decoder surface: it does not require token-position
DataNodes or a `GenerationBundle`.

The generic compiler in `predicate_monitor.py` owns LC parsing, Boolean
evaluation, binding validation, and lazy composition with a grammar. A domain
adapter owns state transitions, symbol interpretation, predicate valuations,
and its acceptance condition. The core has no action names, vocabulary layout,
scene assumptions, or designated EOS symbol.

```python
from domiknows.generation import (
    PredicateBinding, PredicateConstrainedDFA, compile_predicate_constraints,
)

program = compile_predicate_constraints(policy_graph, [
    PredicateBinding(needs_review, "needs_review", "publication"),
    PredicateBinding(approved, "approved", "publication"),
])
compiled = PredicateConstrainedDFA.from_dfa(grammar, program)
decoder_dfa = compiled.bind(document_adapter)
```

An adapter implements `PredicateMonitorAdapter`:

- `start_state`: immutable, hashable request state.
- `step(state, symbol)`: returns `SemanticTransition(new_state, events)` or
  `None` for rejection. Each `PredicateEvent(domain, values, phase)` provides
  complete Boolean values for the rules being evaluated at that point.
- `is_accepting(state)`: the application's structural acceptance condition.
- `acceptance_events(state)`: valuations that must hold in an accepting state.

Supported Boolean LCs are `ifL`, `notL`, `andL`, `orL`, `nandL`, `norL`,
`xorL` (odd parity), `equivalenceL`/`iffL` (all operands agree), and `forAllL`
(universal implication). `notL` is unary; `ifL` and `forAllL` are binary.
`atMostL`, `atLeastL`, `exactL`, and `existsL` count true operand groundings.
Each rule belongs to one event domain. Predicate concepts are registered by identity;
constraint names are diagnostic labels. Unsupported head constraints fail
compilation by name. Missing or non-Boolean observations raise
`MissingSemanticBinding` rather than being treated as satisfied.

Single-variable adapters can continue using `{predicate_name: bool}` values:
these describe one implicit grounding. Multiple variables require explicit
finite domains and tuple-keyed observations; a flat value is never broadcast
over different entities:

```python
from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import ifL
from domiknows.generation import PredicateEvent

with Graph("link_policy") as graph:
    entity = Concept(name="entity")
    link = Concept(name="link")
    link.has_a(source=entity, target=entity)
    approved = entity(name="approved")
    ifL(link("source", "target"), approved("target"), name="approved_target")

program = compile_predicate_constraints(graph, [
    PredicateBinding(link, "link", "publication"),
    PredicateBinding(approved, "approved", "publication"),
])
event = PredicateEvent("publication", {
    ("link", "robot", "document"): True,
    ("approved", "document"): False,
}, variable_domains={"source": ("robot",), "target": ("document",)})
assert not program.check(event)
```

Free variables are universally evaluated over their Cartesian product. Reusing
a variable joins on the same entity. Direct relation arguments and the exact
role projections produced by DomiKnowS graph validation are supported; arbitrary
DataNode paths and joins without recoverable variable mappings remain unsupported.
Domains must be finite tuples, lists, sets, frozensets, or ranges containing
distinct hashable identities. Supply every referenced variable's domain, even
when empty. A universally quantified empty domain is vacuously true; an empty
count is zero. Omitted tuples are unknown bindings and raise an error, including
when the antecedent is false or an earlier grounding already violates the rule.

For a count operand, variables not occurring outside the count are locally
quantified. Thus head `exactL(approved("x"), limit=2)` counts all approved
entities in that event. In `ifL(request("x"), exactL(approved("x"),
eligible("y"), limit=2))`, `x` remains the outer variable and only `y` is
quantified inside the count. Counts sum operand truths independently:
`exactL(a("x"), b("y"), limit=2)` sums the number of true `a` and `b`
groundings, without duplicating counts through a Cartesian product. To count
matching pairs, supply a compound operand such as `andL(a("x"), b("y"))`.
Each event supplies a complete snapshot; the program does not accumulate
observations from earlier events. Accumulated `*AL` counts, value/filter LCs,
multiclass atoms, and arbitrary graph traversal still fail compilation.

`compile_predicate_constraints(..., max_groundings=10000)` limits predicate
evaluations per rule/event, including repeated atoms. Oversized domains or an
exhausted budget raise `PredicateGroundingLimitExceeded`; satisfaction remains
unknown rather than becoming an approximate rejection. This is independent of
the monitor's DFA reachability budget below.

The default predicate phase is `immediate`. A binding can declare an
application-defined deferred phase, such as `seal` for a document signature.
Rules containing that predicate execute in that phase. The adapter must emit
the corresponding event when that entity is finalized; immediate predicates
needed by the deferred formula must be included then too. Distinct deferred
phases cannot be mixed in one formula. Domains with no emitted instances are
vacuous, so supplying the correct event lifecycle is part of the adapter's
semantic contract.

Compilation snapshots active head LCs in the selected graph and descendant
subgraphs. `program.coverage` lists compiled and inactive head constraints; an
unsupported active head aborts compilation by name. Ancestors and siblings are
outside this scope. Recompile after editing the graph and
bind a new adapter for each request. The unbound `PredicateConstrainedDFA`
recognizes the grammar only; use its bound monitor for semantic enforcement.
The monitor rejects violating transitions and never restores forbidden
symbols. With `remaining_steps` supplied, it keeps a candidate only if an
accepting semantic completion is reachable within that many additional
transitions after the candidate. Termination tokens count as transitions, so a
valid final token can be emitted with `remaining_steps=0`. Omitting the budget
checks immediate transitions only.

Bind the semantic monitor before adding task-specific runtime wrappers, and
retain that complete policy for replay of sampled trajectories. Context-free
decoders that receive encoded model tensors should call
`require_bound_predicate_dfa(policy)` before decoding; it rejects an unbound
program with active semantic rules. `find_predicate_monitor(policy)` locates a
monitor beneath wrappers exposing `base_dfa`, so callers can avoid double
binding while preserving outer restrictions. Plain grammars and programs with
no semantic rules remain usable without binding. Scene-dependent policies
must not be cached by entity count or temporary context-object identity.

For lookahead across additional outer restrictions, wrap the fully composed
policy in `ProductiveDFA(policy)`. This generic view includes task wrappers in
its reachability queries and should also be retained for replay. Both wrappers
expose `can_reach_accepting(state, max_steps=...)`, requiring a finite budget.
The iterative search memoizes exact answers and has a separate computational
cap of 10,000 state expansions per query. Set `max_state_expansions` when
binding a generic compiled DFA or constructing `ProductiveDFA` to change it.
Search first tries a small depth-first witness probe, then uses breadth-first
discovery with state dominance to avoid unrolling cycles to the full horizon.
Proven witness lengths apply to larger budgets; exhausted budgets apply to
smaller ones. Successor enumerations are cached only after they finish. Both
search stages and all candidate symbols share the same query expansion cap.
Exceeding this cap raises `SemanticSearchLimitExceeded`: feasibility remains
unknown, no approximate rejection is cached, and no hard rule is relaxed.

`test_regr/generation/test_predicate_monitor.py` provides a complete document
workflow adapter. VLABench's adapter lives in
`test_regr/VLABenchAgentInterface/constraint_compiler.py`; its skill parsing and
world-state predicates remain outside the generic package.

`test_regr/generation/constraint_equivalence.py` is a reusable test audit for
any finite-alphabet grammar, bound monitor, and independent acceptance oracle.
It enumerates syntactic paths using `step`, so semantic masking cannot hide
counterexamples, and compares acceptance plus next-token masks at every budget
within the selected finite horizon. Terminal symbols stop enumeration; padding
past termination is outside this audit.

`test_regr/VLABenchAgentInterface/test_constraint_equivalence.py` applies it to
small scenes and completed-plan DataNode verification, alongside explicit
semantic examples and rule deactivation/addition checks. These bounded tests
establish agreement for the enumerated scenes and horizons, not an unbounded
equivalence proof. Task sequence, initial-target, and press-applicability rules are world LCs
and use the same monitor and completed-plan verification as physical rules.


---

# 1. Vocabulary Defines the DFA Alphabet
The DFA operates on a compact vocabulary rather than raw tokenizer IDs.

A `TokenVocabulary` maps:

```
token string → compact label
```

Example:

```
yes      → 0
no       → 1
<EOS>    → 2
_other   → 3
```

Here, `EOS` means **End Of Sequence** (the sequence-termination token).

The DFA alphabet becomes:

```
Σ = {0,1,2,3}
```

where `_other` represents every token not explicitly included in the constrained vocabulary.

This greatly reduces automaton size and makes DFA construction tractable.

---

# 2. Logical Constraints Become DFA Fragments
The DFA builder analyzes logical constraints stored in the DomiKnowS graph and identifies patterns that correspond to regular languages.

Each recognized constraint pattern is translated directly into a DFA fragment.

Instead of constructing one large automaton directly, the system builds several small DFAs and combines them later.

Conceptually:

```
Logical Constraint
        ↓
Pattern Recognition
        ↓
DFA Fragment
```

---

# 3. Example Constraint DFAs

## EOS Closure
Constraint:

```
After EOS (End Of Sequence), no non-EOS token may appear.
```

Equivalent language:

```
.* EOS EOS*
```

DFA:

```
open ----EOS----> eos_state

eos_state --EOS--> eos_state
eos_state --other--> dead
```

States:

```
open
eos_state
dead
```

This guarantees:

```
A B EOS EOS EOS
```

is valid while

```
A B EOS C
```

is rejected.

---

## Maximum Length
Constraint:

```
At most N non-EOS tokens.
```

Example:

```
N = 3
```

DFA states:

```
0
1
2
3
overflow
```

Each non-EOS token increments the counter:

```
0 → 1 → 2 → 3 → overflow
```

Overflow becomes a dead state.

The automaton therefore rejects sequences longer than the permitted length.

---

## Required Token
Constraint:

```
token "A" must appear at least once
```

DFA:

```
not_seen
seen
```

Transitions:

```
not_seen --A--> seen
not_seen --other--> not_seen

seen --anything--> seen
```

Accepting state:

```
seen
```

A sequence is accepted only if token A was observed.

---

## Forbidden Token
Constraint:

```
token "BAD" may never occur
```

DFA:

```
ok
dead
```

Transition:

```
ok --BAD--> dead
```

All other symbols remain in:

```
ok
```

---

## Conditional Constraint
Constraint:

```
If token X appears,
then at most N additional non-EOS tokens may follow.
```

The automaton maintains:

```
trigger not seen
trigger seen
counter after trigger
```

The DFA therefore encodes a conditional counting rule.

---

# 4. Boolean Logic on DFAs
DomiKnowS supports logical combinations:

```
andL
orL
notL
nandL
norL
xorL
iffL
ifL
```

These are translated using classical automata operations.

---

## AND
Constraint:

```
A AND B
```

Construction:

```
DFA(A) × DFA(B)
```

Product states:

```
(state_A, state_B)
```

Accepting:

```
accept_A AND accept_B
```

Only sequences satisfying both constraints are accepted.

---

## OR
Constraint:

```
A OR B
```

Construction:

```
union(DFA(A), DFA(B))
```

Accepting:

```
accept_A OR accept_B
```

---

## NOT
Constraint:

```
NOT A
```

Construction:

```
complement(DFA(A))
```

Accepting and rejecting states are swapped.

---

## Implication
Constraint:

```
A ⇒ B
```

Rewritten as:

```
NOT(A) OR B
```

and then compiled using the corresponding DFA operations.

---

# 5. Discovering Constraints from a DomiKnowS Graph
The DFA builder traverses:

```
graph.logicalConstrains
```

and attempts to recognize supported patterns.

Examples:

```
existsAL(token)
```

is recognized as a token-presence requirement and translated into a DFA that accepts only sequences containing the token.

---

```
atMostAL(token,0)
```

is recognized as a forbidden-token constraint and translated into a DFA that rejects any sequence containing the token.

---

```
atMostAL(notL(eos),N)
```

is recognized as a maximum non-EOS length constraint and translated into a counting DFA.

---

```
ifL(existsAL(token),
    atMostAL(notL(eos),N))
```

is recognized as a conditional constraint and translated into a DFA that activates a length restriction after the trigger condition is satisfied.

---

Boolean forms are recursively recognized:

```
andL(...)
orL(...)
notL(...)
xorL(...)
iffL(...)
```

and translated into corresponding automata operations.

---

## 5.1 How Nested Logical Constraints Are Processed in Code
The recursive compilation logic lives in `graph_discovery.py` and is driven by `_match_lc_many(...)`.

Compilation flow for nested trees:

```
Head logical constraint (LC) node
      ↓
_match_lc_many(node)
      ↓
Dispatch by LC class name
      ↓
Recursively compile child LCs
      ↓
Compose child DFAs with DFA algebra
```

Key implementation behaviors:

### Recursive Dispatch
- The matcher dispatches on LC (logical constraint) type (`andL`, `orL`, `notL`, `ifL`, `xorL`, `iffL`, `nandL`, `norL`, `atMostAL`, `atLeastAL`, `existsAL`, `exactAL`) and the restricted forms below.
- For nested expressions, child LC nodes are recursively compiled first, then composed.

### Additional count and universal forms

- Head `atMostL`, `atLeastL`, `exactL`, and `existsL` compile when their predicate is a pathless token or token set. Head count constraints accumulate over the full sequence; nested element-wise forms remain unsupported.
- Count limits follow the LC definition: `limit=` takes precedence, followed by a trailing integer, then the default of 1. `existsL` and `existsAL` always use 1.
- `forAllL(A(x), B(x))` compiles when both arguments are pathless unary token predicates bound to the same variable. The DFA forbids tokens in `A` but outside `B`.
- Head `greaterL`, `greaterEqL`, `lessL`, `lessEqL`, `equalCountsL`, and `notEqualCountsL` compile for two pathless unary token predicates on distinct variables when `max_sequence_length` is supplied to `constraints_to_dfa_from_graph`. The bound includes EOS and other labels, and sequences exceeding it are rejected. Comparisons nested in Boolean LCs remain unsupported because their graph semantics are element-wise. A comparison exceeding the 4096-state or 250,000-transition construction budget remains unsupported.

### Request-bound value and selection constraints

`compile_generation_constraint_plan(graph, bundle)` retains LCs that cannot be
compiled into the static DFA. Its `evaluate(labels, context)` method returns
typed `SemanticValue` results; `bind(context, max_sequence_length=N)` returns a
DFA-like object accepted by the existing greedy, sampling, and beam decoders.
The bound is required because request-dependent values are checked against the
generated prefix. Reachability search has an explicit `search_budget` and raises
if it cannot prove a token can still lead to an accepted sequence.

```python
from domiknows.generation import (
    GenerationSemanticContext, compile_generation_constraint_plan,
)

plan = compile_generation_constraint_plan(graph, bundle)
facts = GenerationSemanticContext(
    attributes={0: {"instanceID": "chosen"}},
    bindings={"x": 0},
    fixed_truth={0: True},
    membership_scores={"selector_name": {0: 0.9}},
    expected={"answer_name": "A"},
)
bound = plan.bind(facts, max_sequence_length=8)
```

For ordered candidate DataNodes, the plan can build those facts itself:

```python
bound = plan.bind_data_nodes(
    token_nodes, max_sequence_length=len(token_nodes),
    expected={answer.lcName: "A"},
)
```

The builder copies each node's attributes and `instanceID`/`instanceValue`.
For `fixedL`, it reads observed compact labels from
`<generated_token>/label` and derives truth separately for each pathless token
predicate. For soft `miotaL`, it reads `<selector-name>/score` (one probability
per candidate), or `<generated_token>/local/softmax` for a single-token
selector. A node's `generation_variables` attribute names the `sameL` or
`differentL` variables bound to that position; subclass values come from a
direct concept-name attribute or the standard `<concept>/label` attribute.
Missing or duplicate bindings, missing scores, and invalid labels fail before
decoding. The bound decoder accepts sequences with exactly as many positions
as the DataNode snapshot.

The request snapshot supports `eqL` attribute filters, `fixedL` observed truth,
numeric `sumL`, `iotaL` uniqueness and selection, thresholded `miotaL`,
`queryL` answers, and `sameL`/`differentL` category comparisons. `from_data_nodes`
copies `instanceID`, `instanceValue`, and attributes from ordered token-position
DataNodes. An LC that returns a number, selection set, or answer restricts
generation only when an `expected` value or a Boolean parent consumes it.
`iotaL` always requires exactly one selected candidate. For relation paths,
provide `path_resolver(path, labels)` returning eligible position indexes.
Request values used by existing contextual DFA markers go in `request_values`.
Missing facts and unhandled predicate shapes raise explicit errors.
`analyze_generation_constraints` and `constraints_to_dfa_from_graph` still
describe and build only the static DFA fragment; use the typed plan for these
request-dependent expressions. `from_data_nodes` snapshots attributes but does
not infer arbitrary graph relation paths; supply `path_resolver` for those.

### `andL` (Conjunction)
- Every child that is generation-relevant must compile successfully.
- Non-generation children are ignored by DFA hard decoding (they remain for normal DomiKnowS loss/verification).
- If any generation-relevant child is unsupported, the whole `andL` is rejected for DFA compilation.

### `orL` (Disjunction)
- Each branch must be generation-relevant and fully compilable.
- Child DFAs are first reduced to one DFA per branch (product if needed), then merged via `union_dfa(...)`.
- Any unsupported branch causes the whole `orL` to be unsupported.

### `notL` / `nandL` / `norL`
- `notL` expects exactly one logical child and compiles as DFA complement.
- `nandL` is compiled as complement of the compiled `andL` result.
- `norL` is compiled as complement of the compiled `orL` result.

### `xorL` and `iffL`
- Implemented for exactly two branches.
- `xorL(A,B)` is compiled as `(A ∧ ¬B) ∨ (¬A ∧ B)`.
- `iffL(A,B)` / `equivalenceL(A,B)` is compiled as `(A ∧ B) ∨ (¬A ∧ ¬B)`.

### Nested `ifL`
`ifL` has multiple matching stages in code:

1. Special EOS-closure pattern:
      ```
      ifL(is_before_rel, ifL(eos_x, eos_y))
      ```
2. Special conditional max-length pattern:
      ```
      ifL(existsAL(token), atMostAL(notL(eos), N))
      ```
3. Generic implication fallback when both sides are regular and compilable:
      ```
      ifL(A, B)  =>  union(complement(DFA(A)), DFA(B))
      ```

### Path-Aware Nested Shapes
For ordered constraints, the matcher reads relation-path variables (first/second token roles) and extracts token predicates from nested structures such as:

```
ifL(before, ifL(trigger(first), allowed(second)))
existsAL(andL(before, A(first), B(second)))
```

These compile to dedicated DFAs like ordered-token and after-trigger-allowed automata.

### Unsupported Nested Forms
If a generation-relevant nested form cannot be compiled into a supported regular fragment, the behavior is controlled by policy:

- `warn`: emit warning
- `error`: raise exception
- `ignore`: skip silently

This is why deeply nested constraints are not merely "parsed"; they are recursively validated for regular-language compilability before entering final DFA composition.

---

# 6. Building the Final DFA
After all supported constraints are discovered:

```
constraint_1 → DFA_1
constraint_2 → DFA_2
constraint_3 → DFA_3
...
```

they are combined using DFA algebra:

```
DFA_final =
DFA_1 × DFA_2 × DFA_3 × ...
```

The final DFA accepts only strings satisfying every constraint simultaneously.

Formally:

```
L(final)
=
⋂ L(DFA_i)
```

where:

```
L(DFA)
```

is the language accepted by that automaton.

---

# 7. How Constraints Affect DFA Structure
Each logical constraint changes the automaton in a different way.

## Forbidden Token
Adds:

```
dead state
```

---

## Required Token
Changes:

```
accepting states
```

---

## Maximum Length
Adds:

```
counter states
```

---

## Ordered Token Constraint
Adds:

```
progress states
```

tracking:

```
seen A
seen A then B
seen A then B then C
```

---

## Boolean Combinations
Add:

```
product states
```

or

```
complement structures
```

depending on the operator.

---

# 8. DFA-Guided Decoding
Once constructed, the DFA becomes part of decoding.

At decoding step t:

```
current DFA state = q_t
```

For every candidate token:

```
a ∈ Σ
```

the decoder evaluates:

```
δ(q_t, a)
```

If the transition leads to:

```
dead
```

the token is forbidden.

Only valid transitions remain.

Conceptually:

```
Language model (LM) logits
      ↓
DFA mask
      ↓
valid logits
      ↓
sampling / beam search
```

Thus constraints are enforced during generation rather than after generation.

---

# 9. Relationship to the HMM
The DFA constrains observable sequences.

The Hidden Markov Model (HMM) constrains latent trajectories.

```
HMM:
P(z_t+1 | z_t)
```

describes hidden-state evolution.

```
DFA:
accept(x_1:T)
```

describes observable sequence legality.

Combined:

```
Sequence valid
=
HMM path exists
AND
DFA accepts string
```

The DFA therefore acts as a symbolic controller layered on top of probabilistic sequence modeling.

---

# 10. Summary

Regular constraints can be represented exactly by DFAs.

Instead of approximating these constraints with penalties or post-processing, the system translates supported logical constraints directly into automata and performs Hard Constraint Decoding where illegal prefixes are impossible to generate.


The resulting system is therefore best viewed as a graph-constrained weighted automaton whose language is defined jointly by learned probabilities and symbolic DomiKnowS knowledge.

```text
+---------------------------------------------------------------+
|                    DomiKnowS Constraint Graph                 |
+---------------------------------------------------------------+
|                                                               |
|  atLeastAL(token=A,2)                                         |
|  atMostAL(token=B,0)                                          |
|  existsAL(C before D)                                         |
|  ifL(trigger -> restriction)                                  |
|  EOS Closure                                                  |
|                                                               |
+-----------------------+---------------------------------------+
                        |
                        v
+---------------------------------------------------------------+
|              Constraint Pattern Recognition                   |
+---------------------------------------------------------------+
|                                                               |
|  Constraint -> Regular Language Pattern                       |
|                                                               |
+-----------------------+---------------------------------------+
                        |
                        v
+---------------------------------------------------------------+
|                 DFA Fragment Generation                       |
+---------------------------------------------------------------+
|                                                               |
|  C1 -> DFA1                                                   |
|  C2 -> DFA2                                                   |
|  C3 -> DFA3                                                   |
|                                                               |
+-----------------------+---------------------------------------+
                        |
                        v
+---------------------------------------------------------------+
|                  DFA Algebra Composition                      |
+---------------------------------------------------------------+
|                                                               |
|      DFA = DFA1 ∩ DFA2 ∩ DFA3 ∩ ...                          |
|                                                               |
+-----------------------+---------------------------------------+
                        |
                        v
+---------------------------------------------------------------+
|                Final Constrained Language                     |
+---------------------------------------------------------------+
|                                                               |
|      Accept only sequences satisfying all constraints         |
|                                                               |
+---------------+-------------------------------+---------------+
                |                               |
                v                               v
        Graph-Aware HMM                  Spectral Automaton
```


The standalone and joint embodied planners use `dfa.stop_policy.StopPolicy`
for termination too. Supply `stop_policy=StopPolicy(max_steps=7)` or the legacy
`max_steps=7`, never both. Timeouts, external cancellation callbacks, EOS,
accepting-state stops, and stagnant-state limits use the shared helpers.
Cancellation before acceptance raises an error instead of returning a partial
plan. Compact planner progress reports generated labels and no prompt tokens.

`remaining_steps_for(policy, step_index)` supplies the number of transitions
available **after** the candidate token to every DFA view; zero permits a final
accepting token. HMM lookahead uses `remaining_tokens_for` for its inclusive
horizon and converts at the DFA boundary. Replay reuses the same finite budget,
DFA masks, and token termination rules; it does not rerun collection wall-clock
or external cancellation checks during gradient computation.


VLABench task facts are explicit request data: `task` selects a declared task
contract, `task_skill_sequence` can supply a known-skill sequence, and
`task_target_entity` supplies a scene entity name or pointer. The task adapter
emits Boolean observations; `task_skill_sequence`, `task_initial_target`, and
`task_sequence_completion` logical constraints decide their legality. Press
applicability is declared by `press_requires_button_target`. Deactivating these
rules removes their restrictions after recompilation. A button's presence and
density/friction keywords do not select a task goal or answer.

Composite task declarations live in
`test_regr/VLABenchAgentInterface/task_policy.py`. Completed-plan verification
accepts `task_context=...`, or `policy=...` to reuse the exact bound requirements
from collection. Online sampling retains this same policy for gradient replay.
The handwritten contextual/causal/composite/task-pattern DFA implementations,
separate pointer filter, and automatic button-plan fallback have been removed.
