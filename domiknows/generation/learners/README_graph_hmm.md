# DomiKnowS-Aware HMM and Spectral Automata

This note explains the graph-aware HMM and spectral automata stack under
`domiknows.generation.learners.hmm` and `domiknows.generation.learners.wfa`.

The learner is not only estimating a raw sequence distribution:

```text
P(x_1:T)
```

It estimates a graph-constrained sequence distribution:

```text
P(x_1:T | G, C)
```

where:

- `G` is the DomiKnowS relational graph.
- `C` is the logical or semantic constraint layer that can be compiled into masks, projections, DFA restrictions, or constrained Hankel filtering.

In this view, the result is closer to a typed weighted automaton over a
relational state space than to a plain unconstrained HMM.

## PMD-Compatible Torch Learners

The graph-aware learner code under `domiknows.generation.learners` has two layers:

- EM or spectral learners such as `DomiKnowSAwareHMM` and `GraphSpectralAutomaton`, which fit, score, decode, initialize, and diagnose graph-constrained compact sequences.
- Torch learner heads such as `GraphHMMGenerationHead` and `GraphSpectralGenerationHead`, which are `torch.nn.Module` objects designed to plug directly into DomiKnowS `ModuleLearner`.

The Torch heads are the first-class PMD integration path:

```python
from domiknows.generation.learners import GraphHMMGenerationHead
from domiknows.sensor.pytorch.learners import ModuleLearner

head = GraphHMMGenerationHead(
    graph=graph,
    n_hidden_states=bundle.vocabulary.label_count,
    label_count=bundle.vocabulary.label_count,
    symbols=tuple(range(bundle.vocabulary.label_count)),
    state_names=bundle.vocabulary.labels,
    transition_mask=transition_mask,
    emission_mask=emission_mask,
    label_to_token_id=label_to_token_id,
    pad_size=pad_size,
    trainable=True,
)

token[bundle.generated_token] = ModuleLearner(
    token[bundle.contains],
    text["instruction_tokens"],
    "target_labels",
    module=head,
)
```

The head returns log-probabilities shaped `[seq_len, label_count]`, so
`PrimalDualProgram.cmodel(...)` can compute DomiKnowS logical constraint loss
over populated token DataNodes. The same head also exposes
`next_label_logits(...)`, so it can be used with compact-label DFA decoding.

Fitted EM and spectral learners can initialize PMD heads:

```python
learner = DomiKnowSAwareHMM(...).fit(sequences)
head = learner.to_torch_learner(trainable=True, pad_size=pad_size)

spectral = GraphSpectralAutomaton(...).fit(sequences, prefixes, suffixes, rank=2)
wfa_head = spectral.to_torch_learner(trainable=True, pad_size=pad_size)
```

`GraphSpectralGenerationHead` preserves the learned constrained WFA operators,
but hard graph or DFA legality filtering remains on `GraphSpectralAutomaton`
via `score(..., enforce_constraints=True)`, `hard_score(...)`,
`allowed_symbols(...)`, or an external DFA wrapper.

## Automatic Constraint-DFA HMM State Spaces

For generation-shaped graphs, regular DomiKnowS constraints can now create an
HMM state space automatically by reusing the graph-discovered DFA:

```python
from domiknows.generation.learners import domiknows_hmm_from_generation_constraints

learner = domiknows_hmm_from_generation_constraints(graph, bundle)
```

The compiler builds one hidden HMM state per productive DFA edge:

```text
(dfa_state_before, emitted_symbol, dfa_state_after)
```

Emissions are tied to the edge symbol, and HMM transitions connect edges whose
DFA endpoints line up. This means beliefs are no longer necessarily hand-named
states like `before_B`; for a rule set such as "B at most once" and "C at least
once," the compiler creates readable edge-state names such as:

```text
need_C_no_B__emit_B__to_need_C_seen_B
seen_C_seen_B__emit_END
```

This is exact for finite DFA-compiled constraints under the compact generation
vocabulary. If an EOS token is known, EOS-emitting states are only created when
the DFA successor is accepting, so ending before required constraints are
satisfied has no HMM support.

## What The Graph Contributes

DomiKnowS graphs declare domain knowledge with `Graph`, `Concept`, `Property`,
`Relation`, and `LogicalConstrain` objects. The graph is a declarative
structure, not an executable probabilistic model by itself. The HMM adapter
turns the graph pieces that are safe to interpret as sequence structure into
probabilistic constraints.

Concepts and entity types can become hidden-state types:

```text
Person
Object
Relation
Action
Attribute
```

Relations can become allowed latent transitions:

```text
owns(Person, Object)
left_of(Object, Object)
inside(Object, Container)
```

Logical constraints can become hard projections when they are faithfully
compiled. For example, a rule such as:

```text
inside(x, y) => not outside(x, y)
```

can remove incompatible transitions from a transition operator and redistribute
probability mass over legal alternatives.

Unsupported arbitrary DomiKnowS logic is not approximated silently. It remains
available to DomiKnowS solvers and PMD-style losses, while this package reports
that it was not compiled into an HMM or WFA mask.

## Classical vs DomiKnowS-Aware HMM

A classical HMM learns a free transition distribution:

```text
P(s_t+1 | s_t)
```

A DomiKnowS-aware HMM learns:

```text
P(s_t+1 | s_t, G, C)
```

Equivalently, constraints can be read as conditioning the transition law:

```text
P(s_t | s_t-1, C)
```

The constrained transition matrix is a reshaped version of the learned base
matrix:

```text
A^(C) = g(A, C)
```

In this view, DomiKnowS constraints act as priors over transitions. The result
is effectively a conditional HMM: the automaton still learns transition
probabilities, but graph semantics and logical constraints determine which
transition mass is legal or preferred.

The transition matrix is projected through graph and constraint masks:

```text
A = project_matrix(A, transition_mask)
B = project_matrix(B, emission_mask)
```

Forbidden graph or constraint transitions have probability zero. Forward,
backward, Baum-Welch, Viterbi, scoring, sampling, and DFA export all use the
projected transition and emission distributions, so impossible paths never
receive probability mass.

## Dynamic Relational State Constraints

Static masks are useful when graph topology alone decides what is legal. Many
relational systems need a stronger form:

```text
A_t = A(h_t, G, C)
```

Here the transition operator depends on the current latent relational state,
not only on a fixed graph mask.

Example:

```text
if holding(Person, Cup)
then Cup cannot simultaneously be_on(Table)
```

This cannot always be represented by one fixed transition matrix. The validity
of the next transition depends on the current inferred world state.
`DomiKnowSAwareHMM` supports this with opt-in hooks:

```python
from domiknows.generation.learners import DomiKnowSAwareHMM, DynamicConstraintContext

def dynamic_transition(context: DynamicConstraintContext):
    if context.prefix == ("holding_cup",):
        return [
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    return None

learner = DomiKnowSAwareHMM(
    graph=None,
    n_hidden_states=2,
    symbols=["holding_cup", "cup_on_table"],
    dynamic_transition=dynamic_transition,
)
```

During forward/backward, Viterbi, scoring, and sampling, each transition step
can receive a different projected matrix. If a dynamic hard mask removes every
outgoing transition for a row, that row stays zero for that step, so the
corresponding path receives no probability mass.

## Factorized Hidden State

The internal HMM tensors are flat `[K, K]`, but the state ids can be backed by a
factorized relational state space:

```text
h_t = (h_t_entity, h_t_relation, h_t_constraint)
```

or:

```text
h_t in H_E x H_R x H_C
```

This gives compositional state metadata while preserving ordinary HMM tensor
operations.

```python
from domiknows.generation.learners import FactorizedStateSpace

state_space = FactorizedStateSpace.from_factors({
    "entity": ["Person", "Cup"],
    "relation": ["holding", "on_table"],
})

transition_mask = state_space.transition_mask(
    lambda src, dst: not (
        src["relation"] == "holding"
        and dst["relation"] == "on_table"
    )
)
```

The factorized state space maps flat ids to dictionaries such as:

```text
{entity: Cup, relation: holding}
```

This is the bridge toward factor graphs, CRFs, neuro-symbolic dynamics, and
relational state machines without replacing the HMM core.

## Hard Masks vs Soft Energy

Hard masks are exact:

```text
invalid transition => probability 0
```

Soft constraints instead penalize violations without eliminating them:

```text
P(z_t+1 | z_t) proportional to exp(HMM score - lambda E_C)
```

where `E_C` is a logical violation energy. This connects the HMM package to
DomiKnowS PMD losses, t-norm losses, and differentiable logic.

```python
import torch
from domiknows.generation.learners import DynamicConstraintContext, DomiKnowSAwareHMM

def transition_energy(context: DynamicConstraintContext):
    energy = torch.zeros((2, 2), dtype=torch.float64)
    energy[0, 1] = 4.0
    return energy

learner = DomiKnowSAwareHMM(
    graph=None,
    n_hidden_states=2,
    symbols=["a", "b"],
    transition_energy=transition_energy,
    energy_weight=1.0,
)
```

At a transition step, the effective matrix is:

```text
A_t = normalize_rows(A * dynamic_factor * exp(-lambda E_C))
```

followed by the static graph mask. Hard masks still provide guarantees; soft
energies provide preferences.

## Typed Transition Operators

Instead of treating transitions as untyped matrix entries `A_ij`, graph
relations let you think in terms of relation-specific operators:

```text
Person --owns--> Object
```

as a typed operator:

```text
M_owns
```

Sequence evolution becomes:

```text
h_t+1 = h_t M_r
```

where `r` is a relation or emitted symbol. This connects the implementation to
relational message passing, graph automata, and typed weighted automata.

## Spectral Interpretation

The spectral learner builds a finite Hankel matrix. In a raw sequence model:

```text
H(u, v) = P(uv)
```

In the graph-constrained learner:

```text
H_G(u, v) = P(uv | G, C)
```

Entries whose concatenated prefix/suffix string has no legal hidden path under
the graph masks, or is rejected by an optional DFA, are set to zero. Prefixes
and suffixes are therefore interpreted as compact typed walks or operator
chains, not arbitrary strings alone.

The learned spectral automaton is a finite-rank weighted operator system
recovered from constrained Hankel factorization:

```text
score(x_1:T) = alpha M_x1 M_x2 ... M_xT omega
```

Each symbol has an operator `M_a`, recovered with Torch SVD from the constrained
Hankel blocks.

### Dynamic Spectral Operators

Spectral fitting is finite-basis and static: SVD recovers base operators `M_a`.
At traversal time, those operators can be adapted with opt-in dynamic hooks:

```text
h_t+1 = h_t M_t
M_t = M_a(h_t, G, C)
```

This keeps the learned base operator inspectable while allowing the scoring
path to depend on prefix state, graph metadata, or soft constraint energy.

```python
from domiknows.generation.learners import DynamicConstraintContext, GraphSpectralAutomaton

def operator_transform(context: DynamicConstraintContext, symbol, base):
    if symbol == "object_token" and context.prefix == ("holding_cup",):
        return base * 0.25
    return base

def operator_energy(context: DynamicConstraintContext, symbol):
    return None

spectral = GraphSpectralAutomaton(
    symbols=["holding_cup", "object_token"],
    operator_transform=operator_transform,
    operator_energy=operator_energy,
    energy_weight=1.0,
)
```

`operator(symbol)` still returns the static learned `M_a`. The dynamic path is
used by `score(...)`, `prefix_state(...)`, and `reconstruct_hankel(dynamic=True)`.

## Constraint Geometry

Logical constraints define admissible subspaces of the sequence probability
geometry. Graph projections remove probability mass from impossible relational
trajectories before operator recovery or decoding. In the spectral learner,
this induces a constrained Hankel geometry in which prefixes and suffixes are
interpreted as typed relational operator chains rather than unconstrained token
strings.

## API Overview

### Explicit Mask Inputs (DFA First)

Graph-aware HMM construction is explicit-mask-first.

Recommended path:

- compile generation logical constraints to DFA,
- convert DFA structure to transition/emission masks,
- initialize `DomiKnowSAwareHMM` with those masks.

For planning-style domains, pass masks directly from planning compilers.

### Supported Constraint Specs

The HMM stack accepts explicit mask/spec inputs through typed spec objects.
These are best for constraints representable before inference as transition
masks, emission masks, or observable DFA metadata.

| Spec | Meaning | Effect |
| --- | --- | --- |
| `TransitionMaskSpec(mask)` | Explicit hidden-state transition compatibility matrix | Multiplies the transition mask |
| `EmissionMaskSpec(mask)` | Explicit hidden-state-to-symbol compatibility matrix | Multiplies the emission mask |
| `AllowedTransitionsSpec([(src, dst), ...])` | Only these latent transitions are legal | Intersects the transition mask with the pair set |
| `ForbiddenTransitionsSpec([(src, dst), ...])` | These latent transitions are impossible | Sets those transition entries to zero |
| `AllowedEmissionsSpec([(state, symbol), ...])` | Only these emissions are legal for the listed states | Intersects the emission mask with the pair set |
| `ForbiddenEmissionsSpec([(state, symbol), ...])` | These emissions are impossible | Sets those emission entries to zero |
| `StatePredicateTransitionSpec(predicate)` | Factorized-state transition rule over `src` and `dst` dictionaries | Builds a transition mask from `FactorizedStateSpace` |
| `ConstraintDFAExportSpec(dfa)` | Regular observable constraint carried for DFA export/debugging | Registered in the report; not forced into a local HMM matrix |

Older dict and object-style specs remain source-compatible:

```python
constraints = [
    {"forbid_transition": ("holding", "on_table")},
    {"forbid_emission": ("LOC", "not_a_location")},
]
```

The typed form is clearer for reusable code:

```python
from domiknows.generation.learners import (
    ForbiddenEmissionsSpec,
    ForbiddenTransitionsSpec,
    StatePredicateTransitionSpec,
)

constraints = [
    ForbiddenTransitionsSpec([("holding", "on_table")]),
    ForbiddenEmissionsSpec([("LOC", "not_a_location")]),
    StatePredicateTransitionSpec(
        lambda src, dst: not (
            src["relation"] == "holding"
            and dst["relation"] == "on_table"
        ),
        name="holding-blocks-on-table",
    ),
]
```

Unsupported/non-local forms are reported through
`constraint_report.unsupported`; they are not approximated silently.

### Static Specs vs Dynamic Hooks

Static specs answer: "What transitions or emissions are never legal for this
task?" They are compiled once into `transition_mask` and `emission_mask`.

Dynamic hooks answer: "What is legal now, given the current prefix, belief, or
external relational state?" They can compute per-step masks or energies through
`dynamic_transition` and `transition_energy`.

Expanding the supported static fragment reduces boilerplate for common graph
constraints, but it does not eliminate hooks. Hooks remain the right interface
for belief-dependent logic, runtime world state, project-specific reasoning,
learned constraint selection, or constraints whose legality depends on more
than the fixed local `(state_t, state_t+1, symbol_t)` tuple.

### Constrained HMM

`DomiKnowSAwareHMM` runs constrained Baum-Welch and projects `A` and `B` after
each update:

```python
import torch
from domiknows.generation.learners import DomiKnowSAwareHMM

transition_mask = torch.tensor([
    [1.0, 1.0],
    [0.0, 1.0],
])

emission_mask = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
])

learner = DomiKnowSAwareHMM(
    graph=None,
    n_hidden_states=2,
    state_names=["Person", "Object"],
    symbols=["person_token", "object_token"],
    transition_mask=transition_mask,
    emission_mask=emission_mask,
)

learner.fit([
    ["person_token", "object_token"],
    ["person_token", "object_token", "object_token"],
])

print(learner.score(["person_token", "object_token"]))
print(learner.viterbi(["person_token", "object_token"]).states)
```

The learner can export a hard-support observable DFA for debugging or hard
constraint inspection. The DFA uses subset construction: each DFA state is the
set of HMM hidden states reachable after an observed prefix, so it accepts an
observable string whenever at least one legal positive-probability hidden path
can emit that string.

```python
dfa = learner.to_constraint_dfa()
print(dfa.accepts(["person_token", "object_token"]))
```

This exact export applies to static projected HMM support. If the HMM has an
arbitrary `dynamic_transition` callback or `transition_energy`,
`to_constraint_dfa()` raises by default rather than returning a misleading
automaton. Use `on_unsupported_dynamic="static"` only when you intentionally
want to ignore dynamic or soft behavior and export the static support language.

Dynamic hard constraints can be exported exactly when the caller supplies a
finite-state abstraction:

```python
from domiknows.generation.learners import FiniteStateDynamicConstraint

monitor = FiniteStateDynamicConstraint(
    start_state="open",
    transition_mask=lambda monitor_state, reachable_hmm_states, metadata: transition_mask,
    advance=lambda monitor_state, symbol, next_hmm_states, metadata: monitor_state,
)

dfa = learner.to_constraint_dfa(finite_state_dynamic=monitor)
```

The product DFA state is `(reachable_hmm_states, monitor_state)`. A positive
`support_threshold` can prune low-probability support, but that is an
engineering approximation rather than the exact positive-support language.

### Graph-Constrained Spectral Automaton

`GraphSpectralAutomaton` builds constrained Hankel blocks and recovers one
operator per symbol:

```python
from domiknows.generation.learners import GraphSpectralAutomaton

spectral = GraphSpectralAutomaton(
    symbols=["person_token", "object_token"],
    transition_mask=transition_mask,
    emission_mask=emission_mask,
)

spectral.fit(
    sequences=[
        ["person_token"],
        ["person_token", "object_token"],
        ["object_token", "object_token"],
    ],
    prefixes=[(), ("person_token",), ("object_token",)],
    suffixes=[(), ("person_token",), ("object_token",)],
    rank=2,
)

print(spectral.score(["person_token", "object_token"]))
print(spectral.operator("person_token").shape)
```

The Hankel matrix zeroes graph-invalid entries before SVD:

```python
H = spectral.build_hankel()
```

That makes spectral learning constraint-aware, but the recovered low-rank WFA
is still a signed scorer. Low-rank reconstruction can assign a non-zero signed
score to strings that were filtered out of the constrained Hankel block. Use
hard scoring or DFA or legality filtering when invalid strings must stay invalid:

```python
soft_score = spectral.score(["object_token", "person_token"])
hard_score = spectral.score(["object_token", "person_token"], enforce_constraints=True)

assert hard_score == spectral.hard_score(["object_token", "person_token"])
```

For decoding-style code, use `allowed_symbols(prefix)` or an external DFA
wrapper to filter candidate continuations before applying the WFA score.

## State Meaning

In a classical HMM, a hidden state is often just an arbitrary latent cluster.
In a DomiKnowS-aware HMM, a state can be interpreted as a partially satisfied
relational structure:

```text
{
  entity: Object_3,
  inside: Box_1,
  visible: True,
  ownership: Person_2
}
```

This makes the model closer to a symbolic scene state, logical world model, or
factor graph state than to a purely statistical cluster.

## DFA Coupling

When a DFA is also present, decoding or filtering can track both the
probabilistic graph state and the logical consistency state:

```text
(h_t, q_t)
```

where:

- `h_t` is the probabilistic graph latent state.
- `q_t` is the DFA or logical consistency state.

Transitions must satisfy both:

```text
h_t+1 = h_t M_r
delta(q_t, r) != dead
```

This is the same product-state idea used in constrained decoding and
graph-constrained automata.

## Current Limits

- arbitrary DomiKnowS logical constraints are not reverse-compiled into HMM or WFA structure;
- only graph structure, explicit typed or dict specs, the supported static logical fragment, dynamic hooks, soft transition energies, and optional DFA acceptance affect HMM/WFA learning automatically;
- dynamic relational constraints are opt-in hooks over compact HMM states;
- spectral learning is finite-basis and compact-symbol, not open-vocabulary language model training;
- graph or DFA filters constrain spectral Hankel queries during learning, but the learned WFA remains a signed scorer at inference time unless callers use `score(..., enforce_constraints=True)`, `hard_score(...)`, `allowed_symbols(...)`, or an external DFA wrapper.