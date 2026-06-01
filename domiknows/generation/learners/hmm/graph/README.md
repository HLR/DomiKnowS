# How DomiKnowS Logical Constraints Shape a Graph-Aware HMM

## Overview
This folder implements a **DomiKnowS-aware Hidden Markov Model (HMM)** where
logical constraints do not remain external checks. Instead, they are compiled
into transition/emission support and applied directly inside training,
scoring, Viterbi decoding, sampling, and optional DFA export.

Conceptually:

```text
DomiKnowS Graph
      +
Logical Constraints
      +
(Optional) Generation DFA
      ↓
Constraint Normalization + Pattern Matching
      ↓
Transition / Emission Masks (+ reports)
      ↓
Constrained HMM Parameters
      ↓
Constrained EM, Viterbi, Sampling
```

This is the HMM analogue of the DFA-side compilation flow in
`domiknows/generation/dfa/README.md`, but here the target object is a
probabilistic latent-state model rather than a pure recognizer.

---

## 1. What Is Built in This Folder

- `graphAwareHMM.py`: `DomiKnowSAwareHMM`, constrained Baum-Welch training,
  constrained Viterbi/sampling, optional export to DFA support.
- `constraints.py`: mask/spec data structures and projection helpers.
- `dynamic.py`: runtime dynamic-constraint context, factorized-state space, and
  transition-energy utilities.
- `constraint_compiler.py`: bridge from discovered generation DFA to an
  initialized graph-aware HMM (`from_generation_constraints` path).
- `graph_head.py`: compact-label Torch head (`GraphHMMGenerationHead`) that
  applies the same graph-aware masking semantics in generation heads.
- `graph_head_utils.py`: helper utilities used by graph-aware heads.

---

## 2. Core Idea: Constraints Become HMM Support

A standard discrete HMM has:

- initial distribution: `pi[i]`
- transition matrix: `A[i,j]`
- emission matrix: `B[i,symbol]`

Graph constraints are compiled to masks:

- transition mask `M_T[i,j] >= 0`
- emission mask `M_E[i,symbol] >= 0`

Then parameter updates and runtime probabilities are projected so forbidden
entries remain zero.

In simplified form:

$$
A' = \operatorname{project\_rows}(A, M_T),\qquad
B' = \operatorname{project\_rows}(B, M_E)
$$

This means logical constraints influence model behavior **during** learning and
inference, not only as post-hoc filtering.

---

## 3. Static Constraint Compilation (DFA/Planning -> Masks)

Graph-aware HMM construction in this module is explicit-mask-first.

Primary source:

- `constraint_compiler.py` compiles generation logical constraints through the
  canonical generation DFA compiler, then converts productive DFA edges into
  HMM states and support masks.

Additional source:

- planning helpers can provide explicit transition/emission masks directly.

Typed specs in `constraints.py` are still used to represent and validate mask
structures and related diagnostics (`ConstraintApplicationReport`,
`ConstraintDFAExportSpec`).

---

## 4. Runtime Dynamic Constraints

Static masks handle graph structure known up front. Runtime logic is handled by
callbacks using `DynamicConstraintContext` in `dynamic.py`.

The context includes:

- decoding/training step
- generated prefix
- optional belief over hidden states
- optional full sequence
- metadata

Two runtime hooks are supported in `DomiKnowSAwareHMM` and
`GraphHMMGenerationHead`:

- `dynamic_transition(context)`
  - returns per-step transition mask/weights
- `transition_energy(context)`
  - returns non-negative energy matrix
  - applied as soft penalty via `exp(-weight * energy)`

So the effective transition at a step can be viewed as:

$$
A_t \propto A \odot M_{T,static} \odot M_{T,dynamic}(t)
\odot \exp(-\lambda E_t)
$$

If a user needs exact finite-language export of dynamic constraints,
`FiniteStateDynamicConstraint` supplies the finite control state protocol.

---

## 5. Constraint-Aware HMM Training

`DomiKnowSAwareHMM.fit(...)` performs constrained EM:

1. Build symbol ids and masks.
2. Initialize parameters under masks.
3. Run forward-backward using constrained dynamics.
4. Accumulate expected counts.
5. Re-project M-step updates through masks.

The projection helpers from `constraints.py` enforce:

- finite, non-negative values
- row normalization for legal rows
- exact zeros for forbidden support

So learned transitions/emissions stay graph-consistent over iterations.

---

## 6. Constraint-Aware Decoding and Sampling

`DomiKnowSAwareHMM` applies the same constrained dynamics in:

- `score(...)`
- `viterbi(...)`
- `sample(...)`

This keeps training and inference semantics aligned:

- static graph masks always active
- optional dynamic mask/energy active per step
- impossible transitions/emissions remain impossible

---

## 7. DFA -> HMM Compilation Path for Generation Constraints

`constraint_compiler.py` reuses the generation DFA compiler to seed an HMM.

Flow:

1. Build a generation DFA from constraints.
2. Convert productive DFA edges into HMM hidden states (`ConstraintHMMState`).
3. Build transition/emission masks from edge connectivity and edge labels.
4. Initialize an HMM with those masks and parameters.

This gives an HMM whose hidden support is aligned with DFA structure while
still allowing probabilistic training/scoring over that support.

---

## 8. Compact Generation Head Path

`GraphHMMGenerationHead` in `graph_head.py` exposes graph-aware HMM behavior as
a Torch compact-label head for constrained generation workflows.

It combines:

- explicit transition/emission masks
- optional dynamic transition / energy callbacks
- compact-label utilities for prompt/prefix decoding interfaces

So graph constraints can be enforced in head-level generation APIs without
dropping to raw HMM class usage.

---

## 9. Relationship to DFA Constraints

The DFA side and HMM-graph side are complementary:

- DFA: exact hard acceptance over observable sequences.
- Graph-aware HMM: probabilistic latent modeling with graph-shaped support.

In practice, pipelines can combine both:

```text
HMM latent dynamics constrained by graph masks
AND
DFA constrained decoding over emitted compact labels
```

This yields a system where symbolic constraints and probabilistic sequence
modeling reinforce each other at different layers.

---

## 10. Practical Notes

- Unsupported or non-local forms are surfaced through
  `ConstraintApplicationReport`.
- If you need richer symbolic enforcement than local masks can express,
  combine this module with DFA constraints from the `dfa` package.
- If you need exact export of dynamic behavior, provide
  `FiniteStateDynamicConstraint` rather than arbitrary unbounded callbacks.

---

## Minimal Entry Points

- Build constrained HMM directly:
  - `DomiKnowSAwareHMM(...)`
- Build from generation constraints via DFA bridge:
  - `DomiKnowSAwareHMM.from_generation_constraints(...)`
- Build graph-aware compact generation head:
  - `GraphHMMGenerationHead(...)`
