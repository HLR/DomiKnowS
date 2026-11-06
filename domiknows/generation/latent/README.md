# Latent Soft-Enforcement Utilities

domiknows.generation.latent provides differentiable, product-logic based
constraint enforcement for generation training.

This package complements hard DFA decoding by adding soft losses and latent
automata transition reweighting that can be optimized with gradients.

## What This Package Covers

- product t-norm logic operators and window-formula losses over token
  probabilities;
- graph-annotation helpers for latent-only, DFA-only, or mixed enforcement;
- discovery utilities that collect marked graph constraints into a runtime
  enforcement bundle;
- transition potential helpers for HMM and WFA dynamics;
- task-agnostic training loss aggregation utilities.

## Modules

| Module | Purpose |
| --- | --- |
| __init__.py | Package re-exports for the public latent API. |
| constraints.py | Core soft-logic primitives, formula evaluation, and latent loss breakdowns. |
| enforcement.py | Graph marking and discovery helpers that build GenerationEnforcement bundles. |
| losses.py | Unified generation-training loss combiner and probability conversion helpers. |
| potentials.py | HMM/WFA transition potential definitions and application utilities. |
| compiler_recipes.py | Reusable recipe compilers for custom graph-to-latent mappings. |

## Core Data Structures

- LatentWindowSpec: one soft window-rule term (if-label, formula, window,
  weight, reduction).
- GenerationEnforcement: compiled hard DFA, latent specs, callable latent loss,
  latent diagnostics callable, and optional transition potentials.
- LatentLossBreakdown and LatentLossItem: per-term diagnostics for latent loss.
- GenerationLossWeights and GenerationLossBreakdown: weighted training-loss
  composition for supervised, PMD, latent, allowed-mass, and automata terms.
- LatentTransitionPotential: transition compatibility factors for latent
  automata models.

## Typical Workflow

1. declare graph logical constraints, marking them with mark_for_latent,
   mark_for_dfa, or mark_for_both where needed, and attach any project
   compilers or recipes with register_latent_compilers;
2. discover generation enforcement from the graph to obtain a
   GenerationEnforcement object;
3. run latent_loss or latent_breakdown on model token probabilities during
   training;
4. optionally apply transition potentials to HMM or WFA transitions;
5. combine supervised/PMD/latent components with compute_generation_training_loss.

## Formula Semantics

The soft logic uses product t-norm style semantics:

- NOT(p) = 1 - p
- AND(p, q) = p * q
- OR(p, q) = 1 - (1-p)(1-q)
- implication penalty = p * (1-q)

Formulas can be built from integer label ids and nested and/or expressions.
Cross-concept references use LabelRef.

## Discovery

Discovery modes (`latent_mode` on `discover_generation_enforcement`, `mode` on
`discover_latent_window_specs`):

| Mode | Marked specs | Adjacent `is_next_rel` rules | Reports uncompilable rules |
| --- | --- | --- | --- |
| `default` | yes | yes, unless the rule has marked specs | no |
| `marked` | yes | no | no |
| `auto` | no | yes | yes |
| `marked_and_auto` | yes | yes, alongside any marked specs | yes |

Automatic compilation only accepts rules it can represent faithfully.
`ifL(is_next, ifL(A(current), B(next)))` is compiled, including AND/OR
consequents. Window formulas have no negation, so `A(current) => not B(next)`
produces no window spec; it becomes a transition potential instead.

Compilers and recipes run in every mode. Attach them to the graph with
`register_latent_compilers(graph, ...)`, or pass `extra_latent_compilers` /
`extra_compilers` for a single call. Per-call compilers run first, built-in
support takes precedence over both, and registration never affects other
graphs.

Automatically discovered specs over a concept the caller did not supply are
skipped with a warning, so a bare token tensor is never read as latent-state
probabilities. Pass a mapping such as
`{"generated_token": token_probs, "latent_state": state_probs}` to enforce them.

## Notes

- latent constraints are soft and differentiable; they guide behavior but do
  not guarantee validity on their own;
- hard validity is handled by DFA-side enforcement;
- latent and hard pathways can be combined in one graph via mark_for_both;
- recipe compilers need project-specific labels, so each graph chooses which
  to register rather than having them installed process-wide.
