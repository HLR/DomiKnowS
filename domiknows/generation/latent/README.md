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
| compiler_recipes.py | Optional, reusable recipe compilers for custom graph-to-latent mappings. |

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

1. mark graph logical constraints with latent metadata using mark_for_latent,
   mark_for_dfa, or mark_for_both;
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

## Notes

- latent constraints are soft and differentiable; they guide behavior but do
  not guarantee validity on their own;
- hard validity is handled by DFA-side enforcement;
- latent and hard pathways can be combined in one graph via mark_for_both;
- recipe compilers are opt-in and intended for project-specific extensions.
