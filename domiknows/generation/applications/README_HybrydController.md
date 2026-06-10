# HybridController Architecture

## Overview

The `HybridController` is a neuro-symbolic generation controller that combines:

1. An open-vocabulary language model for candidate generation.
2. A DFA-based constraint system for hard logical validation.
3. A compact DomiKnowS-aware scoring model (HMM or Spectral head).
4. A latent preference and risk evaluation layer.

Rather than relying solely on a language model, the controller performs a generate–verify–rerank workflow in which symbolic constraints and compact learned domain models influence the final output.

It supports both backend paths:

1. HuggingFace path, including hard_dfa, unconstrained, product_compact_learner_dfa, and product_hmm_dfa decode strategies.
2. OpenAI path via OpenAIResponsesAdapter, where candidates are generated through Responses API integration and then normalized into the same HybridController scoring and reranking pipeline.

---

# High-Level Architecture

```text
Prompt
   │
   ▼
HybridController
   │
   ├── Constraint Selection
   ├── Candidate Generation
   ├── DFA Verification
   ├── Compact Head Scoring
   ├── Risk Estimation
   ├── Latent Preference Evaluation
   └── Candidate Reranking
```

The controller acts as an orchestration layer between large language models and DomiKnowS symbolic reasoning components.

---

# Main Workflow

The complete workflow is:

```text
Prompt
   │
   ▼
Generate Candidates
   │
   ▼
Verify Candidates Using DFA
   │
   ▼
Score Candidates
   │
   ├── Compact Head Log Probability
   ├── Latent Preference Score
   ├── Failure Risk Estimate
   └── Constraint Validity
   │
   ▼
Compute Hybrid Score
   │
   ▼
Rank Candidates
   │
   ▼
Return Best Results
```

---

# Step 1: Constraint Selection

The controller may select among multiple precompiled constraint bundles.

```text
Prompt
   │
   ▼
Constraint Selector
   │
   ▼
Selected DFA Bundle
```

Two selectors are supported:

### ManualConstraintSelector

Uses keyword matching.

Example:

```text
"medical" → Medical DFA
"finance" → Finance DFA
```

### CompactConstraintSelector

Uses a small trainable neural classifier.

```text
Prompt Tokens
      │
      ▼
Embedding Layer
      │
      ▼
Classifier
      │
      ▼
Constraint Bundle
```

This allows dynamic selection of constraint sets based on prompt content.

---

# Step 2: Candidate Generation

The controller can obtain candidate sequences from multiple sources.

```text
Prompt
   │
   ▼
Candidate Generator
```

Possible backends:

### HuggingFace Adapter

```text
Prompt
   │
   ▼
HF Model
   │
   ▼
Generated Candidates
```

### OpenAI Adapter

```text
Prompt
   │
   ▼
OpenAI Responses API
   │
   ▼
Generated Candidates
```

OpenAI-backed generation runs through OpenAIResponsesAdapter and uses the shared vocabulary and DFA interfaces to produce constrained/verified candidates before normalization and reranking.

### Precomputed Candidates

Candidates may also be provided externally.

```text
External Candidate Set
   │
   ▼
HybridController
```

All candidates are normalized into:

```python
GenerationCandidate
```

objects.

### HuggingFace Decode Strategies

For HuggingFace-backed generation, the controller supports multiple decode strategies:

1. hard_dfa
2. unconstrained
3. product_compact_learner_dfa
4. product_hmm_dfa

Alias names are also supported for compatibility (for example compact_dfa, hmm_dfa, strict_hmm_dfa).

The strategy is resolved in an internal normalization step, then routed to the appropriate decode loop.

### OpenAI Generation Path

For backend = openai, the controller routes into the OpenAI adapter path:

1. Submit prompt to the OpenAI Responses integration.
2. Generate and verify candidate output using vocabulary and DFA constraints.
3. Normalize adapter output into GenerationCandidate.
4. Reuse the same HybridController scoring, risk estimation, latent preference, and ranking logic.

Conceptually:

```text
Prompt
   │
   ▼
OpenAIResponsesAdapter
   │
   ▼
generate_and_verify(...)
   │
   ▼
GenerationCandidate
   │
   ▼
Hybrid Scoring + Reranking
```

---

# New Product Decode Paths

The new functionality introduces two strict product-state decoding modes that tightly couple symbolic constraints with compact-model scoring during generation.

## product_compact_learner_dfa

This path runs a custom token/label loop where each step:

1. Computes compact next-label logits from the current prefix.
2. Intersects with DFA-allowed labels.
3. Optionally blends one-step HuggingFace backend logits projected into label space.
4. Samples from the masked/filtered label distribution.
5. Commits the transition to token ids and label trace.

Conceptually:

```text
Prefix
   │
   ▼
Compact Head Next-Label Logits
   │
   ▼
DFA Allowed Label Mask
   │
   ▼
(Optional) HF Backend Label Bias
   │
   ▼
Sample Label
   │
   ▼
Append Token + Advance DFA
```

This ensures no sampled label can violate DFA transitions at generation time.

## product_hmm_dfa

This path decodes over an explicit product state:

1. HMM belief
2. DFA state
3. Token prefix

At each step:

1. HMM emission and current belief produce immediate label logits.
2. Optional recursive lookahead estimates downstream DFA success probability.
3. Optional HuggingFace backend token logits are projected to label logits and blended.
4. A label is sampled under DFA masking.
5. Both HMM belief and DFA state are advanced.

Conceptually:

```text
(HMM Belief, DFA State, Prefix)
                     │
                     ▼
          HMM Label Logits
                     │
                     ├── Optional Lookahead (future DFA success)
                     │
                     ├── Optional HF Label Bias
                     │
                     ▼
         DFA-Constrained Sampling
                     │
                     ▼
 Update Belief + DFA + Prefix
```

This provides strict constrained decoding while preserving explicit probabilistic state tracking.

---

# Runtime Controls for New Paths

Both product decode paths support runtime controls:

1. stop policy integration for decoding termination
2. top-k and top-p sampling filters
3. per-candidate random generator seeds
4. optional retention of rejected candidates

Additional controls:

1. product_compact_learner_dfa
    1. compact_logit_weight
    2. backend_logit_weight
2. product_hmm_dfa
    1. hmm_weight
    2. lookahead_weight
    3. hf_weight
    4. lookahead_max_steps
    5. transition_potential

Unsupported keyword arguments in strict product modes are rejected explicitly, which helps catch misspelled or unintended options early.

---

# Step 3: DFA Verification

Each candidate is translated into compact labels.

The DFA determines whether the sequence satisfies all hard constraints.

```text
Candidate Labels
       │
       ▼
DFA.accepts()
       │
 ┌─────┴─────┐
 │           │
 ▼           ▼
Valid     Invalid
```

Validity becomes:

```text
validity = 1.0
```

or

```text
validity = 0.0
```

Rejected candidates can optionally be removed entirely.

---

# Step 4: Compact Head Scoring

The compact head evaluates how well a candidate matches the learned domain model.

Supported compact heads include:

### GraphHMMGenerationHead

Derived from:

```text
DomiKnowSAwareHMM
```

### GraphSpectralGenerationHead

Derived from:

```text
GraphSpectralAutomaton
```

The head computes:

```text
P(labels | prompt)
```

through teacher-forced evaluation.

```text
Prompt + Candidate
        │
        ▼
Compact Head
        │
        ▼
head_logprob
```

This score reflects domain consistency learned from graph-constrained training data.

---

# Step 5: Failure Risk Estimation

The controller predicts the probability that the candidate is moving toward a future DFA violation.

```text
Current Prefix
       │
       ▼
Compact Head
       │
       ▼
Next Label Distribution
       │
       ▼
Probability Mass
Outside DFA Support
```

Risk is computed as:

```text
risk = 1 - allowed_probability_mass
```

A candidate with high risk receives a penalty.

---

# Step 6: Latent Preference Evaluation

The controller may include a latent enforcement model.

```text
Candidate
    │
    ▼
Latent Enforcement
    │
    ▼
Latent Loss
```

The loss is converted into:

```text
latent_preference = - latent_loss
```

Lower latent violations therefore increase the final score.

Examples:

* Style preferences
* Semantic preferences
* Safety preferences
* Domain-specific latent constraints

---

# Step 7: Hybrid Score Computation

The controller combines all scoring components.

```text
Total Score =
    Validity Weight × Validity
  + Head Weight × Head Log Probability
  + Latent Weight × Latent Preference
  - Risk Weight × Risk
  - Length Weight × Length
```

More formally:

```text
score =
    w_validity * validity
  + w_head * head_logprob
  + w_latent * latent_preference
  - w_risk * risk
  - w_length * length
```

where:

```text
w_validity
w_head
w_latent
w_risk
w_length
```

are configurable weights.

---

# Candidate Ranking

After scoring:

```text
Candidate A → Score 15.2
Candidate B → Score 12.4
Candidate C → Score  8.7
```

Candidates are sorted:

```text
Highest Score
      │
      ▼
Lowest Score
```

The best candidates are returned.

---

# Constraint Repair Support

The controller can explain DFA failures.

```text
Invalid Candidate
       │
       ▼
Repair Analyzer
       │
       ▼
Suggested Fixes
```

Examples:

```text
Replace token 5
```

or

```text
Choose one of:
   Person
   Organization
   Location
```

This provides debugging support for constrained generation.

---

# Conceptual Interpretation

The HybridController combines four complementary systems.

```text
                 HybridController

      Open Vocabulary Generator
                    +
         DFA Hard Constraints
                    +
       DomiKnowS Compact Model
                    +
        Latent Preference Layer
```

Each component contributes a different capability.

### Large Language Model

Provides creativity and open-vocabulary generation.

### DFA

Guarantees hard logical consistency.

### DomiKnowS Compact Head

Provides domain-specific statistical knowledge.

### Latent Layer

Encodes semantic preferences and soft requirements.

---

# Neuro-Symbolic Product View

The overall architecture can be viewed as:

```text
                    Large LM
                        │
                        ▼
                 Candidate Text
                        │
                        ▼

             +-------------------+
             | HybridController  |
             +-------------------+

                │      │      │
                ▼      ▼      ▼

              DFA    HMM/WFA  Latent
             Logic   Domain   Semantic
             Rules   Model    Preference

                │      │      │
                └──┬───┴───┬──┘
                   ▼       ▼

                 Hybrid Score

                       │
                       ▼

               Best Candidate
```

This architecture enables open-vocabulary generation while preserving symbolic correctness, domain consistency, and latent semantic preferences.
