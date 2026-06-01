# HybridController Architecture

## Overview

The `HybridController` is a neuro-symbolic generation controller that combines:

1. An open-vocabulary language model for candidate generation.
2. A DFA-based constraint system for hard logical validation.
3. A compact DomiKnowS-aware scoring model (HMM or Spectral head).
4. A latent preference and risk evaluation layer.

Rather than relying solely on a language model, the controller performs a generate–verify–rerank workflow in which symbolic constraints and compact learned domain models influence the final output.

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
