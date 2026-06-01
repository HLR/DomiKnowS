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
- The matcher dispatches on LC (logical constraint) type (`andL`, `orL`, `notL`, `ifL`, `xorL`, `iffL`, `nandL`, `norL`, `atMostAL`, `atLeastAL`, `existsAL`, `exactAL`).
- For nested expressions, child LC nodes are recursively compiled first, then composed.

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

# 10. Why This Works
The key idea is that many DomiKnowS logical constraints correspond to regular languages.

Regular constraints can be represented exactly by DFAs.

Instead of approximating these constraints with penalties or post-processing, the system translates supported logical constraints directly into automata and performs:

```
Hard Constraint Decoding
```

where illegal prefixes are impossible to generate.

This provides:

- Exact enforcement
- Interpretable constraint states
- Efficient decoding
- Compatibility with HMMs
- Compatibility with spectral weighted automata
- Direct integration with DomiKnowS logical constraints

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