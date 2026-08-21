# Current GraphQA / KB-VQA Example

This note records the current object-centered GraphQA example and clarifies why
the recent low zero-shot VLM result is not comparable to the earlier 60--70%
TemporalQA / trained-execution results.

## Goal

The GraphQA / KB-VQA setting should follow the CLEVR-style DomiKnowS design:

- objects are graph entities;
- visual predicates are concepts over objects or object pairs;
- knowledge-base facts are used as symbolic evidence or consistency rules;
- the executable query composes concepts using DomiKnowS logic;
- final answers are produced by `iotaL` for single-answer questions or `miotaL`
  for multi-answer questions.

The current direction is not to ask the model to directly answer the natural
language question. Instead, the model predicts local predicates, and DomiKnowS
executes the logical query.

## Example Question

Example question from GraphQA / VQAR:

```text
QID: 2384837_22

Program:
Relate_Reverse(right, candidate_objects)
Find_Attr(red)

Rendered question:
Which red objects are to the right of another candidate object?

Gold answer set:
{1304282, 1304283}
```

This is a multi-answer question, so it should not be represented with ordinary
`iotaL`, which assumes a unique selected entity. It should use `miotaL`.

## Object-Centered Graph

The graph contains:

| Graph item | Meaning |
| --- | --- |
| `image` | one image / scene |
| `object` | detected or annotated visual object |
| `object_pair` | ordered pair of objects |
| `object_domain(o)` | candidate answer object |
| `name_* (o)` | object-name predicates, e.g. `name_dog(o)` |
| `attr_* (o)` | object-attribute predicates, e.g. `attr_red(o)` |
| `relation_* (p)` | object-pair relation predicates, e.g. `relation_right(p)` |
| `pair_src(p, o)` | source object of an object pair |
| `pair_dst(p, o)` | destination object of an object pair |
| KB-derived concepts | object-level concepts derived from KB facts |

For visual predicates, the VLM should answer predicate-level grounding
questions, not the full QA question.

For example:

```text
Given the image and object box 1304282, is this object red?
Answer yes or no.
```

For relations:

```text
Given the image, object box 1304282, and object box 1304272,
is object 1304282 to the right of object 1304272?
Answer yes or no.
```

## Query Form

For a single-answer query, the intended form is:

```python
queryL(answer_object, iotaL(andL(
    red("o"),
    right_of("o", iotaL(car("x")))
)))
```

For the current multi-answer example, the intended form is closer to:

```python
miotaL(andL(
    attr_red("o"),
    relation_right(path=("p", pair_src)),
    object_domain("o")
))
```

In implementation, relation queries need path wiring through `object_pair`,
`pair_src`, and `pair_dst`, rather than treating `right_of(o, x)` as a direct
binary Python call.

## Why `miotaL` Was Added

Some GraphQA questions return a set of objects, not a single object. For these
questions:

```python
iotaL(...)
```

is too restrictive, because it enforces a unique selected entity.

The new multi-answer selector is:

```python
miotaL(...)
```

It returns all candidate objects satisfying the body. Evaluation should compare
the predicted answer set against the gold answer set.

## Current Diagnostic Result

The recent Qwen3-VL diagnostic run used:

- C2 examples;
- non-trivial visual predicates;
- `miotaL` exact set matching;
- zero-shot Qwen3-VL predicate grounding;
- no predicate fine-tuning.

Observed result:

```text
300 examples
18 exact matches
6.0% exact-set accuracy
```

This result is not comparable to the earlier 60--70% results.

## Why This Is Not Comparable To Earlier 60--70%

The earlier strong numbers came from different settings:

| Result type | What it measured |
| --- | --- |
| TemporalQA 60--70% | temporal relation query accuracy on MATRES |
| GraphQA atomic accuracy | local predicate-family accuracy, not final answer accuracy |
| GraphQA oracle executable | correctness of the symbolic executable graph with gold predicates |
| GraphQA trained execution | learned predicate / execution diagnostics, not raw zero-shot VLM exact sets |

The 6% result is a stricter and currently weaker setting:

```text
zero-shot VLM predicate grounding + exact multi-answer set execution
```

This is useful as a failure analysis, but it should not be used as the main
DomiKnowS trained-execution result.

## What Went Wrong In The Low Run

The main issue is predicate grounding, not the existence of `miotaL`.

Common failure mode:

```text
logic_str: miotaL(attr_bare("o"))
gold_answers: one object
predicted_answers: many objects
```

The VLM binary yes/no scores are poorly calibrated. A fixed threshold such as
`0.5` can produce too many positives or miss the correct object.

This means the current zero-shot VLM predicate scorer is not yet a reliable
replacement for Scallop-style oracle scene graph predicates.

## Correct Comparison Path

To compare fairly with Scallop-style GraphQA:

1. First evaluate oracle predicates plus DomiKnowS execution.
2. Then evaluate learned predicate grounding separately.
3. For multi-answer questions, report exact-set accuracy and optionally
   recall/F1.
4. Do not mix atomic predicate accuracy with final answer-set accuracy.

## Current Implementation Files

| File | Role |
| --- | --- |
| `test_regr/GraphQA/object_centered_pipeline.py` | dynamic object-centered graph construction and logic translation |
| `test_regr/GraphQA/evaluate_object_centered_c2.py` | C2 evaluation runner and prediction logging |
| `test_regr/GraphQA/test_object_centered_pipeline.py` | unit tests, including `miotaL` multi-answer execution |
| `domiknows/graph/logicalConstrain.py` | `miotaL` logical constraint |
| `domiknows/solver/lcLossBooleanMethods.py` | differentiable multi-selection support for `miotaL` |

## Recommended Next Step

Use this sequence:

1. Validate C2 with oracle predicates and `miotaL`.
2. Run Qwen3-VL predicate grounding on a small non-trivial subset and log every
   sample.
3. Inspect false positives and false negatives by predicate type.
4. Replace raw yes/no VLM grounding with grouped multiple-choice predicate
   scoring or fine-tuned predicate heads.
5. Only then run large C2--C6 experiments.

