# Constraint-guided active learning

Select whole examples for annotation, with ranked entity/relation targets and
classifier label candidates inside each example. Selection never requests
annotations, changes public predictions, trains a model, or accesses an oracle.

```python
from domiknows.solver.active_learning import ConstraintActiveLearner

learner = ConstraintActiveLearner(
    selected, eligible, mode="ilp",
    weights={"uncertainty": .35, "constraint_conflict": .35,
             "inference_disagreement": .30},
)
queue = learner.select(
    pool, budget=20,
    sample_id=lambda root: root.instanceID,
    annotations=annotation_mask,
    embeddings=sample_embeddings,  # optional; omit for score-only selection
)
print(queue.render_text())
payload = queue.to_dict()
json_text = queue.to_json(indent=2)
```

`pool` is an iterable of individual sample DataNode roots with predictions.
Flatten a batch wrapper's samples into the pool before selection. Sample IDs
must be unique strings or integers. Duplicate grounded targets are rejected by
diagnostic inference and reported as a failed example. Supply `sample_id` when
root instance IDs are not globally unique across your pool.

## Annotation masks and model inputs

Get exact target IDs from `payload["ranked"][i]["targets"]`. Each ID includes
the sample ID, concept path, and instance ID. Enum targets carry class names.
`annotations` maps sample IDs to collections of annotated target IDs:

```python
example = payload["selected"][0]
annotation_mask = {
    example["sample_id"]: [target["id"] for target in example["targets"]]
}
```

This is an exclusion mask, not a mapping of gold labels. Labels belong to your
annotation/training pipeline. Partially annotated examples remain eligible for
their remaining targets. Unknown samples, unknown target IDs, and malformed
masks raise `ValueError`. Fully annotated examples are excluded.

Already populated values at the inference `key` are authoritative. Missing local
values are computed privately from logits. If you retain both logits and local
probabilities, update the selected probabilities after retraining or remove them
to request recomputation. Selection does not run sensors or learners.

## Scores and backend semantics

Every component has a bounded 0-to-1 interpretation within its backend:

| Component | Measurement |
| --- | --- |
| `uncertainty` | Classifier entropy divided by log(class count) |
| `constraint_conflict` | Boolean violations for ILP; rule dissatisfaction for soft modes |
| `inference_disagreement` | ILP constrained label differs from classifier argmax |

Classifier entropy requires finite, nonnegative, normalized probabilities. It
is unavailable for arbitrary unnormalized scores. Hard ILP disagreement is a
label-change indicator, including enum label changes; it is not an objective gap.

ILP conflict comes from discrete verification of active head logical rules
under classifier argmax labels. Compiled joined groundings supply entity-level
associations; legacy or aggregated domains use domain involvement. Implicit
ontology constraints can affect the ILP label-change signal, but are not included
as separate local conflict scores in this version.

Circuit conflict is `1 - P(rule satisfaction)` under the existing circuit
semantics, using shared leaves and categorical exclusivity. It does not conjoin
all rules into a globally conditioned prior. T-norm conflict is `1 - truth_score`
under the selected t-norm; it is not a probability. Soft conflict uses the
complete head-rule evaluation and associates it with involved grounded targets.
It does not claim a separate exact marginal violation probability per target.
False executable-query answers are not automatically interpreted as rule violations.

Circuit and t-norm target-aligned constrained-label disagreement is unavailable
in this version. Its weight is explicitly omitted and recorded in `unsupported`.
Unsupported non-Boolean conflict outputs and targets without relevant rules also
have unavailable conflict signals. Circuit diagnostic size limits never silently
substitute t-norm acquisition scores.

Weights are renormalized independently for each target over its available
signals. Targets report `components`, `effective_weights`, `unavailable_signals`,
`score`, `related_rules`, and available `evidence_ids`. Rule links indicate
involvement, not causal responsibility. Evidence IDs are namespaced by sample.
These scores are within-backend acquisition heuristics, not calibrated estimates
of annotation value. Comparing scores across different backend modes is not a
probability comparison.

The example score is the mean of its highest-scoring `top_targets` eligible
targets (default five). This avoids rewarding an example simply by summing more
entities. All remaining annotation targets are still returned, with label
candidates ranked by classifier probability. Suggested labels are predictions,
never annotation answers.

Strategies: `weighted` (default), `uncertainty`, `constraint_conflict`,
`inference_disagreement`, and seeded `random`. Selecting a single unavailable
signal excludes that example rather than substituting a different signal.
Ties use stable encoded sample/target IDs. Random order is independent of pool
iteration order for a fixed seed.

## Optional diversity

`embeddings` is a mapping covering exactly the pool sample IDs with finite,
nonzero vectors of equal dimension. Torch vectors and ordinary lists work.
The selector normalizes them, selects the highest acquisition example first,
then greedily balances its score against minimum cosine distance to already
selected examples. `diversity_weight` defaults to `.25`; distance is scaled to
0-to-1. Results report the combined `selection_score` and `diversity_score`.
Embeddings are caller supplied; there is no provider dependency or network call.

## Repeated rounds

```python
with learner.session() as session:
    first = session.select(pool, budget=10)
    # Obtain annotations externally, update the mask and model predictions.
    second = session.select(pool, budget=10, annotations=updated_mask)
```

The session owns one incremental inference engine per sample and caches detached
explanation evidence. Unchanged samples skip inference and evidence collection;
annotation-mask changes only require rescoring. Prediction, node, relation,
rule, or inference-setting changes invalidate the appropriate sample cache.
Evicted samples and closed sessions release owned engines. Custom objectives
and opaque inputs conservatively disable evidence reuse.

Changed samples currently perform incremental inference **and a cold diagnostic
pass** to refresh evidence. Grounding evidence is not yet collected directly
from reused incremental rule blocks. Therefore, a speedup for changed samples
is not promised. `stats.incremental`, `explanations_reused`, and `timing.total_ms`
make this cost visible. Sessions are intended for serial use; source graphs must
not be mutated concurrently with acquisition.

## Reports and failures

`AnnotationSelection` provides detached `to_dict()`, strict `to_json()`, and
deterministic `render_text()`. Reports include schema version, requested mode,
actual per-example backend, settings, weights, ranked and selected examples,
excluded reasons, evidence, truncation, timing, and cache statistics.

Failed and interrupted evaluations are excluded by default. Proven infeasibility
is excluded with `review_required=True` and conflict evidence; it is not assigned
an automatic maximum score. Failures preserve previously committed evidence and
allow later recovery. Invalid API configuration, masks, IDs, and embeddings raise
instead of producing a misleading queue. No results are populated onto public
DataNodes.

`evidence_limit` defaults to 1,000 displayed records per example. A streaming
collector computes acquisition associations and statistics from all evaluations,
independently of this limit. Reducing the limit changes displayed evidence and
references, not scores or rankings. Timing is reported separately from deterministic
rendered text. `budget=0` still returns the ranked pool for inspection and masks.

## Examples, tests, and benchmark

Run the three-backend annotation-mask example:
`python -m test_regr.examples.active_learning_demo`.

Tests in `test_regr/solver/test_active_learning.py` cover manual implication scores,
disjointness/counting, enum entropy, relation targets, annotation exclusion,
diversity, graph-size aggregation, truncation independence, output isolation,
failure recovery, and repeated-round caches.

Run `python -m test_regr.solver.benchmark_active_learning --output results.json`
for a reproducible annotation loop comparing random, uncertainty, weighted
constraint-guided, and weighted-plus-diversity selection. The benchmark uses
three seeds, a pool of 12 examples, an independent 48-example test set, and
whole-example budgets of zero, two, and four. A simulated oracle accesses gold
only after selection; the selector never receives test labels. A small classifier
is retrained from the same initial weights at each budget point. Results measure
unconstrained label/example accuracy and logical satisfaction on the held-out set.

The checked-in `test_regr/solver/active_learning_benchmark_results.json` contains
per-seed measurements and means/standard deviations. This small synthetic run
validates the annotation loop and comparison methodology; it does not establish
gains on a production dataset or imply constraint-guided acquisition always wins.
At budget four, the measured mean and sample standard deviation across three
seeds were (percent):

| Strategy | Label accuracy | Whole-example accuracy | Constraint satisfaction |
| --- | ---: | ---: | ---: |
| Random | 75.35 +/- 12.86 | 63.19 +/- 15.77 | 97.22 +/- 2.41 |
| Uncertainty | 78.12 +/- 9.08 | 56.25 +/- 18.16 | 100.00 +/- 0.00 |
| Weighted | 84.72 +/- 7.82 | 70.83 +/- 14.58 | 100.00 +/- 0.00 |
| Weighted with diversity | 90.62 +/- 5.80 | 81.25 +/- 11.60 | 100.00 +/- 0.00 |

An annotation UI, automatic retraining service, learned acquisition policies,
and soft target-aligned disagreement are deferred.
