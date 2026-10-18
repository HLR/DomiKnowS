# Incremental inference

An incremental session owns inference state for a changing DataNode graph.
It automatically compares snapshots on every call, including tensor contents,
node identity and order, relation destinations, ontology declarations, logical
rule expressions, activity, and priorities. Explicit hints widen invalidation;
they never suppress automatically detected changes.

```python
with root.incrementalInference(selected, eligible, mode="ilp") as session:
    initial = session.infer()  # publish normal inference outputs
    child.attributes["<selected>/local/softmax"].copy_(new_probabilities)
    updated = session.infer(populate=False)  # return detached values only
    rule.active = False
    edited = session.infer(changes={"rules": [rule]})
    print(edited.status, edited.answers, edited.stats)
    print(edited.to_json(indent=2))
```

Use `mode="circuit"` or `mode="tnorm"` for the other execution semantics.
Run the complete three-mode example from the repository root:
`python -m test_regr.examples.incremental_demo`.

## Inputs and results

Concept arguments select inference inputs using the existing DataNode convention.
Omitting them collects available predicted concepts. `queries` accepts the
existing ad hoc executable-query mapping; queries are registered temporarily,
then removed. Otherwise, active registered executable constraints are evaluated.

Supported options are `key` (default `("local", "softmax")`), `compiled`
(default `True`), `tnorm`, `counting_tnorm`, `threshold`, `circuit_backend`,
`circuit_max_nodes`, `circuit_aggregation`, `circuit_size_limit_action`,
`queryNamespace`, `constraints`, `fun`, `epsilon`, `minimizeObjective`, and
`ignorePinLCs`. Existing camel-case circuit option aliases also work. Backend
options keep their existing inference meanings. `queries` and `constraints`
cannot be combined. ILP constraint selection must match active executable names.
Values already present at the selected `key` are authoritative. Missing local
values are derived from logits on the private sample. If both logits and local
probabilities are stored, update the selected probabilities or remove them to
request recomputation; changing logits alone does not override explicit inputs.

`infer(changes=None, populate=True)` returns an `IncrementalInferenceResult`.
For a batch root, it returns one result per sample in current input order, with
independent sessions and caches. Results contain JSON-safe `status`, `mode`,
`answers`, ILP `assignments` and `objective`, detected `changes`, reuse/timing
`stats`, and an optional `error`. `to_dict()` makes a detached copy;
`to_json()` serializes it. Soft answers contain distributions and decoded
answers following the ordinary backend's semantics. T-norm scores are not
probabilities. Circuit fallback remains explicit in answer metadata and stats.

Hints accept `nodes`, `concepts`, `relations`, and `rules` lists. Concept and
rule objects or their names can be supplied; relations use link names.
`nodes` hints conservatively invalidate attribute-dependent evaluation.
Changing the selected source probabilities directly, adding/removing children,
rewiring relations, and registering/removing/editing logical constraints are
detected without hints. A hint does not make snapshot scanning optional.

## Reuse and invalidation

* ILP retains session-owned models and rule blocks. Prediction edits refresh
  the objective and reoptimize with previous assignments as starts. Rule edits
  replace affected blocks; topology edits rebuild grounded structural blocks
  and update predictive variables. Executable hypotheses are selected again.
  Schema, input-layout, or priority-layout changes use a fresh model. Optimization
  remains global, so a local change can alter distant assignments. Warm starts
  may select a different equally optimal assignment when the objective ties.
* Circuit retains symbolic formula construction and refreshes leaf weights.
  Unaffected rule/query results are reused. Structural changes invalidate whole
  affected rule domains. Manager compaction and size-limit fallback can require
  a cold reevaluation, recorded under `full_rebuild_reasons`.
* T-norm reuses unaffected rule/query results and evaluates dependent rules
  again. Formula plans and eligible relation adjacency tensors are cached.

Invalidation granularity is a complete logical constraint, rather than individual
groundings. Topology edits, arbitrary attributes, opaque candidate selectors,
and custom objective callbacks conservatively invalidate broader state.
Custom callbacks are evaluated again because external closure state cannot be
verified. Ordinary inference's legacy ILP reuse also checks topology/rule
signatures, preventing reuse based only on equal entity counts.

Statistics report actual model/structure/constraint and adjacency reuse,
formula-plan hits/misses, variable additions/removals, and rebuild reasons.
`total_ms` includes snapshot detection and publication. `compilation_ms` is a
subset of construction time, so timing fields should not be summed. ILP
`evaluation_ms` measures optimization; circuit measures weighted model counting;
t-norm includes formula traversal and evaluation. No-change calls skip evaluation.

## Lifetime and failure behavior

Use the context manager or call `close()` to release owned models and caches.
Successful published outputs remain after closing; later `infer()` calls reject
a closed session. Sessions do not modify a retained ordinary solver model.
Graph registration, active flags, and temporary DataNode state are restored
after diagnostic execution. Return-only calls preserve public outputs.

Only completed inference commits cache state. Infeasible, interrupted/ambiguous,
or failed calls preserve the last committed result, discard staged models, and
remove stale outputs still owned by that session. A later call can recover.
Exceptions become `status="error"` with an error message; invalid API options
and closed-session use raise immediately. ILP status distinguishes proven
infeasibility from interruption/ambiguity.

Do not mutate source graphs concurrently with inference or share a session
between threads. Snapshot checking detects DataNode mutations during evaluation,
but it is not a synchronization mechanism for arbitrary ontology mutation.
Tensor snapshot hashing requires reading contents, including device-to-host
transfer for CUDA tensors. Sessions are inference-only and run without gradients;
they do not rerun sensors, learn parameters, or cache across processes.

## Validation and benchmark

`test_regr/solver/test_incremental.py` compares updates with fresh inference for
all modes, compiled/legacy relation paths, enum/counting queries, rule edits,
priorities, fixed labels, same-count rewiring, batch membership, schema changes,
failure recovery, and sensor-built skeleton samples.

Run `python -m test_regr.solver.benchmark_incremental --output results.json`
for a reproducible small CPU benchmark. The checked-in results in
`test_regr/solver/incremental_benchmark_results.json` use 16 entities, six binary
predicates, and five prediction updates, with fresh parity checked every update.
The fresh baseline includes session/solver initialization and closure; incremental
time includes snapshot detection. These measurements are workload-specific.

| Backend | Workload | Incremental median ms | Fresh median ms | Ratio |
| --- | --- | ---: | ---: | ---: |
| ILP | Independent | 115.20 | 472.55 | 4.10x |
| ILP | Global count | 112.85 | 468.02 | 4.15x |
| Circuit | Independent | 45.35 | 514.22 | 11.34x |
| Circuit | Global count | 57.48 | 578.76 | 10.07x |
| T-norm | Independent | 56.76 | 319.68 | 5.63x |
| T-norm | Global count | 55.12 | 386.53 | 7.01x |
