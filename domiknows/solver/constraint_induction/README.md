# Constraint induction with human review

`ConstraintInducer` mines bounded, structured constraint proposals from examples
and domain guidance. It never activates a proposal automatically. Predictive
correlations are proposals, not established domain truths or causal claims.

```python
from domiknows.solver.constraint_induction import ConstraintInducer

inducer = ConstraintInducer(person, organization, employment)
proposals = inducer.propose(
    discovery_examples,
    guidance=["A person cannot also be an organization."],
    validation_examples=held_out_examples,
    sample_id=lambda root: root.instanceID,
)
print(proposals.render_text())
payload = proposals.to_dict()
json_text = proposals.to_json(indent=2)

candidate_id = payload["candidates"][0]["id"]
preview = proposals.preview(candidate_id, preview_examples,
                            modes=["ilp", "circuit", "tnorm"])
print(preview.render_text())

# Call only after receiving the human review decision.
decision = proposals.approve(candidate_id, reviewer="reviewer-id",
                             rationale="Checked the domain definition and evidence.")
activation = proposals.activate(decision, graph)
proposals.deactivate(activation, reviewer="reviewer-id", rationale="Pause this rule.")
```

Reviewer identity is caller-provided audit metadata. This Python workflow does
not authenticate reviewers or distinguish a human caller from an automated
script. Your application must enforce its review/access policy. A detached JSON
report cannot manufacture a trusted approval in another proposal set.

## Examples and observations

Supply individual sample DataNode roots, an allowed concept vocabulary, and
unique string/integer sample IDs. Discovery and validation IDs and DataNode
objects must be disjoint. The API detects identity overlap; callers must prevent
semantic duplicate examples under different IDs. Flatten batch wrappers into
individual roots. Sensor-built skeleton samples are privately materialized.

Annotations come from exact `<concept>/label` attributes or corresponding
skeleton variable/property tensors. Binary labels accept Boolean/0/1 scalars or
one-hot vectors. Enum labels accept class names, integer indices, or one-hot
vectors. Invalid labels raise; absent labels and NaN placeholders remain unknown.
Extraction avoids
DataNode's fuzzy attribute lookup, which can otherwise return another predicate's
label when a requested label is missing.

Prediction observations use argmax at `key=("local", "softmax")`, configurable
on the inducer. Missing or nonfinite predictions remain unknown; prediction
observations never replace annotation observations. Source tensors, retained
solver models, graph registries, and inference outputs remain unchanged during
proposal generation and previews. No sensors, model training, provider SDKs, or
network calls are enabled by default.

## Templates and guidance

Supported `templates` are `implication`, `disjointness`, `relation_typing`, and
`counting`; enum restrictions use implication/disjointness with class literals.
Unary rules require matching entity domains. Relation typing supports one to
four declared roles. Counting supports `at_most` and `at_least`, per example for
entity predicates or per declared role for relation predicates. General arbitrary
logical expressions and multi-hop path synthesis are outside this version.

The deterministic guidance grammar supports:

* `employee implies person`
* `A person cannot also be an organization`
* `person and organization are disjoint`
* `color=red implies warning`
* `At most 1 employee per example`
* `At least 1 employment per arg1`

Names are resolved against the allowed vocabulary; ambiguous names remain
unresolved. Full graph/concept paths are accepted. Unsupported prose is retained
with an error for human editing, rather than guessed. Guidance-derived domain
requirements stay visible even when the data contradicts them.

For broader language, provide `guidance_adapter(request)`. It receives only the
guidance text, its ID, allowed concept paths/enum classes/role names, and enabled
templates. Return `{"candidates": [specification, ...]}`. The adapter is trusted
caller-supplied Python code, but its returned specifications are validated and
compiled directly through an allowlist. No generated Python or DSL text is
evaluated. Unknown fields/symbols, invalid domains, disabled templates, malformed
responses, or adapter failures become unresolved proposals. Invalid responses
are rejected as a whole; ordinary statistical mining still proceeds.

Specifications have these shapes (priority defaults to 100):

```python
{"template": "implication", "antecedent": "employee", "consequent": "person"}
{"template": "disjointness", "left": "person", "right": "organization"}
{"template": "relation_typing", "relation": "employment",
 "roles": {"arg1": "person", "arg2": "organization"}}
{"template": "counting", "predicate": "employment", "operator": "at_most",
 "limit": 1, "group_by": "arg1"}
{"template": "implication", "antecedent": {"concept": "color", "class": "red"},
 "consequent": "warning", "priority": 90}
```

Automatic mining considers compatible unary pairs, declared relation-role
types, and observed counting minima/maxima. It skips existing categorical
exclusivity rules. Guidance is processed first; `max_candidates` defaults to 256.
Truncation reports a lower bound on omitted search candidates, not an exhaustive
search count. Deduplicated specifications retain all generation origins.

## Counting completeness

Declare closed domains explicitly:

```python
complete_domains = {sample_id: ["employment"] for sample_id in sample_ids}
```

Pass this mapping to `propose`. A count is unknown without that declaration,
when any member label is missing, or when grouping links are missing/ambiguous.
Declarations cover the complete possible member domain, including negatives;
observing only known positive relations is insufficient to validate an upper
bound. Empty declared domains count as zero. For per-owner counts, declared
owner nodes with no related members are included. Reviewers are responsible for
whether completeness declarations match the real dataset.

## Metrics and ranking

Discovery and held-out metrics are reported separately, each split into
annotation and prediction sources. Reports contain applicable/unknown/vacuous
groundings, violations, supporting/violating/covered examples, coverage,
conditional confidence, bootstrap stability, and grounded support/contradiction
records. They include dataset IDs, completeness declarations, policy/settings,
candidate IDs/versions, origins, and review history; arbitrary sample text and
attributes are omitted. Guidance text is retained intentionally as provenance.

Implication confidence considers true antecedents with known consequents; false
antecedents do not inflate it through vacuous satisfaction. Disjointness considers
known groundings where at least one predicate is true. Relation typing considers
true relations with known, uniquely linked endpoint types.

Confidence averages per-example satisfaction among applicable examples. The
separate `micro_confidence` weights groundings. Bootstrap resampling operates on
examples, avoiding claims that 100 correlated groundings are 100 independent
examples. The rank score is confidence times coverage times one minus bootstrap
standard deviation. It is a review heuristic, not a calibrated probability of
a rule being universally valid.

Defaults require at least two applicable held-out examples, coverage `.1`,
violation rate at most `.05`, and bootstrap standard deviation at most `.2`.
`bootstrap_rounds` defaults to 50. Prediction-only validation is not accepted by
default; enable `allow_prediction_only=True` explicitly to use it when no
applicable held-out annotation evidence exists. This source is recorded in the
review decision. `evidence_limit` defaults to 100 observations per split/source;
changing it does not change metrics, stability, or ranking.

## Review, versions, and activation

`approve`, `reject`, and `edit` require a reviewer and nonempty rationale.
Approval requires a resolved specification and a held-out validation pass.
When computed validation misses policy thresholds, `approve(..., override=True)`
records an explicit review override and the failed validation status. It does
not pretend the statistical checks passed. Missing held-out data or unresolved
symbols cannot be overridden.

`edit(id, specification, reviewer=..., rationale=...)` creates a new version,
archives the old one, and invalidates approval and metrics. Call `revalidate()`
before reviewing it again. Changes to source examples, ontology/rules, or
induction settings also require revalidation and renewed review. Revalidation
invalidates prior inactive approvals; activated provenance remains a frozen
record of what was reviewed.

`activate(decision, graph)` accepts an unchanged approval issued by this proposal
set for its current validated version. Constraints are registered under owned,
collision-free names; ordinary incremental inference detects these rule edits.
The head and nested DSL constraints retain induction provenance, validation
metrics, and the review decision. A requested graph must contain the vocabulary
concepts involved in the rule. Existing models are not modified directly.

`deactivate(activation, ...)` and `rollback(activation, ...)` disable the registered
version and append audit events, preserving history. Active versions must be
deactivated before editing. An unmodified deactivated version may receive a new
approval and be reactivated; older activation handles cannot undo that newer
activation. Externally replaced/edited owned rules require a new reviewed version.
Use JSON exports as audit artifacts; restoring live approval authority from JSON
is deliberately not implemented.

## Impact previews

`preview(id_or_ids, examples, modes=..., **inference_options)` compares baseline
inference with the selected candidate group in a temporary registry context.
It returns JSON-safe `InductionPreview` with complete explanation reports,
changed labels/answers, timing, and unsupported sections. Previews of already
active candidates explicitly list them; their baseline already includes them.

ILP differences include changed optimal labels/hypotheses and objective deltas
only for matching configurations and completed optimal solves. Custom objective
transforms omit deltas because external mutable state cannot be verified. Proven
infeasibility retains IIS provenance; an IIS is an irreducible conflict, not a
smallest conflict or exhaustive repair. Preview groups together before activation
because individually acceptable rules can conflict.

Circuit and t-norm previews retain ordinary backend semantics. Circuit rule
evaluation does not automatically condition other queries on newly proposed
rules; those query probabilities may remain unchanged. T-norm changes are fuzzy
scores, not probabilities or costs. Full proposed reports include the additional
rule evaluations even when existing answers stay unchanged. Previews are logical
and computational comparisons, not real-world causal explanations.

Use this workflow serially and avoid concurrent source/graph mutation. The
Python objects retain live source roots for revalidation and are not cross-process
sessions. A review UI, unrestricted rule synthesis, automatic activation, and
automatic repair/retraining are deferred.

## Validation and example

Run `python -m test_regr.examples.constraint_induction_demo` for guidance,
held-out validation, a three-mode group preview, explicit simulated review,
activation, provenance inspection, and deactivation on a synthetic graph.

`test_regr/solver/test_constraint_induction.py` checks annotations versus
predictions, missing labels, vacuous implications, example-clustered confidence,
truncation, enum/disjointness rules, relations/per-owner counts, complete-domain
requirements, adapter failures and unsafe references, stale/forged approvals,
version edits, infeasible groups/IIS, exception restoration, and incremental
invalidation after activation.
