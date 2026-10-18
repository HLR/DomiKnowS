# Inference explanations

`DataNode.explainInference()` runs on private prediction tensors and fresh solver
state. It returns an `ExplanationReport`, or a list of reports in sample order
for a batch root. Existing `/ILP`, executable answers, and retained solver models
are preserved. The input must have populated prediction tensors.

```python
report = dn.explainInference(person, organization, mode="ilp")
print(report.render_text())
payload = report.to_dict()
json_text = report.to_json(indent=2)

# Use the exact identifier supplied by the report, rather than constructing one.
prediction = payload["predictions"][0]
alternative = report.counterfactual({prediction["id"]: False})
print(alternative.render_text())
```

Binary assignments must be Python `bool` values. Enum assignments must be names
from the prediction's `class_names`. Each ID encodes the sample index, full graph
and concept path, and entity instance ID. Ambiguous duplicate IDs are rejected.
Successive counterfactuals on an alternative report retain its previous
assignments; counterfactuals on the original report start from the original
inputs. The original report records detached counterfactual results.

## Semantics

| Mode | Explanation | Counterfactual |
|---|---|---|
| `ilp` | Local and selected labels, objective, executable-hypothesis answers, grounded verification and solver conflicts | Force labels and select a new optimal world, including a new executable-hypothesis search |
| `circuit` | Exact query probability/distribution, symbolic operator trace and grounded probability leaves | Evaluate `P(query AND evidence) / P(evidence)` with the existing circuit variable identities |
| `tnorm` | Fuzzy operator inputs/results and decoded scores | Replace specified probability inputs with Boolean endpoints and re-evaluate |

Circuit conditioning preserves shared-leaf identity and categorical exclusivity.
It conditions each query/rule evaluation under the existing executable circuit
semantics; it does not implicitly conjoin all graph rules into a global prior.
`undefined_evidence` means a logically possible event has zero probability;
`impossible_evidence` means the evidence event is logically false, including a
contradiction with a fixed label. A size limit produces `size_limit`, with no
automatic t-norm fallback.

T-norm fields are `score`, `truth_score`, and `position_scores`, rather than
probability claims. `distribution` retains the existing decoder's vector of
scores; it is not necessarily a normalized probability distribution. For a
Boolean query, `score` describes the selected answer and `truth_score` describes
the proposition being true. Circuit reports similarly distinguish `probability`
of the selected answer from `truth_probability`.

ILP reports preserve the ordinary raw-probability objective and the
executable-hypothesis log-probability objective defaults. `fun` keeps its existing
custom-transform meaning. Objective differences are reported only when both
runs completed optimally. Infeasible individual executable hypotheses may have
conflicts even when another hypothesis supplies an optimal answer. Runs and
conflicts include priorities; they do not imply every rule was enforced in the
selected priority run.

An IIS is an irreducible conflict, not necessarily the smallest conflict. Solver
rows identify their originating logical rule, counterfactual assignment, or
implicit ontology constraint. Bounds identify concept, entity and class.
Grounded rule references associated with changed labels establish involvement,
not unique causal responsibility. No automatic rule relaxation is performed.

## Options and evidence

Supported common options: `key=("local", "softmax")`, `compiled=True`,
`evidence_limit=1000`, and `queryNamespace` for ad hoc query strings.

T-norm/circuit options: `tnorm`, `counting_tnorm`, `threshold`, `circuit_backend`,
`circuit_max_nodes`, and `circuit_aggregation` (existing inference semantics).
Existing DataNode spellings `circuitBackend`, `circuitMaxNodes`, and
`circuitAggregation` are also accepted. `circuitSizeLimitAction` must be `"raise"`.
ILP options: `fun`, `epsilon`, `minimizeObjective`, and `ignorePinLCs`.

Registered executable queries use the sample's activation labels. Ad hoc queries
use the same expression/string mapping as `inferExecutableResults()`:
`constraints` filters registered soft queries; for ILP it must exactly match
the active executable names. `queries` and `constraints` are mutually exclusive.

```python
report = dn.explainInference(
    person, mode="circuit",
    queries={"any_person": "existsL(person('x'))"},
    queryNamespace={"person": person},
)
```

Schema version 1 contains backend settings, completion status, predictions,
answers, evidence with stable IDs, solver runs, conflicts, timing, unsupported
sections, and truncation counts. Leaf evidence records actual evaluated inputs,
including fixed-label substitutions. Evaluation evidence records candidate
bindings, operand values and operator results for compiled and legacy traversal.
Aggregating operators retain their candidate domains rather than inventing a
single-entity explanation. The evidence limit caps recorded evidence, not
inference work; larger reports record omitted record counts.

The report retains a private baseline session for counterfactuals. Serialization
contains no session, DataNode, tensor, or Gurobi object. Graph changes invalidate
the session: create a new explanation after editing rules. Diagnostic calls are
serialized with one another; callers must not concurrently mutate the graph or
run ordinary inference on the same graph during a diagnostic call.

## Optional LLM interpretation

```python
def adapter(evidence):
    # Configure your provider here. No provider is called by DomiKnowS itself.
    return {
        "summary": "The counterfactual forces this grounded prediction.",
        "evidence_ids": [evidence["evidence"][0]["id"]],
    }

interpretation = report.summarize(adapter)
```

The adapter receives a detached JSON-safe report, without raw sample text or
arbitrary attributes. It must return text and at least one known evidence ID.
Unknown references or provider errors return a failed interpretation and leave
the report intact. Accepted text is labeled `LLM-generated`; reference validation
does not prove that the prose correctly interprets the evidence.

See [the runnable three-mode example](../../../../test_regr/examples/explanation_demo.py).
