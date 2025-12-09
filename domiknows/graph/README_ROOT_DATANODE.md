# DataNode Root Selection Algorithm

The `findRootDataNode` method identifies the root DataNode from a collection using a hierarchical selection process.

## Selection Criteria (in order of priority)

### 1. Impact Links Filter (Soft Filter)
**Candidates must have no incoming `impactLinks`, or only `"contains"` type links.**

*Reason:* In a graph structure, root nodes typically have no incoming edges. The `"contains"` relationship is hierarchical (parent→child), so having only `"contains"` impacts still qualifies a node as root. Nodes with other incoming links are dependents, not roots.

*Fallback:* If no nodes qualify, **all nodes become candidates**. This ensures `root_candidates` is never empty.

**Important implication:** Because of this fallback, Step 1 does not guarantee filtering. When fallback occurs, all original nodes remain as candidates for subsequent steps.

### 2. Initial Instance ID Match (High Priority)
**If a `READER` key exists in the builder, attempt to select the node matching that `instanceID` from the candidates.**

*Reason:* The `READER` key indicates an externally specified entry point (e.g., from data loading). This respects explicit configuration over heuristic selection, ensuring deterministic behavior when the root is known in advance.

**In practice, this step often succeeds** because:
- Step 1's fallback ensures all nodes remain candidates when filtering fails
- Default `instanceID=0` exists among all nodes

Steps 3-5 are only reached when Step 1 successfully filters candidates **AND** the filtered set excludes both:
- The READER-specified instance (if configured), **AND**
- The default instance (`instanceID=0`)

#### Instance ID Resolution Logic

```python
initialInstanceID = -1
if "READER" in self:
    initialInstanceID = dict.__getitem__(self, "READER")
else:
    initialInstanceID = 0
```

| Condition | `initialInstanceID` | Behavior |
|-----------|---------------------|----------|
| `READER` key exists | Value stored in `READER` | Use explicitly configured root instance |
| `READER` key absent | `0` | Default to first instance (index 0) |

- **`READER` present:** The builder was configured with a specific starting point (e.g., by a data reader that knows which instance is the entry point). That value is used directly.
- **`READER` absent:** Assumes the first created instance (`instanceID=0`) is the root—a reasonable default when no explicit configuration exists.
- **Uses `dict.__getitem__`:** Bypasses any custom `__getitem__` override in the builder class to access the raw dictionary value.
- **Can fail silently:** If no candidate matches `initialInstanceID`, selection continues to Step 3.

### 3. Rarest Ontology Type
**Among remaining candidates, prefer nodes whose `ontologyType` appears least frequently.**

*Reason:* Common ontology types (e.g., `Token`, `Span`) are typically lower-level entities. Rare types often represent higher-level, aggregate concepts like `Sentence` or `Document`—more likely to be the semantic root.

### 4. Most Outgoing Relations
**Select the node with the most outgoing `relationLinks` (excluding `"contains"`).**

*Reason:* Root nodes typically govern or reference other entities. More outgoing relations suggest a coordinator role, aggregating information from multiple children or related nodes.

### 5. Most Children (Final Tiebreaker)
**Prefer the node with the most child DataNodes.**

*Reason:* A node with more children has broader structural coverage, indicating it sits higher in the containment hierarchy—characteristic of a root.

## Guarantees

- **Always returns a DataNode** if the input list is non-empty
- **Deterministic** — same input produces same output through consistent priority ordering