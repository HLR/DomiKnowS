# TemporalRelation / MATRES Adapter

This adapter models MATRES temporal relation classification in a CLEVR-style DomiKnowS pipeline:

1. Build generic graph concepts in `graph.py`.
2. Convert MATRES rows into document events and event pairs in `dataset.py`.
3. Build query logic and learner-facing examples in `execution.py`.
4. Run the learned front-end through multiple-choice small-LLM prompts in `llm_inference.py`.
5. Check oracle labels and global temporal consistency in `oracle.py`.
6. Validate behavior in `test_temporal_relation_adapter.py` and `smoke_test_dataset.py`.

## Graph Concepts

Defined in `graph.py`:

- `document`, `sentence`, `token`: text structure.
- `event`: event mentions over tokens/spans.
- `query_event1`, `query_event2`: learned marker predicates over event nodes.
- `EventPair`: pair carrier with `pair_event1` and `pair_event2` edges.
- `temporal_relation`: multiclass parent concept over event pairs.
- `Before`, `After`, `Equal`, `Vague`: label concepts under `temporal_relation`.

MATRES gives oracle event ids, so `event(e)` can be treated as observed. In a learned pipeline, an LLM/question interpreter predicts `query_event1(e)` and `query_event2(e)` similarly to CLEVR compositional concepts selecting objects.

## Example

Text:

```text
John packed his bag before he left the house.
```

MATRES-style event ids:

```python
events = [
    {"id": "eiid12", "text": "packed"},
    {"id": "eiid18", "text": "left"},
]

query_event1(eiid12)
query_event2(eiid18)
```

The local pair learner receives text with event markers:

```text
John [E1]packed[/E1] his bag before he [E2]left[/E2] the house.
```

and predicts logits over:

```python
Before(p), After(p), Equal(p), Vague(p)
```

for the candidate event pair `p = EventPair(eiid12, eiid18)`.

## Final Query

The final query selects the event pair whose endpoints satisfy the learned query-event markers:

```python
queryL(
    temporal_relation,
    iotaL(
        andL(
            EventPair("p"),
            event(path=("p", pair_event1)),
            event(path=("p", pair_event2)),
            query_event1(path=("p", pair_event1)),
            query_event2(path=("p", pair_event2))
        )
    )
)
```

Expected label for the example:

```python
Before
```

## Learner Interface

`execution.py` exposes two helper layers.

`create_query_event_groundings(instance)` returns oracle labels for query marker predicates:

```python
[
    {"event_id": "eiid12", "query_event1": True, "query_event2": False},
    {"event_id": "eiid18", "query_event1": False, "query_event2": True},
]
```

In the learned version, a small LM predicts these markers from the input question/task text.

`create_pair_learner_examples(instance)` returns one classification example per ordered event pair:

```python
{
    "e1": "eiid12",
    "e2": "eiid18",
    "text_with_event_markers": "John [E1]packed[/E1] ... [E2]left[/E2] ...",
    "target_concept": "temporal_relation",
    "label_concepts": ("Before", "After", "Equal", "Vague"),
    "label": "Before",
}
```

This is the normal pair-classification learner target.

## Small-LLM Inference

`llm_inference.py` adds the actual learned front-end used by the adapter:

- `build_query_event_choice_examples(instance)` asks the model to select exactly one event from all candidate events for `query_event1`, then exactly one event for `query_event2`.
- `build_temporal_relation_choice_examples(instance)` asks the model to classify every ordered candidate event pair into one label from `Before`, `After`, `Equal`, `Vague`.
- `SmallCausalLMChoiceBackend` wraps a Hugging Face causal LM such as `Qwen/Qwen2.5-0.5B-Instruct`.
- `StaticChoiceBackend` is used in tests to verify the parsing and grounding loop without loading a model.

The prompt style is deliberately bounded multiple choice. For query-event selection, the model sees all detected document events as candidate answers and must return one candidate by letter or answer text. The selected events become the learned predicate groundings for:

```python
query_event1(e)
query_event2(e)
```

For temporal-relation classification, the model sees text with `[E1]...[/E1]` and `[E2]...[/E2]` markers and returns one temporal label. These predictions provide the learner-side values/logits for `temporal_relation(p)`.

Run a real small-LLM pass on one grouped MATRES document:

```bash
conda run -n CLEVER python -m test_regr.TemporalRelation.run_llm_inference \
  --root /egr/research-hlr2/premsrit/TemporalRelation \
  --limit 1 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cpu
```

If the model is not already cached, this command may need network access and can take time to download weights.

## Global Consistency

The oracle consistency checker in `oracle.py` covers examples of:

- inverse consistency: `Before(x, y) -> After(y, x)`
- equality symmetry: `Equal(x, y) -> Equal(y, x)`
- mutual exclusion: one temporal label per pair
- transitivity/no-cycle checks for `Before`

These rules operate over all document-level event pairs, not only the final queried pair. The smoke test applies the audit per grouped document/sample so event ids from different documents are never mixed.

## Run Tests

From the DomiKnowS repo root:

```bash
conda run -n CLEVER python -m unittest test_regr.TemporalRelation.test_temporal_relation_adapter
```

Run the MATRES smoke test with per-sample consistency checks:

```bash
conda run -n CLEVER python -m test_regr.TemporalRelation.smoke_test_dataset \
  --root /egr/research-hlr2/premsrit/TemporalRelation \
  --materialize-candidates \
  --check-consistency
```

Expected current MATRES result: `convert_failures=0`, `oracle_failures=0`, and `consistency_failure_samples=0` for `aquaint.txt`, `platinum.txt`, and `timebank.txt`.

Combined adapter regression:

```bash
conda run -n CLEVER python -m unittest \
  test_regr.GraphQA.test_graphqa_adapter \
  test_regr.TemporalRelation.test_temporal_relation_adapter
```

## Implementation Map

- `graph.py`: DomiKnowS concepts and relations.
- `dataset.py`: MATRES/TB-Dense style file discovery and normalization.
- `execution.py`: final `queryL` logic, candidate event-pair generation, query-event marker labels, and local learner examples.
- `llm_inference.py`: multiple-choice small-LLM loop for `query_event1`, `query_event2`, and `temporal_relation`.
- `run_llm_inference.py`: CLI for running the LLM loop over real MATRES files.
- `oracle.py`: expected answer lookup and temporal consistency checks.
- `test_temporal_relation_adapter.py`: unit tests for graph, query, learner examples, oracle, and dataset grouping.
- `smoke_test_dataset.py`: whole-dataset conversion/oracle smoke.
