# TemporalRelation / MATRES Adapter

This adapter models MATRES temporal relation classification in a CLEVR-style DomiKnowS pipeline:

1. Build generic graph concepts in `graph.py`.
2. Convert MATRES rows into document events and event pairs in `dataset.py`.
3. Build query logic and learner-facing examples in `execution.py`.
4. Predict graph-aligned predicate logits with `modules.py`.
5. Optionally run the zero-shot text-generation baseline in `llm_inference.py`.
6. Check oracle labels and global temporal consistency in `oracle.py`.
7. Validate behavior in `test_temporal_relation_adapter.py` and `smoke_test_dataset.py`.

## Configuration

All machine-specific defaults live in `config.json`:

```json
{
  "python_path": "../..",
  "data_root": "data",
  "training_model": "Qwen/Qwen3-8B",
  "inference_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "output_dir": "models"
}
```

Relative filesystem paths, including `python_path`, are resolved from the directory containing the config file. Model values beginning with `.` or `~` are treated as filesystem paths; values such as `Qwen/Qwen3-8B` remain Hugging Face model identifiers.

The checked-in config uses the bundled dataset and portable model identifiers. To use another environment, copy the file, edit the copy, and select it before running a command:

```powershell
$env:TEMPORAL_RELATION_CONFIG = "C:\experiments\temporal-relation.json"
python test_regr/TemporalRelation/launcher.py smoke
```

```bash
export TEMPORAL_RELATION_CONFIG=/cluster/config/temporal-relation.json
python test_regr/TemporalRelation/launcher.py smoke
```

The launcher prepends `python_path` to both Python's import search path and the `PYTHONPATH` environment variable before loading DomiKnowS. Python handles path resolution and environment separators on Windows, Linux, and macOS.

Available launcher commands are:

```text
smoke
predicate-train
program-train
inference
```

Arguments after the command are forwarded unchanged. CLI options such as `--path`, `--model-path`, and `--output` still override the configured defaults for an individual run.

## Bundled MATRES Data

The processed MATRES split files are included with this adapter so the examples run without relying on lab-local paths:

```text
test_regr/TemporalRelation/data/MATRES/timebank.txt
test_regr/TemporalRelation/data/MATRES/aquaint.txt
test_regr/TemporalRelation/data/MATRES/platinum.txt
test_regr/TemporalRelation/data/MATRES/README.md
```

`dataset.py` reads its default root from `config.json`. The checked-in value is `data`, which resolves to `test_regr/TemporalRelation/data`.

## Local TB-Dense Data

The public [`muk343/TimeBank-dense`](https://github.com/muk343/TimeBank-dense)
mirror is a TimeML tree, not a four-column
relation file. Clone it outside this repository and convert all official splits:

```powershell
git clone https://github.com/muk343/TimeBank-dense.git "$HOME/datasets/TimeBank-dense"
python -m test_regr.TemporalRelation.convert_tbdense `
  --source "$HOME/datasets/TimeBank-dense" `
  --output-dir test_regr/TemporalRelation/data/TBDense `
  --conflict-policy last
```

This writes `train.jsonl`, `dev.jsonl`, and `test.jsonl`. The converter keeps
only event-to-event `TLINK` records, maps the mirror's `NONE` relation to
`Vague`, and prints document, relation, byte, label, and source-anomaly counts.
Strict conflict rejection is the default. The explicit `last` policy is needed
for this mirror because it contains a small number of pairs whose later dense
annotation conflicts with an earlier TimeBank `TLINK`; the summary reports the
exact number selected this way.

Before mixed-corpus training, require a real transitivity grounding:

```powershell
python test_regr/TemporalRelation/launcher.py program-train `
  --path test_regr/TemporalRelation/data/TBDense/train.jsonl `
  --dataset tbdense `
  --limit 1 `
  --max-events-per-instance 8 `
  --pair-selection all `
  --grounding-only
```

The command fails unless the diagnostic reports
`temporal_before_transitive=<positive number>`.

For a mixed MATRES/TB-Dense run, identify each training source explicitly:

```powershell
python test_regr/TemporalRelation/launcher.py program-train `
  --train-paths test_regr/TemporalRelation/data/MATRES/timebank.txt,test_regr/TemporalRelation/data/TBDense/train.jsonl `
  --train-datasets matres,tbdense
```

The program uses the seven-label union head and applies a different legal-label
mask to every example. MATRES cannot train or predict TB-Dense-only containment
or simultaneity labels, while TB-Dense cannot train or predict MATRES `Equal`.

Recommended official-style setup:

```bash
python test_regr/TemporalRelation/launcher.py program-train \
  --train-paths test_regr/TemporalRelation/data/MATRES/timebank.txt,test_regr/TemporalRelation/data/MATRES/aquaint.txt \
  --path test_regr/TemporalRelation/data/MATRES/timebank.txt

python test_regr/TemporalRelation/launcher.py program-train \
  --eval-only \
  --path test_regr/TemporalRelation/data/MATRES/platinum.txt \
  --checkpoint test_regr/TemporalRelation/models/qwen3_8b_temporal_domiknows_program.pt
```

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
            event("p1", path=("p", pair_event1)),
            event("p2", path=("p", pair_event2)),
            query_event1("p1"),
            query_event2("p2")
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

In the learned version, a Qwen-style classifier head predicts these markers as binary logits over event nodes.

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

## Predicate Classifier

`modules.py` is the CLEVR-style learner-facing implementation. It does not generate text. It returns tensors whose dimensions match the DomiKnowS concepts exactly:

```python
event_logits:             [num_events, 2]
query_event1_logits:      [num_events, 2]
query_event2_logits:      [num_events, 2]
temporal_relation_logits: [num_event_pairs, 4]
```

The four temporal classes are:

```python
("Before", "After", "Equal", "Vague")
```

`OracleTemporalPredicateClassifier` is the perfect predicate module used by tests and oracle execution. `TemporalPredicateClassifier` is the Qwen-backed module: it encodes event and event-pair prompts, pools hidden states, and applies linear heads for the graph concepts. This is the correct path for training/fine-tuning because DomiKnowS receives concept logits directly.

## Predicate Training

Fast oracle smoke test:

```bash
python test_regr/TemporalRelation/launcher.py predicate-train \
  --limit 3 \
  --max-events 30 \
  --oracle \
  --device cpu
```

Train Qwen3-8B as the underlying predicate model on CUDA 6 with a frozen backbone and learned DomiKnowS concept heads:

```bash
CUDA_VISIBLE_DEVICES=6 python test_regr/TemporalRelation/launcher.py predicate-train \
  --limit 20 \
  --max-events 30 \
  --device cuda \
  --freeze-backbone \
  --epochs 3 \
  --lr 1e-3
```

For full MATRES training, remove `--limit` and either remove `--max-events` or raise it after the smoke run is healthy. `--freeze-backbone` trains only the predicate heads; use `--no-freeze-backbone` for full fine-tuning, which is much more expensive. The checkpoint saves the four concept heads by default and only saves the full 8B model if `--save-full-model` is passed.



## DomiKnowS `program.train` Workflow

The production TemporalQA path follows the CLEVR-style DomiKnowS execution route:

```text
MATRES file -> document/event/pair instances -> graph.compile_executable(...)
-> InferenceProgram(..., TemporalSolverModel) -> program.train(...)
```

The final executable query is stored in each compiled example as `logic_str`, and the answer label is stored as `logic_label`. This is important: the learner is not trained by parsing generated text. Qwen encodes event-pair prompts and returns logits aligned with DomiKnowS concepts, then DomiKnowS applies `queryL(...)`, `iotaL(...)`, and optional global temporal constraints.

Current graph-level execution query:

```python
queryL(
    temporal_relation,
    iotaL(
        andL(
            EventPair("p"),
            event("p1", path=("p", pair_event1)),
            event("p2", path=("p", pair_event2)),
            query_event1("p1"),
            query_event2("p2"),
        )
    ),
)
```

### 8B Warmup Training

This trains the `temporal_relation` ModuleLearner on the target pair only. It is the stable supervised warmup stage.

```bash
CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n CLEVER \
python test_regr/TemporalRelation/launcher.py program-train \
  --device cuda \
  --freeze-backbone \
  --lora-r 4 \
  --lora-alpha 8 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,v_proj \
  --max-length 96 \
  --encode-batch-size 1 \
  --max-events-per-instance 8 \
  --pair-selection target \
  --max-pairs-per-instance 2 \
  --warmup-epochs 2 \
  --constraint-epochs 0 \
  --lr 3e-5 \
  --skip-condition-eval
```

### Execution / Global-Constraint Continuation

This starts from a warmup checkpoint and trains through executable DomiKnowS logic over query-related event pairs. Keeping `pair-selection=related` lets the global rules see inverse/related pairs while avoiding all-pairs OOM.

```bash
CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n CLEVER \
python test_regr/TemporalRelation/launcher.py program-train \
  --device cuda \
  --freeze-backbone \
  --lora-r 4 \
  --lora-alpha 8 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,v_proj \
  --max-length 96 \
  --encode-batch-size 1 \
  --max-events-per-instance 16 \
  --pair-selection related \
  --max-pairs-per-instance 8 \
  --warmup-epochs 0 \
  --constraint-epochs 2 \
  --lr 5e-7 \
  --beta 0.3 \
  --checkpoint test_regr/TemporalRelation/models/qwen3_8b_temporal_domiknows_program.pt \
  --skip-condition-eval
```

Use `--no-global-consistency` for the execution-only ablation.

### Evaluation on Platinum

`--eval-only` evaluates the full file passed with `--path`; it does not split off a random dev set.

```bash
CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n CLEVER \
python test_regr/TemporalRelation/launcher.py program-train \
  --path test_regr/TemporalRelation/data/MATRES/platinum.txt \
  --device cuda \
  --freeze-backbone \
  --lora-r 4 \
  --lora-alpha 8 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,v_proj \
  --max-length 96 \
  --encode-batch-size 1 \
  --max-events-per-instance 16 \
  --pair-selection related \
  --max-pairs-per-instance 8 \
  --eval-only \
  --checkpoint test_regr/TemporalRelation/models/qwen3_8b_temporal_domiknows_program.pt \
  --skip-condition-eval
```

### ILP Inference

The Qwen runner exposes DomiKnowS inference outputs through `--infer-types`. To request ILP output in addition to local predictions, add:

```bash
--infer-types local/softmax,local/argmax,ILP
```

A fast CPU oracle smoke test for the ILP path is:

```bash
conda run --no-capture-output -n CLEVER python -c \
"from test_regr.TemporalRelation.program import make_packed_left_example, temporal_program_declaration; \
ds, ctx, program = temporal_program_declaration([make_packed_left_example()], device='cpu', infer_types=['local/argmax','ILP']); \
print(program.evaluate_condition(ds, device='cpu'))"
```

Expected result: `100.0` on the toy `packed before left` example. ILP requires the DomiKnowS ILP solver stack to be available; if Gurobi/licensing is unavailable, keep `local/softmax,local/argmax` for training/evaluation and use the oracle smoke above to verify solver availability.

### 1B / 1.5B Model Run

For a smaller underlying LLM, use the same DomiKnowS program path and swap `--model-path`. A true 1B-ish Qwen command is:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n CLEVER \
python test_regr/TemporalRelation/launcher.py program-train \
  --limit 200 \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda \
  --freeze-backbone \
  --lora-r 4 \
  --lora-alpha 8 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,v_proj \
  --max-length 96 \
  --encode-batch-size 1 \
  --max-events-per-instance 8 \
  --pair-selection target \
  --max-pairs-per-instance 2 \
  --warmup-epochs 1 \
  --constraint-epochs 0 \
  --lr 3e-5 \
  --skip-condition-eval
```

If the 1.5B model is not cached, omit `HF_HUB_OFFLINE=1` for the first download. On this machine, the currently verified cached small Qwen fallback is `Qwen/Qwen2.5-0.5B-Instruct`; use it for a quick smoke test by replacing the `--model-path` value.

## Text-Generation Baseline

`llm_inference.py` is kept as a zero-shot/debug baseline only. It asks a causal LM to return one candidate answer as text and then parses that text. Because it uses generation, it has `--max-new-tokens`; that flag should not appear in the final trained predicate-classifier path.

Run the text-generation baseline on one grouped MATRES document:

```bash
python test_regr/TemporalRelation/launcher.py inference \
  --limit 1 \
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
python -m unittest test_regr.TemporalRelation.test_temporal_relation_adapter
```

Run the MATRES smoke test with per-sample consistency checks:

```bash
python test_regr/TemporalRelation/launcher.py smoke \
  --materialize-candidates \
  --check-consistency
```

Expected current MATRES result: `convert_failures=0`, `oracle_failures=0`, and `consistency_failure_samples=0` for `aquaint.txt`, `platinum.txt`, and `timebank.txt`.

Combined adapter regression:

```bash
python -m unittest \
  test_regr.GraphQA.test_graphqa_adapter \
  test_regr.TemporalRelation.test_temporal_relation_adapter
```

## Implementation Map

- `config.json`: portable dataset, model, and output defaults.
- `config.py`: config loading, validation, environment override, and relative-path resolution.
- `launcher.py`: cross-platform `python_path`/`PYTHONPATH` bootstrap and command dispatch.
- `graph.py`: DomiKnowS concepts and relations.
- `dataset.py`: MATRES/TB-Dense style file discovery and normalization.
- `convert_tbdense.py`: strict TimeML/delimited conversion with split preservation.
- `execution.py`: final `queryL` logic, candidate event-pair generation, query-event marker labels, and local learner examples.
- `modules.py`: CLEVR-style predicate classifiers returning DomiKnowS-aligned logits.
- `train_predicate_classifier.py`: training/evaluation runner for Qwen-backed concept heads and oracle smoke checks.
- `llm_inference.py`: zero-shot multiple-choice text-generation baseline for `query_event1`, `query_event2`, and `temporal_relation`.
- `run_llm_inference.py`: CLI for running the text-generation baseline over real MATRES files.
- `oracle.py`: expected answer lookup and temporal consistency checks.
- `test_temporal_relation_adapter.py`: unit tests for graph, query, learner examples, oracle, and dataset grouping.
- `smoke_test_dataset.py`: whole-dataset conversion/oracle smoke.
