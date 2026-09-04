# GraphQA / KB-VQA Adapter

This adapter converts Scallop-style VQAR / KB-VQA tasks into bounded DomiKnowS logic, following the CLEVR implementation style.

## Files

- `graph.py`: graph concepts, object-symbol concepts, object-pair relations, and dynamically discovered relation predicates.
- `dataset.py`: VQAR task loading and conversion into GraphQA instances.
- `execution.py`: bounded query construction with `queryL`, `iotaL`, `andL`, and `orL`.
- `oracle.py`: perfect symbolic oracle for converted instances.
- `program_qwen_train.py`: DomiKnowS `program.train` runner with Qwen predicate learners.
- `train_direct_qwen_answer.py`: direct-answer Qwen baseline.
- `test_graphqa_adapter.py`: unit tests for bounded propagation, query conversion, and oracle behavior.

## Bundled Data

The full VQAR task pickles and visual features are too large to commit. The repository includes only the small knowledge-base support files:

```text
test_regr/GraphQA/data/gqa_info.json
test_regr/GraphQA/data/knowledge_base/is_a.facts
test_regr/GraphQA/data/knowledge_base/in_oa_rel.facts
```

The full dataset is expected externally, for example:

```text
/egr/research-hlr2/premsrit/VQAR_data
```

You can override the default root with:

```bash
export GRAPHQA_VQAR_ROOT=/path/to/VQAR_data
```

## Logic Mapping

The adapter uses bounded, non-recursive propagation:

```text
Name(o, x) and TypeOf(x, y) -> ObjectType(o, y)
ObjectType(o, x) and TypeOf(x, y) -> ObjectCategory(o, y)
```

Queries are represented with DomiKnowS executable logic. For example, a white animal left of `o2` becomes conceptually:

```python
queryL(
    answer_object,
    iotaL(
        andL(
            ObjectCategory(o, animal),
            Attribute(o, white),
            LeftOf(o, o2),
        )
    )
)
```

The implementation supports VQAR clause functions currently mapped in `dataset.py`:

```text
Initial, Find_Name, Find_Attr, Hypernym_Find, Relate, Relate_Reverse, And, Or, KG_Find
```

## Tests

Run the unit tests:

```bash
conda run --no-capture-output -n CLEVER python -m unittest test_regr.GraphQA.test_graphqa_adapter
```

Run a real-dataset smoke test when the full VQAR data is available:

```bash
conda run --no-capture-output -n CLEVER python -m test_regr.GraphQA.smoke_test_real_dataset \
  --root /egr/research-hlr2/premsrit/VQAR_data \
  --limit 25
```

## Training

DomiKnowS predicate training example:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/DomiKnowS \
conda run --no-capture-output -n CLEVER python -m test_regr.GraphQA.program_qwen_train \
  --root /egr/research-hlr2/premsrit/VQAR_data \
  --task-path /egr/research-hlr2/premsrit/VQAR_data/data/dataset/task_list/train_tasks_c2_10000.pkl \
  --kb-dir /egr/research-hlr2/premsrit/VQAR_data/data/knowledge_base \
  --limit 2500 \
  --kb-depth 2 \
  --max-extra-kg-facts 256 \
  --model-path /path/to/Qwen3-8B \
  --device cuda \
  --freeze-backbone \
  --lora-r 4 \
  --lora-alpha 8 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,v_proj \
  --max-length 96 \
  --encode-batch-size 1 \
  --warmup-epochs 2 \
  --constraint-epochs 0 \
  --lr 3e-5 \
  --skip-condition-eval \
  --output /path/to/graphqa_domiknows.pt
```
