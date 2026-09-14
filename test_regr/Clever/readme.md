# Clevr Example for InferenceProgram

Read about the dataset here: https://cs.stanford.edu/people/jcjohns/clevr/

1. **Download the image the data**

Use this link https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip to download the images and copy the images of the training segment into the 'train/images' folder.
Then cd into the Clever folder.

2. **Install the examples Requirements**
   ```
   pip install -r requirements.txt
   ```
3. **Install Jacinle and PreciseRolPooling**

   ```
   git clone https://github.com/vacancy/Jacinle --recursive
   cd Jacinle
   pip install -e .
   ```
   Note: please also check https://github.com/vacancy/PreciseRoIPooling for installing PreciseRoIPooling if need
4. **Run the program**

The initial run may take longer as necessary files are extracted.

Quick test (sanity check):
   ```bash
   python main.py --train-size 10 --test-size 10 --epochs 4 --batch-size 2 --dummy
   ```
Train the Model:
   ```bash
   python main.py --train-size 5000 --test-size 1000 --epochs 2 --batch-size 20 --lr 1e-6 --tnorm G
   ```
or 
   ```bash
   python main.py --train-size 5000 --test-size 1000 --epochs 2 --batch-size 20 --lr 1e-6 --tnorm P
   ```

To perform evaluation using a trained checkpoint, add the '--eval-only' flag to the commands above.

5 **Change Dataset Filter**

In preprocess_dataset function in preprocessor.py, there is comment for each filter type of function. Considering uncomment/comment filter that you want to test data with. Currently, we test with 1 relation.

---

## Currently working execution

There are two simple type of query that current working with CLEVR

1. Object property

The current CLEVR with question asking the existence of properties of single objects are already working in DomiKnowS
We provide the example of execution for this type of question below.

**Example** "Does there blue big square in the image?"

```python
existL(is_blue('x'), is_big(path=('x')), is_square(path=('x')))
```

2. One Relation

We currently support question with using one relation between two objects. Each object can have multiple properties.
We provide the example of execution for this type of question below.

**Example** "Does there blue big square in front of red small thing in the image?"

```python
existL(is_blue('x'), is_big(path=('x')), is_square(path=('x')),
       is_left('rel1', path=("x", obj1.reversed)),
       is_red('y', path=('rel1', obj2), is_small(path=('y')))
       )
```

## Work in Progress

- Counting
- Automatic Conversion from natural langauge question into DomiKnowS execution
- Multiple Relations
---

## 3D-FORCE (Puzzle and REF)

`force3d_dataset.py` adapts the 3D-FORCE dataset (`/localscratch/kamalida/projects/SaPy/datasets/3D-FORCE`)
to this pipeline: it loads one view per question (camera 0), translates the lambda-style programs into
DomiKnowS logic (`existsL` for Puzzle, `miotaL` with a one-hot object label for REF), swaps the CLEVR
vocabulary for 7 colors / 12 shapes / 24 spatial relations (`left`, `obj_left`, `left_0`..`left_3`, ...),
and derives oracle relation labels from the 3D geometry (`scene.json` + `camera.json`). Train/test are
split by whole scenes. Every pair of logical variables gets a fixed `distinct('x', 'y')` relation (identity, never learned): the dataset requires each variable to bind a different object, and that also links relation-free variables into one connected formula.

Tests (CPU): `python -m pytest test_force3d_adapter.py -q`

Oracle sanity check (expect ~100%):
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset force3d --oracle-mode --train-size 200 --epochs 1 \
  --curriculum none --disable-plugins --skip-train-eval --tensorboard false --step-notebook false
```
Learned run, Puzzle:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset force3d --force3d-test-scenes 20 --epochs 10 \
  --batch-size 10 --lr 1e-3 --tnorm G --curriculum none --disable-plugins --skip-train-eval \
  --tensorboard false --step-notebook false
```
REF: add `--force3d-split ref --force3d-test-scenes 100`. The log then also prints `REF top-1 accuracy`
(argmax of the miotaL selection vs. the answer index) next to the framework's exact-match score.

Notes: use `--curriculum none` (the CLEVR curriculum buckets are empty for 8-12 object scenes);
`--train-size/--test-size` cap the scene-split parts; the first load caches to `dataset_cache/force3d_*.pkl`.

### Generated training data and training options

`gen_force3d_free.py` writes free yes/no puzzles (no distractor or uniqueness constraints) over the
4,459 rendered scenes the released splits do not use. Every question is a contrastive pair: a positive
anchored on real objects and a negative made by one structure-preserving edit (flip one relation on the
same axis and perspective, or swap one descriptor value). Question structure therefore predicts the
answer at chance; the generator prints this structure-only baseline as a self-check. Answers come from
the scene geometry (100% agreement with the released labels). Load with `--force3d-json <file>`:
```bash
python gen_force3d_free.py --out generated/free_pairs_all.json --num-scenes 5000 --per-scene 10 --seed 1
DOMIKNOWS_JOINT_SOFT_PRUNE_ROWS=20000 python main.py --dataset force3d \
  --force3d-json $PWD/generated/free_pairs_all.json --train-size 3000 --force3d-test-scenes 60 --test-size 300 \
  --epochs 5 --batch-size 2 --lr 3e-2 --tnorm P --init-prior --infer-type local --curriculum none \
  --disable-plugins --skip-train-eval --tensorboard false --step-notebook false
```

Things that matter when reading results:
- **Structure-only baseline.** At load time the log prints the held-out accuracy of answering from question
  structure alone (variables, descriptors, relations; no image). Compare the model against this, not 50%:
  the released Puzzle split scores 65% this way, and the old unpaired generator 74%.
- **Camera relations are direction bins.** A question's `left_k` means "left, seen from its k-th view", whose
  angle to camera 0 depends on the view setup. The model learns one head per 30-degree direction in camera
  0's frame (`dir0` right, `dir90` front, `dir180` left, `dir270` behind) and the loader maps each question's
  camera relations onto them with its cameras. This reproduces all 1,150 released Puzzle answers, including
  random-view puzzles.
- **Boxes.** `objects_raw` holds raw pixel boxes (clipped to the image); `main.py` rescales them into the
  ResNet backbone's 224x224 input frame (`modules.boxes_in_backbone_frame`). Before this fix the ROI boxes
  missed most objects (median pixel correlation 0.08), for CLEVR as well.
- `--init-prior` (heads start at class priors), the product t-norm, small batches and lr ~3e-2. The log also
  prints a "soft accuracy" (P(satisfied) > 0.5); the exact evaluator thresholds each predicate at 0.5.
  `DOMIKNOWS_JOINT_SOFT_PRUNE_ROWS` / `_TOPK` control loss-path pruning of joint tables.
