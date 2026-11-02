#!/usr/bin/env bash
# Run inside the configured training environment; select a free GPU by UUID.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Select a free GPU before starting this run}"
: "${STAGE1_CHECKPOINT:?Supply a checkpoint with verified training-data lineage}"
: "${PLANNING_DIR:?Supply independently sourced training scenes}"
: "${CONTROL_SOURCE:?Supply the controller training dataset}"
: "${OUTPUT_DIR:?Supply a new run output directory}"

exec "${PYTHON:-python}" -u -m test_regr.JointEmbodiedAgentInterface.main train-agent \
  --two-stage --resume "$STAGE1_CHECKPOINT" \
  --stage1-epochs 0 --stage1-eai-warmup-steps 0 --controller-warmup-steps 0 \
  --stage2-epochs "${STAGE2_EPOCHS:-4}" --stage2-rounds-per-epoch "${STAGE2_ROUNDS:-10}" \
  --stage2-dfa off --stage2-parameter-policy freeze_inactive \
  --stage2-eai-anchor-weight 0 --stage2-planner-anchor-weight 0 \
  --stage2-controller-bc-weight 0 --dagger-weight 0 --feasibility-weight 0 \
  --execution-assistance off --stage2-eval-rollouts-per-task 10 \
  --stage2-eval-planning --stage2-eval-planning-split validation \
  --planner-model "${PLANNER_MODEL:-Qwen/Qwen3-VL-8B-Instruct}" \
  --vision-model google/siglip-base-patch16-384 --image-size 384 \
  --vlabench-planning-dir "$PLANNING_DIR" --control-source "$CONTROL_SOURCE" \
  --task all --device cuda:0 --load-in-4bit --seed 42 \
  --batch-size 8 --workers 0 --output "$OUTPUT_DIR"
