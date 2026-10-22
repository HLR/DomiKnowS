#!/usr/bin/env bash
set -euo pipefail

cd /home/auszok/DomiKnowS

docker exec \
  -e CUDA_VISIBLE_DEVICES=4 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  vigorous_easley bash -lc '
cd /workspace
mkdir -p test_regr/JointEmbodiedAgentInterface/results test_regr/JointEmbodiedAgentInterface/checkpoints/full_384_gpu4
exec /opt/dominows/venv/bin/python -u -m test_regr.JointEmbodiedAgentInterface.main train-agent \
  --two-stage \
  --resume test_regr/JointEmbodiedAgentInterface/checkpoints/full_384_gpu4/joint_stage2_epoch_000.pt \
  --planner-model /home/auszok/models/Qwen/Qwen3-VL-8B-Instruct \
  --vision-model google/siglip-base-patch16-384 \
  --image-size 384 \
  --eai-dataset all \
  --eai-split train \
  --vlabench-planning-dir test_regr/VLABenchAgentInterface/data/planning \
  --control-source test_regr/VLABenchAgentInterface/data/control \
  --task all \
  --device cuda:0 \
  --load-in-4bit \
  --batch-size 8 \
  --workers 0 \
  --stage1-epochs 5 \
  --stage2-epochs 5 \
  --stage1-rounds-per-epoch 10 \
  --stage2-rounds-per-epoch 10 \
  --eai-samples 8 \
  --vlabench-planner-samples 4 \
  --vlabench-rollouts 8 \
  --stage2-eval-rollouts-per-task 3 \
  --stage2-max-baseline-success-regression 1.0 \
  --stage2-max-eai-success-regression 0.02 \
  --stage2-max-eai-recall-regression 0.02 \
  --stage2-planner-anchor-weight 0.25 \
  --stage2-controller-bc-weight 0.25 \
  --output test_regr/JointEmbodiedAgentInterface/checkpoints/full_384_gpu4
' >> /home/auszok/DomiKnowS/test_regr/JointEmbodiedAgentInterface/results/full_384_gpu4.log 2>&1

echo "FULL_384_EXIT_CODE=$?"
