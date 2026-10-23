#!/usr/bin/env bash
# E5: E6 data (cleaned + yield-edited ped_v2, ped_v2/cache/train, 61,088 samples) with the E4 (old) loss and pick:
# final-score BCE only, no sub5 / InfoNCE / 1.5 m pedestrian rule / no-score tokens, pick by final score.
# Ego acceleration input as recorded (like E4 and E6). Separates the data change from the loss change:
# E4 vs E5 = data only, E5 vs E6 = loss only. GPUs 0-3.
#
# Smoke test (1 GPU, a few batches, separate folder, wandb offline):   SMOKE=1 bash run_train_E5_pedV2yield_oldLoss.sh
#
# Everything for this run lives in:
#   /datahdd2/karthik/RAP/experiments/E5_pedV2yield_oldLoss/
set -euo pipefail

EXP_NAME=E5_pedV2yield_oldLoss
DEVICES='[0,1,2,3]'
EPOCHS=20
EXTRA=()
if [ "${SMOKE:-0}" = "1" ]; then
  EXP_NAME=${EXP_NAME}_smoketest
  DEVICES='[0]'
  export WANDB_MODE=offline
  EPOCHS=1
  EXTRA=(trainer.params.limit_train_batches=30 trainer.params.limit_val_batches=3 +trainer.params.log_every_n_steps=5)
fi
EXP_DIR=/datahdd2/karthik/RAP/experiments/$EXP_NAME

export PYTHONPATH=/datahdd/karthik/RAP:${PYTHONPATH:-}
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/datahdd/karthik/dataset/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/datahdd2/karthik/RAP/exp"
export NAVSIM_DEVKIT_ROOT="/datahdd/karthik/RAP/navsim"
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The system disk (/tmp) can fill up: keep temp files on /datahdd2
export TMPDIR=/datahdd2/karthik/tmp
mkdir -p "$TMPDIR"
export ip_head="128.2.220.69:6379"
export redis_password="dummy"
export num_nodes="1"

# Lightning saves checkpoints to ./rap/<wandb id>/checkpoints (on /datahdd, which is full): link it to /datahdd2.
export RAP_WANDB_ID=$EXP_NAME
mkdir -p "$EXP_DIR/checkpoints"
[ -e "rap/$EXP_NAME" ] || ln -s "$EXP_DIR" "rap/$EXP_NAME"

# Keep a copy of this script with the experiment, so the run can be reproduced later
cp "$0" "$EXP_DIR/$(basename "$0")"

python navsim/planning/script/run_training.py \
  agent=rap_agent \
  dataset=navsim_dataset \
  agent.config.trajectory_sampling.time_horizon=5 \
  agent.config.cache_data=False \
  agent.config.distill_feature=True \
  agent.config.pdm_scorer=True \
  agent.config.use_resnet=True \
  agent.config.train_metric_cache_path=./cache/train_metric_cache \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  cache_path=/datahdd2/karthik/RAP/cache/ego_50k \
  cache_path_perturbed=/datahdd/karthik/RAP/cache/ablation/perturbed \
  cache_path_others=/datahdd/karthik/RAP/cache/ablation/aug \
  cache_path_ped=/datahdd2/karthik/ped_v2/cache/train \
  train_test_split=navtrain \
  experiment_name=$EXP_NAME \
  trainer.params.max_epochs=$EPOCHS \
  trainer.params.num_sanity_val_steps=0 \
  "+trainer.params.devices=$DEVICES" \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  dataloader.params.batch_size=16 \
  dataloader.params.num_workers=4 \
  dataloader.params.drop_last=True \
  seed=42 "${EXTRA[@]}" 2>&1 | tee -a "$EXP_DIR/train.log"
