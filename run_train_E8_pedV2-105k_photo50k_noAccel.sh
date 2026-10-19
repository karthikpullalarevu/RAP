#!/usr/bin/env bash
# E8: E4 (run_train_E4_pedV2-105k_photo50k.sh) with the ego acceleration input (ax, ay) set to 0 in training and
# testing (zero_ego_accel=True). Same data as E4: the ped_v2 set before the cleanup and the yield edit, rebuilt as
# hard links in ped_v2/cache/train_original_E4 (process_data/build_e4_original_cache.py). Same loss and pick as E4.
# Runs on GPUs 4-7 next to E7 (E6 + zero accel) on GPUs 0-3.
#
# Smoke test (1 GPU, a few batches, separate folder, wandb offline):   SMOKE=1 bash run_train_E8_pedV2-105k_photo50k_noAccel.sh
#
# Everything for this run lives in:
#   /datahdd2/karthik/RAP/experiments/E8_pedV2-105k_photo50k_noAccel/
set -euo pipefail

EXP_NAME=E8_pedV2-105k_photo50k_noAccel
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
export CUDA_VISIBLE_DEVICES=4,5,6,7
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
  +agent.config.zero_ego_accel=True \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  cache_path=/datahdd2/karthik/RAP/cache/ego_50k \
  cache_path_perturbed=/datahdd/karthik/RAP/cache/ablation/perturbed \
  cache_path_others=/datahdd/karthik/RAP/cache/ablation/aug \
  cache_path_ped=/datahdd2/karthik/ped_v2/cache/train_original_E4 \
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
