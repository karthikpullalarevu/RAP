#!/usr/bin/env bash
# E7: same as E6 (below) but the ego acceleration input (ax, ay) is set to 0 in training and testing
#   (zero_ego_accel=True), to remove the "already braking -> stop" shortcut found in E6.
# E6: ResNet RAP, same data setup as E4 but with
#   - the cleaned + yield-edited ped_v2 train cache (lead-car / ped-behind / too-late / side-pass samples moved out,
#     51,259 labels rewritten to slow or stop; see Notion "PedV2 Training Data Cleanup + Yield Labels")
#   - the new score loss (score_mode=sub5): weighted BCE on the 5 sub-scores [NC, DAC, EP, TTC, C] + multi-positive
#     InfoNCE on NC across the 64 proposals; on ped_v2 samples NC also fails when the proposal comes within 1.5 m of
#     the inserted pedestrian while moving; ped_v2 samples that start within 2 m skip the score loss
#   - the proposal is picked from the 5 predicted sub-scores (pick_mode=gated_pdms; log_barrier can be tried at eval)
# Trained from scratch.
#
# Smoke test (1 GPU, a few batches, separate folder, wandb offline):   SMOKE=1 bash run_train_E6_pedV2yield_sub5_infonce.sh
#
# Everything for this run lives in:
#   /datahdd2/karthik/RAP/experiments/E6_pedV2yield_sub5_infonce/
set -euo pipefail

EXP_NAME=E7_pedV2yield_sub5_infonce_noAccel
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
# The system disk (/tmp) is full: keep temp files on /datahdd2 (a full /tmp crashed epoch 2 at validation)
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
  +agent.config.score_mode=sub5 \
  +agent.config.sub_score_weight=1 \
  +agent.config.final_score_weight=0 \
  +agent.config.pick_mode=gated_pdms \
  +agent.config.zero_ego_accel=True \
  +agent.config.no_score_tokens_path=/datahdd2/karthik/ped_v2/analysis/10_e6/no_score_loss_tokens.txt \
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
