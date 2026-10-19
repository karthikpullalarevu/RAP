#!/usr/bin/env bash
# Evaluate E6 (epoch 15) with the two sub-score pick modes, in parallel on GPUs 0-3:
#   GPU 0: navtest, pick_mode=gated_pdms        GPU 1: navtest, pick_mode=log_barrier
#   GPU 2: ped_v2 held-out val, gated_pdms      GPU 3: ped_v2 held-out val, log_barrier
# E4 (epoch 15, pick_mode=final) results already exist in E4_pedV2-105k_photo50k/{navtest,ped_val_batch3_epoch15}.
# Gemini qualitative eval is run separately afterwards.
set -uo pipefail

EXP=E6_pedV2yield_sub5_infonce
EXP_DIR=/datahdd2/karthik/RAP/experiments/$EXP
CKPT=$EXP_DIR/epoch15.ckpt      # symlink to checkpoints/epoch=15-step=27824.ckpt ('=' in the name breaks Hydra overrides)
[ -e "$CKPT" ] || ln -s "$EXP_DIR/checkpoints/epoch=15-step=27824.ckpt" "$CKPT"

export PYTHONPATH=/datahdd/karthik/RAP:${PYTHONPATH:-}
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/datahdd/karthik/dataset/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/datahdd/karthik/RAP/exp"
export NAVSIM_DEVKIT_ROOT="/datahdd/karthik/RAP/navsim"
export OPENSCENE_DATA_ROOT="/datahdd2/karthik/dataset/nuplan-v1.1"
export HYDRA_FULL_ERROR=1
export ip_head="128.2.220.69:6379"
export redis_password="dummy"
export num_nodes="1"
# The system disk can fill up: keep temp files on /datahdd2
export TMPDIR=/datahdd2/karthik/tmp
mkdir -p "$TMPDIR"

navtest() {   # $1 = gpu, $2 = pick mode
    local name=eval_${EXP}_ep15_$2
    CUDA_VISIBLE_DEVICES=$1 python navsim/planning/script/run_pdm_score.py \
        agent=rap_agent \
        agent.config.trajectory_sampling.time_horizon=5 \
        agent.config.use_resnet=True \
        agent.config.distill_feature=False \
        agent.config.pdm_scorer=False \
        +agent.config.pick_mode=$2 \
        agent.config.train_metric_cache_path=./cache/train_metric_cache \
        agent.checkpoint_path="$CKPT" \
        train_test_split=navtest \
        metric_cache_path=/datahdd/karthik/RAP/exp/metric_cache \
        sensor_blobs_path=/datahdd2/karthik/dataset/nuplan-v1.1/sensor_blobs/test \
        experiment_name="$name"
    local csv; csv=$(find "exp/$name" -name "*.csv" -type f 2>/dev/null | sort | tail -1)
    if [ -n "$csv" ]; then
        mkdir -p "$EXP_DIR/navtest_ep15_$2"
        cp "$csv" "$EXP_DIR/navtest_ep15_$2/results.csv"
        python analyze_eval.py "$EXP_DIR/navtest_ep15_$2/results.csv" > "$EXP_DIR/navtest_ep15_$2/analysis.txt"
    fi
    echo "NAVTEST_DONE $2"
}

pedval() {    # $1 = gpu, $2 = pick mode
    CUDA_VISIBLE_DEVICES=$1 python eval_ped_val.py --exp-name $EXP --ckpt "$CKPT" --pick-mode $2 --tag ep15_$2
    echo "PEDVAL_DONE $2"
}

navtest 0 gated_pdms  > "$EXP_DIR/eval_navtest_gated_pdms.log"  2>&1 &
navtest 1 log_barrier > "$EXP_DIR/eval_navtest_log_barrier.log" 2>&1 &
pedval  2 gated_pdms  > "$EXP_DIR/eval_pedval_gated_pdms.log"   2>&1 &
pedval  3 log_barrier > "$EXP_DIR/eval_pedval_log_barrier.log"  2>&1 &
wait
echo ALL_EVAL_DONE
