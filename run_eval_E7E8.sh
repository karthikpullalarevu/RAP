#!/usr/bin/env bash
# Evaluate E7 (epoch 16, gated_pdms) and E8 (epoch 18, final), both with zero_ego_accel=True, in parallel on GPUs 0-3:
#   GPU 0: E7 navtest      GPU 1: E8 navtest      GPU 2: E7 ped_v2 held-out val      GPU 3: E8 ped_v2 held-out val
# Compare with E6 (navtest_ep15_gated_pdms, ped_val_batch3_ep15_gated_pdms) and E4 (navtest, ped_val_batch3_epoch15).
set -uo pipefail

X=/datahdd2/karthik/RAP/experiments
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
export TMPDIR=/datahdd2/karthik/tmp
mkdir -p "$TMPDIR"

navtest() {   # $1 = gpu, $2 = exp, $3 = epoch tag, $4 = pick mode
    local EXP_DIR=$X/$2 name=eval_$2_$3_$4
    CUDA_VISIBLE_DEVICES=$1 python navsim/planning/script/run_pdm_score.py \
        agent=rap_agent \
        agent.config.trajectory_sampling.time_horizon=5 \
        agent.config.use_resnet=True \
        agent.config.distill_feature=False \
        agent.config.pdm_scorer=False \
        +agent.config.pick_mode=$4 \
        +agent.config.zero_ego_accel=True \
        agent.config.train_metric_cache_path=./cache/train_metric_cache \
        agent.checkpoint_path="$EXP_DIR/$3.ckpt" \
        train_test_split=navtest \
        metric_cache_path=/datahdd/karthik/RAP/exp/metric_cache \
        sensor_blobs_path=/datahdd2/karthik/dataset/nuplan-v1.1/sensor_blobs/test \
        experiment_name="$name"
    local csv; csv=$(find "exp/$name" -name "*.csv" -type f 2>/dev/null | sort | tail -1)
    if [ -n "$csv" ]; then
        mkdir -p "$EXP_DIR/navtest_$3_$4"
        cp "$csv" "$EXP_DIR/navtest_$3_$4/results.csv"
        python analyze_eval.py "$EXP_DIR/navtest_$3_$4/results.csv" > "$EXP_DIR/navtest_$3_$4/analysis.txt"
    fi
    echo "NAVTEST_DONE $2"
}

pedval() {    # $1 = gpu, $2 = exp, $3 = epoch tag, $4 = pick mode
    CUDA_VISIBLE_DEVICES=$1 python eval_ped_val.py --exp-name $2 --ckpt "$X/$2/$3.ckpt" --pick-mode $4 \
        --zero-ego-accel --tag $3_$4
    echo "PEDVAL_DONE $2"
}

E7=E7_pedV2yield_sub5_infonce_noAccel
E8=E8_pedV2-105k_photo50k_noAccel
navtest 0 $E7 epoch16 gated_pdms > $X/$E7/eval_navtest.log 2>&1 &
navtest 1 $E8 epoch18 final      > $X/$E8/eval_navtest.log 2>&1 &
pedval  2 $E7 epoch16 gated_pdms > $X/$E7/eval_pedval.log  2>&1 &
pedval  3 $E8 epoch18 final      > $X/$E8/eval_pedval.log  2>&1 &
wait
echo ALL_EVAL_DONE
