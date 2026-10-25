#!/usr/bin/env bash
# Evaluate E5 (epoch 19, final pick, recorded accel input) on GPUs 0-3:
#   GPU 0: navtest   GPU 1: ped_v2 held-out val   GPU 2: Gemini RGB + raster   GPU 3: pedestrian sweep
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
        --tag $3_$4
    echo "PEDVAL_DONE $2"
}

E5=E5_pedV2yield_oldLoss
CK=$X/$E5/epoch19.ckpt
navtest 0 $E5 epoch19 final > $X/$E5/eval_navtest.log 2>&1 &
pedval  1 $E5 epoch19 final > $X/$E5/eval_pedval.log  2>&1 &
# Gemini RGB + raster, pedestrian walking left, recorded accel input (same setting as E4/E6 in the headline table)
CUDA_VISIBLE_DEVICES=2 python visualize_gemini_rgb_vs_raster.py --model "E5=$CK::final" --closer 1.5 --walk left \
    --output-dir $X/$E5/gemini_walk_left > $X/$E5/eval_gemini.log 2>&1 &
# pedestrian sweep, 20 Gemini raster scenes, accel input 0 (same as the other models' sweep)
CUDA_VISIBLE_DEVICES=3 python visualize_ped_grid_sweep.py --model "E5=$CK::final" --scenes 20 \
    --output-dir $X/ped_grid_sweep/run20_E5 > $X/$E5/eval_sweep.log 2>&1 &
wait
echo ALL_EVAL_DONE
