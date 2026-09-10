#!/usr/bin/env bash
# Usage: bash cache_dataset.sh [--split navtrain] [--task ego|aug|perturb|metric|all]

SPLIT="navtrain"
TASK="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split) SPLIT="$2"; shift 2 ;;
        --task)  TASK="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

export PYTHONPATH=/datahdd/karthik/RAP:${PYTHONPATH}
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/datahdd/karthik/dataset/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/datahdd/karthik/RAP/exp"
export NAVSIM_DEVKIT_ROOT="/datahdd/karthik/RAP/navsim"
export HYDRA_FULL_ERROR=1

SENSOR_BLOBS="/datahdd/karthik/dataset/nuplan-v1.1/sensor_blobs"
SENSOR_SPLIT="trainval"  # on-disk folder name (was renamed from navtrain)
CACHE_ROOT="./cache"

mkdir -p "$CACHE_ROOT" exp

run_ego() {
    echo "==> [ego] split=$SPLIT"
    export OPENSCENE_DATA_ROOT="/datahdd/karthik/dataset/nuplan-v1.1"
    python navsim/planning/script/run_dataset_caching.py \
        agent=rap_agent \
        dataset=navsim_dataset \
        agent.config.trajectory_sampling.time_horizon=5 \
        agent.config.cache_data=True \
        train_test_split=$SPLIT \
        train_test_split.scene_filter.has_route=false \
        experiment_name=trainval_test \
        worker=single_machine_thread_pool \
        worker.max_workers=64 \
        sensor_blobs_path=$SENSOR_BLOBS/$SENSOR_SPLIT \
        cache_path=$CACHE_ROOT/rap_ego
}

run_aug() {
    echo "==> [aug] split=$SPLIT"
    export OPENSCENE_DATA_ROOT="/datahdd/karthik/dataset_aug"
    python navsim/planning/script/run_dataset_caching.py \
        agent=rap_agent \
        dataset=navsim_dataset \
        agent.config.trajectory_sampling.time_horizon=5 \
        agent.config.cache_data=True \
        train_test_split=$SPLIT \
        train_test_split.scene_filter.has_route=false \
        train_test_split.scene_filter.tokens=null \
        train_test_split.scene_filter.log_names=null \
        experiment_name=trainval_test \
        worker=single_machine_thread_pool \
        worker.max_workers=64 \
        sensor_blobs_path=/datahdd/karthik/dataset/nuplan-v1.1/rendered_sensor_blobs_augmented/$SENSOR_SPLIT \
        cache_path=$CACHE_ROOT/rap_aug
}

run_perturb() {
    echo "==> [perturb] split=$SPLIT"
    export OPENSCENE_DATA_ROOT="/datahdd/karthik/dataset_perturbed"
    python navsim/planning/script/run_dataset_caching.py \
        agent=rap_agent \
        dataset=navsim_dataset \
        agent.config.trajectory_sampling.time_horizon=5 \
        agent.config.cache_data=True \
        train_test_split=$SPLIT \
        train_test_split.scene_filter.has_route=false \
        train_test_split.scene_filter.tokens=null \
        train_test_split.scene_filter.log_names=null \
        experiment_name=trainval_test \
        worker=single_machine_thread_pool \
        worker.max_workers=64 \
        sensor_blobs_path=$SENSOR_BLOBS/$SENSOR_SPLIT \
        cache_path=$CACHE_ROOT/rap_perturbed
}

run_metric() {
    echo "==> [metric cache] split=$SPLIT"
    export OPENSCENE_DATA_ROOT="/datahdd/karthik/dataset/nuplan-v1.1"
    python navsim/planning/script/run_train_metric_caching.py \
        train_test_split=$SPLIT \
        cache.cache_path=$CACHE_ROOT/train_metric_cache \
        worker=single_machine_thread_pool \
        worker.max_workers=64
}

case "$TASK" in
    ego)     run_ego ;;
    aug)     run_aug ;;
    perturb) run_perturb ;;
    metric)  run_metric ;;
    all)
        run_ego
        run_aug
        run_perturb
        echo "==> All caching done. Running metric cache..."
        run_metric
        ;;
    *) echo "Unknown task: $TASK (choose: ego, aug, perturb, metric, all)"; exit 1 ;;
esac
