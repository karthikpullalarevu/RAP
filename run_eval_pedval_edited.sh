#!/usr/bin/env bash
# Held-out ped_v2 with yield-edited labels (ped_v2/cache/val_yield_edited, made by process_data/yield_edit_val.py;
# too-late / side-pass samples left out, as in training). All five models in parallel on GPUs 0-4.
# Results: <experiment>/ped_val_batch3_<tag>_editedval/
set -uo pipefail
cd /datahdd/karthik/RAP
X=/datahdd2/karthik/RAP/experiments
export PYTHONPATH=/datahdd/karthik/RAP:${PYTHONPATH:-}
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0" NUPLAN_MAPS_ROOT="/datahdd/karthik/dataset/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/datahdd/karthik/RAP/exp" NAVSIM_DEVKIT_ROOT="/datahdd/karthik/RAP/navsim"
export OPENSCENE_DATA_ROOT="/datahdd2/karthik/dataset/nuplan-v1.1" TMPDIR=/datahdd2/karthik/tmp
export ip_head="128.2.220.69:6379" redis_password="dummy" num_nodes="1"
# one maths thread per scoring worker: 5 models x 32 workers otherwise oversubscribe the CPUs badly
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CACHE=/datahdd2/karthik/ped_v2/cache/val_yield_edited
SAMPLES=/datahdd2/karthik/ped_v2/analysis/09_yield_edit/val/val_samples_edited.csv

run() {   # $1 gpu, $2 experiment, $3 checkpoint, $4 pick mode, $5 tag, $6 extra flag
    CUDA_VISIBLE_DEVICES=$1 python eval_ped_val.py --exp-name $2 --ckpt "$X/$2/$3" --pick-mode $4 \
        --val-cache $CACHE --val-samples $SAMPLES --tag ${5}_editedval ${6:-} > "$X/$2/eval_pedval_editedval.log" 2>&1
    echo "DONE $2"
}
run 0 E4_pedV2-105k_photo50k            epoch15.ckpt final      epoch15 &
run 1 E5_pedV2yield_oldLoss              epoch19.ckpt final      epoch19 &
run 2 E6_pedV2yield_sub5_infonce         epoch15.ckpt gated_pdms ep15_gated_pdms &
run 3 E7_pedV2yield_sub5_infonce_noAccel epoch16.ckpt gated_pdms epoch16_gated_pdms --zero-ego-accel &
run 4 E8_pedV2-105k_photo50k_noAccel     epoch18.ckpt final      epoch18_final --zero-ego-accel &
wait
echo ALL_EDITEDVAL_DONE
