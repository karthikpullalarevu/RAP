# RAP Data Preprocessing Pipeline

This document traces every step we ran to go from raw NuPlan data to training-ready caches.

## Prerequisites

- Conda env: `rap` (Python 3.9, PyTorch 2.1, CUDA 12.1)
- Raw NuPlan data already downloaded and placed at `/datahdd/karthik/dataset/nuplan-v1.1/`
- NuPlan maps at `/datahdd/karthik/dataset/nuplan-maps-v1.0/`

### Environment Variables

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/datahdd/karthik/dataset/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/datahdd/karthik/RAP/exp"
export NAVSIM_DEVKIT_ROOT="/datahdd/karthik/RAP/navsim"
export PYTHONPATH=/datahdd/karthik/RAP:${PYTHONPATH}
```

---

## Phase 1: Generate OpenScene Metadata + Rendered Views

**Script:** `process_data/process.sh`

Run from inside the `process_data/` directory. Processes all 1,192 NuPlan `.db` log files (split: `trainval`) through 3 stages using 32 threads each.

### Step 1 — Ego Vehicle Metadata (`create_openscene_metadata.py`)

```bash
python -u create_openscene_metadata.py \
  --nuplan-root-path /datahdd/karthik/dataset/nuplan-v1.1 \
  --nuplan-db-path /datahdd/karthik/dataset/nuplan-v1.1/splits/trainval \
  --nuplan-sensor-path /datahdd/karthik/dataset/nuplan-v1.1/sensor_blobs/trainval \
  --nuplan-map-version nuplan-maps-v1.0 \
  --nuplan-map-root /datahdd/karthik/dataset/nuplan-maps-v1.0 \
  --out-dir /datahdd/karthik/dataset/nuplan-v1.1/navsim_logs/trainval \
  --split trainval --thread-num 32 --start-index 0 --end-index 14561
```

**What it does:**
- Reads each NuPlan log `.db` file
- For each frame: extracts ego pose, CAN bus data, camera info, traffic lights, map features (lanes, crosswalks, etc.), 3D object annotations
- Renders 3D rasterized camera views using `ScenarioRenderer` (single front camera `CAM_F0`)
- Saves rendered images to `rendered_sensor_blobs/trainval/`
- Saves per-log `.pkl` files to `navsim_logs/trainval/`

**Checkpoint:** `checkpoint_ego.txt` (tracks completed log indices)

### Step 2 — Recovery Perturbation Data (`create_openscene_metadata_purturbed.py`)

```bash
python -u create_openscene_metadata_purturbed.py \
  --nuplan-root-path /datahdd/karthik/dataset/nuplan-v1.1 \
  --nuplan-db-path /datahdd/karthik/dataset/nuplan-v1.1/splits/trainval \
  --nuplan-sensor-path /datahdd/karthik/dataset/nuplan-v1.1/sensor_blobs/trainval \
  --nuplan-map-version nuplan-maps-v1.0 \
  --nuplan-map-root /datahdd/karthik/dataset/nuplan-maps-v1.0 \
  --out-dir /datahdd/karthik/dataset_perturbed/navsim_logs/trainval \
  --split trainval --thread-num 32 --start-index 0 --end-index 14561
```

**What it does:**
- For each training token (from `navtrain.yaml`), takes a 14-frame window
- Perturbs the ego pose at the current frame (position ±0.5m, yaw ±15°, velocity ±20%, acceleration ±10%)
- Re-renders the perturbed frame's camera view
- Saves to `/datahdd/karthik/dataset_perturbed/navsim_logs/trainval/`

**Purpose:** Teaches the model to recover from off-trajectory states.

**Checkpoint:** `checkpoint_perturbed.txt`

### Step 3 — Cross-Agent Augmentation (`create_openscene_metadata_aug.py`)

```bash
python -u create_openscene_metadata_aug.py \
  --nuplan-root-path /datahdd/karthik/dataset/nuplan-v1.1 \
  --nuplan-db-path /datahdd/karthik/dataset/nuplan-v1.1/splits/trainval \
  --nuplan-sensor-path /datahdd/karthik/dataset/nuplan-v1.1/sensor_blobs/trainval \
  --nuplan-map-version nuplan-maps-v1.0 \
  --nuplan-map-root /datahdd/karthik/dataset/nuplan-maps-v1.0 \
  --out-dir /datahdd/karthik/dataset_aug/navsim_logs/trainval \
  --split trainval --thread-num 32 --start-index 0 --end-index 14561
```

**What it does:**
- For each log, finds all vehicles that appear in the scene
- Treats each neighboring vehicle as a "virtual ego" and re-creates the full scene from its perspective
- Filters out boring trajectories using a constant-velocity model (keeps only vehicles with mean CV prediction error > 0.5m over 5s)
- Re-renders 4 camera views (`CAM_F0`, `CAM_L0`, `CAM_R0`, `CAM_B0`) from the virtual ego's perspective
- Adds the original ego vehicle back into the scene as an obstacle
- Saves rendered images to `rendered_sensor_blobs_augmented/trainval/`
- Saves per-vehicle `.pkl` files to `/datahdd/karthik/dataset_aug/navsim_logs/trainval/`

**Purpose:** Multiplies training data by using other vehicles' viewpoints without needing extra real data.

**Checkpoint:** `checkpoint_aug.txt`

---

## Phase 2: Cache Dataset for Training

**Script:** `cache_dataset.sh`

```bash
bash cache_dataset.sh --split navtrain --task all
```

This runs 4 caching tasks sequentially using Hydra + the NavSim framework.

### Step 4 — Cache Ego Data

```bash
OPENSCENE_DATA_ROOT=/datahdd/karthik/dataset/nuplan-v1.1
python navsim/planning/script/run_dataset_caching.py \
  agent=rap_agent dataset=navsim_dataset \
  agent.config.trajectory_sampling.time_horizon=5 \
  agent.config.cache_data=True \
  train_test_split=navtrain \
  train_test_split.scene_filter.has_route=false \
  experiment_name=trainval_test \
  worker=single_machine_thread_pool worker.max_workers=64 \
  sensor_blobs_path=/datahdd/karthik/dataset/nuplan-v1.1/sensor_blobs/navtrain \
  cache_path=./cache/rap_ego
```

**Output:** `./cache/rap_ego/` — cached features + targets for ego vehicle scenes.

### Step 5 — Cache Augmented Data

```bash
OPENSCENE_DATA_ROOT=/datahdd/karthik/dataset_aug
python navsim/planning/script/run_dataset_caching.py \
  agent=rap_agent dataset=navsim_dataset \
  agent.config.trajectory_sampling.time_horizon=5 \
  agent.config.cache_data=True \
  train_test_split=navtrain \
  train_test_split.scene_filter.has_route=false \
  train_test_split.scene_filter.tokens=null \
  train_test_split.scene_filter.log_names=null \
  experiment_name=trainval_test \
  worker=single_machine_thread_pool worker.max_workers=64 \
  sensor_blobs_path=/datahdd/karthik/dataset/nuplan-v1.1/rendered_sensor_blobs_augmented/navtrain \
  cache_path=./cache/rap_aug
```

**Output:** `./cache/rap_aug/` — cached features + targets for cross-agent augmented scenes.

### Step 6 — Cache Perturbed Data

```bash
OPENSCENE_DATA_ROOT=/datahdd/karthik/dataset_perturbed
python navsim/planning/script/run_dataset_caching.py \
  agent=rap_agent dataset=navsim_dataset \
  agent.config.trajectory_sampling.time_horizon=5 \
  agent.config.cache_data=True \
  train_test_split=navtrain \
  train_test_split.scene_filter.has_route=false \
  train_test_split.scene_filter.tokens=null \
  train_test_split.scene_filter.log_names=null \
  experiment_name=trainval_test \
  worker=single_machine_thread_pool worker.max_workers=64 \
  sensor_blobs_path=/datahdd/karthik/dataset/nuplan-v1.1/sensor_blobs/navtrain \
  cache_path=./cache/rap_perturbed
```

**Output:** `./cache/rap_perturbed/` — cached features + targets for perturbed recovery scenes.

### Step 7 — Cache Training Metrics

```bash
OPENSCENE_DATA_ROOT=/datahdd/karthik/dataset/nuplan-v1.1
python navsim/planning/script/run_train_metric_caching.py \
  train_test_split=navtrain \
  cache.cache_path=./cache/train_metric_cache \
  worker=single_machine_thread_pool worker.max_workers=64
```

**Output:** `./cache/train_metric_cache/` — precomputed metrics for training loss computation.

---

## Summary of Outputs

| Step | Script | Output Location |
|------|--------|-----------------|
| 1. Ego metadata + renders | `create_openscene_metadata.py` | `dataset/nuplan-v1.1/navsim_logs/trainval/` + `rendered_sensor_blobs/` |
| 2. Perturbed metadata + renders | `create_openscene_metadata_purturbed.py` | `dataset_perturbed/navsim_logs/trainval/` + `rendered_sensor_blobs/` |
| 3. Augmented metadata + renders | `create_openscene_metadata_aug.py` | `dataset_aug/navsim_logs/trainval/` + `rendered_sensor_blobs_augmented/` |
| 4. Ego cache | `run_dataset_caching.py` | `./cache/rap_ego/` |
| 5. Aug cache | `run_dataset_caching.py` | `./cache/rap_aug/` |
| 6. Perturbed cache | `run_dataset_caching.py` | `./cache/rap_perturbed/` |
| 7. Metric cache | `run_train_metric_caching.py` | `./cache/train_metric_cache/` |

## Next Step: Start Ray & Train

### Step 8 — Start Ray

Run the `ray_start` function (defined in `~/.zshrc`):

```bash
ray_start
```

This runs:
```bash
ulimit -u 65535
ray start --head --port 6396 --num-gpus 8 \
          --dashboard-host 0.0.0.0 \
          --redis-password "<auto-generated>"
export ip_head="127.0.0.1:6396"
export redis_password="<auto-generated>"
export num_nodes=1
```

### Step 9 — Run Training

```bash
python navsim/planning/script/run_training.py \
  ...  # see training config for full arguments
```
