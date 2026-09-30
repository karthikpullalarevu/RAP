# InterPlan Pedestrian Data Pipeline

Template for processing new InterPlan simulation batches into RAP training data.

---

## 1. Folder Structure

Every new batch gets ONE folder on `/datahdd2/karthik/`:

```
/datahdd2/karthik/ped_v2/                    # <-- batch root
  raw/                                        # raw InterPlan simulation output (all batches)
    _batch_meta/{batch}/                      # manifest.csv, metrics_per_scene.csv per batch
    {scene_name}__{hash}/                     # e.g. 2021.05.12.19.36.12_veh-35_00005_00204__30330416a2e6
      simulation_log/{hash}-s0.msgpack.xz     # pickled SimulationLog
      rendered/rasters/frame_XXXX/CAM_*.png   # camera rasters (ped-in-view frames only)
      rendered/fov_frames.json                # frames where ped is in camera view
      metadata/                               # result.json, scene_config.yaml, manifest
      metrics/                                # nuPlan closed-loop metrics (parquet)
      aggregator_metric/
      ground_truth/                           # insertion debug + resolved ped config
  analysis/                                   # all analysis outputs go here
    01_scene_summary.csv                      # A: one row per scene (all frames)
    02_ped_frames.csv                         # B: one row per frame with an inserted ped
    03_plots/                                 # A_*.png overview, B_*.png ped frames
    04_raster_grid.png                        # random front-camera rasters
    analysis_report.json                      # headline numbers
    _partial.jsonl                            # per-scene results (resume)
  cache/                                      # RAP-format cache (rap_feature.gz + rap_target.gz)
    {scene_name}_{anchor_idx}/
      {ped_token}/
        rap_feature.gz
        rap_target.gz
  metric_cache/                               # PDM metric cache entries (symlinked into main)
```

## 2. Raw Data Format

Each scene directory contains a `simulation_log/*.msgpack.xz` file. This is an lzma-compressed msgpack containing a pickled `SimulationLog` object from InterPlan.

**Loading a simulation log:**
```python
import io, lzma, msgpack, pickle, sys, types

# Need dummy classes for interplan/tuplan_garage modules
# (see process_data/build_ped_cache.py for _PatchingUnpickler)
from process_data.build_ped_cache import load_simulation_log, _patch_map_path

_patch_map_path()  # redirect /ocean/ paths to local NUPLAN_MAPS_ROOT
sim_log = load_simulation_log(log_path)
```

**What's inside a SimulationLog:**
```
sim_log.simulation_history          # SimulationHistory
  .data                             # list of SimulationHistoryBuffers (one per frame)
    [i].ego_state                   # EgoState (NuPlan object)
      .rear_axle.x, .y, .heading   # position in UTM coords
      .dynamic_car_state.speed      # ego speed in m/s
      .time_point                   # TimePoint
    [i].observation                 # DetectionsTracks
      .tracked_objects.tracked_objects  # list of TrackedObject
        [j].tracked_object_type     # TrackedObjectType (VEHICLE, PEDESTRIAN, BICYCLE, ...)
        [j].center.x, .y           # object center position
        [j].velocity.x, .y         # object velocity
        [j].box                     # OrientedBox (length, width, height)
    [i].traffic_light_status        # list of TrafficLightStatusData
  .map_api                          # NuPlanMap (shared across all frames)
sim_log.planner
  ._route_roadblock_dict            # dict: route roadblock IDs -> data
```

**Key numbers:**
- 299 frames per scene at 20Hz (~15 seconds)
- `database_interval = 0.5s` means training iterations map to every 10th sim frame
- Anchor frames 0-198, each generates one training token
- Token format: `{manifest_hex}_{frame_index:04d}`

## 3. Analysis Pipeline

### Step 0: Put all scenes in one folder

Batches arrive as folders or tars. `process_data/extract_ped_v2.sh` extracts the batch 2 tars
and moves batch 1 scenes into `/datahdd2/karthik/ped_v2/raw/`. Each batch's `manifest.csv` and
`metrics_per_scene.csv` go to `raw/_batch_meta/<batch>/` (used for the `batch` column).
Safe to re-run (per-tar `.done` markers). Tars are kept. Run in tmux `ped-v2-extract` (~3.5 h).

Scene folders are named `{nuplan_scene}__{hash}`; rasters live in `rendered/rasters/frame_XXXX/`
(only frames where a ped is in camera view, listed in `rendered/fov_frames.json`).

### Step 1: Run analysis (`process_data/analyze_batch.py`)

```bash
python process_data/analyze_batch.py \
    --batch-root /datahdd2/karthik/ped_v2/raw \
    --output-dir /datahdd2/karthik/ped_v2/analysis \
    --workers 48          # add --limit 20 for a quick test
```
~5 s per scene per worker. Results are appended to `analysis/_partial.jsonl`; a re-run skips done scenes.

### Step 2: Outputs

| Output | What it is |
|------|-----------------|
| `01_scene_summary.csv` | **A. Whole-scene overview** (all frames), one row per scene: batch, status, context/type/group, configured ped speed/angle, ego speed/braking/stopped/stuck, vehicle & ped counts, collisions, closest gap, min TTC, ego reaction, nuPlan metrics |
| `02_ped_frames.csv` | **B. Pedestrian frames**, one row per frame with an inserted ped: ego speed/accel, gap to nearest inserted ped (car body to ped box), ped side (front/left/right/back), ped speed, TTC, in camera view, raster exists |
| `03_plots/A1..A5` | Status per batch, context/type, score/speed/braking/vehicle histograms, variants per base scene, scene flags |
| `03_plots/B1_speed_vs_distance_2d.png` | **2D histogram: ego speed vs distance to ped**, all batches + per batch |
| `03_plots/B2, B3` | Closest gap, ped speed, min TTC, speed at closest approach; ego reaction, ped side, peds present |
| `04_raster_grid.png` | 20 random front-camera rasters for a visual check |
| `analysis_report.json` | Headline numbers |

Notes:
- Only **inserted** peds (track tokens in `metadata/result.json`) count as "peds" in part B; original NuPlan peds are counted separately (`max_orig_peds`).
- Ego reaction: `stop` if speed drops below 0.5 m/s while a ped is present, `slow down` if below 70% of the speed when the ped appeared, else `keep going`.
- TTC assumes constant velocity (gap / closing speed); blank when not closing.

**Key questions to answer:**
- How many scenes are healthy (succeeded, no ego collision)?
- How close / how fast does ego get to peds (B1)? Close + fast = hard cases.
- Does ego mostly stop, slow down, or keep going?
- Is the scene mix (context, base scenes) varied enough?

---

## 4. Environment Setup

Always run from the RAP project root with these env vars:

```bash
cd /datahdd/karthik/RAP
conda activate rap
export PYTHONPATH=/datahdd/karthik/RAP
export NUPLAN_MAPS_ROOT=/datahdd/karthik/dataset/nuplan-maps-v1.0
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
```

For long-running tasks (> 5 min), use tmux:
```bash
tmux new-session -d -s ped-v2-analysis
tmux send-keys -t ped-v2-analysis '...' Enter
```

## 5. Data Format Reference

### SimulationLog structure
```
sim_log.simulation_history.data[frame_idx]
  .ego_state
    .rear_axle.x, .y, .heading     # UTM coordinates
    .dynamic_car_state.speed        # m/s
    .time_point                     # TimePoint
  .observation
    .tracked_objects.tracked_objects # list of TrackedObject
      .tracked_object_type          # VEHICLE, PEDESTRIAN, BICYCLE, etc
      .center.x, .y                # UTM coordinates
      .velocity.x, .y              # m/s
      .box.length, .width, .height  # meters
  .traffic_light_status             # list of TrafficLightStatusData

sim_log.planner._route_roadblock_dict  # route info
sim_log.simulation_history.map_api     # NuPlanMap
```

### Loading requires patching
InterPlan simulation logs reference `interplan.*` and `tuplan_garage.*` modules that we don't have installed. The `_PatchingUnpickler` in `process_data/build_ped_cache.py` stubs these out. Also, map paths in the pickle point to `/ocean/...` (the original cluster), so `_patch_map_path()` redirects them to our local `NUPLAN_MAPS_ROOT`.

### Frame timing
- 299 frames at 20Hz = ~15 seconds per scene
- Training uses `database_interval=0.5s` = every 10th sim frame
- Anchor frames for training tokens: 0, 1, 2, ..., 198
- Each anchor needs frames [anchor, anchor+100] (11 iterations x 10 sim frames)

### Token format
- `{manifest_hex}_{anchor_idx:04d}` e.g. `b3660ca245e65774_0030`
- Unique per (scene, anchor_frame) pair
- Different format from NuPlan tokens (16-char hex like `0ea47b4586de59a9`)

## 6. Existing v1 Data Locations

| What | Path |
|------|------|
| Raw simulation | `/datahdd2/karthik/pedestrian_simulation_v1_1809/` (510 scenes) |
| RAP cache | `/datahdd2/karthik/RAP/cache/rap_ped_v1_ped_only/` (28,423 tokens) |
| Metric cache | `/datahdd2/karthik/RAP/cache/train_metric_cache/` (symlinked from `/datahdd/karthik/RAP/cache/train_metric_cache`) |
| Trainval ped scan | `/datahdd2/karthik/output_trainval/summary.csv` |
| Navtest ped scan | `/datahdd2/karthik/output_navtest/summary.csv` |

## 7. Next Steps After Analysis

Once analysis is reviewed, the remaining pipeline steps are:

1. **Build RAP cache** (`process_data/build_ped_cache.py`) - extracts features/targets
2. **Build metric cache** (`process_data/build_ped_metric_cache.py`) - computes PDM scores
3. **Update metadata CSV** - append new entries to metric cache metadata
4. **Add to training config** - point `cache_path_ped` to the new cache

These are documented separately. This document covers **analysis only**.
