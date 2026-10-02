#!/usr/bin/env python3
"""Build RAP cache from pedestrian simulation data.

Reads simulation logs + raster images from pedestrian_simulation_v1_1809,
extracts ego trajectories, and creates rap_feature.gz / rap_target.gz files
compatible with RAP's CacheOnlyDataset.

Usage:
    python process_data/build_ped_cache.py \
        --dataset-root /datahdd2/karthik/pedestrian_simulation_v1_1809 \
        --ego-cache /datahdd/karthik/RAP/cache/rap_ego \
        --navsim-pkl-dir /datahdd/karthik/dataset/nuplan-v1.1/navsim_logs/trainval \
        --output-cache /datahdd2/karthik/RAP/cache/rap_ped \
        --workers 8
"""

import argparse
import csv
import gzip
import io
import os
import pickle
import sys
import types
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import math

import lzma
import msgpack
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Dummy module stubs for interplan/tuplan_garage (needed to unpickle sim logs)
# ---------------------------------------------------------------------------

class _FlexibleMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls)
        obj.__dict__.update(kwargs)
        return obj


def _make_dummy_class(name):
    return _FlexibleMeta(
        name, (),
        {"__setstate__": lambda self, state: self.__dict__.update(state)
         if isinstance(state, dict) else None},
    )


class _PatchingUnpickler(pickle.Unpickler):
    """Unpickler that stubs out interplan/tuplan_garage classes."""

    def find_class(self, module, name):
        if module.startswith(("interplan", "tuplan_garage")):
            parts = module.split(".")
            root = parts[0]
            if root not in sys.modules:
                sys.modules[root] = types.ModuleType(root)
            current = sys.modules[root]
            for i, part in enumerate(parts[1:], 1):
                full = ".".join(parts[: i + 1])
                if full not in sys.modules:
                    mod = types.ModuleType(full)
                    setattr(current, part, mod)
                    sys.modules[full] = mod
                current = sys.modules[full]
            cls = _make_dummy_class(name)
            setattr(current, name, cls)
            return cls
        return super().find_class(module, name)


def _patch_map_path():
    """Patch GPKGMapsDB to redirect PSC map paths to local."""
    import nuplan.database.maps_db.gpkg_mapsdb as gpkg_mod
    _original_init = gpkg_mod.GPKGMapsDB.__init__

    def _patched_init(self, map_version, map_root=None):
        if map_root and "/ocean/" in str(map_root):
            map_root = os.environ.get(
                "NUPLAN_MAPS_ROOT",
                "/datahdd/karthik/dataset/nuplan-maps-v1.0",
            )
        _original_init(self, map_version, map_root)

    gpkg_mod.GPKGMapsDB.__init__ = _patched_init


def load_simulation_log(log_path: Path):
    """Load a SimulationLog from msgpack.xz, stubbing interplan deps."""
    with lzma.open(str(log_path), "rb") as f:
        raw_bytes = next(iter(msgpack.Unpacker(f, raw=True)))
    return _PatchingUnpickler(io.BytesIO(raw_bytes)).load()


# ---------------------------------------------------------------------------
# Image processing (mirrors bev_feature_build.py transforms)
# ---------------------------------------------------------------------------

_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
_SCALE = 0.4


def _process_image(img_array: np.ndarray) -> np.ndarray:
    """Normalize, scale, and pad a single camera image to (3, 448, 768)."""
    img = img_array.astype(np.float32)
    # Normalize
    img = (img - _MEAN) / _STD
    # Scale (resize to 40% of original: 1080*0.4=432, 1920*0.4=768)
    h, w = img.shape[:2]
    new_h, new_w = int(h * _SCALE), int(w * _SCALE)
    pil_img = Image.fromarray((img * _STD + _MEAN).clip(0, 255).astype(np.uint8))
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    img = np.array(pil_img).astype(np.float32)
    img = (img - _MEAN) / _STD
    # Pad to multiple of 32: 432 -> 448, 768 already divisible
    pad_h = (32 - new_h % 32) % 32  # 432 % 32 = 16, pad = 16
    if pad_h > 0:
        img = np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode="constant")
    # Transpose to (3, H, W)
    img = img.transpose(2, 0, 1)
    return img


def load_and_process_cameras(raster_dir: Path, frame_idx: int) -> np.ndarray:
    """Load 4 camera PNGs for a frame and process to (4, 3, 448, 768)."""
    frame_dir = raster_dir / f"frame_{frame_idx:04d}"
    cameras = []
    for cam_name in ["CAM_B0", "CAM_F0", "CAM_L0", "CAM_R0"]:
        img_path = frame_dir / f"{cam_name}.png"
        img = np.array(Image.open(img_path))
        cameras.append(_process_image(img))
    return np.stack(cameras, axis=0)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def global_to_local(positions: np.ndarray, headings: np.ndarray,
                    anchor_x: float, anchor_y: float, anchor_heading: float):
    """Convert global (x, y, heading) to local coordinates relative to anchor.

    Args:
        positions: (N, 2) global x, y
        headings: (N,) global headings
        anchor_x, anchor_y, anchor_heading: anchor frame's global pose

    Returns:
        local_poses: (N, 3) local [x, y, heading]
    """
    cos_a = np.cos(-anchor_heading)
    sin_a = np.sin(-anchor_heading)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    dx = positions[:, 0] - anchor_x
    dy = positions[:, 1] - anchor_y
    local_xy = np.stack([dx, dy], axis=-1) @ rot.T

    local_heading = headings - anchor_heading
    # Normalize to [-pi, pi]
    local_heading = (local_heading + np.pi) % (2 * np.pi) - np.pi

    return np.column_stack([local_xy, local_heading]).astype(np.float32)


# ---------------------------------------------------------------------------
# Driving command lookup
# ---------------------------------------------------------------------------

def load_driving_commands(pkl_path: Path):
    """Load navsim pkl and return (timestamps, driving_commands) arrays."""
    with open(pkl_path, "rb") as f:
        frames = pickle.load(f)
    timestamps = np.array([fr["timestamp"] for fr in frames])
    commands = np.array([fr["driving_command"] for fr in frames])
    return timestamps, commands


def lookup_driving_command(sim_timestamp_us: float,
                           pkl_timestamps: np.ndarray,
                           pkl_commands: np.ndarray) -> np.ndarray:
    """Find closest pkl frame's driving command by timestamp."""
    idx = np.argmin(np.abs(pkl_timestamps - sim_timestamp_us))
    return pkl_commands[idx]


# ---------------------------------------------------------------------------
# Borrow lidar2img and img_shape from existing ego cache
# ---------------------------------------------------------------------------

def borrow_from_ego_cache(ego_cache_dir: Path, scene_name: str):
    """Load lidar2img and img_shape from an existing ego cache entry."""
    scene_dir = ego_cache_dir / scene_name
    # Pick any token subdirectory
    token_dirs = [d for d in scene_dir.iterdir() if d.is_dir()]
    if not token_dirs:
        raise FileNotFoundError(f"No tokens in ego cache for {scene_name}")
    feat_path = token_dirs[0] / "rap_feature.gz"
    with gzip.open(feat_path, "rb") as f:
        feat = pickle.load(f)
    return feat["lidar2img"], feat["img_shape"]


# ---------------------------------------------------------------------------
# Front-camera pedestrian filter
# ---------------------------------------------------------------------------

def classify_camera(local_x, local_y):
    angle = math.degrees(math.atan2(local_y, local_x))
    if -45 <= angle <= 45:
        return 'FRONT'
    elif 45 < angle <= 135:
        return 'LEFT'
    elif -135 <= angle < -45:
        return 'RIGHT'
    else:
        return 'BACK'


def has_front_pedestrian(sample, ego_state):
    tracked = sample.observation.tracked_objects.tracked_objects
    peds = [obj for obj in tracked
            if getattr(obj.tracked_object_type, 'name', '') == 'PEDESTRIAN']
    if not peds:
        return False
    heading = ego_state.center.heading
    for ped in peds:
        dx = ped.center.x - ego_state.center.x
        dy = ped.center.y - ego_state.center.y
        local_x = dx * math.cos(-heading) - dy * math.sin(-heading)
        local_y = dx * math.sin(-heading) + dy * math.cos(-heading)
        if classify_camera(local_x, local_y) == 'FRONT':
            return True
    return False


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

HISTORY_STEPS = 3       # 3 past frames + current = 4 history frames
FUTURE_STEPS = 10       # 10 future poses
SUBSAMPLE = 10          # 20Hz -> 2Hz (every 10th frame)


def process_scene(scene_name: str, dataset_root: Path, ego_cache_dir: Path,
                  navsim_pkl_dir: Path, output_cache: Path, frames=None):
    """Process one scene and write cache entries.

    frames: optional list of anchor frame indices (from --sample-list). If given,
    only those anchors are cached and the front-pedestrian filter is skipped.
    """
    scene_dir = dataset_root / scene_name
    # v2 scene folders are "{nuplan_scene}__{hash}"; ego cache / navsim pkl use the nuplan name
    base_name = scene_name.split("__")[0]

    # 1. Load simulation log
    sim_logs = list((scene_dir / "simulation_log").glob("*.msgpack.xz"))
    if not sim_logs:
        return 0, scene_name, "no simulation log"
    sim_log = load_simulation_log(sim_logs[0])
    history = sim_log.simulation_history
    samples = history.data
    ego_states = history.extract_ego_state
    n_frames = len(samples)

    # 2. Extract all ego data at 20Hz
    all_x = np.array([s.center.x for s in ego_states])
    all_y = np.array([s.center.y for s in ego_states])
    all_heading = np.array([s.center.heading for s in ego_states])
    all_vx = np.array([s.dynamic_car_state.rear_axle_velocity_2d.x for s in ego_states])
    all_vy = np.array([s.dynamic_car_state.rear_axle_velocity_2d.y for s in ego_states])
    all_ax = np.array([s.dynamic_car_state.rear_axle_acceleration_2d.x for s in ego_states])
    all_ay = np.array([s.dynamic_car_state.rear_axle_acceleration_2d.y for s in ego_states])
    all_ts = np.array([s.time_us for s in ego_states])

    # 3. Borrow camera transforms from ego cache
    try:
        lidar2img, img_shape = borrow_from_ego_cache(ego_cache_dir, base_name)
    except FileNotFoundError:
        return 0, scene_name, "not in ego cache"

    # 4. Load driving commands from navsim pkl
    pkl_path = navsim_pkl_dir / f"{base_name}.pkl"
    if pkl_path.exists():
        pkl_ts, pkl_cmds = load_driving_commands(pkl_path)
    else:
        pkl_ts, pkl_cmds = None, None

    # 5. Determine valid anchor frames
    # Need HISTORY_STEPS * SUBSAMPLE frames before and FUTURE_STEPS * SUBSAMPLE after
    min_idx = HISTORY_STEPS * SUBSAMPLE  # 30
    max_idx = n_frames - 1 - FUTURE_STEPS * SUBSAMPLE  # 299 - 1 - 100 = 198

    # v1 keeps rasters in rasters/, v2 in rendered/rasters/
    raster_dir = scene_dir / "rasters"
    if not raster_dir.exists():
        raster_dir = scene_dir / "rendered" / "rasters"
    count = 0
    checked = 0

    if frames is None:
        anchors = range(min_idx, max_idx + 1)
    else:
        anchors = [f for f in sorted(frames) if min_idx <= f <= max_idx]

    for anchor_idx in anchors:
        checked += 1
        if frames is None and not has_front_pedestrian(samples[anchor_idx], ego_states[anchor_idx]):
            continue
        # History frame indices (4 frames at 0.5s intervals)
        hist_indices = [anchor_idx - i * SUBSAMPLE for i in range(HISTORY_STEPS, -1, -1)]
        # Future frame indices (10 frames at 0.5s intervals)
        fut_indices = [anchor_idx + (i + 1) * SUBSAMPLE for i in range(FUTURE_STEPS)]

        if fut_indices[-1] >= n_frames:
            continue

        # Check raster exists
        frame_dir = raster_dir / f"frame_{anchor_idx:04d}"
        if not frame_dir.exists():
            continue

        # --- Build ego_status (4, 11) ---
        anchor_x = all_x[anchor_idx]
        anchor_y = all_y[anchor_idx]
        anchor_heading = all_heading[anchor_idx]

        ego_features = []
        for hidx in hist_indices:
            # Pose in local coords
            pos = np.array([[all_x[hidx], all_y[hidx]]])
            hdg = np.array([all_heading[hidx]])
            local_pose = global_to_local(pos, hdg, anchor_x, anchor_y, anchor_heading)[0]

            # Velocity and acceleration (already in ego frame from simulation)
            velocity = np.array([all_vx[hidx], all_vy[hidx]], dtype=np.float32)
            acceleration = np.array([all_ax[hidx], all_ay[hidx]], dtype=np.float32)

            # Driving command
            if pkl_ts is not None:
                cmd = lookup_driving_command(all_ts[hidx], pkl_ts, pkl_cmds).astype(np.float32)
            else:
                cmd = np.array([0, 1, 0, 0], dtype=np.float32)  # default: straight

            ego_feat = np.concatenate([local_pose, velocity, acceleration, cmd])
            ego_features.append(ego_feat)

        ego_status = torch.tensor(np.stack(ego_features), dtype=torch.float32)  # (4, 11)

        # --- Build trajectory (10, 3) ---
        fut_pos = np.column_stack([all_x[fut_indices], all_y[fut_indices]])
        fut_hdg = all_heading[fut_indices]
        trajectory = torch.tensor(
            global_to_local(fut_pos, fut_hdg, anchor_x, anchor_y, anchor_heading),
            dtype=torch.float64,
        )  # (10, 3)

        # --- Build camera features ---
        camera_imgs = load_and_process_cameras(raster_dir, anchor_idx)
        camera_feature = torch.tensor(
            np.ascontiguousarray(camera_imgs), dtype=torch.float32
        )  # (4, 3, 448, 768)

        # --- Token ---
        manifest_token = sim_logs[0].stem.split("-")[0]  # anchor_token from filename
        token = f"{manifest_token}_{anchor_idx:04d}"

        # --- Assemble feature dict ---
        features = {
            "camera_feature": camera_feature,
            "camera_valid": torch.tensor(True),
            "rendered_camera_feature": camera_feature.clone(),
            "img_shape": img_shape,
            "lidar2img": lidar2img,
            "ego_status": ego_status,
        }

        # --- Assemble target dict ---
        targets = {
            "trajectory": trajectory,
            "token": token,
        }

        # --- Write cache ---
        cache_log_name = f"{scene_name}_{anchor_idx:04d}"
        cache_dir = output_cache / cache_log_name / token
        if (cache_dir / "rap_feature.gz").exists() and (cache_dir / "rap_target.gz").exists():
            count += 1  # already built (resume)
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)

        for name, data in [("rap_feature", features), ("rap_target", targets)]:
            out_path = cache_dir / f"{name}.gz"
            with gzip.open(out_path, "wb", compresslevel=1) as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        count += 1

    return count, scene_name, f"ok ({checked} checked, {count} cached)"


def main():
    parser = argparse.ArgumentParser(description="Build RAP cache from pedestrian simulation data")
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/datahdd2/karthik/pedestrian_simulation_v1_1809"))
    parser.add_argument("--ego-cache", type=Path,
                        default=Path("/datahdd/karthik/RAP/cache/rap_ego"))
    parser.add_argument("--navsim-pkl-dir", type=Path,
                        default=Path("/datahdd/karthik/dataset/nuplan-v1.1/navsim_logs/trainval"))
    parser.add_argument("--output-cache", type=Path,
                        default=Path("/datahdd2/karthik/RAP/cache/rap_ped_v1_ped_only"))
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1, sim log loading is memory-heavy)")
    parser.add_argument("--sample-list", type=Path, default=None,
                        help="CSV with scene_name,frame_idx (from select_ped_v2_samples.py). "
                             "If given, only these anchors are cached.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N scenes (for testing)")
    args = parser.parse_args()

    os.environ["NUPLAN_MAPS_ROOT"] = "/datahdd/karthik/dataset/nuplan-maps-v1.0"
    _patch_map_path()

    # Scenes (and optionally anchor frames) to process
    scene_frames = {}
    if args.sample_list:
        with open(args.sample_list, newline="") as f:
            for row in csv.DictReader(f):
                scene_frames.setdefault(row["scene_name"], []).append(int(row["frame_idx"]))
        scenes = sorted(scene_frames)
    else:
        manifest_path = args.dataset_root / "manifest.csv"
        with open(manifest_path, newline="") as f:
            scenes = [row["scenario_name"] for row in csv.DictReader(f)]

    if args.limit:
        scenes = scenes[:args.limit]

    args.output_cache.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    errors = []

    if args.workers <= 1:
        for i, scene_name in enumerate(scenes):
            count, name, status = process_scene(
                scene_name, args.dataset_root, args.ego_cache,
                args.navsim_pkl_dir, args.output_cache, scene_frames.get(scene_name),
            )
            total_samples += count
            if not status.startswith("ok"):
                errors.append((name, status))
            print(f"[{i+1}/{len(scenes)}] {name}: {count} samples ({status})")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_scene, scene_name, args.dataset_root,
                    args.ego_cache, args.navsim_pkl_dir, args.output_cache,
                    scene_frames.get(scene_name),
                ): scene_name
                for scene_name in scenes
            }
            for i, future in enumerate(as_completed(futures), 1):
                count, name, status = future.result()
                total_samples += count
                if not status.startswith("ok"):
                    errors.append((name, status))
                print(f"[{i}/{len(scenes)}] {name}: {count} samples ({status})")

    print(f"\nDone! Total samples: {total_samples}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for name, status in errors:
            print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
