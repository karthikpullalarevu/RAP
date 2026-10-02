#!/usr/bin/env python3
"""Build metric cache entries for pedestrian simulation data.

Uses MetricCacheProcessor.compute_metric_cache() with a lightweight scenario
adapter that wraps InterPlan simulation data. Uses Ray for parallel processing.

Usage:
    python process_data/build_ped_metric_cache.py \
        --sim-root /datahdd2/karthik/pedestrian_simulation_v1_1809 \
        --ped-cache /datahdd2/karthik/RAP/cache/rap_ped_v1_ped_only \
        --metric-cache /datahdd2/karthik/RAP/cache/train_metric_cache

    # ped_v2: take tokens from sample lists (no need to wait for the feature cache)
    # and write to a separate folder that can be linked into the main cache later
    python process_data/build_ped_metric_cache.py \
        --sim-root /datahdd2/karthik/ped_v2/raw \
        --sample-list .../05_selection/train_samples.csv .../05_selection/val_samples.csv \
        --metric-cache /datahdd2/karthik/ped_v2/metric_cache
"""

import argparse
import csv
import io
import os
import pickle
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import lzma
import msgpack
import ray

os.environ.setdefault("NUPLAN_MAPS_ROOT", "/datahdd/karthik/dataset/nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

# ---------------------------------------------------------------------------
# Simulation log loader
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


def load_simulation_log(log_path):
    with lzma.open(str(log_path), "rb") as f:
        raw_bytes = next(iter(msgpack.Unpacker(f, raw=True)))
    return _PatchingUnpickler(io.BytesIO(raw_bytes)).load()


# ---------------------------------------------------------------------------
# Scenario adapter
# ---------------------------------------------------------------------------

from nuplan.common.actor_state.state_representation import StateSE2


class PedScenarioAdapter:
    def __init__(self, scene_name, ped_token, anchor_frame,
                 sim_history, route_roadblock_ids, map_api):
        self._scene_name = scene_name
        self._token = ped_token
        self._anchor = anchor_frame
        self._history = sim_history
        self._route_ids = route_roadblock_ids
        self._map_api = map_api

    @property
    def log_name(self):
        return self._scene_name

    @property
    def scenario_type(self):
        return "unknown"

    @property
    def token(self):
        return self._token

    @property
    def map_api(self):
        return self._map_api

    @property
    def database_interval(self):
        return 0.5

    @property
    def initial_ego_state(self):
        return self._history.data[self._anchor].ego_state

    @property
    def initial_tracked_objects(self):
        return self._history.data[self._anchor].observation

    @property
    def start_time(self):
        return self._history.data[self._anchor].ego_state.time_point

    def get_route_roadblock_ids(self):
        return self._route_ids

    def get_mission_goal(self):
        return StateSE2(0, 0, 0)

    def get_traffic_light_status_at_iteration(self, iteration):
        frame = self._anchor + iteration * 10
        return iter(self._history.data[frame].traffic_light_status)

    def get_tracked_objects_at_iteration(self, iteration, **kwargs):
        frame = self._anchor + iteration * 10
        return self._history.data[frame].observation


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_ped_cache(ped_cache_root):
    """Scan ped cache dirs to build {scene_name: [(anchor_frame, ped_token), ...]}."""
    scene_frames = defaultdict(list)
    for entry in sorted(ped_cache_root.iterdir()):
        if not entry.is_dir():
            continue
        dir_name = entry.name
        parts = dir_name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        try:
            anchor_idx = int(parts[1])
        except ValueError:
            continue

        scene_name = parts[0]
        subdirs = list(entry.iterdir())
        if len(subdirs) != 1:
            continue
        ped_token = subdirs[0].name
        scene_frames[scene_name].append((anchor_idx, ped_token))

    return scene_frames


def read_sample_lists(paths):
    """Read scene_name,frame_idx,token CSVs into {scene_name: [(anchor_frame, ped_token), ...]}."""
    scene_frames = defaultdict(set)
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                scene_frames[row["scene_name"]].add((int(row["frame_idx"]), row["token"]))
    return {k: sorted(v) for k, v in scene_frames.items()}


def metadata_path(file_path, metric_cache_path):
    """Path as stored in the metadata CSV: cache/train_metric_cache/<log>/unknown/<token>/metric_cache.pkl.

    Always uses the main-cache prefix, so entries built in a separate folder can be
    linked into cache/train_metric_cache and their CSV lines appended as they are.
    """
    rel = os.path.relpath(str(file_path), str(metric_cache_path))
    return f"cache/train_metric_cache/{rel}"


def find_sim_log(sim_root, scene_name):
    """Find the simulation log file for a scene."""
    scene_dir = sim_root / scene_name / "simulation_log"
    if not scene_dir.exists():
        return None
    logs = list(scene_dir.glob("*.msgpack.xz"))
    if len(logs) == 1:
        return logs[0]
    logs = list(scene_dir.glob("*.pkl.xz"))
    return logs[0] if len(logs) == 1 else None


# ---------------------------------------------------------------------------
# Ray remote: process one scene
# ---------------------------------------------------------------------------

@ray.remote
def process_scene(scene_name, frames, sim_root, metric_cache_path):
    """Process all ped tokens for one scene. Runs as a Ray task."""
    _patch_map_path()

    from navsim.planning.metric_caching.train_cache_processor import MetricCacheProcessor

    processor = MetricCacheProcessor(
        cache_path=str(metric_cache_path),
        force_feature_computation=False,
    )

    log_path = find_sim_log(Path(sim_root), scene_name)
    if log_path is None:
        return {"scene": scene_name, "done": 0, "errors": len(frames),
                "skipped": 0, "paths": []}

    try:
        sim_log = load_simulation_log(log_path)
    except Exception as e:
        return {"scene": scene_name, "done": 0, "errors": len(frames),
                "skipped": 0, "paths": [], "error_msg": str(e)}

    history = sim_log.simulation_history
    map_api = history.map_api
    route_ids = list(sim_log.planner._route_roadblock_dict.keys())
    n_sim_frames = len(history.data)

    paths = []
    done = 0
    errors = 0
    skipped = 0

    for anchor_idx, ped_token in sorted(frames):
        max_frame = anchor_idx + 100
        if max_frame >= n_sim_frames:
            errors += 1
            continue

        out_path = Path(metric_cache_path) / scene_name / "unknown" / ped_token / "metric_cache.pkl"
        if out_path.exists():
            paths.append(metadata_path(out_path, metric_cache_path))
            skipped += 1
            done += 1
            continue

        adapter = PedScenarioAdapter(
            scene_name=scene_name,
            ped_token=ped_token,
            anchor_frame=anchor_idx,
            sim_history=history,
            route_roadblock_ids=route_ids,
            map_api=map_api,
        )

        try:
            entry = processor.compute_metric_cache(adapter)
            if entry is not None:
                paths.append(metadata_path(entry.file_name, metric_cache_path))
                done += 1
            else:
                errors += 1
        except Exception as e:
            print(f"  ERROR {ped_token}: {e}", flush=True)
            errors += 1
            done += 1

    del sim_log, history
    return {"scene": scene_name, "done": done, "errors": errors,
            "skipped": skipped, "paths": paths}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build metric cache for ped data")
    parser.add_argument("--sim-root", type=Path,
                        default=Path("/datahdd2/karthik/pedestrian_simulation_v1_1809"))
    parser.add_argument("--ped-cache", type=Path,
                        default=Path("/datahdd2/karthik/RAP/cache/rap_ped_v1_ped_only"))
    parser.add_argument("--metric-cache", type=Path,
                        default=Path("./cache/train_metric_cache"))
    parser.add_argument("--sample-list", type=Path, nargs="+", default=None,
                        help="CSV(s) with scene_name,frame_idx,token. Used instead of scanning --ped-cache.")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    if args.sample_list:
        print("Reading sample lists...", flush=True)
        scene_frames = read_sample_lists(args.sample_list)
    else:
        print("Scanning ped cache...", flush=True)
        scene_frames = scan_ped_cache(args.ped_cache)
    total_frames = sum(len(v) for v in scene_frames.values())
    print(f"Found {total_frames} ped tokens across {len(scene_frames)} scenes", flush=True)

    if not ray.is_initialized():
        ray.init(address="auto")
    print(f"Ray connected: {ray.cluster_resources()}", flush=True)

    t0 = time.time()
    sim_root_str = str(args.sim_root)
    metric_cache_str = str(args.metric_cache)

    # Submit all scenes as Ray tasks
    futures = {}
    for scene_name, frames in sorted(scene_frames.items()):
        fut = process_scene.remote(scene_name, frames, sim_root_str, metric_cache_str)
        futures[fut] = scene_name

    # Collect results as they complete
    all_paths = []
    total_done = 0
    total_errors = 0
    total_skipped = 0
    completed = 0

    pending = list(futures.keys())
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        for ref in ready:
            result = ray.get(ref)
            completed += 1
            total_done += result["done"]
            total_errors += result["errors"]
            total_skipped += result["skipped"]
            all_paths.extend(result["paths"])

            elapsed = time.time() - t0
            rate = total_done / elapsed if elapsed > 0 else 0
            remaining = total_frames - total_done - total_errors
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{completed}/{len(scene_frames)}] {result['scene']}: "
                  f"{result['done']} done, {result['errors']} err | "
                  f"total={total_done}/{total_frames} | "
                  f"{rate:.1f} frames/s | ETA {eta/60:.0f}m",
                  flush=True)

    # Append new paths to metadata CSV
    if all_paths:
        metadata_dir = args.metric_cache / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        csv_path = metadata_dir / "train_metric_cache_metadata_node_0.csv"
        if not csv_path.exists():
            csv_path.write_text("file_name\n")

        existing = set()
        if csv_path.exists():
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row:
                        existing.add(row[0])

        to_add = [p for p in all_paths if p not in existing]

        if to_add:
            with open(csv_path, "a") as f:
                for p in to_add:
                    f.write(p + "\n")
            print(f"\nAppended {len(to_add)} new paths to {csv_path}", flush=True)
        else:
            print(f"\nAll paths already in metadata CSV", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone! {total_done} entries ({total_errors} errors, {total_skipped} skipped) in {elapsed/60:.1f}m")


if __name__ == "__main__":
    main()
