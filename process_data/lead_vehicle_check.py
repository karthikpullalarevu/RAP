#!/usr/bin/env python3
"""At each sequence's closest moment to an inserted pedestrian: is another vehicle between ego and the ped?

Ego's path = straight corridor ahead of the car (car width + 0.3 m each side, 30 m long).
Writes <analysis>/06_close_pass/closest_moment_lead_vehicle.csv with one row per sequence:
  category A: ped in ego's path, nothing between     B: ped ahead but off to the side, nothing between
           C: another vehicle between ego and ped    D: ped beside or behind ego

Usage: python process_data/lead_vehicle_check.py --workers 64
"""
import argparse, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import box as sbox

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/datahdd/karthik/dataset/nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
from process_data.build_ped_cache import load_simulation_log, _patch_map_path

RAW = Path("/datahdd2/karthik/ped_v2/raw")
MPH = 2.23694


def check(args):
    name, t0 = args
    try:
        _patch_map_path()
        ins = set(json.loads((RAW / name / "metadata/result.json").read_text())["saved_log_summary"]["pedestrian_track_tokens"])
        data = load_simulation_log(next((RAW / name / "simulation_log").glob("*.xz"))).simulation_history.data
        ego = data[t0].ego_state
        cx, cy, h = ego.center.x, ego.center.y, ego.center.heading
        co, si = math.cos(-h), math.sin(-h)
        foot = ego.car_footprint.geometry
        ob = ego.car_footprint.oriented_box
        hl, hw = ob.length / 2, ob.width / 2
        loc = lambda o: ((o.center.x - cx) * co - (o.center.y - cy) * si, (o.center.x - cx) * si + (o.center.y - cy) * co)
        objs = data[t0].observation.tracked_objects.tracked_objects
        ped = min([o for o in objs if o.track_token in ins], key=lambda o: o.box.geometry.distance(foot))
        px, py = loc(ped)
        ahead = affinity.translate(affinity.rotate(sbox(hl, -(hw + 0.3), hl + 30, hw + 0.3), h, origin=(0, 0), use_radians=True), cx, cy)
        lead = None
        for o in objs:
            if o.tracked_object_type.name == "VEHICLE" and o.box.geometry.intersects(ahead):
                g = o.box.geometry.distance(foot)
                if lead is None or g < lead[0]:
                    lead = (g, math.hypot(o.velocity.x, o.velocity.y), loc(o)[0])
        between = lead is not None and lead[2] < px
        if px <= 0:
            cat = "D"
        elif between:
            cat = "C"
        elif ped.box.geometry.intersects(ahead):
            cat = "A"
        else:
            cat = "B"
        return dict(scene_name=name, category=cat, ped_x=px, ped_y=py,
                    lead_gap=lead[0] if lead else np.nan, lead_speed_mph=lead[1] * MPH if lead else np.nan)
    except Exception as ex:
        return dict(scene_name=name, category="error", err=repr(ex))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, default=Path("/datahdd2/karthik/ped_v2/analysis"))
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    p = pd.read_csv(args.analysis_dir / "02_ped_frames.csv", usecols=["scene_name", "frame_idx", "ego_speed_ms", "gap_m"])
    c = p.loc[p.groupby("scene_name").gap_m.idxmin()]
    with ProcessPoolExecutor(args.workers) as ex:
        r = pd.DataFrame(list(ex.map(check, [(x.scene_name, int(x.frame_idx)) for x in c.itertuples()], chunksize=4)))
    r = r.merge(c.assign(ego_mph=c.ego_speed_ms * MPH)[["scene_name", "frame_idx", "gap_m", "ego_mph"]], on="scene_name")
    out = args.analysis_dir / "06_close_pass" / "closest_moment_lead_vehicle.csv"
    r.to_csv(out, index=False)
    print(r.category.value_counts().to_dict())
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
