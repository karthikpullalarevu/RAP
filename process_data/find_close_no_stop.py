#!/usr/bin/env python3
"""Scenes where ego passes close to an inserted pedestrian without stopping (candidates for editing the label).

Per scene (InterPlan ped_v2, 20 Hz frames from 02_ped_frames.csv):
  closest moment = frame with the smallest ego-to-pedestrian gap
  close          = closest gap < --gap metres
  not stopped    = ego never below --stop-mph from the first pedestrian frame up to the closest moment
Scenes with another car between ego and the pedestrian (lead_vehicle_check.py, category C) are left out.

Writes <analysis>/07_close_no_stop/close_no_stop_scenes.csv and prints counts by speed at the closest moment.
Usage: python process_data/find_close_no_stop.py --gap 3 --stop-mph 1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MPH = 2.23694
SPEED_BINS = [0, 5, 10, 15, 20, 99]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, default=Path("/datahdd2/karthik/ped_v2/analysis"))
    ap.add_argument("--gap", type=float, default=3.0)
    ap.add_argument("--stop-mph", type=float, default=1.0)
    args = ap.parse_args()
    a = args.analysis_dir

    p = pd.read_csv(a / "02_ped_frames.csv", usecols=["scene_name", "frame_idx", "ego_speed_ms", "gap_m", "ped_side",
                                                       "ped_local_x_m", "ped_local_y_m", "ped_speed_ms"])
    p["ego_mph"] = p.ego_speed_ms * MPH
    closest = p.loc[p.groupby("scene_name").gap_m.idxmin()].set_index("scene_name")
    p = p.join(closest.frame_idx.rename("t_close"), on="scene_name")
    before = p[p.frame_idx <= p.t_close]
    lowest = before.groupby("scene_name").ego_mph.min().rename("min_mph_before_closest")

    s = closest.join(lowest)
    lead = pd.read_csv(a / "06_close_pass" / "closest_moment_lead_vehicle.csv").set_index("scene_name")
    s = s.join(lead.category.rename("lead_category"))
    info = pd.read_csv(a / "01_scene_summary.csv", usecols=["scene_name", "batch", "reaction", "context"]).set_index("scene_name")
    s = s.join(info)
    sel = pd.read_csv(a / "05_selection" / "train_samples.csv")
    s["split"] = np.where(s.index.isin(sel.scene_name), "train", "val/unused")
    s["train_samples"] = sel.scene_name.value_counts().reindex(s.index).fillna(0).astype(int)

    s["close"] = s.gap_m < args.gap
    s["not_stopped"] = s.min_mph_before_closest >= args.stop_mph
    s["lead_car"] = s.lead_category == "C"
    pick = s[s.close & s.not_stopped & ~s.lead_car].copy()
    pick["speed_bin_mph"] = pd.cut(pick.ego_mph, SPEED_BINS, right=False)

    out = a / "07_close_no_stop"
    out.mkdir(exist_ok=True)
    pick.reset_index().to_csv(out / "close_no_stop_scenes.csv", index=False)

    n = len(s)
    print(f"All scenes: {n}")
    print(f"  closest gap < {args.gap} m:                 {s.close.sum()} ({s.close.mean():.1%})")
    print(f"  ... and never below {args.stop_mph} mph before it: {(s.close & s.not_stopped).sum()}")
    print(f"  ... and no car between (final):         {len(pick)} ({len(pick)/n:.1%} of all, "
          f"{len(pick)/(~s.lead_car).sum():.1%} of scenes without a lead car)")
    print(f"  of these in training: {(pick.split == 'train').sum()} scenes, {pick.train_samples.sum()} samples")
    print("\nBy ego speed at the closest moment (mph):")
    print(pick.groupby("speed_bin_mph", observed=False).agg(scenes=("gap_m", "size"), median_gap_m=("gap_m", "median")).to_string())
    print("\nPedestrian side at the closest moment:", pick.ped_side.value_counts().to_dict())
    print("Ego reaction label:", pick.reaction.value_counts().to_dict())
    print(f"\nSaved {out / 'close_no_stop_scenes.csv'}")


if __name__ == "__main__":
    main()
