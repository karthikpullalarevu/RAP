#!/usr/bin/env python3
"""Does telling the model "ego is already braking" make it stop?

The model reads ego's current speed and acceleration (ego_status, last row: x, y, heading,
vx, vy, ax, ay, 4 command flags). This overrides ax and re-runs the model on held-out ped_v2
samples with ego at >= 10 mph:
  NATIVE - a real pedestrian 4-12 m ahead in ego's path
  FAR    - the pedestrian is >= 20 m ahead (nothing close)
Also reports how stopping relates to the recorded acceleration in the labels.

Usage:
    CUDA_VISIBLE_DEVICES=7 RAY_TMPDIR=/datahdd2/tmp/ray_eval python debug_ego_accel_override.py \
        --model "Latest=/a.ckpt" --output-dir <dir>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import visualize_ped_comparison as V
from debug_ped_box_control import PED_V2, MPH, load_sample, predict

AX = 5          # index of longitudinal acceleration in the ego_status row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--accels", type=float, nargs="+", default=[0.0, -1.0, -2.0, -3.0, -4.0])
    ap.add_argument("--max-per-group", type=int, default=150)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(PED_V2 / "analysis/05_selection/val_samples.csv")
    frames = pd.read_csv(PED_V2 / "analysis/02_ped_frames.csv",
                         usecols=["scene_name", "frame_idx", "ego_speed_ms", "ped_local_x_m", "ped_local_y_m",
                                  "n_inserted_peds"])
    s = val.merge(frames, on=["scene_name", "frame_idx"])
    s = s[(s.ego_speed_ms * MPH >= 10) & (s.n_inserted_peds == 1)]
    groups = {"native": s[(s.ped_local_x_m >= 4) & (s.ped_local_x_m <= 12) & (s.ped_local_y_m.abs() < 1.5)],
              "far": s[s.ped_local_x_m >= 20]}
    groups = {k: v.sample(min(len(v), args.max_per_group), random_state=0) for k, v in groups.items()}
    agents = [(m.split("=", 1)[0], V.load_agent(m.split("=", 1)[1])) for m in args.model]

    rows = []
    for group, df in groups.items():
        for r in df.itertuples():
            feat, label = load_sample(r.scene_name, int(r.frame_idx), r.token)
            ax0 = float(feat["ego_status"][-1, AX])
            lv = np.linalg.norm(np.diff(np.vstack([[0, 0], label[:, :2]]), axis=0), axis=1) / 0.5
            for name, agent in agents:
                for a in [None] + list(args.accels):
                    f = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in feat.items()}
                    if a is not None:
                        f["ego_status"][-1, AX] = a
                    traj = predict(agent, f)
                    v = np.linalg.norm(np.diff(np.vstack([[0, 0], traj[:, :2]]), axis=0), axis=1) / 0.5
                    rows.append(dict(group=group, model=name, token=r.token, ego_mph=r.ego_speed_ms * MPH,
                                     ped_x=r.ped_local_x_m, recorded_ax=ax0,
                                     accel_input="recorded" if a is None else f"{a:+.0f} m/s2",
                                     dist_5s=float(traj[:, 0].max()), end_mph=float(v[-1] * MPH),
                                     label_dist_5s=float(label[:, 0].max()), label_end_mph=float(lv[-1] * MPH)))
    d = pd.DataFrame(rows)
    d.to_csv(out / "accel_override_results.csv", index=False)

    def summ(g):
        return pd.Series({"samples": len(g), "ego mph now": g.ego_mph.mean(),
                          "distance in 5 s (m)": g.dist_5s.mean(), "speed at 5 s (mph)": g.end_mph.mean(),
                          "near-stop (<5 m) %": 100 * (g.dist_5s < 5).mean(),
                          "stops before the pedestrian %": 100 * (g.dist_5s < g.ped_x - 4.1).mean()})
    pd.set_option("display.width", 250)
    t = d.groupby(["model", "group", "accel_input"], sort=False).apply(summ).round(1)
    lab = d.drop_duplicates(["group", "token"]).groupby("group").agg(
        samples=("token", "size"), recorded_ax_mean=("recorded_ax", "mean"),
        label_dist_5s=("label_dist_5s", "mean"), label_end_mph=("label_end_mph", "mean")).round(2)
    text = t.to_string() + "\n\nLabels and recorded acceleration per group:\n" + lab.to_string() + "\n"
    (out / "accel_override_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
