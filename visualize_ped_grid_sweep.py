#!/usr/bin/env python3
"""Where does the model stop for a pedestrian? Sweep a pedestrian box over a grid of positions.

Gemini test scenes, raster input in every camera slot (as in visualize_gemini_rgb_vs_raster.py). For each scene
the model runs once without a pedestrian and once per grid position, with a pedestrian box (InterPlan size,
facing left) drawn into every camera that sees it. Ego acceleration input is set to 0 for every model.

Per model, heatmaps over (metres ahead x metres sideways):
  - stop rate: plan ends below 1 mph
  - plan shortening: 5 s plan length without pedestrian - with pedestrian (m)

Usage:
    CUDA_VISIBLE_DEVICES=0 python visualize_ped_grid_sweep.py \
        --model "E4=/a.ckpt::final" --model "E7=/b.ckpt::gated_pdms::noaccel" --scenes 20 --output-dir <dir>
"""

import argparse
import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import visualize_ped_comparison as V
import visualize_ped_gemini_bev as B
import visualize_gemini_raster_ped as G
from visualize_gemini_rgb_vs_raster import predict, end_mph, box_centre, CAMS

AHEAD = [5, 10, 15, 20, 30]                       # metres ahead of ego (far side of the box)
SIDE = [6, 3, 1.5, 0, -1.5, -3, -6]               # metres to the left (+) / right (-)
STOP_MPH = 1.0


def make_input(loader, token, rasters):
    a = loader.get_agent_input_from_token(token)
    a.ego_statuses[-1].ego_acceleration[0] = 0.0   # no braking cue
    a.ego_statuses[-1].ego_acceleration[1] = 0.0
    cams = a.cameras[-1]
    for name in CAMS:
        getattr(cams, name).image = rasters[name]
    return a


def heatmaps(df, models, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("stopped", "stop rate with pedestrian", "Blues", (0, 1), "{:.0%}"),
               ("shortening", "plan shortened by pedestrian (m)", "Oranges", None, "{:.1f}")]
    lim = {"shortening": (0, max(1.0, df.groupby(["model", "ahead", "side"]).shortening.mean().max()))}
    fig, axes = plt.subplots(len(metrics), len(models), figsize=(4.3 * len(models), 8.6), squeeze=False)
    for j, m in enumerate(models):
        d = df[df.model == m]
        for i, (col, title, cmap, vlim, fmt) in enumerate(metrics):
            ax = axes[i, j]
            grid = d.groupby(["ahead", "side"])[col].mean().unstack().reindex(index=AHEAD[::-1], columns=SIDE)
            lo, hi = vlim or lim[col]
            ax.imshow(grid.values, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
            for (r, c), v in np.ndenumerate(grid.values):
                ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=9,
                        color="white" if (v - lo) / (hi - lo + 1e-9) > 0.6 else "#222222")
            ax.set_xticks(range(len(SIDE)), [f"{abs(s):g}{'L' if s > 0 else 'R' if s < 0 else ''}" for s in SIDE], fontsize=8)
            ax.set_yticks(range(len(AHEAD)), [f"{a} m" for a in AHEAD[::-1]], fontsize=8)
            ax.axvline(1.5, color="#222222", lw=0.8, ls=":"); ax.axvline(4.5, color="#222222", lw=0.8, ls=":")
            base = d.drop_duplicates("token").stopped_no_ped.mean()
            ax.set_title(f"{m}: {title}" + (f"\n(no pedestrian: {base:.0%})" if col == "stopped" else ""), fontsize=10)
            if i == len(metrics) - 1:
                ax.set_xlabel("metres sideways (L = left, R = right)", fontsize=8)
            if j == 0:
                ax.set_ylabel("metres ahead", fontsize=8)
    fig.suptitle(f"Pedestrian box swept over positions · {df.token.nunique()} Gemini raster scenes · braking input 0\n"
                 "dotted lines: our car is 2.3 m wide, so 0 and 1.5 m are in our path", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help='"Label=/path.ckpt::pick_mode[::noaccel]"')
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scenes", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def load(spec):
        p = spec.split("::")
        return V.load_agent(p[0], p[1] if len(p) > 1 else "final", len(p) > 2 and p[2] == "noaccel")
    models = [m.split("=", 1) for m in args.model]
    agents = [(name, load(ckpt)) for name, ckpt in models]

    from navsim.common.dataloader import SceneLoader, SceneFilter
    tokens = {d for d in os.listdir(B.ABLATION_DIR) if os.path.exists(os.path.join(B.ABLATION_DIR, d, "case.json"))}
    logs, found = set(), set()
    for pkl in sorted(glob.glob(os.path.join(B.TEST_LOGS, "*.pkl"))):
        for fr in pickle.load(open(pkl, "rb")):
            if fr["token"] in tokens:
                logs.add(Path(pkl).stem)
                found.add(fr["token"])
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True)
    sf.log_names, sf.tokens = sorted(logs), sorted(found)
    loader = SceneLoader(sensor_blobs_path=Path(B.TEST_BLOBS), data_path=Path(B.TEST_LOGS), scene_filter=sf,
                         sensor_config=agents[0][1].get_sensor_config())

    rows, n_done = [], 0
    for token in sorted(loader.tokens):
        if n_done >= args.scenes:
            break
        base_in = loader.get_agent_input_from_token(token)
        cams = {n: getattr(base_in.cameras[-1], n) for n in CAMS}
        if cams["cam_f0"].image is None or cams["cam_f0"].image.max() == 0:
            continue
        base = {n: np.asarray(c.rendered_image).astype(np.uint8) for n, c in cams.items()}
        mats = {n: B.lidar2img(c) for n, c in cams.items()}
        no_ped = {name: predict(agent, make_input(loader, token, base)) for name, agent in agents}
        for ahead in AHEAD:
            for side in SIDE:
                xy = (float(ahead), float(side))
                rasters, seen = {}, []
                for n in CAMS:
                    r = G.draw_ped_cuboid(base[n].copy(), mats[n], xy)
                    rasters[n] = r
                    if (r != base[n]).any():
                        seen.append(n)
                if ahead == 15 and side == 0:
                    import cv2
                    cv2.imwrite(str(out / f"{token}_front_15m.jpg"), cv2.cvtColor(rasters["cam_f0"], cv2.COLOR_RGB2BGR))
                centre = box_centre(xy)
                for name, agent in agents:
                    traj = predict(agent, make_input(loader, token, rasters))
                    gap, _, _ = B.closest_gap(traj, centre)
                    d0, d1 = float(no_ped[name][:, 0].max()), float(traj[:, 0].max())
                    rows.append(dict(token=token, model=name, ahead=ahead, side=side, seen_in=",".join(seen),
                                     dist_5s_no_ped=d0, dist_5s_ped=d1, shortening=d0 - d1,
                                     end_mph_no_ped=end_mph(no_ped[name]), end_mph_ped=end_mph(traj),
                                     stopped_no_ped=end_mph(no_ped[name]) < STOP_MPH, stopped=end_mph(traj) < STOP_MPH,
                                     closest_gap=gap, hits=int(gap <= 0),
                                     ego_mph=float(base_in.ego_statuses[-1].ego_velocity[0]) * B.MPH,
                                     traj=json.dumps(np.round(traj, 3).tolist()),
                                     traj_no_ped=json.dumps(np.round(no_ped[name], 3).tolist())))
        n_done += 1
        print(f"done {n_done}/{args.scenes} {token}", flush=True)
        pd.DataFrame(rows).to_csv(out / "grid_sweep.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out / "grid_sweep.csv", index=False)
    heatmaps(df, [m[0] for m in models], out / "grid_heatmaps.png")
    print(f"Saved grid_sweep.csv and grid_heatmaps.png to {out}")


if __name__ == "__main__":
    main()
