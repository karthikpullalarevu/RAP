#!/usr/bin/env python3
"""Control for the Gemini raster test: does a hand-drawn pedestrian box make the model stop
inside the InterPlan rasters it was trained on?

Uses held-out ped_v2 validation samples (cache/val), all with ego moving at >= 10 mph:
  FAR      - the real pedestrian is >= 20 m ahead. Run as is, then with a box drawn in ego's
             path at each of --box-distances (far side of the box at that distance).
  NATIVE   - a real pedestrian (drawn by the InterPlan renderer) is 4-12 m ahead in ego's path.
             Run as is. This is what a genuine close pedestrian does to the model.

Distances here are from the car centre, the convention the InterPlan rasters use.

Usage:
    CUDA_VISIBLE_DEVICES=7 RAY_TMPDIR=/datahdd2/tmp/ray_eval python debug_ped_box_control.py \
        --model "Latest=/a.ckpt" --model "Baseline=/b.ckpt" --output-dir <dir>
"""

import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import visualize_ped_comparison as V
import visualize_gemini_raster_ped as G

PED_V2 = Path("/datahdd2/karthik/ped_v2")
MPH = 2.23694
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
IMG_ROWS = 432                                   # rows 432..447 of the cached image are padding


def load_sample(scene, frame, token):
    d = PED_V2 / "cache/val" / f"{scene}_{frame:04d}" / token
    feat = pickle.load(gzip.open(d / "rap_feature.gz"))
    target = pickle.load(gzip.open(d / "rap_target.gz"))
    return feat, target["trajectory"].float().numpy()


def front_image(feat):
    """Front raster as uint8 RGB (432 x 768), from the normalised cached tensor."""
    t = feat["camera_feature"][1].numpy().transpose(1, 2, 0)[:IMG_ROWS]
    return np.clip(t * STD + MEAN, 0, 255).astype(np.uint8)


def with_front(feat, img):
    """Copy of the features with the front raster replaced by img (uint8 RGB, 432 x 768)."""
    out = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in feat.items()}
    norm = (img.astype(np.float32) - MEAN) / STD
    out["camera_feature"][1, :, :IMG_ROWS] = torch.from_numpy(norm.transpose(2, 0, 1))
    return out


def predict(agent, feat):
    with torch.no_grad():
        f = {k: v.unsqueeze(0).cuda() for k, v in feat.items()
             if isinstance(v, torch.Tensor) and k not in ("camera_valid", "rendered_camera_feature")}
        return agent.forward(f)["trajectory"].squeeze(0).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--box-distances", type=float, nargs="+", default=[6.0, 9.0, 12.0])
    ap.add_argument("--max-per-group", type=int, default=150)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    val = pd.read_csv(PED_V2 / "analysis/05_selection/val_samples.csv")
    frames = pd.read_csv(PED_V2 / "analysis/02_ped_frames.csv",
                         usecols=["scene_name", "frame_idx", "ego_speed_ms", "ped_local_x_m", "ped_local_y_m",
                                  "ped_side", "n_inserted_peds"])
    s = val.merge(frames, on=["scene_name", "frame_idx"])
    s = s[(s.ego_speed_ms * MPH >= 10) & (s.n_inserted_peds == 1)]
    far = s[s.ped_local_x_m >= 20]
    native = s[(s.ped_local_x_m >= 4) & (s.ped_local_x_m <= 12) & (s.ped_local_y_m.abs() < 1.5)]
    far = far.sample(min(len(far), args.max_per_group), random_state=0)
    native = native.sample(min(len(native), args.max_per_group), random_state=0)
    print(f"FAR samples: {len(far)}   NATIVE close-pedestrian samples: {len(native)}", flush=True)

    agents = [(m.split("=", 1)[0], V.load_agent(m.split("=", 1)[1])) for m in args.model]
    rows, examples = [], []
    for group, df in (("far", far), ("native", native)):
        for r in df.itertuples():
            feat, label = load_sample(r.scene_name, int(r.frame_idx), r.token)
            M = feat["lidar2img"][1][:3].numpy().astype(np.float64)       # for the 0.4-scaled image
            img0 = front_image(feat)
            variants = [("as is", feat, np.nan)]
            if group == "far":
                for dist in args.box_distances:
                    img = G.draw_ped_cuboid(img0.copy(), M, (dist, 0.0))
                    variants.append((f"box at {dist:g} m", with_front(feat, img), dist))
                    if len(examples) < 6 and dist == args.box_distances[0]:
                        examples.append(np.concatenate([img0, img], axis=1))
            for name, agent in agents:
                for vname, f, dist in variants:
                    traj = predict(agent, f)
                    v = np.linalg.norm(np.diff(np.vstack([[0, 0], traj[:, :2]]), axis=0), axis=1) / 0.5
                    rows.append(dict(group=group, variant=vname, model=name, token=r.token,
                                     ego_mph=r.ego_speed_ms * MPH, real_ped_x=r.ped_local_x_m, box_x=dist,
                                     dist_5s=float(traj[:, 0].max()), end_mph=float(v[-1] * MPH),
                                     min_mph=float(v.min() * MPH), label_dist_5s=float(label[:, 0].max())))
    df = pd.DataFrame(rows)
    df.to_csv(out / "control_results.csv", index=False)

    import cv2
    if examples:
        sheet = np.concatenate(examples, axis=0)
        cv2.imwrite(str(out / "control_examples_before_after.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    def summ(g):
        return pd.Series({"samples": len(g), "ego mph now": g.ego_mph.mean(), "distance in 5 s (m)": g.dist_5s.mean(),
                          "speed at 5 s (mph)": g.end_mph.mean(), "near-stop (<5 m) %": 100 * (g.dist_5s < 5).mean(),
                          "stops before the box/ped %": 100 * (g.dist_5s < g.box_x.fillna(g.real_ped_x) - 4.1).mean(),
                          "label distance in 5 s (m)": g.label_dist_5s.mean()})
    table = df.groupby(["model", "group", "variant"], sort=False).apply(summ).round(1)
    pd.set_option("display.width", 250)
    text = table.to_string()
    (out / "control_summary.txt").write_text(text + "\n")
    print(text)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
