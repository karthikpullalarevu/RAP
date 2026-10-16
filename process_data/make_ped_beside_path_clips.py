#!/usr/bin/env python3
"""Short video clips of training samples where the pedestrian is ahead but 1.5-3 m to the side of our path.

Each clip runs from --before s before the sample moment to --after s after it, at 20 fps (real time):
  left:  InterPlan top-down frame, zoomed around ego (ego = yellow outline, pedestrians = red)
  right: front camera raster (what the model sees), when one exists for that frame
  text:  time from the sample moment, ego speed (mph), gap to the pedestrian, where the pedestrian is

Usage: python process_data/make_ped_beside_path_clips.py --n 6
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAW = Path("/datahdd2/karthik/ped_v2/raw")
PED = Path("/datahdd2/karthik/ped_v2")
MPH = 2.23694
HZ = 20
CROP = 150                       # half-size of the top-down crop, pixels around ego
H = 600                          # output height
INK = (34, 34, 34)


def bev_crop(path):
    img = cv2.imread(str(path))
    cy, cx = img.shape[0] // 2 + 15, img.shape[1] // 2 + 24          # ego sits slightly off the image centre
    return cv2.resize(img[cy - CROP:cy + CROP, cx - CROP:cx + CROP], (H, H), interpolation=cv2.INTER_AREA)


def put(img, text, y, scale=0.8, color=INK, thick=2):
    cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def make_clip(scene, t0, frames, out_path, before, after):
    rdir = RAW / scene / "rendered"
    vdir = next((rdir / "video").glob("*_frames"))
    n_frames = len(list(vdir.glob("frame_*.png")))
    rw = int(H * 16 / 9)
    blank = np.zeros((H, rw, 3), np.uint8)
    info = frames.set_index("frame_idx")
    with tempfile.TemporaryDirectory(dir=str(out_path.parent)) as tmp:
        k = 0
        for i in range(max(0, t0 - before * HZ), min(n_frames, t0 + after * HZ + 1)):
            left = bev_crop(vdir / f"frame_{i:05d}.png")
            rp = rdir / "rasters" / f"frame_{i:04d}" / "CAM_F0.png"
            right = cv2.resize(cv2.imread(str(rp)), (rw, H)) if rp.exists() else blank.copy()
            img = np.hstack([left, right])
            dt = (i - t0) / HZ
            put(img, f"t = {dt:+.1f} s" + ("   <- training sample" if i == t0 else ""), 34,
                color=(52, 104, 235) if i == t0 else INK)
            if i in info.index:
                r = info.loc[i]
                put(img, f"ego {r.ego_speed_ms * MPH:.0f} mph   gap {r.gap_m:.1f} m", 70)
                put(img, f"ped {r.ped_local_x_m:.1f} m ahead, {abs(r.ped_local_y_m):.1f} m "
                         f"{'left' if r.ped_local_y_m > 0 else 'right'}", 104)
            else:
                put(img, "no inserted pedestrian this frame", 70)
            if not rp.exists():
                cv2.putText(img, "no raster for this frame", (H + 20, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (200, 200, 200), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(tmp, f"{k:05d}.png"), img)
            k += 1
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(HZ), "-i", os.path.join(tmp, "%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_path)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--before", type=int, default=3)
    ap.add_argument("--after", type=int, default=5)
    ap.add_argument("--out", type=Path, default=PED / "analysis" / "08_ped_beside_path_clips")
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    names = os.listdir(PED / "cache" / "train")
    t = pd.DataFrame({"scene_name": [n.rsplit("_", 1)[0] for n in names],
                      "frame_idx": [int(n.rsplit("_", 1)[1]) for n in names]})
    p = pd.read_csv(PED / "analysis" / "02_ped_frames.csv",
                    usecols=["scene_name", "frame_idx", "ego_speed_ms", "gap_m", "ped_side", "ped_local_x_m", "ped_local_y_m"])
    m = t.merge(p, on=["scene_name", "frame_idx"])
    y = m.ped_local_y_m.abs()
    cand = m[(m.ped_side == "FRONT") & (y >= 1.5) & (y < 3)]
    picks = cand.drop_duplicates("scene_name").sample(args.n, random_state=args.seed)

    rows = []
    for k, x in enumerate(picks.itertuples(), 1):
        out = args.out / f"{k:02d}_{x.scene_name}_f{x.frame_idx}.mp4"
        make_clip(x.scene_name, int(x.frame_idx), p[p.scene_name == x.scene_name], out, args.before, args.after)
        rows.append(dict(clip=out.name, scene=x.scene_name, frame=x.frame_idx, ego_mph=round(x.ego_speed_ms * MPH, 1),
                         ped_ahead_m=round(x.ped_local_x_m, 1), ped_side_m=round(x.ped_local_y_m, 1), gap_m=x.gap_m))
        print(f"saved {out.name}", flush=True)
    pd.DataFrame(rows).to_csv(args.out / "clips.csv", index=False)
    print(pd.DataFrame(rows).drop(columns="scene").to_string(index=False))


if __name__ == "__main__":
    main()
