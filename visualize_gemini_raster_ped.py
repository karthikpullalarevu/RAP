#!/usr/bin/env python3
"""Gemini scenes as rasters: draw the added pedestrian into the scene's raster as a box.

For a debug test of "does the model react to the pedestrian in raster form?". The box is drawn the
way the InterPlan training rasters draw agents: a cuboid with one flat colour per face
(front pink, back cyan, left purple, right blue, top indigo), pedestrian size 0.9 x 0.79 x 1.79 m,
standing on the ground (z = 0 in the lidar frame).

Makes a 2x2 picture per scene:  original photo | original raster
                                 pedestrian photo | raster with the pedestrian box

Usage:
    python visualize_gemini_raster_ped.py --output-dir <dir> [--scenes token_v0 token_v1 ...]
"""

import argparse
import json
import math
import pickle
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/datahdd/karthik/sem_1/RAP_analysis_visualedits")
LIDAR_RESULTS = BASE / "images/lidar/results.json"
ABLATION_DIR = BASE / "pedestrian_ablation"
GENERATED_DIR = BASE / "generated_pedestrian"
TEST_LOGS = Path("/datahdd2/karthik/dataset/nuplan-v1.1/navsim_logs/test")

PED_LWH = (0.9, 0.79, 1.79)                 # pedestrian box in the InterPlan sims
PED_Z0 = 0.07                               # box bottom in the lidar frame (measured on a training raster)
FACE_RGB = {"front": (209, 31, 112), "back": (68, 182, 217), "left": (108, 8, 174),
            "right": (59, 86, 211), "top": (58, 10, 154)}
PHOTO_H = 1080                              # saved rasters are 1120 rows: 1080 + 40 rows of padding


def front_lidar2img(token):
    """3x4 lidar-frame -> front-image-pixel matrix from the navsim log (as lidar_pedestrian_depth.py)."""
    case = json.loads((ABLATION_DIR / token / "case.json").read_text())
    frames = pickle.load(open(TEST_LOGS / f"{case['log_name']}.pkl", "rb"))
    cal = next(f for f in frames if f["token"] == token)["cams"]["CAM_F0"]
    Rinv = np.linalg.inv(np.array(cal["sensor2lidar_rotation"], dtype=np.float64))
    t = np.array(cal["sensor2lidar_translation"], dtype=np.float64)
    return np.array(cal["cam_intrinsic"], dtype=np.float64) @ np.hstack([Rinv, (-Rinv @ t).reshape(3, 1)])


def ped_positions(dets, M):
    """(x, y) in the lidar frame for every detected person: lidar range along the image bearing."""
    out = []
    for d in dets:
        if d.get("distance_m") is None or d.get("bbox") is None:
            continue
        u, v = (d["bbox"][0] + d["bbox"][2]) / 2.0, d["bbox"][3]
        A = np.array([[M[0, 0] - u * M[2, 0], M[0, 1] - u * M[2, 1]],
                      [M[1, 0] - v * M[2, 0], M[1, 1] - v * M[2, 1]]])
        b = np.array([u * M[2, 3] - M[0, 3], v * M[2, 3] - M[1, 3]])
        r = float(d["distance_m"])
        x, y = np.linalg.solve(A, b) if abs(np.linalg.det(A)) > 1e-6 else (r, 0.0)
        n = math.hypot(x, y)
        out.append((r * x / n, r * y / n) if n > 1e-6 and x > 0 else (r, 0.0))
    return sorted(out, key=lambda p: math.hypot(*p))


def draw_ped_cuboid(img_rgb, M, xy, heading=math.pi / 2, lwh=PED_LWH):
    """Paint a pedestrian cuboid into an RGB raster (in place). heading: pi/2 = walking to the left."""
    L, W, H = lwh
    c, s = math.cos(heading), math.sin(heading)
    # corners: index = (front/back, left/right, bottom/top)
    pts = {}
    for fx in (1, -1):
        for ly in (1, -1):
            for tz in (0, 1):
                px = xy[0] + fx * L / 2 * c - ly * W / 2 * s
                py = xy[1] + fx * L / 2 * s + ly * W / 2 * c
                pts[(fx, ly, tz)] = np.array([px, py, PED_Z0 + tz * H, 1.0])
    faces = {
        "front": [(1, 1, 0), (1, -1, 0), (1, -1, 1), (1, 1, 1)],
        "back": [(-1, 1, 0), (-1, -1, 0), (-1, -1, 1), (-1, 1, 1)],
        "left": [(1, 1, 0), (-1, 1, 0), (-1, 1, 1), (1, 1, 1)],
        "right": [(1, -1, 0), (-1, -1, 0), (-1, -1, 1), (1, -1, 1)],
        "top": [(1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, 1)],
    }
    order = []
    for name, idx in faces.items():
        P = np.array([pts[i] for i in idx])
        proj = (M @ P.T).T
        if (proj[:, 2] <= 0.3).any():
            continue
        order.append((proj[:, 2].mean(), name, (proj[:, :2] / proj[:, 2:3]).astype(np.int32)))
    for _, name, poly in sorted(order, reverse=True):          # far faces first
        cv2.fillPoly(img_rgb, [poly], FACE_RGB[name], lineType=cv2.LINE_AA)
    return img_rgb


def load_raster(token):
    r = cv2.cvtColor(cv2.imread(str(ABLATION_DIR / token / "raster" / "front.jpg")), cv2.COLOR_BGR2RGB)
    return r[:PHOTO_H].copy()                                   # drop the padding rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scenes", nargs="+", required=True, help="token_vN, e.g. 045f4ecec234564b_v0")
    ap.add_argument("--all-peds", action="store_true", help="Draw every detected person, not only the nearest")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dets = {(e["token"], e["suffix"]): e["detections"] for e in json.load(open(LIDAR_RESULTS))}

    for sc in args.scenes:
        token, v = sc.rsplit("_v", 1)
        M = front_lidar2img(token)
        peds = ped_positions(dets.get((token, f"front_{v}"), []), M)
        peds = peds if args.all_peds else peds[:1]
        orig = cv2.cvtColor(cv2.imread(str(ABLATION_DIR / token / "rgb" / "front.jpg")), cv2.COLOR_BGR2RGB)
        ped_photo = cv2.cvtColor(cv2.imread(str(GENERATED_DIR / f"{token}_front_{v}.jpg")), cv2.COLOR_BGR2RGB)
        raster = load_raster(token)
        raster_ped = raster.copy()
        for xy in sorted(peds, key=lambda p: -math.hypot(*p)):  # far first
            draw_ped_cuboid(raster_ped, M, xy)

        fig, axes = plt.subplots(2, 2, figsize=(17, 10))
        where = ", ".join(f"{math.hypot(*p):.1f} m" for p in peds)
        panels = [(orig, "Original photo"), (raster, "Original raster"),
                  (ped_photo, "Photo with pedestrian (Gemini)"),
                  (raster_ped, f"Raster with pedestrian box added ({where} away)")]
        for ax, (im, title) in zip(axes.ravel(), panels):
            ax.imshow(im)
            ax.set_title(title, fontsize=13)
            ax.axis("off")
        fig.suptitle(f"{token} · variant {v}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"{token}_v{v}_raster_ped.jpg", dpi=80)
        plt.close(fig)
        print(f"saved {token}_v{v}: {len(peds)} box(es) at {where}")


if __name__ == "__main__":
    main()
