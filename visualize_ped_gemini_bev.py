#!/usr/bin/env python3
"""Gemini pedestrian scenes: front view + bird's-eye view with the pedestrian's position.

For each Gemini-edited image (a pedestrian added to a real navtest front photo) this draws
  left  - the edited photo with each model's predicted path projected onto it
  right - a top-down view: ego, the pedestrian (from last semester's lidar distance)
          and each model's path
The closest gap between each path and the pedestrian is saved in the CSV only, not drawn.

Pedestrian position follows last semester's method (sem_1/.../RAP_v2/eval_ablation.py):
range from lidar ground returns at the feet of the YOLO box (images/lidar/results.json),
direction from the bottom-centre of the box assuming it is on the ground.
The pedestrian is treated as standing still - there is only one image.

Usage:
    CUDA_VISIBLE_DEVICES=7 RAY_TMPDIR=/datahdd2/tmp/ray_eval python visualize_ped_gemini_bev.py \
        --model "Baseline=/path/a.ckpt" --model "New=/path/b.ckpt" --output-dir <dir>
"""

import argparse
import glob
import json
import math
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import visualize_ped_comparison as V   # load_agent, project_trajectory_to_image (also sets env vars)

LIDAR_RESULTS = "/datahdd/karthik/sem_1/RAP_analysis_visualedits/images/lidar/results.json"
ABLATION_DIR = "/datahdd/karthik/sem_1/RAP_analysis_visualedits/pedestrian_ablation"
GENERATED_DIR = "/datahdd/karthik/sem_1/RAP_analysis_visualedits/generated_pedestrian"
TEST_LOGS = "/datahdd2/karthik/dataset/nuplan-v1.1/navsim_logs/test"
TEST_BLOBS = "/datahdd2/karthik/dataset/nuplan-v1.1/sensor_blobs/test"

HALF_LEN, HALF_WID, AXLE_TO_CENTER = 2.588, 1.1485, 1.461   # ego box, rear axle -> box centre
PED_RADIUS = 0.35
MPH = 2.23694
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]                  # fixed order, one per model
INK, MUTED = "#222222", "#777777"


def predict(agent, agent_input, front_path=None):
    """Predicted poses (10, 3) = x forward, y left, heading; optionally with the front photo replaced."""
    if front_path is not None:
        # RGB, like the data loader (PIL). cv2.imread alone gives BGR, i.e. red and blue swapped.
        agent_input.cameras[-1].cam_f0.image = cv2.cvtColor(cv2.imread(front_path), cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        feats = {}
        for b in agent.get_feature_builders():
            feats.update(b.compute_features(agent_input))
        feats = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v for k, v in feats.items()}
        return agent.forward(feats)["trajectory"].squeeze(0).float().cpu().numpy()


def lidar2img(cam):
    """3x4 matrix taking ego-frame points to front-image pixels (same as project_trajectory_to_image)."""
    Rinv = np.linalg.inv(cam.sensor2lidar_rotation)
    return cam.intrinsics @ np.hstack([Rinv, (-Rinv @ cam.sensor2lidar_translation).reshape(3, 1)])


def ped_position(dets, M):
    """(x, y) of the nearest detected person in the ego frame, or None."""
    valid = [d for d in dets if d.get("distance_m") is not None and d.get("bbox") is not None]
    if not valid:
        return None
    d = min(valid, key=lambda d: d["distance_m"])
    u, v = (d["bbox"][0] + d["bbox"][2]) / 2.0, d["bbox"][3]        # bottom-centre of the box, on the ground
    A = np.array([[M[0, 0] - u * M[2, 0], M[0, 1] - u * M[2, 1]],
                  [M[1, 0] - v * M[2, 0], M[1, 1] - v * M[2, 1]]])
    b = np.array([u * M[2, 3] - M[0, 3], v * M[2, 3] - M[1, 3]])
    r = float(d["distance_m"])
    if abs(np.linalg.det(A)) < 1e-6:
        return r, 0.0
    x, y = np.linalg.solve(A, b)
    n = math.hypot(x, y)
    return (r * x / n, r * y / n) if n > 1e-6 and x > 0 else (r, 0.0)   # lidar range along the image bearing


def car_corners(pose):
    x, y, h = pose
    c, s = math.cos(h), math.sin(h)
    cx, cy = x + AXLE_TO_CENTER * c, y + AXLE_TO_CENTER * s
    loc = np.array([[HALF_LEN, HALF_WID], [HALF_LEN, -HALF_WID], [-HALF_LEN, -HALF_WID], [-HALF_LEN, HALF_WID]])
    return loc @ np.array([[c, s], [-s, c]]) + [cx, cy]


def closest_gap(traj, ped):
    """Smallest gap (m) between the car body and the pedestrian along the path, and the pose where it happens.

    Path is interpolated to 0.05 s steps from the current pose (0, 0, 0). Negative gap = overlap.
    """
    poses = np.vstack([[0.0, 0.0, 0.0], traj])
    t = np.linspace(0, len(poses) - 1, (len(poses) - 1) * 10 + 1)
    i = np.minimum(t.astype(int), len(poses) - 2)
    w = (t - i)[:, None]
    dense = poses[i] * (1 - w) + poses[i + 1] * w
    best = (1e9, None, None)
    for k, (x, y, h) in enumerate(dense):
        c, s = math.cos(h), math.sin(h)
        dx, dy = ped[0] - (x + AXLE_TO_CENTER * c), ped[1] - (y + AXLE_TO_CENTER * s)
        lx, ly = dx * c + dy * s, -dx * s + dy * c
        gap = math.hypot(max(abs(lx) - HALF_LEN, 0), max(abs(ly) - HALF_WID, 0)) - PED_RADIUS
        if gap < best[0]:
            best = (gap, (x, y, h), t[k] * 0.5)
    return best


def draw(img_path, cam, ped, results, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(19, 8.6), gridspec_kw={"width_ratios": [1.55, 1]})

    # --- left: photo with projected paths ---
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    ax_img.imshow(img)
    H, W = img.shape[:2]
    for r in results:
        px, valid = V.project_trajectory_to_image(r["traj"][:, :2], cam)
        ok = valid & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        ax_img.plot(px[ok, 0], px[ok, 1], "-o", color=r["color"], lw=3, ms=8, mec="white", mew=1.5)
    ax_img.set_xlim(0, W)
    ax_img.set_ylim(H, 0)
    ax_img.axis("off")
    ax_img.set_title("Front camera (pedestrian added with Gemini)", fontsize=13, color=INK)

    # --- right: bird's-eye view, ego pointing up ---
    P = lambda x, y: (-np.asarray(y), np.asarray(x))               # ego (x fwd, y left) -> plot
    ax_bev.add_patch(Polygon(np.column_stack(P(*car_corners((0, 0, 0)).T)), fc="#bbbbbb", ec=INK, lw=1.2))
    ax_bev.annotate("", xy=P(AXLE_TO_CENTER + HALF_LEN - 0.3, 0), xytext=P(AXLE_TO_CENTER - 1.2, 0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5, mutation_scale=11))
    for r in results:
        path = np.vstack([[0, 0], r["traj"][:, :2]])
        ax_bev.plot(*P(path[:, 0], path[:, 1]), "-o", color=r["color"], lw=2.5, ms=5, label=r["name"])
    top = max(30.0, max(r["traj"][:, 0].max() for r in results) + 3)
    if ped is not None:
        ax_bev.add_patch(Circle(P(*ped), PED_RADIUS, fc=INK, ec="white", lw=1, zorder=6))
        ax_bev.annotate(f"pedestrian\n{math.hypot(*ped):.1f} m away", xy=P(*ped), xytext=(10, 0),
                        textcoords="offset points", fontsize=10, color=INK, va="center", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        top = max(top, ped[0] + 5)
    half = max(8.0, top / 3.2)
    ax_bev.set_xlim(-half, half)
    ax_bev.set_ylim(-4, top)
    ax_bev.set_aspect("equal")
    ax_bev.grid(color="#e5e5e5", lw=0.6)
    ax_bev.set_axisbelow(True)
    ax_bev.set_xlabel("left  ←  metres  →  right", fontsize=10, color=MUTED)
    ax_bev.set_ylabel("metres ahead", fontsize=10, color=MUTED)
    for sp in ("top", "right"):
        ax_bev.spines[sp].set_visible(False)
    ax_bev.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False, fontsize=12, ncol=2)
    ax_bev.set_title("Top-down view", fontsize=13, color=INK)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help='"Label=/path/to.ckpt", up to 3')
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-scenes", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    models = [m.split("=", 1) for m in args.model]
    assert len(models) <= len(COLORS)
    agents = [(name, V.load_agent(ckpt), COLORS[i]) for i, (name, ckpt) in enumerate(models)]
    ped_results = {(e["token"], e["suffix"]): e["detections"] for e in json.load(open(LIDAR_RESULTS))}

    # scene loader for the navtest tokens behind the Gemini scenes (as in visualize_ped_comparison.py)
    from navsim.common.dataloader import SceneLoader, SceneFilter
    tokens = {d for d in os.listdir(ABLATION_DIR) if os.path.exists(os.path.join(ABLATION_DIR, d, "case.json"))}
    logs, found = set(), set()
    for pkl in sorted(glob.glob(os.path.join(TEST_LOGS, "*.pkl"))):
        for fr in pickle.load(open(pkl, "rb")):
            if fr["token"] in tokens:
                logs.add(Path(pkl).stem)
                found.add(fr["token"])
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True)
    sf.log_names, sf.tokens = sorted(logs), sorted(found)
    loader = SceneLoader(sensor_blobs_path=Path(TEST_BLOBS), data_path=Path(TEST_LOGS), scene_filter=sf,
                         sensor_config=agents[0][1].get_sensor_config())
    scene_tokens = sorted(loader.tokens)
    if args.max_scenes:
        scene_tokens = scene_tokens[:args.max_scenes]
    print(f"{len(scene_tokens)} scenes, models: {[m[0] for m in models]}")

    rows = []
    for token in scene_tokens:
        cam = loader.get_agent_input_from_token(token).cameras[-1].cam_f0
        if cam.image is None or cam.image.max() == 0:
            continue
        M = lidar2img(cam)
        ego = loader.get_agent_input_from_token(token).ego_statuses[-1]
        ego_mph = float(ego.ego_velocity[0]) * MPH
        orig = {name: predict(agent, loader.get_agent_input_from_token(token)) for name, agent, _ in agents}
        for v in range(3):
            front = os.path.join(GENERATED_DIR, f"{token}_front_{v}.jpg")
            if not os.path.exists(front):
                continue
            ped = ped_position(ped_results.get((token, f"front_{v}"), []), M)
            results = []
            for name, agent, color in agents:
                traj = predict(agent, loader.get_agent_input_from_token(token), front)
                gap, pose, t_close = closest_gap(traj, ped) if ped is not None else (float("nan"), None, float("nan"))
                results.append(dict(name=name, color=color, traj=traj, gap=gap, pose=pose,
                                    dist=float(traj[:, 0].max())))
                rows.append(dict(token=token, variant=v, model=name, ego_mph=ego_mph,
                                 ped_x=ped[0] if ped else np.nan, ped_y=ped[1] if ped else np.nan,
                                 ped_dist=math.hypot(*ped) if ped else np.nan,
                                 dist_5s_orig=float(orig[name][:, 0].max()), dist_5s_ped=float(traj[:, 0].max()),
                                 closest_gap=gap, t_closest=t_close, hits=int(gap <= 0) if ped else np.nan,
                                 traj=json.dumps(np.round(traj, 3).tolist())))
            draw(front, cam, ped, results, f"{token} · variant {v} · ego at {ego_mph:.0f} mph",
                 str(out / f"{token}_v{v}.jpg"))
        print(f"done {token}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "gemini_bev_summary.csv", index=False)
    print(f"Saved {len(df) // max(len(models), 1)} pictures and gemini_bev_summary.csv to {out}")


if __name__ == "__main__":
    main()
