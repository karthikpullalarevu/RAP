#!/usr/bin/env python3
"""Gemini pedestrian scenes: does the model react to the pedestrian as a photo vs as a raster?

For each Gemini image the model is run on
  - photos:  the 4 camera photos, front one = Gemini photo with the pedestrian
  - rasters: the 4 camera rasters, front one = scene raster with a pedestrian box drawn in
and also on both without the pedestrian, for the numbers in the CSV.

Picture per image (2x2):   photo + predicted paths    | top-down view
                           raster + predicted paths   | top-down view

The pedestrian box is drawn like the InterPlan training rasters (visualize_gemini_raster_ped.py),
`--closer` metres nearer than the lidar-measured distance, far side of the box at that distance.

Usage:
    CUDA_VISIBLE_DEVICES=7 RAY_TMPDIR=/datahdd2/tmp/ray_eval python visualize_gemini_rgb_vs_raster.py \
        --model "Baseline=/a.ckpt" --model "Latest=/b.ckpt" --closer 1.5 --output-dir <dir>
"""

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import visualize_ped_comparison as V          # load_agent, project_trajectory_to_image
import visualize_ped_gemini_bev as B          # scene loader pieces, geometry helpers, colours
import visualize_gemini_raster_ped as G       # draw_ped_cuboid

CAMS = ["cam_b0", "cam_f0", "cam_l0", "cam_r0"]
INK, MUTED = B.INK, B.MUTED


def predict(agent, agent_input):
    with torch.no_grad():
        feats = {}
        for b in agent.get_feature_builders():
            feats.update(b.compute_features(agent_input))
        feats = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v for k, v in feats.items()}
        return agent.forward(feats)["trajectory"].squeeze(0).float().cpu().numpy()


EGO_MPH = None      # set by --ego-mph: overrides the speed the model is told ego has now
EGO_ACCEL = None    # set by --ego-accel: overrides ego's current longitudinal acceleration input (m/s2)


def end_mph(traj):
    """Speed over the last half second of the predicted path."""
    return float(np.linalg.norm(traj[-1, :2] - traj[-2, :2]) / 0.5 * B.MPH)


def make_input(loader, token, mode, front=None):
    """mode 'photo': real photos (front optionally replaced). mode 'raster': rasters in every camera slot."""
    a = loader.get_agent_input_from_token(token)
    if EGO_MPH is not None:
        a.ego_statuses[-1].ego_velocity[0] = EGO_MPH / B.MPH
        a.ego_statuses[-1].ego_velocity[1] = 0.0
    if EGO_ACCEL is not None:
        a.ego_statuses[-1].ego_acceleration[0] = EGO_ACCEL
    cams = a.cameras[-1]
    if mode == "raster":
        for name in CAMS:
            cam = getattr(cams, name)
            cam.image = np.asarray(cam.rendered_image).astype(np.uint8)
    if front is not None:
        cams.cam_f0.image = front
    return a


def box_centre(xy):
    """Centre of the drawn box: far side at xy, heading left (matches G.draw_ped_cuboid)."""
    n = math.hypot(*xy)
    half = abs(xy[1] / n) * G.PED_LWH[0] / 2 + abs(xy[0] / n) * G.PED_LWH[1] / 2
    return xy[0] * (1 - half / n), xy[1] * (1 - half / n)


def draw(rows, title, out_path):
    """rows: list of (image_rgb, cam, ped_xy, ped_label, results, row_title)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    fig, axes = plt.subplots(2, 2, figsize=(18, 13.5), gridspec_kw={"width_ratios": [1.75, 1]})
    P = lambda x, y: (-np.asarray(y), np.asarray(x))
    top = max(30.0, max(r["traj"][:, 0].max() for row in rows for r in row[4]) + 3,
              max(row[2][0] for row in rows) + 5)
    half = max(8.0, top / 3.2)
    for (ax_img, ax_bev), (img, cam, ped, ped_label, results, row_title) in zip(axes, rows):
        ax_img.imshow(img)
        H, W = img.shape[:2]
        for r in results:
            px, valid = V.project_trajectory_to_image(r["traj"][:, :2], cam)
            ok = valid & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
            ax_img.plot(px[ok, 0], px[ok, 1], "-o", color=r["color"], lw=3, ms=8, mec="white", mew=1.5)
        ax_img.set_xlim(0, W)
        ax_img.set_ylim(H, 0)
        ax_img.axis("off")
        ax_img.set_title(row_title, fontsize=14, color=INK)

        ax_bev.add_patch(Polygon(np.column_stack(P(*B.car_corners((0, 0, 0)).T)), fc="#bbbbbb", ec=INK, lw=1.2))
        ax_bev.annotate("", xy=P(B.AXLE_TO_CENTER + B.HALF_LEN - 0.3, 0), xytext=P(B.AXLE_TO_CENTER - 1.2, 0),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5, mutation_scale=11))
        for r in results:
            path = np.vstack([[0, 0], r["traj"][:, :2]])
            ax_bev.plot(*P(path[:, 0], path[:, 1]), "-o", color=r["color"], lw=2.5, ms=5, label=r["name"])
        ax_bev.add_patch(Circle(P(*ped), B.PED_RADIUS, fc=INK, ec="white", lw=1, zorder=6))
        ax_bev.annotate(ped_label, xy=P(*ped), xytext=(10, 0), textcoords="offset points", fontsize=10.5,
                        color=INK, va="center", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax_bev.set_xlim(-half, half)
        ax_bev.set_ylim(-4, top)
        ax_bev.set_aspect("equal")
        ax_bev.grid(color="#e5e5e5", lw=0.6)
        ax_bev.set_axisbelow(True)
        ax_bev.set_ylabel("metres ahead", fontsize=10, color=MUTED)
        for sp in ("top", "right"):
            ax_bev.spines[sp].set_visible(False)
        ax_bev.set_title("Top-down view", fontsize=13, color=INK)
    axes[1, 1].set_xlabel("left  ←  metres  →  right", fontsize=10, color=MUTED)
    axes[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=12, ncol=2)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help='"Label=/path/to.ckpt", or "Label=/path/to.ckpt::pick_mode" (final, gated_pdms, log_barrier), add "::noaccel" for E7/E8')
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--closer", type=float, default=1.5)
    ap.add_argument("--ego-mph", type=float, default=None, help="Override ego's current speed input (mph)")
    ap.add_argument("--ego-accel", type=float, default=None,
                    help="Override ego's current longitudinal acceleration input, m/s2 (e.g. -2 = braking)")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--walk", choices=["left", "right"], default="left",
                    help="Which way the drawn pedestrian box faces (walks), seen from ego")
    args = ap.parse_args()
    heading = math.pi / 2 if args.walk == "left" else -math.pi / 2
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    global EGO_MPH, EGO_ACCEL
    EGO_MPH, EGO_ACCEL = args.ego_mph, args.ego_accel

    models = [m.split("=", 1) for m in args.model]
    def load(spec):     # "path[::pick_mode[::noaccel]]"
        parts = spec.split("::")
        return V.load_agent(parts[0], parts[1] if len(parts) > 1 else "final", len(parts) > 2 and parts[2] == "noaccel")
    agents = [(name, load(ckpt), B.COLORS[i]) for i, (name, ckpt) in enumerate(models)]
    dets = {(e["token"], e["suffix"]): e["detections"] for e in json.load(open(B.LIDAR_RESULTS))}

    import glob, pickle
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
    scene_tokens = sorted(loader.tokens)[:args.max_scenes] if args.max_scenes else sorted(loader.tokens)
    print(f"{len(scene_tokens)} scenes, models {[m[0] for m in models]}, box {args.closer} m closer, "
          f"walking {args.walk}", flush=True)

    rows = []
    for token in scene_tokens:
        base_in = loader.get_agent_input_from_token(token)
        cam = base_in.cameras[-1].cam_f0
        if cam.image is None or cam.image.max() == 0:
            continue
        M = B.lidar2img(cam)
        raster0 = np.asarray(cam.rendered_image).astype(np.uint8)
        recorded_mph = float(base_in.ego_statuses[-1].ego_velocity[0]) * B.MPH
        ego_mph = recorded_mph if EGO_MPH is None else EGO_MPH
        no_ped = {(name, mode): predict(agent, make_input(loader, token, mode))
                  for name, agent, _ in agents for mode in ("photo", "raster")}
        for v in range(3):
            front_path = os.path.join(B.GENERATED_DIR, f"{token}_front_{v}.jpg")
            ped = B.ped_position(dets.get((token, f"front_{v}"), []), M)
            if not os.path.exists(front_path) or ped is None:
                continue
            n = math.hypot(*ped)
            ped_drawn = (ped[0] * (1 - args.closer / n), ped[1] * (1 - args.closer / n))
            photo = cv2.cvtColor(cv2.imread(front_path), cv2.COLOR_BGR2RGB)
            raster = G.draw_ped_cuboid(raster0.copy(), M, ped_drawn, heading=heading)
            ped_raster = box_centre(ped_drawn)

            res = {"photo": [], "raster": []}
            for name, agent, color in agents:
                for mode, front, ped_xy in (("photo", photo, ped), ("raster", raster, ped_raster)):
                    traj = predict(agent, make_input(loader, token, mode, front))
                    gap, _, _ = B.closest_gap(traj, ped_xy)
                    res[mode].append(dict(name=name, color=color, traj=traj))
                    rows.append(dict(token=token, variant=v, model=name, input=mode, walk=args.walk, ego_accel=args.ego_accel, ego_mph=ego_mph,
                                     ped_x=ped_xy[0], ped_y=ped_xy[1], ped_dist=math.hypot(*ped_xy),
                                     dist_5s_no_ped=float(no_ped[(name, mode)][:, 0].max()),
                                     dist_5s_ped=float(traj[:, 0].max()), closest_gap=gap, hits=int(gap <= 0),
                                     recorded_mph=recorded_mph, end_mph_no_ped=end_mph(no_ped[(name, mode)]),
                                     end_mph_ped=end_mph(traj),
                                     traj=json.dumps(np.round(traj, 3).tolist())))
            draw([(photo, cam, ped, f"pedestrian\n{n:.1f} m away", res["photo"],
                   "Photo input (pedestrian added with Gemini)"),
                  (raster, cam, ped_raster, f"pedestrian box\n{math.hypot(*ped_raster):.1f} m away", res["raster"],
                   f"Raster input (pedestrian box drawn {args.closer:g} m closer than measured, walking {args.walk})")],
                 f"{token} · variant {v} · ego at {ego_mph:.0f} mph"
                 + ("" if EGO_MPH is None else f" (speed input set; recorded {recorded_mph:.0f} mph)")
                 + ("" if EGO_ACCEL is None else f" · braking input {EGO_ACCEL:g} m/s²"),
                 str(out / f"{token}_v{v}.jpg"))
        print(f"done {token}", flush=True)

    pd.DataFrame(rows).to_csv(out / "rgb_vs_raster_summary.csv", index=False)
    print(f"Saved pictures and rgb_vs_raster_summary.csv to {out}")


if __name__ == "__main__":
    main()
