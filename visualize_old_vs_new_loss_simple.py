#!/usr/bin/env python3
"""Simple slides: old loss (E4) vs new loss (E6) on the camera image, 4 scenes per input type.

One figure per input (RGB, raster): 2 x 4 grid, top row with the pedestrian, bottom row the same scene without it.
Both plans are drawn on each image; the only text is a small legend. Same scenes as visualize_old_vs_new_loss.py.

Usage:
    CUDA_VISIBLE_DEVICES=0 python visualize_old_vs_new_loss_simple.py --output-dir <dir>
"""

import argparse
import glob
import json
import math
import os
import pickle
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import visualize_ped_comparison as V
import visualize_ped_gemini_bev as B
import visualize_gemini_raster_ped as G
from visualize_gemini_rgb_vs_raster import predict, make_input, box_centre
from visualize_old_vs_new_loss import MODELS as _MODELS, EXAMPLES, CLOSER, SURFACE, INK, GRID, draw_image
from matplotlib.patches import Polygon

COLORS = {"old loss": "#184f95", "new loss": "#eb6834"}          # dark blue, orange
MODELS = [(n, c, m, COLORS[n]) for n, c, m, _ in _MODELS]
PED_C = INK                                                       # pedestrian (orange is taken by the new loss)


def draw_image_full(ax, img, cam, plans):
    """Camera image with both plans, no crop."""
    draw_image(ax, img, cam, plans)
    ax.set_ylim(img.shape[0], 0)


def true_scale_limits(fig, axes):
    """1 m forward = 1 m sideways: widen each top-down view's sideways range to its panel shape."""
    W, H = fig.get_size_inches()
    for ax in axes:
        pos = ax.get_position()
        ratio = (pos.height * H) / (pos.width * W)
        half = (ax._bev_top + 4) / ratio / 2
        ax.set_xlim(-half, half)


def closest_distance_now(ped):
    """Car body (box) to pedestrian edge, now (m)."""
    from shapely.geometry import Point, Polygon as SPoly
    return max(0.0, SPoly(B.car_corners((0, 0, 0))).distance(Point(*ped)) - B.PED_RADIUS)


def draw_bev(ax, plans, ped, true_scale=False, mph=None):
    """Top-down: car, both plans (dots every 0.5 s), pedestrian if given. Car points up."""
    P = lambda x, y: (-np.asarray(y), np.asarray(x))
    ax.add_patch(Polygon(np.column_stack(P(*B.car_corners((0, 0, 0)).T)), fc="#d6d5cf", ec=INK, lw=1, zorder=3))
    for _, traj, color in plans:
        path = np.vstack([[0, 0], traj[:, :2]])
        ax.plot(*P(path[:, 0], path[:, 1]), "-o", color=color, lw=3.5, ms=6, mec="white", mew=1, zorder=4)
    top = max([25.0] + [t[:, 0].max() + 3 for _, t, _ in plans] + ([ped[0] + 4] if ped is not None else []))
    if ped is not None:
        ax.plot(*P(*ped), "o", color=PED_C, ms=12, mec="white", mew=1.5, zorder=5)
    ax.set_ylim(-4, top)
    ax.set_xlim(-8, 8)
    if true_scale:
        ax._bev_top = top                      # sideways range is set by true_scale_limits() once the layout is fixed
    else:
        ax.set_aspect("auto")              # forward axis squeezed to fit; sideways range fixed at +-8 m
    if mph is not None:
        txt = f"vx = {mph:.0f} mph" + (f"\ndistance to pedestrian = {closest_distance_now(ped):.0f} m" if ped is not None else "")
        ax.text(0.03, 0.98, txt, transform=ax.transAxes, fontsize=10.5, color=INK, va="top", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f4f4f2")
    for s_ in ax.spines.values():
        s_.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--grid2x2", action="store_true",
                    help="only with the pedestrian: 2 x 2 scenes, full camera image + true-scale top-down view")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})

    agents = [(n, V.load_agent(str(c), m), col) for n, c, m, col in MODELS]
    dets = {(e["token"], e["suffix"]): e["detections"] for e in json.load(open(B.LIDAR_RESULTS))}
    from navsim.common.dataloader import SceneLoader, SceneFilter
    want = {t for _, t, _, _ in EXAMPLES}
    logs, found = set(), set()
    for pkl in sorted(glob.glob(os.path.join(B.TEST_LOGS, "*.pkl"))):
        for fr in pickle.load(open(pkl, "rb")):
            if fr["token"] in want:
                logs.add(Path(pkl).stem)
                found.add(fr["token"])
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True)
    sf.log_names, sf.tokens = sorted(logs), sorted(found)
    loader = SceneLoader(sensor_blobs_path=Path(B.TEST_BLOBS), data_path=Path(B.TEST_LOGS), scene_filter=sf,
                         sensor_config=agents[0][1].get_sensor_config())

    for inp, name in (("photo", "rgb"), ("raster", "raster")):
        exs = [e for e in EXAMPLES if e[0] == inp]
        if args.grid2x2:
            fig, axes = plt.subplots(2, 4, figsize=(20, 8.0), gridspec_kw={"width_ratios": [16 / 9, 1, 16 / 9, 1]})
        else:
            fig, axes = plt.subplots(2, 2 * len(exs), figsize=(6.4 * len(exs), 4.4),
                                     gridspec_kw={"width_ratios": [3.0, 1.3] * len(exs)})
        for k, (_, token, v, _) in enumerate(exs):
            base = loader.get_agent_input_from_token(token)
            cam = base.cameras[-1].cam_f0
            M = B.lidar2img(cam)
            if inp == "photo":
                with_ped = cv2.cvtColor(cv2.imread(os.path.join(B.GENERATED_DIR, f"{token}_front_{v}.jpg")),
                                        cv2.COLOR_BGR2RGB)
                no_ped = np.asarray(cam.image).astype(np.uint8)
            else:
                ped = B.ped_position(dets.get((token, f"front_{v}"), []), M)
                n = math.hypot(*ped)
                drawn = (ped[0] * (1 - CLOSER / n), ped[1] * (1 - CLOSER / n))
                no_ped = np.asarray(cam.rendered_image).astype(np.uint8)
                with_ped = G.draw_ped_cuboid(no_ped.copy(), M, drawn)
            mph = abs(float(base.ego_statuses[-1].ego_velocity[0])) * B.MPH
            ped_xy = B.ped_position(dets.get((token, f"front_{v}"), []), M) if inp == "photo" else box_centre(drawn)
            if args.grid2x2:                                  # with the pedestrian only, 2 x 2 scenes
                row, col = divmod(k, 2)
                plans = [(n_, predict(a, make_input(loader, token, inp, with_ped)), c) for n_, a, c in agents]
                draw_image_full(axes[row, 2 * col], with_ped, cam, plans)
                draw_bev(axes[row, 2 * col + 1], plans, ped_xy, true_scale=True, mph=mph)
            else:
                for row, (img, front, ped) in enumerate([(with_ped, with_ped, ped_xy), (no_ped, None, None)]):
                    plans = [(n_, predict(a, make_input(loader, token, inp, front)), c) for n_, a, c in agents]
                    draw_image(axes[row, 2 * k], img, cam, plans)
                    draw_bev(axes[row, 2 * k + 1], plans, ped, mph=mph)
            print("done", inp, token, v, flush=True)
        fig.legend(handles=[Line2D([], [], color=c, lw=4, marker="o", ms=7, mec="white", label=n_)
                            for n_, _, _, c in MODELS]
                   + [Line2D([], [], color=PED_C, marker="o", ls="none", ms=9, mec="white", label="pedestrian")],
                   loc="lower center", ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 0.0))
        if args.grid2x2:
            fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.07, wspace=0.03, hspace=0.04)
            true_scale_limits(fig, [axes[r, c] for r in range(2) for c in (1, 3)])
            fig.savefig(out / f"old_vs_new_loss_{name}_2x2.png", dpi=130)
        else:
            fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.11, wspace=0.03, hspace=0.04)
            fig.savefig(out / f"old_vs_new_loss_{name}.png", dpi=130)
        plt.close(fig)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
