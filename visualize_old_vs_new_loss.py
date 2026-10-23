#!/usr/bin/env python3
"""Slide figures: old loss (E4) vs new loss (E6) on Gemini scenes, photo and raster, with / without pedestrian.

Each example: camera image with both plans (pedestrian present) | top-down view with both plans, with the
pedestrian (solid) and without it (dashed); an X marks where a plan comes to a stop. Ego inputs as recorded.

Usage:
    CUDA_VISIBLE_DEVICES=0 python visualize_old_vs_new_loss.py --output-dir <dir>
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
from matplotlib.patches import Polygon
import numpy as np

import visualize_ped_comparison as V
import visualize_ped_gemini_bev as B
import visualize_gemini_raster_ped as G
from visualize_gemini_rgb_vs_raster import predict, make_input, box_centre

X = Path("/datahdd2/karthik/RAP/experiments")
MODELS = [("old loss", X / "E4_pedV2-105k_photo50k/epoch15.ckpt", "final", "#2a78d6"),
          ("new loss", X / "E6_pedV2yield_sub5_infonce/epoch15.ckpt", "gated_pdms", "#1baf7a")]
# (input, token, Gemini variant, outcome of the new loss)
EXAMPLES = [("photo", "9e2ca6b821065c6c", 2, "stops for pedestrian"), ("photo", "f02aa91af18752f8", 1, "stops (also without pedestrian)"),
            ("photo", "19d7afdb02055502", 1, "doesn't stop"), ("photo", "50c88f8f4a535591", 2, "doesn't stop"),
            ("raster", "565c0481a2a5540a", 0, "stops for pedestrian"), ("raster", "f02aa91af18752f8", 0, "stops (also without pedestrian)"),
            ("raster", "266d9993badd5b9c", 1, "doesn't stop"), ("raster", "32e05d8aad845364", 2, "doesn't stop")]
CLOSER = 1.5
SURFACE, INK, MUTED, GRID, PED = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#eb6834"
STOP_MPH = 1.0


def stop_index(traj):
    """Index of the first pose after which the plan stays below 1 mph, else None."""
    full = np.vstack([[0, 0, 0], traj])
    v = np.linalg.norm(np.diff(full[:, :2], axis=0), axis=-1) / 0.5 * B.MPH
    for i in range(len(v)):
        if (v[i:] < STOP_MPH).all():
            return i
    return None


def verdict(traj, ped):
    """What a plan does with the pedestrian there: stops before it, drives through it, or passes it."""
    s = stop_index(traj)
    if s is not None and np.vstack([[0, 0, 0], traj])[s + 1, 0] < ped[0] - 3.0:
        return "stops"
    gap, _, _ = B.closest_gap(traj, ped)
    return "drives through" if gap <= 0 else "passes"


def subtitle(r):
    return "   ".join(f"{name}: {verdict(t, r['ped'])}" for name, t, _ in r["plans_ped"])


def draw_image(ax, img, cam, plans):
    ax.imshow(img)
    H, W = img.shape[:2]
    for name, traj, color in plans:
        px, ok = V.project_trajectory_to_image(np.vstack([[0, 0], traj[:, :2]]), cam)
        ok = ok & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        ax.plot(px[ok, 0], px[ok, 1], "-o", color=color, lw=4, ms=7, mec="white", mew=1.5)
    ax.set_xlim(0, W)
    ax.set_ylim(H * 0.92, H * 0.3)       # crop sky and bonnet
    ax.axis("off")


def draw_bev(ax, ped, plans_ped, plans_np):
    P = lambda x, y: (-np.asarray(y), np.asarray(x))
    ax.add_patch(Polygon(np.column_stack(P(*B.car_corners((0, 0, 0)).T)), fc="#d6d5cf", ec=INK, lw=1))
    for (name, traj, color), ls, alpha in [(p, "--", 0.55) for p in plans_np] + [(p, "-", 1.0) for p in plans_ped]:
        path = np.vstack([[0, 0], traj[:, :2]])
        ax.plot(*P(path[:, 0], path[:, 1]), ls, color=color, lw=2.5, alpha=alpha, marker="o", ms=3.5)
        s = stop_index(traj)
        if s is not None:
            ax.plot(*P(path[s + 1, 0], path[s + 1, 1]), "X", color=color, ms=15, mec=INK, mew=1.2, alpha=alpha, zorder=6)
    ax.plot(*P(*ped), "o", color=PED, ms=13, mec="white", mew=2, zorder=7)
    ax.annotate("", xy=P(ped[0], ped[1] + 2.2), xytext=P(*ped),
                arrowprops=dict(arrowstyle="-|>", color=PED, lw=2, mutation_scale=14), zorder=7)
    top = max(30.0, ped[0] + 8, max(t[:, 0].max() for _, t, _ in plans_ped + plans_np) + 3)
    half = top / 2.6
    ax.set_xlim(-half, half)
    ax.set_ylim(-4, top)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)


def legend(fig, y):
    h = [Line2D([], [], color=c, lw=3, label=n) for n, _, _, c in MODELS]
    h += [Line2D([], [], color=INK, lw=2, ls="-", label="with pedestrian"),
          Line2D([], [], color=INK, lw=2, ls="--", alpha=0.55, label="no pedestrian"),
          Line2D([], [], color=INK, marker="X", ls="none", ms=11, label="stops"),
          Line2D([], [], color=PED, marker="o", ls="none", ms=11, mec="white", label="pedestrian")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, y), ncol=6, frameon=False, fontsize=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE})

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

    results = []
    for inp, token, v, outcome in EXAMPLES:
        base = loader.get_agent_input_from_token(token)
        cam = base.cameras[-1].cam_f0
        M = B.lidar2img(cam)
        ped = B.ped_position(dets.get((token, f"front_{v}"), []), M)
        n = math.hypot(*ped)
        if inp == "photo":
            img = cv2.cvtColor(cv2.imread(os.path.join(B.GENERATED_DIR, f"{token}_front_{v}.jpg")), cv2.COLOR_BGR2RGB)
            ped_xy = ped
        else:
            drawn = (ped[0] * (1 - CLOSER / n), ped[1] * (1 - CLOSER / n))
            img = G.draw_ped_cuboid(np.asarray(cam.rendered_image).astype(np.uint8).copy(), M, drawn)
            ped_xy = box_centre(drawn)
        plans_ped = [(name, predict(a, make_input(loader, token, inp, img)), c) for name, a, c in agents]
        plans_np = [(name, predict(a, make_input(loader, token, inp)), c) for name, a, c in agents]
        mph = abs(float(base.ego_statuses[-1].ego_velocity[0])) * B.MPH
        results.append(dict(inp=inp, token=token, v=v, outcome=outcome, img=img, cam=cam, ped=ped_xy,
                            plans_ped=plans_ped, plans_np=plans_np, mph=mph))
        print("done", inp, token, v, flush=True)

    for r in results:                                   # one slide per example
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.9, 1]})
        draw_image(a1, r["img"], r["cam"], r["plans_ped"])
        draw_bev(a2, r["ped"], r["plans_ped"], r["plans_np"])
        fig.text(0.01, 0.97, f"{r['inp']}  ·  {r['mph']:.0f} mph  ·  new loss {r['outcome']}", fontsize=15,
                 color=INK, va="top")
        fig.text(0.01, 0.925, subtitle(r), fontsize=12, color=MUTED, va="top")
        legend(fig, 0.0)
        fig.tight_layout(rect=(0, 0.06, 1, 0.9))
        fig.savefig(out / f"{r['inp']}_{''.join(ch if ch.isalnum() else '_' for ch in r['outcome']).strip('_')}_{r['token'][:8]}_v{r['v']}.png",
                    dpi=130)
        plt.close(fig)

    for inp in ("photo", "raster"):                     # 4 examples per input on one slide
        rs = [r for r in results if r["inp"] == inp]
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 4, width_ratios=[1.9, 1, 1.9, 1], left=0.01, right=0.99, top=0.93, bottom=0.08,
                              wspace=0.06, hspace=0.18)
        for k, r in enumerate(rs):
            row, col = divmod(k, 2)
            a1, a2 = fig.add_subplot(gs[row, col * 2]), fig.add_subplot(gs[row, col * 2 + 1])
            draw_image(a1, r["img"], r["cam"], r["plans_ped"])
            draw_bev(a2, r["ped"], r["plans_ped"], r["plans_np"])
            a1.set_title(f"new loss {r['outcome']}  ·  {r['mph']:.0f} mph\n{subtitle(r)}", fontsize=12, color=INK,
                         loc="left")
        fig.suptitle(f"Old loss (E4) vs new loss (E6) · {inp} input", fontsize=16, x=0.01, ha="left")
        legend(fig, 0.0)
        fig.savefig(out / f"grid_{inp}.png", dpi=130)
        plt.close(fig)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
