#!/usr/bin/env python3
"""Gemini pedestrian scenes: all 64 proposals, not just the picked one. Is any of them stopping before the pedestrian?

There is no real PDM score here (the pedestrian was added by Gemini, it is not in the logs), so every
proposal is judged by geometry against the pedestrian position measured from lidar:
  hits      the car body touches the pedestrian (gap <= 0)
  stops     never touches, and slows below STOP_MPH before reaching the pedestrian
  slows     never touches, ends at less than half the current speed
  passes    everything else

For each scene (variant 0 = first Gemini edit) the model is run twice, like visualize_gemini_rgb_vs_raster.py:
  photo:  front camera = Gemini photo
  raster: front camera = scene raster with a pedestrian box drawn --closer metres nearer

Usage:
    CUDA_VISIBLE_DEVICES=7 RAY_TMPDIR=/datahdd2/tmp/ray_eval python debug_gemini_all_proposals.py \
        --ckpt /datahdd2/karthik/RAP/experiments/E4_pedV2-105k_photo50k/epoch15.ckpt \
        --output-dir /datahdd2/karthik/RAP/experiments/E4_pedV2-105k_photo50k/gemini_all_proposals --scenes 10
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
import visualize_ped_gemini_bev as B          # scene loader pieces, geometry helpers
import visualize_gemini_raster_ped as G       # draw_ped_cuboid
import visualize_gemini_rgb_vs_raster as R    # make_input, box_centre

DT = 0.5
STOP_MPH = 2.0
INK, MUTED = B.INK, B.MUTED
GROUP_COLORS = {"hits": "#d62728", "stops": "#1baf7a", "slows": "#2a78d6", "passes": "#9a9a9a"}
PICKED = "#eb6834"


def all_proposals(agent, agent_input):
    with torch.no_grad():
        feats = {}
        for b in agent.get_feature_builders():
            feats.update(b.compute_features(agent_input))
        feats = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v for k, v in feats.items()}
        o = agent.forward(feats, return_score=True)
    return o["trajectory"][0].float().cpu().numpy(), o["score"][0].float().cpu().numpy()


def speeds_mph(traj):
    xy = np.vstack([[0, 0], traj[:, :2]])
    return np.linalg.norm(np.diff(xy, axis=0), axis=1) / DT * B.MPH


def classify(traj, ped, ego_mph):
    gap, _, _ = B.closest_gap(traj, ped)
    v = speeds_mph(traj)
    front_x = traj[:, 0] + B.AXLE_TO_CENTER + B.HALF_LEN          # front bumper, roughly (paths are near straight)
    before = front_x < ped[0] - B.PED_RADIUS
    if gap <= 0:
        group = "hits"
    elif np.any((v < STOP_MPH) & before):
        group = "stops"
    elif v[-1] < 0.5 * ego_mph:
        group = "slows"
    else:
        group = "passes"
    return group, gap, v


def draw(rows, title, out_path):
    """rows: (image, cam, ped_xy, ped_label, props, scores, groups, speeds, row_title) per input type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon

    fig, axes = plt.subplots(2, 3, figsize=(24, 13), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    P = lambda x, y: (-np.asarray(y), np.asarray(x))
    top = max(30.0, max(r[4][:, :, 0].max() for r in rows) + 3, max(r[2][0] for r in rows) + 5)
    half = max(8.0, top / 3.2)
    for (ax_img, ax_bev, ax_v), (img, cam, ped, ped_label, props, scores, groups, speeds, row_title) in zip(axes, rows):
        pick = int(np.argmax(scores))
        order = sorted(range(len(props)), key=lambda i: scores[i])     # draw high scores last, on top
        n = {g: groups.count(g) for g in GROUP_COLORS}

        ax_img.imshow(img)
        H, W = img.shape[:2]
        for i in order + [pick]:
            px, valid = V.project_trajectory_to_image(props[i][:, :2], cam)
            ok = valid & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
            if i == pick:
                ax_img.plot(px[ok, 0], px[ok, 1], "-o", color=PICKED, lw=3, ms=7, mec="white", mew=1.2)
            else:
                ax_img.plot(px[ok, 0], px[ok, 1], "-", color=GROUP_COLORS[groups[i]], lw=1, alpha=0.6)
        ax_img.set_xlim(0, W)
        ax_img.set_ylim(H, 0)
        ax_img.axis("off")
        ax_img.set_title(row_title, fontsize=14, color=INK)

        ax_bev.add_patch(Polygon(np.column_stack(P(*B.car_corners((0, 0, 0)).T)), fc="#bbbbbb", ec=INK, lw=1.2))
        for i in order:
            path = np.vstack([[0, 0], props[i][:, :2]])
            ax_bev.plot(*P(path[:, 0], path[:, 1]), "-", color=GROUP_COLORS[groups[i]], lw=1.2, alpha=0.7)
        path = np.vstack([[0, 0], props[pick][:, :2]])
        ax_bev.plot(*P(path[:, 0], path[:, 1]), "-o", color=PICKED, lw=3, ms=5, zorder=5)
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
        ax_bev.set_title(f"All 64 from above · hits {n['hits']} · stops {n['stops']} · "
                         f"slows {n['slows']} · passes {n['passes']}", fontsize=12, color=INK)

        t = np.arange(len(speeds[0]) + 1) * DT
        for i in order:
            ax_v.plot(t, np.r_[speeds[i][0], speeds[i]], "-", color=GROUP_COLORS[groups[i]], lw=1, alpha=0.6)
        ax_v.plot(t, np.r_[speeds[pick][0], speeds[pick]], "-o", color=PICKED, lw=3, ms=5)
        ax_v.axhline(STOP_MPH, color=MUTED, lw=0.8, ls=":")
        ax_v.set_ylim(bottom=0)
        ax_v.set_xlabel("seconds", fontsize=10, color=MUTED)
        ax_v.set_ylabel("speed (mph)", fontsize=10, color=MUTED)
        ax_v.grid(color="#e5e5e5", lw=0.6)
        stoppers = [i for i, g in enumerate(groups) if g == "stops"]
        best_stop = (f"best stopping one: predicted {scores[max(stoppers, key=lambda i: scores[i])]:.2f}"
                     if stoppers else "no proposal stops before the pedestrian")
        ax_v.set_title(f"Speed · picked: predicted {scores[pick]:.2f}, {groups[pick]}\n{best_stop}",
                       fontsize=12, color=INK)
        for ax in (ax_bev, ax_v):
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

    handles = [Line2D([], [], color=PICKED, lw=3, marker="o", label="picked (highest predicted score)")]
    handles += [Line2D([], [], color=c, lw=2, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=12)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=75)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--closer", type=float, default=1.5)
    ap.add_argument("--scenes", type=int, default=10, help="Number of scenes (variant 0 of each)")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    agent = V.load_agent(args.ckpt)
    dets = {(e["token"], e["suffix"]): e["detections"] for e in json.load(open(B.LIDAR_RESULTS))}

    from navsim.common.dataloader import SceneLoader, SceneFilter
    # case.json names the log, so there is no need to scan all test logs for the tokens (slow on a busy disk).
    # Take a few spare scenes in case some are skipped (no photo / no lidar detection).
    cases = {}
    for d in sorted(os.listdir(B.ABLATION_DIR)):
        case = os.path.join(B.ABLATION_DIR, d, "case.json")
        if (os.path.exists(case) and os.path.exists(os.path.join(B.GENERATED_DIR, f"{d}_front_0.jpg"))
                and dets.get((d, "front_0"))):
            cases[d] = json.load(open(case))["log_name"]
        if len(cases) >= args.scenes + 5:
            break
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True)
    sf.log_names, sf.tokens = sorted(set(cases.values())), sorted(cases)
    loader = SceneLoader(sensor_blobs_path=Path(B.TEST_BLOBS), data_path=Path(B.TEST_LOGS), scene_filter=sf,
                         sensor_config=agent.get_sensor_config())

    rows, done = [], 0
    for token in sorted(loader.tokens):
        if done >= args.scenes:
            break
        base_in = loader.get_agent_input_from_token(token)
        cam = base_in.cameras[-1].cam_f0
        front_path = os.path.join(B.GENERATED_DIR, f"{token}_front_0.jpg")
        M = B.lidar2img(cam)
        ped = B.ped_position(dets.get((token, "front_0"), []), M)
        if cam.image is None or cam.image.max() == 0 or not os.path.exists(front_path) or ped is None:
            continue
        ego_mph = float(base_in.ego_statuses[-1].ego_velocity[0]) * B.MPH
        n = math.hypot(*ped)
        ped_drawn = (ped[0] * (1 - args.closer / n), ped[1] * (1 - args.closer / n))
        photo = cv2.cvtColor(cv2.imread(front_path), cv2.COLOR_BGR2RGB)
        raster = G.draw_ped_cuboid(np.asarray(cam.rendered_image).astype(np.uint8).copy(), M, ped_drawn)
        ped_raster = R.box_centre(ped_drawn)

        panels = []
        for mode, front, ped_xy, label, row_title in (
                ("photo", photo, ped, f"pedestrian\n{n:.1f} m away", "Photo input (pedestrian added with Gemini)"),
                ("raster", raster, ped_raster, f"pedestrian box\n{math.hypot(*ped_raster):.1f} m away",
                 f"Raster input (pedestrian box drawn {args.closer:g} m closer)")):
            props, scores = all_proposals(agent, R.make_input(loader, token, mode, front))
            res = [classify(p, ped_xy, ego_mph) for p in props]
            groups, gaps, speeds = [r[0] for r in res], [r[1] for r in res], [r[2] for r in res]
            pick = int(np.argmax(scores))
            for i in range(len(props)):
                rows.append(dict(token=token, input=mode, proposal=i, ego_mph=round(ego_mph, 1),
                                 ped_ahead_m=round(ped_xy[0], 2), ped_side_m=round(ped_xy[1], 2),
                                 group=groups[i], predicted_score=round(float(scores[i]), 3),
                                 closest_gap_m=round(gaps[i], 2), min_mph=round(float(speeds[i].min()), 1),
                                 end_mph=round(float(speeds[i][-1]), 1), picked=i == pick))
            panels.append((front if mode == "photo" else raster, cam, ped_xy, label, props, scores, groups, speeds,
                           row_title))
        draw(panels, f"{token} · ego at {ego_mph:.0f} mph · pedestrian {n:.1f} m away (lidar)",
             str(out / f"gemini_all_proposals_{token}.jpg"))
        done += 1
        print(f"done {token}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "gemini_all_proposals.csv", index=False)

    def per_scene(g):
        p = g[g.picked].iloc[0]
        st = g[g.group == "stops"]
        return pd.Series(dict(ego_mph=p.ego_mph, ped_ahead_m=p.ped_ahead_m, ped_side_m=p.ped_side_m,
                              hits=(g.group == "hits").sum(), stops=len(st), slows=(g.group == "slows").sum(),
                              picked_group=p.group, picked_score=p.predicted_score, picked_end_mph=p.end_mph,
                              best_stop_score=st.predicted_score.max() if len(st) else np.nan))
    summary = df.groupby(["token", "input"]).apply(per_scene).reset_index()
    summary.to_csv(out / "gemini_all_proposals_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.to_string(index=False))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
