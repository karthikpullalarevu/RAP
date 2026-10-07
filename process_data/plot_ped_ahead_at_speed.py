#!/usr/bin/env python3
"""Training-style rasters where a pedestrian is 5-10 m ahead in ego's path while ego is fast,
with a top-down view of what happens next (ego and pedestrian over the following 5 s).

Usage: python process_data/plot_ped_ahead_at_speed.py [--min-mph 15] [--n 8]
"""
import argparse, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/datahdd/karthik/dataset/nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
from process_data.build_ped_cache import load_simulation_log, _patch_map_path

RAW = Path("/datahdd2/karthik/ped_v2/raw")
A = Path("/datahdd2/karthik/ped_v2/analysis")
MPH, HZ = 2.23694, 20
EGO_C, PED_C, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#777777"


def extract(args):
    scene, t0 = args
    _patch_map_path()
    ins = set(json.loads((RAW / scene / "metadata/result.json").read_text())["saved_log_summary"]["pedestrian_track_tokens"])
    data = load_simulation_log(next((RAW / scene / "simulation_log").glob("*.xz"))).simulation_history.data
    e0 = data[t0].ego_state
    cx, cy, h = e0.center.x, e0.center.y, e0.center.heading
    c, s = math.cos(-h), math.sin(-h)
    loc = lambda x, y: ((x - cx) * c - (y - cy) * s, (x - cx) * s + (y - cy) * c)
    foot0 = e0.car_footprint.geometry
    ped0 = min([o for o in data[t0].observation.tracked_objects.tracked_objects if o.track_token in ins],
               key=lambda o: o.box.geometry.distance(foot0))
    ego, ped, gaps = {}, {}, {}
    for f in range(max(0, t0 - HZ), min(len(data), t0 + 5 * HZ + 1)):
        e = data[f].ego_state
        ego[f - t0] = (*loc(e.center.x, e.center.y), e.center.heading - h, e.dynamic_car_state.speed * MPH)
        for o in data[f].observation.tracked_objects.tracked_objects:
            if o.track_token == ped0.track_token:
                ped[f - t0] = loc(o.center.x, o.center.y)
                gaps[f - t0] = o.box.geometry.distance(e.car_footprint.geometry)
    box = e0.car_footprint.oriented_box
    return dict(scene=scene, t0=t0, ego=ego, ped=ped, gaps=gaps, L=box.length, W=box.width,
                ped_mph=math.hypot(ped0.velocity.x, ped0.velocity.y) * MPH)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mph", type=float, default=15)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-ped-mph", type=float, default=None, help="Only pedestrians moving slower than this")
    ap.add_argument("--min-ped-mph", type=float, default=None, help="Only pedestrians moving faster than this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--simple", action="store_true", help="Slide version: two short labels per example, nothing else")
    args = ap.parse_args()

    p = pd.read_csv(A / "02_ped_frames.csv", usecols=["scene_name", "frame_idx", "gap_m", "ego_speed_ms", "has_raster",
                                                      "ped_local_x_m", "ped_local_y_m", "n_inserted_peds", "ped_speed_ms"])
    q = p[(p.has_raster == 1) & (p.n_inserted_peds == 1) & (p.ped_local_y_m.abs() < 1.5)
          & (p.ped_local_x_m >= 5) & (p.ped_local_x_m <= 10) & (p.ego_speed_ms * MPH >= args.min_mph)
          & (p.frame_idx <= 198)]
    if args.max_ped_mph is not None:
        q = q[q.ped_speed_ms * MPH < args.max_ped_mph]
    if args.min_ped_mph is not None:
        q = q[q.ped_speed_ms * MPH > args.min_ped_mph]
    first = q.sort_values("frame_idx").groupby("scene_name", as_index=False).first()
    # what happens over the next 5 s, for every such sequence (from the per-frame table)
    m = first[["scene_name", "frame_idx"]].rename(columns={"frame_idx": "t0"}).merge(p, on="scene_name")
    nxt = m[(m.frame_idx > m.t0) & (m.frame_idx <= m.t0 + 5 * HZ)].groupby("scene_name").gap_m.min()
    print(f"sequences with a pedestrian 5-10 m ahead in ego's path while ego >= {args.min_mph:g} mph: {len(first)}")
    print("closest gap over the next 5 s (m):", nxt.describe(percentiles=[.1, .25, .5, .75]).round(2).to_dict())
    print("  under 0.5 m: %.0f%% | under 1 m: %.0f%% | contact (0 m): %d" % (100 * (nxt < 0.5).mean(), 100 * (nxt < 1).mean(), (nxt <= 0.005).sum()))

    picks = first.sample(min(args.n, len(first)), random_state=args.seed).sort_values("ego_speed_ms")
    with ProcessPoolExecutor(8) as ex:
        tracks = list(ex.map(extract, [(r.scene_name, int(r.frame_idx)) for r in picks.itertuples()]))

    rows = math.ceil(len(tracks) / 2)
    fig, axes = plt.subplots(rows, 4, figsize=(21, 4.6 * rows), gridspec_kw={"width_ratios": [1.9, 1, 1.9, 1]})
    P = lambda x, y: (-np.asarray(y), np.asarray(x))
    for k, d in enumerate(tracks):
        ax_img, ax = axes[k // 2, (k % 2) * 2], axes[k // 2, (k % 2) * 2 + 1]
        ax_img.imshow(Image.open(RAW / d["scene"] / "rendered/rasters" / f"frame_{d['t0']:04d}" / "CAM_F0.png"))
        ax_img.axis("off")
        ego = pd.DataFrame(d["ego"], index=["x", "y", "h", "mph"]).T.sort_index()
        ped = pd.DataFrame(d["ped"], index=["x", "y"]).T.sort_index()
        gap = pd.Series(d["gaps"]).sort_index()
        fut_gap = gap[gap.index >= 0]
        gmin, tmin = fut_gap.min(), fut_gap.idxmin() / HZ
        L, W = d["L"] / 2, d["W"] / 2
        ax.add_patch(Polygon(np.column_stack(P([L, L, -L, -L], [W, -W, -W, W])), fc=EGO_C, ec=EGO_C, alpha=0.35, lw=1.5))
        ax.annotate("", xy=P(L - 0.3, 0), xytext=P(-L / 2, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5, mutation_scale=11))
        ef = ego[ego.index >= 0]
        ax.plot(*P(ef.x, ef.y), color=EGO_C, lw=2)
        ax.plot(*P(ef.x[::10], ef.y[::10]), "o", color=EGO_C, ms=4)
        pp, pf = ped[ped.index <= 0], ped[ped.index >= 0]
        ax.plot(*P(pp.x, pp.y), color=PED_C, lw=2)
        ax.plot(*P(pf.x, pf.y), color=PED_C, lw=1.5, ls=(0, (2, 2)))
        ax.plot(*P(pf.x[::10], pf.y[::10]), "o", color=PED_C, ms=4, mfc="white")
        ax.add_patch(Circle(P(ped.loc[0].x, ped.loc[0].y), 0.4, fc=PED_C, ec="white", lw=1, zorder=5))
        ax.set_xlim(-9, 9)
        ax.set_ylim(-4, 32)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_ylabel("metres ahead", fontsize=9, color=MUTED)
        end_mph = ef.mph.iloc[-1]
        if args.simple:
            ax.set_ylabel("")
            ax_img.set_title(f"ego {ego.loc[0].mph:.0f} mph · pedestrian {d['ped_mph']:.0f} mph",
                             fontsize=17, fontweight="bold", color=INK)
            ax.set_title(f"passes {gmin:.1f} m away", fontsize=17, fontweight="bold", color=PED_C)
            continue
        ax_img.set_title(f"pedestrian {ped.loc[0].x:.1f} m ahead · ego {ego.loc[0].mph:.0f} mph · ped {d['ped_mph']:.0f} mph",
                         fontsize=13, fontweight="bold", color=INK)
        ax.set_title(f"next 5 s: closest gap {gmin:.1f} m at {tmin:.1f} s\nego ends at {end_mph:.0f} mph",
                     fontsize=11, color=INK)
    for k in range(len(tracks), rows * 2):
        axes[k // 2, (k % 2) * 2].axis("off"); axes[k // 2, (k % 2) * 2 + 1].axis("off")
    kind = ("slow pedestrian " if args.max_ped_mph is not None else "fast pedestrian " if args.min_ped_mph is not None
            else "Pedestrian ")
    if not args.simple:
      fig.suptitle(f"{kind.capitalize()}5–10 m ahead in ego's path while ego is fast: what the simulation does next",
                   fontsize=15, fontweight="bold", y=0.995)
      fig.text(0.5, 0.004, "Left: front raster at that moment. Right: top-down, ego (blue) and pedestrian (orange) over the next 5 s, "
               "a dot every 0.5 s. Solid orange = pedestrian's last second, dashed = what it does next.",
               ha="center", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 1) if args.simple else (0, 0.015, 1, 0.98))
    tag = "" if args.max_ped_mph is None else f"_ped_under_{args.max_ped_mph:g}mph"
    tag += "" if args.min_ped_mph is None else f"_ped_over_{args.min_ped_mph:g}mph"
    tag += "_simple" if args.simple else ""
    out = A / "06_close_pass" / f"ped_ahead_at_speed_{int(args.min_mph)}mph{tag}.png"
    fig.savefig(out, dpi=85)
    print("Saved", out)


if __name__ == "__main__":
    main()
