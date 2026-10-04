#!/usr/bin/env python3
"""Bird's-eye-view grid of close passes: ego within 1 m of an inserted ped while moving.

Each panel is drawn at the closest moment, in the ego frame (ego points up):
  - ego as a rectangle with a heading arrow, its past 2 s and future 4 s path
  - the pedestrian now, its past 2 s and future 2 s path
  - faint "1 s earlier" ghosts of both, to show who was where when

Also writes the full list of such scenes and the speed-vs-distance histogram in mph.

Usage:
    python process_data/plot_close_pass_bev.py \
        --batch-root /datahdd2/karthik/ped_v2/raw \
        --analysis-dir /datahdd2/karthik/ped_v2/analysis
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/datahdd/karthik/dataset/nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

from process_data.build_ped_cache import load_simulation_log, _patch_map_path

MPH = 2.23694
HZ = 20
PAST, FUT_EGO, FUT_PED, GHOST = 2 * HZ, 4 * HZ, 2 * HZ, 1 * HZ
SPEED_BINS = [(1, 5), (5, 10), (10, 20), (20, 99)]      # mph
EGO_C, PED_C, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#777777"
YMIN = -9   # bottom of each panel, metres behind ego centre


def extract(args):
    """Ego and nearest-ped tracks around the closest frame, in the ego frame at that frame."""
    scene_dir, t0 = args
    _patch_map_path()
    inserted = set(json.loads((scene_dir / "metadata/result.json").read_text())
                   ["saved_log_summary"]["pedestrian_track_tokens"])
    data = load_simulation_log(next((scene_dir / "simulation_log").glob("*.msgpack.xz"))).simulation_history.data

    ego0 = data[t0].ego_state
    foot = ego0.car_footprint.geometry
    peds0 = [o for o in data[t0].observation.tracked_objects.tracked_objects if o.track_token in inserted]
    ped0 = min(peds0, key=lambda o: o.box.geometry.distance(foot))
    cx, cy, h = ego0.center.x, ego0.center.y, ego0.center.heading
    c, s = math.cos(-h), math.sin(-h)

    def local(x, y):
        dx, dy = x - cx, y - cy
        return dx * c - dy * s, dx * s + dy * c

    ego, ped = {}, {}
    ghost_mph = data[t0 - GHOST].ego_state.dynamic_car_state.speed * MPH if t0 >= GHOST else None
    for f in range(max(0, t0 - PAST), min(len(data), t0 + FUT_EGO + 1)):
        e = data[f].ego_state
        ego[f - t0] = (*local(e.center.x, e.center.y), e.center.heading - h)
        for o in data[f].observation.tracked_objects.tracked_objects:
            if o.track_token == ped0.track_token:
                ped[f - t0] = local(o.center.x, o.center.y)
    box = ego0.car_footprint.oriented_box
    return {"scene": scene_dir.name, "ego": ego, "ped": ped, "length": box.length, "width": box.width,
            "ghost_mph": ghost_mph,
            "ped_radius": max(ped0.box.length, ped0.box.width) / 2}


def draw_panel(ax, d, row):
    from matplotlib.patches import Circle, Polygon

    def P(x, y):          # ego frame (x forward, y left) -> plot (left is left, forward is up)
        return -np.asarray(y), np.asarray(x)

    def ego_rect(x, y, hd, **kw):
        L, W = d["length"] / 2, d["width"] / 2
        pts = np.array([[L, W], [L, -W], [-L, -W], [-L, W]])
        rot = np.array([[math.cos(hd), -math.sin(hd)], [math.sin(hd), math.cos(hd)]])
        pts = pts @ rot.T + [x, y]
        ax.add_patch(Polygon(np.column_stack(P(pts[:, 0], pts[:, 1])), closed=True, **kw))

    ego = pd.DataFrame(d["ego"], index=["x", "y", "h"]).T.sort_index()
    ped = pd.DataFrame(d["ped"], index=["x", "y"]).T.sort_index()

    # ego: past (thin), future (solid, dot every 0.5 s)
    ax.plot(*P(ego.x[ego.index <= 0], ego.y[ego.index <= 0]), color=EGO_C, lw=1, alpha=0.45)
    fut = ego[ego.index >= 0]
    ax.plot(*P(fut.x, fut.y), color=EGO_C, lw=2)
    ax.plot(*P(fut.x[::10], fut.y[::10]), "o", color=EGO_C, ms=3)
    if -GHOST in ego.index:
        g = ego.loc[-GHOST]
        ego_rect(g.x, g.y, g.h, fill=False, ec=EGO_C, lw=1, ls=(0, (3, 2)), alpha=0.7)
        if d["ghost_mph"] is not None:
            # speed 1 s earlier, written in the dashed box. If that box overlaps the ego box (slow)
            # put it just below; if it is off the panel (fast) pin it to the bottom edge.
            L = d["length"] / 2
            if g.x > -2 * L:
                ty = g.x - L - 1.1
            else:
                ty = max(g.x, YMIN + 1.2)
            tx = -g.y if g.x >= YMIN + 1.2 else 0.0
            ax.text(tx, ty, f"{d['ghost_mph']:.0f}\nmph", ha="center", va="center", fontsize=10.5,
                    color=INK, linespacing=0.95, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ego_rect(0, 0, 0, fc=EGO_C, ec=EGO_C, alpha=0.35, lw=1.5)
    ax.annotate("", xy=P(d["length"] / 2 - 0.3, 0), xytext=P(-d["length"] / 4, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, mutation_scale=12))

    # ped: past (solid), future (dashed), now (filled), 1 s earlier (hollow)
    ax.plot(*P(ped.x[ped.index <= 0], ped.y[ped.index <= 0]), color=PED_C, lw=2)
    pf = ped[(ped.index >= 0) & (ped.index <= FUT_PED)]
    ax.plot(*P(pf.x, pf.y), color=PED_C, lw=1.5, ls=(0, (2, 2)))
    r = max(d["ped_radius"], 0.3)
    if -GHOST in ped.index:
        g = ped.loc[-GHOST]
        ax.add_patch(Circle(P(g.x, g.y), r, fill=False, ec=PED_C, lw=1.2))
    now = ped.loc[0]
    ax.add_patch(Circle(P(now.x, now.y), r, fc=PED_C, ec="white", lw=1, zorder=5))

    ax.set_xlim(-7, 7)
    ax.set_ylim(YMIN, 11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.set_title(f"gap {row.gap_m:.2f} m\nego {row.mph:.0f} mph",
                 fontsize=13, fontweight="bold", color=INK, pad=5, linespacing=1.15)
    ax.text(0.04, 0.97, f"ped {row.ped_mph:.0f} mph", transform=ax.transAxes,
            fontsize=8, color=MUTED, va="top", ha="left")


def bev_grid(picks, batch_root, out_path, workers, cols=9):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    jobs = [(batch_root / r.scene_name, int(r.frame_idx)) for r in picks.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        tracks = list(ex.map(extract, jobs))

    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 4.0 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, d, row in zip(np.ravel(axes), tracks, picks.itertuples()):
        ax.axis("on")
        draw_panel(ax, d, row)

    handles = [
        Patch(fc=EGO_C, ec=EGO_C, alpha=0.35, label="Ego at closest moment (arrow = heading)"),
        Line2D([], [], color=EGO_C, lw=2, marker="o", ms=3, label="Ego next 4 s (dot every 0.5 s)"),
        Patch(fill=False, ec=EGO_C, ls=(0, (3, 2)), label="Ego 1 s earlier (number = its speed then)"),
        Line2D([], [], color=PED_C, marker="o", ms=8, lw=0, mec="white", label="Pedestrian at closest moment"),
        Line2D([], [], color=PED_C, lw=2, label="Pedestrian last 2 s"),
        Line2D([], [], color=PED_C, lw=1.5, ls=(0, (2, 2)), label="Pedestrian next 2 s"),
        Line2D([], [], color=PED_C, marker="o", ms=8, lw=0, mfc="none", label="Pedestrian 1 s earlier"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(f"Close passes: ego within 1 m of a pedestrian while moving ({len(picks)} random sequences, slow → fast)",
                 fontsize=15, fontweight="bold", y=0.998)
    fig.text(0.5, 0.012, "Each panel is 14 m wide × 20 m tall, ego frame, ego pointing up.",
             ha="center", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.955), h_pad=1.2)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


def speed_distance_hist(frames, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    blues = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(10, 6.5))
    h = ax.hist2d(frames.gap_m, frames.ego_speed_ms * MPH,
                  bins=[np.arange(0, 30.5, 0.5), np.arange(0, 42, 1)], cmap=blues, norm=LogNorm(), cmin=1)
    fig.colorbar(h[3], ax=ax, label="Frames (log scale)")
    ax.axvline(1, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(1.2, 39.5, "1 m", fontsize=10, color=INK, va="top")
    ax.set_xlabel("Gap between car body and nearest inserted pedestrian (m)")
    ax.set_ylabel("Ego speed (mph)")
    ax.set_title("Ego speed vs distance to pedestrian — all frames with an inserted pedestrian",
                 fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def gap_speed_heatmap(closest, out_path):
    """One count per sequence at its closest moment: gap (columns) x ego speed (rows), numbers in cells."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    gap_lab = ["under 0.5 m", "0.5–1 m", "1–2 m", "2–5 m", "5–10 m"]
    spd_lab = ["stopped\n(<1 mph)", "1–5 mph", "5–10 mph", "10–20 mph", "over 20 mph"]
    g = pd.cut(closest.gap_m, [-1, 0.5, 1, 2, 5, 1e9], labels=gap_lab)
    v = pd.cut(closest.mph, [-1, 1, 5, 10, 20, 999], labels=spd_lab)
    t = pd.crosstab(v, g).reindex(index=spd_lab, columns=gap_lab, fill_value=0)
    n = int(t.values.sum())

    blues = LinearSegmentedColormap.from_list("seq_blue", ["#eef5fe", "#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab"])
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.imshow(t.values, cmap=blues, origin="lower", aspect="auto", vmin=0, vmax=t.values.max())
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            c = int(t.values[i, j])
            dark = c > 0.6 * t.values.max()
            ax.text(j, i + 0.08, f"{c:,}", ha="center", va="center", fontsize=17, fontweight="bold",
                    color="white" if dark else INK)
            ax.text(j, i - 0.24, f"{100 * c / n:.1f}%", ha="center", va="center", fontsize=10,
                    color="white" if dark else MUTED)
    # white gaps between cells
    ax.set_xticks(np.arange(-0.5, t.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, t.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(t.shape[1]), gap_lab, fontsize=12)
    ax.set_yticks(range(t.shape[0]), spd_lab, fontsize=12)
    ax.set_xlabel("Closest gap between car and pedestrian", fontsize=12, labelpad=10)
    ax.set_ylabel("Ego speed at that moment", fontsize=12, labelpad=10)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # near-miss zone: under 1 m and over 10 mph
    near = int(t.iloc[3:, :2].values.sum())
    ax.add_patch(Rectangle((-0.5, 2.5), 2, 2, fill=False, ec=INK, lw=2.5, zorder=5, clip_on=False))
    ax.text(-0.5, 4.58, f"Near-miss zone: under 1 m at over 10 mph — {near:,} sequences ({100 * near / n:.0f}%)",
            fontsize=11.5, fontweight="bold", color=INK, va="bottom", ha="left", clip_on=False)
    fig.suptitle(f"How close and how fast: {n:,} sequences at their closest moment to a pedestrian",
                 fontsize=14, fontweight="bold", x=0.5, y=0.985)
    fig.text(0.5, 0.005, "Number = sequences, small number = share of all sequences. Gap is car edge to pedestrian edge. "
             "Bin widths are uneven.", ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def gap_histograms(closest, out_path):
    """Histogram of the closest gap, one row per ego-speed range (one count per sequence)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("Ego stopped (<1 mph)", -1, 1), ("Ego 1–5 mph", 1, 5), ("Ego 5–10 mph", 5, 10),
              ("Ego 10–20 mph", 10, 20), ("Ego over 20 mph", 20, 999)]
    bins = np.arange(0, 10.25, 0.25)
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 11), sharex=True, sharey=True)
    for ax, (name, lo, hi) in zip(axes, groups):
        g = closest[(closest.mph > lo) & (closest.mph <= hi)].gap_m
        ax.hist(g, bins=bins, color=EGO_C, edgecolor="white", linewidth=0.8)
        ax.axvline(1, color=INK, lw=1.2, ls=(0, (4, 3)))
        under = int((g < 1).sum())
        ax.text(0.995, 0.9, name, transform=ax.transAxes, ha="right", va="top", fontsize=13, fontweight="bold", color=INK)
        ax.text(0.995, 0.66, f"{len(g):,} sequences\n{under:,} under 1 m ({100 * under / max(len(g), 1):.0f}%)",
                transform=ax.transAxes, ha="right", va="top", fontsize=11, color=MUTED)
        ax.set_ylabel("Sequences", fontsize=10)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].text(1.08, axes[0].get_ylim()[1] * 0.98, "1 m", fontsize=10, color=INK, va="top")
    axes[-1].set_xlabel("Closest gap between car and pedestrian (m)", fontsize=12)
    axes[-1].set_xticks(range(0, 11))
    axes[-1].set_xlim(0, 10)
    fig.suptitle(f"Closest gap to a pedestrian, by ego speed at that moment ({len(closest):,} sequences)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.004, "One count per sequence, at the moment the pedestrian is closest. Bars are 0.25 m wide. "
             "Gap is car edge to pedestrian edge.", ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-root", type=Path, default=Path("/datahdd2/karthik/ped_v2/raw"))
    ap.add_argument("--analysis-dir", type=Path, default=Path("/datahdd2/karthik/ped_v2/analysis"))
    ap.add_argument("--per-bin", type=int, default=12)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.analysis_dir / "06_close_pass"
    out.mkdir(exist_ok=True)
    frames = pd.read_csv(args.analysis_dir / "02_ped_frames.csv",
                         usecols=["scene_name", "frame_idx", "ego_speed_ms", "gap_m", "ped_speed_ms", "ped_side"])

    closest = frames.loc[frames.groupby("scene_name").gap_m.idxmin()].copy()
    closest["mph"] = closest.ego_speed_ms * MPH
    closest["ped_mph"] = closest.ped_speed_ms * MPH
    close = closest[(closest.gap_m < 1) & (closest.mph > 1)].sort_values("mph")
    close.to_csv(out / "close_pass_scenes.csv", index=False)
    print(f"{len(close)} sequences under 1 m while moving (>1 mph)")

    picks = pd.concat(close[(close.mph > lo) & (close.mph <= hi)].sample(args.per_bin, random_state=args.seed)
                      for lo, hi in SPEED_BINS)
    # top up with extra sequences (spread over the speed bins) so the last row of the grid is full
    extra = -len(picks) % args.cols
    rest = close.drop(picks.index)
    for i in range(extra):
        lo, hi = SPEED_BINS[[1, 2, 0, 3][i % 4]]
        add = rest[(rest.mph > lo) & (rest.mph <= hi)].sample(1, random_state=args.seed + 1 + i)
        picks, rest = pd.concat([picks, add]), rest.drop(add.index)
    picks = picks.sort_values("mph")
    bev_grid(picks, args.batch_root, out / "bev_close_passes.png", args.workers, args.cols)
    speed_distance_hist(frames, out / "speed_vs_distance_mph.png")
    gap_speed_heatmap(closest, out / "gap_vs_speed_closest_moment.png")
    gap_histograms(closest, out / "gap_histogram_by_speed.png")


if __name__ == "__main__":
    main()
