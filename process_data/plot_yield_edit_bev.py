#!/usr/bin/env python3
"""One top-down slide of the yield edit: original vs edited 5 s trajectory for one training sample.

Car frame at the sample (car drives to the right). Dots every 1 s on both trajectories and on the pedestrian, with car
boxes drawn at the moment the original trajectory comes closest to the pedestrian.

Usage:
    python process_data/plot_yield_edit_bev.py \
        --scene 2021.06.23.18.23.38_veh-26_00069_00642__eafbd36ac211 --anchor 100
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import process_data.edit_yield_label as Y

ORIG, EDIT, PEDC, INK, MUTED, GRID, SURFACE = "#9a9a9a", "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="2021.06.23.18.23.38_veh-26_00069_00642__eafbd36ac211")
    ap.add_argument("--anchor", type=int, default=100, help="frame (20 Hz) of the training sample")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    sc = Y.Scene(args.scene)
    ta = args.anchor
    r = Y.solve_anchor(sc, ta)
    m = Y.LABEL_STEPS * Y.STEP + 1                      # 5 s at 20 Hz
    ox, oy, oh = sc.ego[ta, :3]
    c, s = math.cos(oh), math.sin(oh)

    def P(x, y):                                       # world -> car frame, plotted with the car pointing up
        dx, dy = np.asarray(x) - ox, np.asarray(y) - oy
        fwd, left = dx * c + dy * s, -dx * s + dy * c
        return fwd, left                               # car drives to the right, left is up

    orig = sc.ego[ta:ta + m, :3]
    ex, ey, eh = sc.path.pose(r["s"][:m])
    edit = np.column_stack([ex, ey, eh])
    g_orig = sc.gaps(ta, sc.s_pdm[ta:ta + m])
    g_edit = sc.gaps(ta, r["s"][:m])
    k = int(np.argmin(g_orig))                         # moment the original comes closest

    def car(ax, pose, color, fill):
        x, y, h = pose
        cc, ss = math.cos(h), math.sin(h)
        pts = [(x + a * cc - b * ss, y + a * ss + b * cc) for a, b in
               [(Y.HALF_LEN, Y.HALF_WID), (Y.HALF_LEN, -Y.HALF_WID), (-Y.HALF_LEN, -Y.HALF_WID), (-Y.HALF_LEN, Y.HALF_WID)]]
        ax.add_patch(Polygon(np.column_stack(P(*np.array(pts).T)), fc=fill, ec=color, lw=2, zorder=4))

    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, ax = plt.subplots(figsize=(16, 7))
    car(ax, sc.ego[ta, :3], INK, "#d6d5cf")                                   # car now
    for traj, color, name in [(orig, ORIG, "original trajectory"), (edit, EDIT, "edited trajectory")]:
        ax.plot(*P(traj[:, 0], traj[:, 1]), color=color, lw=4, label=name, solid_capstyle="round", zorder=3)
        ax.plot(*P(traj[::Y.HZ, 0], traj[::Y.HZ, 1]), "o", color=color, ms=8, mec="white", mew=1.5, zorder=5)
    car(ax, orig[k], ORIG, "none")
    car(ax, edit[k], EDIT, "none")
    for tr in sc.peds:
        p = tr[ta:ta + m]
        ok = ~np.isnan(p[:, 0])
        if not ok.any():
            continue
        ax.plot(*P(p[ok, 0], p[ok, 1]), color=PEDC, lw=2, ls=":", zorder=3)
        for i in range(0, m, Y.HZ):
            if ok[i]:
                ax.plot(*P(p[i, 0], p[i, 1]), "o", color=PEDC, ms=8 if i else 12, mec="white", mew=1.5, zorder=6)
                ax.annotate(f"{i // Y.HZ} s", P(p[i, 0], p[i, 1]), xytext=(7, 5), textcoords="offset points",
                            fontsize=10, color=PEDC)
        if ok[k]:
            ax.plot(*P(p[k, 0], p[k, 1]), "o", color=PEDC, ms=14, mec=INK, mew=1.5, zorder=7)
    ax.plot([], [], "o", color=PEDC, ms=10, mec="white", label="pedestrian")
    ax.annotate(f"{g_orig[k]:.1f} m", P(*orig[k, :2]), xytext=(0, -38), textcoords="offset points", fontsize=14,
                color=MUTED, fontweight="bold", ha="center")
    ax.annotate(f"{g_edit[k]:.1f} m", P(*edit[k, :2]), xytext=(0, -38), textcoords="offset points", fontsize=14,
                color=EDIT, fontweight="bold", ha="center")

    v0 = sc.v_pdm[ta] * Y.MPH
    d0 = np.inf                                        # body gap to the pedestrian, now or when it first appears
    for tr in sc.peds:
        p = tr[ta:ta + m]
        ok = np.where(~np.isnan(p[:, 0]))[0]
        if len(ok):
            i = ok[0]
            d0 = min(d0, float(Y.box_gap(*sc.ego[ta, :3], p[i, 0], p[i, 1], p[i, 2])))
    ax.text(0.02, 0.98, f"vx = {v0:.0f} mph\ndistance to pedestrian = {d0:.0f} m", transform=ax.transAxes,
            fontsize=15, color=INK, va="top")
    ax.text(0.02, 0.80, f"boxes: both cars at {k / Y.HZ:.1f} s  ·  dots every 1 s", transform=ax.transAxes,
            fontsize=11, color=MUTED, va="top")
    xs, ys = P(np.r_[orig[:, 0], edit[:, 0]], np.r_[orig[:, 1], edit[:, 1]])
    right = max(xs.max(), 10) + 6
    ax.set_xlim(-6, right)
    ax.set_ylim(-right * 7 / 16 / 2 * 1.6, right * 7 / 16 / 2 * 1.6)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("metres ahead", color=MUTED)
    ax.set_ylabel("metres (left ↑ ↓ right)", color=MUTED)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    fig.tight_layout()
    out = args.out or str(Y.OUT / f"bev_original_vs_edited_{args.scene[-12:]}_{ta}.png")
    fig.savefig(out, dpi=150)
    print(out, f"vx {v0:.1f} mph, gap now {d0:.1f} m, closest original {g_orig.min():.1f} m, edited {g_edit.min():.1f} m")


if __name__ == "__main__":
    main()
