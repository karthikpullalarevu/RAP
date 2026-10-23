#!/usr/bin/env python3
"""One top-down slide of the yield edit: original vs edited 5 s trajectory for one training sample.

Car frame at the sample, car pointing up. Distance to pedestrian = car body (front) to pedestrian, now or when the
pedestrian first appears in the 5 s.

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

    def P(x, y):                                       # world -> car frame, car pointing up
        dx, dy = np.asarray(x) - ox, np.asarray(y) - oy
        return -(-dx * s + dy * c), dx * c + dy * s

    orig = sc.ego[ta:ta + m, :2]
    ex, ey, _ = sc.path.pose(r["s"][:m])

    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, ax = plt.subplots(figsize=(7, 7.5))
    cc, ss = math.cos(oh), math.sin(oh)
    corners = [(ox + a * cc - b * ss, oy + a * ss + b * cc) for a, b in
               [(Y.HALF_LEN, Y.HALF_WID), (Y.HALF_LEN, -Y.HALF_WID), (-Y.HALF_LEN, -Y.HALF_WID), (-Y.HALF_LEN, Y.HALF_WID)]]
    ax.add_patch(Polygon(np.column_stack(P(*np.array(corners).T)), fc="#d6d5cf", ec=INK, lw=1.5, zorder=4))
    ax.plot(*P(orig[:, 0], orig[:, 1]), color=ORIG, lw=5, label="original trajectory", solid_capstyle="round", zorder=3)
    ax.plot(*P(ex, ey), color=EDIT, lw=5, label="edited trajectory", solid_capstyle="round", zorder=3)
    for x, y, col in [(orig[:, 0], orig[:, 1], ORIG), (ex, ey, EDIT)]:      # dots every 1 s
        ax.plot(*P(x[Y.HZ::Y.HZ], y[Y.HZ::Y.HZ]), "o", color=col, ms=9, mec="white", mew=1.5, zorder=4)

    d0 = np.inf
    for tr in sc.peds:
        p = tr[ta:ta + m]
        ok = np.where(~np.isnan(p[:, 0]))[0]
        if not len(ok):
            continue
        i0, i1 = ok[0], ok[-1]
        d0 = min(d0, float(Y.box_gap(*sc.ego[ta, :3], p[i0, 0], p[i0, 1], p[i0, 2])))
        ax.annotate("", xy=P(p[i1, 0], p[i1, 1]), xytext=P(p[i0, 0], p[i0, 1]),
                    arrowprops=dict(arrowstyle="-|>", color=PEDC, lw=2.5, mutation_scale=20), zorder=5)
        ax.plot(*P(p[i0, 0], p[i0, 1]), "o", color=PEDC, ms=13, mec="white", mew=2, zorder=6)
        sec = [i for i in range(0, m, Y.HZ) if i > i0 and not np.isnan(p[i, 0])]
        ax.plot(*P(p[sec, 0], p[sec, 1]), "o", color=PEDC, ms=8, mec="white", mew=1.5, zorder=6)
    ax.plot([], [], "o", color=PEDC, ms=11, mec="white", label="pedestrian")

    ax.text(0.03, 0.98, f"vx = {sc.v_pdm[ta] * Y.MPH:.0f} mph\ndistance to pedestrian = {d0:.0f} m",
            transform=ax.transAxes, fontsize=14, color=INK, va="top")
    xs, ys = P(np.r_[orig[:, 0], ex], np.r_[orig[:, 1], ey])
    top = max(ys.max(), 10) + 8
    ax.set_ylim(-4, top)
    pxs = [P(tr[ta:ta + m, 0], tr[ta:ta + m, 1])[0] for tr in sc.peds]
    lo = min([-6.0] + [np.nanmin(v) - 1.5 for v in pxs if np.isfinite(v).any()])
    hi = max([6.0] + [np.nanmax(v) + 1.5 for v in pxs if np.isfinite(v).any()])
    ax.set_xlim(lo, hi)
    ax.set_aspect(0.45)                               # metres ahead drawn at 0.45x scale so the plot isn't so tall
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel("metres ahead", color=MUTED)
    ax.set_xlabel("metres", color=MUTED)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(loc="lower left", frameon=False, fontsize=12)
    fig.tight_layout()
    out = args.out or str(Y.OUT / f"bev_original_vs_edited_{args.scene[-12:]}_{ta}.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
