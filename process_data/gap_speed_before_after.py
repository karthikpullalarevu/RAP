#!/usr/bin/env python3
"""Closest gap to the inserted pedestrian vs ego speed at that moment, per training sample, before and after
the yield edit (same 5 s label window the model is trained on).

before = PDM-Closed's original label from each anchor (every sample in the close scenes, incl. later-dropped ones)
after  = the label in the E6 training set: edited label, or PDM-Closed's own if unchanged; dropped samples left out

Usage:
  python process_data/gap_speed_before_after.py --workers 48     # computes 10_gap_speed/gap_speed_samples.csv
  python process_data/gap_speed_before_after.py --plot           # pictures from that csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from process_data import edit_yield_label as Y

BATCH = Y.OUT / "batch"
OUT = Y.PED / "analysis" / "11_gap_speed_before_after"
M = Y.LABEL_STEPS * Y.STEP + 1          # 20 Hz steps in the 5 s label


def closest(g, v):
    j = int(np.nanargmin(g))
    return float(g[j]), float(v[j] * Y.MPH)


def one(job):
    scene, anchors = job
    try:
        sc = Y.Scene(scene)
        g_pdm = sc.gaps(0, sc.s_pdm)
        rows = []
        for r in Y.solve_scene(sc, sorted(anchors)):
            ta = r["anchor"]
            row = dict(scene=scene, anchor=ta, outcome=r["outcome"])
            row["gap_before"], row["mph_before"] = closest(g_pdm[ta:ta + M], sc.v_pdm[ta:ta + M])
            if r["outcome"] in ("edited", "edited_close"):
                row["gap_after"], row["mph_after"] = closest(sc.gaps(ta, r["s"][:M]), r["v"][:M])
            elif r["outcome"] == "unchanged":
                row["gap_after"], row["mph_after"] = row["gap_before"], row["mph_before"]
            rows.append(row)
        return rows, None
    except Exception as ex:
        return [], f"{scene}: {ex!r}"


def compute(workers):
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    a = pd.concat([pd.read_csv(BATCH / "yield_edit_all_anchors.csv"),
                   pd.read_csv(BATCH / "yield_edit_all_anchors_keyframes.csv")])
    a = a[a.in_train_cache == True].drop_duplicates(["scene", "anchor"])
    jobs = [(s, set(g.anchor)) for s, g in a.groupby("scene")]
    print(f"{len(a)} samples in {len(jobs)} scenes")
    rows, errs = [], []
    with ProcessPoolExecutor(workers) as ex:
        for r, e in tqdm(ex.map(one, jobs, chunksize=4), total=len(jobs)):
            rows += r
            if e:
                errs.append(e)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "gap_speed_samples.csv", index=False)
    (OUT / "errors.txt").write_text("\n".join(errs))
    print(f"saved {len(rows)} rows, {len(errs)} scene errors")


UNSAFE_MPH = 1.0                                   # red box: under 1 m from the pedestrian while moving faster than this


A_SAFE, MARGIN, BAND = 4.0, 1.0, 5.0     # m/s2 firm braking, m spare room, m width of the ideal training band


def safe_distance(mph):
    """Smallest safe closest distance at this speed: spare room + stopping distance at A_SAFE (0.5 m when creeping)."""
    v = np.asarray(mph) / Y.MPH
    return np.where(np.asarray(mph) <= 1.0, 0.5, MARGIN + v ** 2 / (2 * A_SAFE))


def plot_zones():
    """Before / after with speed-dependent zones: unsafe (below the stopping-distance curve), ideal for training
    (just above it), safe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(OUT / "gap_speed_samples.csv")
    before = d[["gap_before", "mph_before"]].dropna().to_numpy()
    after = d[["gap_after", "mph_after"]].dropna().to_numpy()
    BLUE, INK, MUTED, RED, GREEN = "#2a78d6", "#222222", "#777777", "#d62728", "#1baf7a"
    sp = np.linspace(0, 42, 400)
    curve = safe_distance(sp)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True)
    for ax, x, title in ((axes[0], before, "Before: original labels (PDM-Closed)"),
                         (axes[1], after, "After: edited labels (used to train E6)")):
        ax.fill_between(sp, 0, curve, color=RED, alpha=0.12, lw=0)
        ax.fill_between(sp, curve, curve + BAND, color=GREEN, alpha=0.16, lw=0)
        ax.plot(sp, curve, color=RED, lw=1.5)
        ax.scatter(x[:, 1], x[:, 0], s=4, alpha=0.18, color=BLUE, lw=0)
        need = safe_distance(x[:, 1])
        unsafe = (x[:, 0] < need).mean()
        ideal = ((x[:, 0] >= need) & (x[:, 0] < need + BAND)).mean()
        ax.set_title(f"{title}\n{len(x):,} samples · unsafe {unsafe:.0%} · ideal band {ideal:.0%} · "
                     f"far {1 - unsafe - ideal:.0%}", loc="left", fontsize=13)
        ax.set_xlim(-0.5, 42)
        ax.set_ylim(0, 15)
        ax.set_xlabel("Ego speed at the closest moment (mph)", fontsize=12)
        ax.grid(color="#eeeeee")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].set_ylabel("Closest distance, car body to pedestrian (m)", fontsize=12)
    fig.text(0.5, 0.005, f"Red curve = {MARGIN:g} m + stopping distance at {A_SAFE:g} m/s² ({A_SAFE * Y.MPH:.0f} mph/s); "
             "0.5 m when creeping (under 1 mph). One dot per training sample at its closest moment within the 5 s label.",
             ha="center", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "gap_vs_speed_before_after_zones.png", dpi=150)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(OUT / "gap_speed_samples.csv")
    before = d[["gap_before", "mph_before"]].dropna().to_numpy()
    after = d[["gap_after", "mph_after"]].dropna().to_numpy()
    near = lambda x: ((x[:, 0] < 1) & (x[:, 1] > UNSAFE_MPH)).sum()
    BLUE, INK, MUTED = "#2a78d6", "#222222", "#777777"
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, x, title in ((axes[0], before, "Before: original labels (PDM-Closed)"),
                         (axes[1], after, "After: edited labels (used to train E6)")):
        ax.scatter(x[:, 1], x[:, 0], s=4, alpha=0.18, color=BLUE, lw=0)
        ax.axhline(5, color=INK, lw=1, ls="--")
        ax.text(41, 5.15, "5 m", ha="right", fontsize=11, color=INK)
        ax.add_patch(plt.Rectangle((UNSAFE_MPH, 0), 45 - UNSAFE_MPH, 1, fc="#d62728", alpha=0.08, ec="#d62728", lw=1))
        n = near(x)
        ax.set_title(f"{title}\n{len(x):,} samples · under 1 m at over {UNSAFE_MPH:g} mph: {n:,} ({n / len(x):.1%})",
                     loc="left", fontsize=13)
        ax.set_xlim(-0.5, 42)
        ax.set_ylim(0, 10)
        ax.set_xlabel("Ego speed at the closest moment (mph)", fontsize=12)
        ax.grid(color="#eeeeee")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Closest distance, car body to pedestrian (m)", fontsize=12)
    fig.text(0.5, 0.005, "One dot per training sample at its closest moment within the 5 s label. Red box = under 1 m while "
             f"moving (over {UNSAFE_MPH:g} mph). After leaves out the samples that were too late to stop (dropped).", ha="center",
             fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "gap_vs_speed_before_after.png", dpi=150)

    # same numbers as the closest-moment heatmap bins
    gb = [0, 0.5, 1, 2, 5, 10, np.inf]
    sb = [0, 1, 5, 10, 20, np.inf]
    for name, x in (("before", before), ("after", after)):
        t = pd.crosstab(pd.cut(x[:, 1], sb, right=False), pd.cut(x[:, 0], gb, right=False))
        print(f"\n{name} ({len(x)} samples), rows = mph, cols = gap m\n{t}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--zones", action="store_true", help="before/after with speed-dependent safe / ideal / unsafe zones")
    args = ap.parse_args()
    plot_zones() if args.zones else plot() if args.plot else compute(args.workers)
