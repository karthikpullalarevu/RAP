#!/usr/bin/env python3
"""Slide-ready figures (16:9 PNG) for the "why don't the models stop for pedestrians" report.

Reads results already on disk (pedestrian grid sweep, Gemini braking runs, E6 over-stop analysis)
and writes to /datahdd2/karthik/ped_v2/analysis/14_slides/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

X = Path("/datahdd2/karthik/RAP/experiments")
A = Path("/datahdd2/karthik/ped_v2/analysis")
OUT = A / "14_slides"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODEL_C = {"E4": "#2a78d6", "E8": "#eb6834", "E6": "#1baf7a", "E7": "#eda100"}       # categorical slots 1-4
MODEL_NAME = {"E4": "E4", "E8": "E8 (E4, no braking input)", "E6": "E6", "E7": "E7 (E6, no braking input)"}
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
AHEAD = [5, 10, 15, 20, 30]
SIDE = [6, 3, 1.5, 0, -1.5, -3, -6]

plt.rcParams.update({"font.size": 12, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.titlecolor": INK})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def sweep_heatmaps(df, col, title, fmt, vmax, name):
    models = ["E4", "E8", "E6", "E7"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 6.2))
    for ax, m in zip(axes, models):
        d = df[df.model == m]
        g = d.groupby(["ahead", "side"])[col].mean().unstack().reindex(index=AHEAD[::-1], columns=SIDE)
        ax.imshow(g.values, cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
        for (r, c), v in np.ndenumerate(g.values):
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=10.5,
                    color="white" if v / vmax > 0.55 else INK)
        ax.add_patch(plt.Rectangle((1.5, -0.5), 3, 5, fill=False, ec=INK, lw=1.6, ls="--"))
        ax.set_xticks(range(7), [f"{abs(s):g}{'L' if s > 0 else 'R' if s < 0 else ''}" for s in SIDE])
        ax.set_yticks(range(5), [f"{a} m" for a in AHEAD[::-1]])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(MODEL_NAME[m], fontsize=13)
        ax.set_xlabel("metres sideways")
    axes[0].set_ylabel("metres ahead")
    fig.suptitle(title, fontsize=16, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.015, "Dashed box = our car's path (car is 2.3 m wide).  20 Gemini scenes rendered as rasters, "
             "ego acceleration input set to 0.", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def sweep_bars(df):
    d = df.copy()
    d["zone"] = None
    d.loc[d.side.abs().le(1.5) & d.ahead.between(5, 20), "zone"] = "in path"
    d.loc[d.side.abs().eq(6), "zone"] = "side"
    g = d.dropna(subset=["zone"]).groupby(["model", "zone"]).agg(short=("shortening", "mean"), hits=("hits", "mean"))
    models = ["E4", "E8", "E6", "E7"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    x = np.arange(len(models))
    for ax, (col, ylab, title, fmt) in zip(axes, [
            ("short", "plan shortened (m)", "How much the 5 s plan shrinks because of the pedestrian", "{:.1f} m"),
            ("hits", "share of plans", "Plan still drives through the pedestrian (in path, 5-20 m)", "{:.0%}")]):
        if col == "short":
            for k, (zone, hatch, lab) in enumerate([("in path", None, "pedestrian in our path, 5-20 m ahead"),
                                                    ("side", "//", "pedestrian 6 m to the side")]):
                vals = [g.loc[(m, zone), col] for m in models]
                bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=[MODEL_C[m] for m in models],
                              alpha=1 if zone == "in path" else 0.35, hatch=hatch, ec=SURFACE, lw=2, label=lab)
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.08, fmt.format(v), ha="center", fontsize=11, color=INK2)
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(fc=INK2, label="pedestrian in our path, 5-20 m ahead"),
                               Patch(fc=INK2, alpha=0.35, hatch="//", ec=SURFACE, label="pedestrian 6 m to the side")],
                      frameon=False, fontsize=11, loc="upper right")
            ax.set_ylim(0, max(g["short"]) * 1.25)
        else:
            vals = [g.loc[(m, "in path"), col] for m in models]
            bars = ax.bar(x, vals, 0.55, color=[MODEL_C[m] for m in models], ec=SURFACE, lw=2)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, fmt.format(v), ha="center", fontsize=11, color=INK2)
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xticks(x, [MODEL_NAME[m].replace(" (", "\n(") for m in models])
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=13, loc="left")
        clean(ax)
    fig.suptitle("Pedestrian sweep: the models react to a pedestrian in their path, but rarely stop for it",
                 fontsize=16, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "03_sweep_in_path_vs_side.png", dpi=150)
    plt.close(fig)


def gemini_braking():
    src = {"0": "gemini_noAccel_E4E6E7E8/walk_left", "-1": "E6_pedV2yield_sub5_infonce/gemini_raster_walk_left_brake1",
           "-2": "E6_pedV2yield_sub5_infonce/gemini_raster_walk_left_brake2"}
    ds = []
    for a, p in src.items():
        d = pd.read_csv(X / p / "rgb_vs_raster_summary.csv")
        d = d[d.model.isin(["E4", "E6 gated"])].copy()
        d["accel"] = float(a)
        ds.append(d)
    d = pd.concat(ds)
    d["model"] = d.model.str.replace(" gated", "")
    d["stop_ped"] = d.end_mph_ped < 1
    d["stop_np"] = d.end_mph_no_ped < 1
    g = d.groupby(["input", "model", "accel"])[["stop_ped", "stop_np"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), sharey=True)
    accels = [0.0, -1.0, -2.0]
    for ax, inp, title in zip(axes, ["photo", "raster"], ["Photo input (pedestrian added by Gemini)",
                                                          "Raster input (pedestrian box)"]):
        for m in ["E4", "E6"]:
            for col, ls, mk, lab in [("stop_ped", "-", "o", "with pedestrian"), ("stop_np", "--", "s", "no pedestrian")]:
                v = [g.loc[(inp, m, a), col] for a in accels]
                ax.plot(range(3), v, ls, color=MODEL_C[m], lw=2, marker=mk, ms=8, mec=SURFACE, mew=2,
                        label=f"{m}, {lab}")
                ax.text(2.08, v[-1], f"{m} {lab.split()[0]}", fontsize=10.5, color=INK2, va="center")
        ax.set_xticks(range(3), ["0", "-1 m/s²\n(-2.2 mph/s)", "-2 m/s²\n(-4.5 mph/s)"])
        ax.set_xlim(-0.2, 2.7)
        ax.set_xlabel("ego braking given to the model as input")
        ax.set_ylim(0, 0.7)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, fontsize=13, loc="left")
        clean(ax)
    axes[0].set_ylabel("plans that stop")
    axes[0].legend(frameon=False, fontsize=10.5, loc="upper left", ncol=2)
    fig.suptitle("Gemini scenes: the braking input, not the pedestrian, decides whether E4 / E6 stop "
                 "(144 scenes)", fontsize=16, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "04_gemini_braking_shortcut.png", dpi=150)
    plt.close(fig)


def overstop_groups():
    d = pd.read_csv(A / "13_e6_overstops/e6_overstops.csv")
    order = ["crosses our path", "near our path", "away from path"]
    labels = ["crosses our path\n(within 1.2 m)", "near our path\n(1.2-3 m)", "away from our path\n(> 3 m)"]
    n = d.group.value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(16, 6.2))
    bars = ax.barh(range(3)[::-1], n.values, 0.6, color="#2a78d6", ec=SURFACE, lw=2)
    for b, v in zip(bars, n.values):
        ax.text(v + 10, b.get_y() + b.get_height() / 2, f"{v}  ({v / n.sum():.0%})", va="center", fontsize=13, color=INK2)
    ax.set_yticks(range(3)[::-1], labels, fontsize=13)
    ax.set_xlabel("held-out ped_v2 samples where E6 stops but the label keeps moving")
    ax.set_xlim(0, n.max() * 1.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"E6 stops in {len(d)} held-out samples where the label doesn't — "
                 "where is the pedestrian in the next 4 s?", fontsize=16, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "05_e6_overstop_groups.png", dpi=150)
    plt.close(fig)


def crop_examples():
    """First 2 example rows of the over-stop grids, for slides."""
    for src, dst in [("1_crosses_our_path.jpg", "06_e6_overstop_examples_crosses.png"),
                     ("3_away_from_path.jpg", "07_e6_overstop_examples_away.png")]:
        im = Image.open(A / "13_e6_overstops" / src)
        w, h = im.size
        row = (h - 100) / 8
        im.crop((0, 0, w, int(100 + 2 * row) - 14)).save(OUT / dst)


def sweep_speed(df):
    """In-path reaction by the car's current speed, and per-scene speed vs stop rate."""
    d = df[df.side.abs().le(1.5) & df.ahead.between(5, 20)].copy()
    bands = ["< 10 mph", "10-20 mph", "> 20 mph"]
    d["band"] = pd.cut(d.ego_mph, [-1, 10, 20, 99], labels=bands)
    models = ["E4", "E8", "E6", "E7"]
    g = d.groupby(["band", "model"], observed=True).agg(stop=("stopped", "mean"), n=("token", "nunique"))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    x = np.arange(len(bands))
    w = 0.2
    for k, m in enumerate(models):
        vals = [g.loc[(b, m), "stop"] for b in bands]
        bars = ax.bar(x + (k - 1.5) * w, vals, w * 0.92, color=MODEL_C[m], ec=SURFACE, lw=1.5, label=MODEL_NAME[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.0%}", ha="center", fontsize=9.5, color=INK2)
    ax.set_xticks(x, [f"{b}\n({g.loc[(b, 'E4'), 'n']} scenes)" for b in bands])
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("plans that stop")
    ax.set_xlabel("car's speed now")
    ax.set_title("Stop rate with a pedestrian in our path (5-20 m ahead)", fontsize=13, loc="left")
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    clean(ax)
    ax = axes[1]
    per = d.groupby(["token", "model"]).agg(stop=("stopped", "mean"), mph=("ego_mph", "first")).reset_index()
    for k, m in enumerate(models):
        p = per[per.model == m]
        ax.scatter(p.mph + (k - 1.5) * 0.35, p.stop, s=70, color=MODEL_C[m], ec=SURFACE, lw=1.5, label=m, zorder=3)
    ax.set_xlabel("car's speed now (mph)")
    ax.set_ylabel("plans that stop (per scene)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("Each dot = one scene", fontsize=13, loc="left")
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    clean(ax)
    fig.suptitle("Pedestrian sweep: the models stop almost only when the car is already slow",
                 fontsize=16, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "08_sweep_by_speed.png", dpi=150)
    plt.close(fig)


def val_curves():
    c = pd.read_csv(OUT / "val_curves.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(16, 6.4))
    for m in ["E4", "E8", "E6", "E7"]:
        s = c[m].dropna()
        ax.plot(s.index, s.values, "-", color=MODEL_C[m], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.2,
                label=MODEL_NAME[m])
        ax.text(s.index[-1] + 0.25, s.values[-1], m, fontsize=11, color=INK2, va="center")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation PDM score")
    ax.set_xticks(range(0, 20, 2))
    ax.set_xlim(-0.5, 20.5)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    clean(ax)
    ax.set_title("Validation PDM score during training (val/score logged each epoch)", fontsize=16, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "09_val_curves.png", dpi=150)
    plt.close(fig)


def gemini_noaccel():
    ds = [pd.read_csv(X / f"gemini_noAccel_E4E6E7E8/walk_{w}/rgb_vs_raster_summary.csv") for w in ("left", "right")]
    d = pd.concat(ds)
    d["model"] = d.model.str.replace(" gated", "")
    d["stop_ped"] = d.end_mph_ped < 1
    d["stop_np"] = d.end_mph_no_ped < 1
    d["react"] = d.dist_5s_no_ped - d.dist_5s_ped
    g = d.groupby(["input", "model"]).agg(stop_ped=("stop_ped", "mean"), stop_np=("stop_np", "mean"),
                                          react=("react", "mean"), hits=("hits", "mean"))
    models = ["E4", "E8", "E6", "E7"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4))
    x = np.arange(2)
    for ax, (col, title, fmt, top) in zip(axes, [
            ("stop_ped", "Stops, with pedestrian", "{:.0%}", 0.45),
            ("react", "Plan shortened by pedestrian (m)", "{:.1f}", 5.5),
            ("hits", "Plan goes through pedestrian's spot", "{:.0%}", 1.0)]):
        for k, m in enumerate(models):
            vals = [g.loc[(i, m), col] for i in ("photo", "raster")]
            bars = ax.bar(x + (k - 1.5) * 0.2, vals, 0.18, color=MODEL_C[m], ec=SURFACE, lw=1.5, label=MODEL_NAME[m])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + top * 0.012, fmt.format(v), ha="center", fontsize=9,
                        color=INK2)
        ax.set_xticks(x, ["photo", "raster"])
        ax.set_ylim(0, top)
        if fmt.endswith("%}"):
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, fontsize=13, loc="left")
        clean(ax)
    axes[0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.suptitle("Gemini scenes with braking input 0: E7 / E8 react more to the pedestrian (288 runs per bar)",
                 fontsize=16, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "10_gemini_noaccel_4models.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(X / "ped_grid_sweep/run20/grid_sweep.csv")
    vmax = float(np.ceil(df.groupby(["model", "ahead", "side"]).shortening.mean().max()))
    sweep_heatmaps(df, "shortening", "Pedestrian sweep: how many metres shorter the 5 s plan gets", "{:.1f}", vmax,
                   "01_sweep_plan_shortening.png")
    sweep_heatmaps(df, "stopped", "Pedestrian sweep: share of plans that stop (below 1 mph at 5 s)", "{:.0%}", 1.0,
                   "02_sweep_stop_rate.png")
    sweep_bars(df)
    gemini_braking()
    overstop_groups()
    crop_examples()
    sweep_speed(df)
    val_curves()
    gemini_noaccel()
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
