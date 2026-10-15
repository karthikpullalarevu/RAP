#!/usr/bin/env python3
"""Look inside one prediction: all 64 proposals, their real PDM scores, and which one gets picked.

For one held-out ped_v2 validation sample (ego moving >= 10 mph, a real pedestrian 4-12 m ahead
in ego's path, same group as the "native" samples in debug_ped_box_control.py):
  - runs the model and keeps all 64 proposals with the head's predicted scores
  - scores every proposal, and the label, with the real PDM scorer (metric cache)
  - draws one figure: front raster, top-down view of all 64 coloured by real PDM score,
    the four trajectories that matter, predicted vs real score, speed over time, and a table

The four trajectories:
  label           what PDM-Closed drove in InterPlan (the ground truth)
  picked          highest predicted score: what the model actually outputs
  best real PDMS  highest real PDM score among the 64
  closest to GT   smallest L1 distance to the label: the one the imitation loss trains

Usage:
    CUDA_VISIBLE_DEVICES=7 RAY_TMPDIR=/datahdd2/tmp/ray_eval python debug_all_proposals.py \
        --ckpt /datahdd2/karthik/RAP/experiments/E4_pedV2-105k_photo50k/epoch15.ckpt \
        --output-dir /datahdd2/karthik/RAP/experiments/E4_pedV2-105k_photo50k/all_proposals --pick 0 1 2
"""

import argparse
import glob
import lzma
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon
from shapely import affinity

import debug_ped_box_control as B
from eval_ped_val import load_agent

PED_V2 = Path("/datahdd2/karthik/ped_v2")
MPH = 2.23694
DT = 0.5
SCORE_NAMES = ["No collision", "Drivable area", "Progress", "TTC ok", "Comfort", "PDM score"]
SCORE_HEADERS = ["No\ncollision", "Drivable\narea", "Progress", "TTC ok", "Comfort", "PDM\nscore"]

# ego size (Chrysler Pacifica, the nuPlan ego), poses are at the rear axle
EGO_LEN, EGO_WID, REAR_TO_CENTRE = 5.176, 2.297, 1.461

# colours (dataviz reference palette, light mode)
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
ROAD, OTHER_CAR, PED = "#ecebe7", "#b9b8b2", "#d03b3b"
KEY = {"label": INK, "picked": "#eb6834", "best real PDMS": "#1baf7a", "closest to GT": "#6250d6"}
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blues", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])


def to_local(xy, origin):
    """Global xy (N, 2) -> ego frame (x forward, y left) at the rear axle."""
    c, s = np.cos(origin.heading), np.sin(origin.heading)
    d = np.asarray(xy) - [origin.x, origin.y]
    return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], axis=1)


def ego_box(x, y, h):
    """Ego footprint polygon for a rear-axle pose in the ego frame."""
    box = Polygon([(-EGO_LEN / 2, -EGO_WID / 2), (EGO_LEN / 2, -EGO_WID / 2),
                   (EGO_LEN / 2, EGO_WID / 2), (-EGO_LEN / 2, EGO_WID / 2)])
    box = affinity.translate(box, REAR_TO_CENTRE, 0)
    box = affinity.rotate(box, h, origin=(0, 0), use_radians=True)
    return affinity.translate(box, x, y)


def scene_from_cache(path):
    """Pedestrian boxes over time, other agents now, and road polygons, all in the ego frame."""
    m = pickle.load(lzma.open(path, "rb"))
    o, origin = m.observation, m.ego_state.rear_axle
    peds = [t for t, a in o._unique_objects.items() if "PEDESTRIAN" in str(a.tracked_object_type)]

    def local_poly(g):
        return Polygon(to_local(np.array(g.exterior.coords), origin))

    ped_track = {}                                          # token -> list of (time s, polygon)
    for i, occ in enumerate(o._occupancy_maps):
        for t in peds:
            if t in occ.token_to_idx:
                ped_track.setdefault(t, []).append((i * 0.1, local_poly(occ._geometries[occ.token_to_idx[t]])))
    now = o._occupancy_maps[0]
    others = [local_poly(now._geometries[now.token_to_idx[t]]) for t in now.tokens if t not in peds]
    road = [local_poly(g) for g in m.drivable_area_map._geometries]
    return ped_track, others, road


def closest_gap(traj, ped_track):
    """Smallest gap (m) between the ego footprint along traj and any pedestrian at the same time."""
    best = np.inf
    for track in ped_track.values():
        at = {round(t, 1): p for t, p in track}
        for k, (x, y, h) in enumerate(traj[:8]):
            p = at.get(round((k + 1) * DT, 1))
            if p is not None:
                best = min(best, ego_box(x, y, h).distance(p))
    return best


def flip(poly):
    """Ego-frame polygon -> plot coordinates (ahead on the x axis, left up)."""
    return np.asarray(poly.exterior.coords)[:, :2]


def speeds_mph(traj):
    xy = np.vstack([[0, 0], traj[:, :2]])
    return np.linalg.norm(np.diff(xy, axis=0), axis=1) / DT * MPH


def pick_samples():
    """Same 'native' group as debug_ped_box_control.py, same random order."""
    val = pd.read_csv(PED_V2 / "analysis/05_selection/val_samples.csv")
    frames = pd.read_csv(PED_V2 / "analysis/02_ped_frames.csv",
                         usecols=["scene_name", "frame_idx", "ego_speed_ms", "ped_local_x_m", "ped_local_y_m",
                                  "n_inserted_peds"])
    s = val.merge(frames, on=["scene_name", "frame_idx"])
    s = s[(s.ego_speed_ms * MPH >= 10) & (s.n_inserted_peds == 1)]
    native = s[(s.ped_local_x_m >= 4) & (s.ped_local_x_m <= 12) & (s.ped_local_y_m.abs() < 1.5)]
    return native.sample(min(len(native), 150), random_state=0).reset_index(drop=True)


def run_one(agent, r, out):
    from navsim.agents.rap_dino.score_module.compute_navsim_score import get_sub_score

    feat, label = B.load_sample(r.scene_name, int(r.frame_idx), r.token)
    with torch.no_grad():
        f = {k: v.unsqueeze(0).cuda() for k, v in feat.items()
             if isinstance(v, torch.Tensor) and k not in ("camera_valid", "rendered_camera_feature")}
        o = agent.forward(f, return_score=True)
    props = o["trajectory"][0].float().cpu().numpy()          # (64, 8, 3)
    pred_score = o["score"][0].float().cpu().numpy()          # (64,)

    cache = glob.glob(str(PED_V2 / "metric_cache" / "*" / "*" / r.token / "metric_cache.pkl"))[0]
    import navsim.agents.rap_dino.score_module.compute_navsim_score as CNS
    real = get_sub_score(cache, np.concatenate([props, label[None]]), True)[0]   # (65, 6)
    obj_type = {t: str(a.tracked_object_type).split(".")[-1].lower()
                for t, a in pickle.load(lzma.open(cache, "rb")).observation._unique_objects.items()}

    def culprit(i):
        """What the at-fault collision / TTC failure of proposal i is with."""
        hit = [obj_type.get(t, "object") for t in CNS.scorer.proposal_fault_collided_track_ids[i]]
        ttc = [obj_type.get(t, "object") for t in CNS.scorer.ttc_collided_track_ids[i]]
        parts = ([f"hits {hit[0]}"] if hit else []) + ([f"TTC: {ttc[0]}"] if ttc and not hit else [])
        return ", ".join(parts) or "—"
    culprits = [culprit(i) for i in range(65)]
    real_props, real_label = real[:64], real[64]
    ped_track, others, road = scene_from_cache(cache)

    i_pick = int(np.argmax(pred_score))
    i_best = int(np.argmax(real_props[:, -1]))
    i_close = int(np.argmin(np.abs(props[:, :8] - label[None, :8]).sum(-1).mean(-1)))
    why = {"label": culprits[64], "picked": culprits[i_pick], "best real PDMS": culprits[i_best],
           "closest to GT": culprits[i_close]}
    key = {"label": (label, real_label, np.nan), "picked": (props[i_pick], real_props[i_pick], pred_score[i_pick]),
           "best real PDMS": (props[i_best], real_props[i_best], pred_score[i_best]),
           "closest to GT": (props[i_close], real_props[i_close], pred_score[i_close])}

    gaps = np.array([closest_gap(p, ped_track) for p in props])
    end_mph = np.array([speeds_mph(p)[-1] for p in props])
    n_hit = int((real_props[:, 0] == 0).sum())
    n_ttc = int((real_props[:, 3] == 0).sum())
    n_slow = int((end_mph < 5).sum())
    ego_mph = r.ego_speed_ms * MPH

    # ---------------- figure ----------------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                         "axes.titleweight": "bold", "axes.titlesize": 11})
    fig = plt.figure(figsize=(20, 14), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.38, wspace=0.2)
    fig.suptitle(f"One sample, all 64 proposals  ·  ego {ego_mph:.0f} mph, pedestrian {r.ped_local_x_m:.1f} m ahead"
                 f"  ·  {r.reaction} scene  ·  token {r.token}", fontsize=14, fontweight="bold", color=INK, y=0.985)

    # A: what the model sees
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(B.front_image(feat))
    ax.set_title("A. What the model sees (front raster)", loc="left")
    ax.axis("off")

    xmax = max(45, float(np.max(props[:, :, 0])) + 5, float(label[:, 0].max()) + 5)
    lat = max(8.0, float(np.abs(props[:, :, 1]).max()) + 3, float(np.abs(label[:, 1]).max()) + 3)
    xlim, ylim = (-6, xmax), (-lat, lat)

    def bev(ax, title):
        ax.set_facecolor(SURFACE)
        for p in road:
            ax.add_patch(MplPolygon(flip(p), closed=True,
                                    fc=ROAD, ec="none", zorder=0))
        for p in others:
            ax.add_patch(MplPolygon(flip(p), closed=True,
                                    fc=OTHER_CAR, ec="none", zorder=1))
        for track in ped_track.values():
            pts = np.array([[t, *p.centroid.coords[0]] for t, p in track if t <= 4.0 + 1e-6])
            if len(pts) == 0:
                continue
            ax.plot(pts[:, 1], pts[:, 2], color=PED, lw=1.5, ls=":", zorder=6)
            for t, p in track:
                if abs(t) < 1e-6:
                    ax.add_patch(MplPolygon(flip(p), closed=True,
                                            fc=PED, ec=SURFACE, lw=1, zorder=7))
            ax.scatter([pts[0, 1]], [pts[0, 2]], s=45, color=PED, ec=SURFACE, lw=1, zorder=9)
            ax.annotate("ped", (pts[0, 1], pts[0, 2]), xytext=(4, 5), textcoords="offset points",
                        color=INK, fontsize=9, zorder=9)
        e = ego_box(0, 0, 0)
        ax.add_patch(MplPolygon(flip(e), closed=True,
                                fc="none", ec=INK, lw=1.5, zorder=8))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_xlabel("metres ahead"); ax.set_ylabel("← right · left → (m)")
        ax.set_title(title, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # B: all 64 coloured by real PDM score
    ax = fig.add_subplot(gs[0, 1:])
    bev(ax, f"B. All 64 proposals, coloured by real PDM score  ·  {n_hit} hit something, "
            f"{n_ttc} fail time-to-collision, {n_slow} end below 5 mph")
    norm = matplotlib.colors.Normalize(0, 1)
    for i in np.argsort(real_props[:, -1]):
        p = np.vstack([[0, 0], props[i, :, :2]])
        ax.plot(p[:, 0], p[:, 1], color=BLUES(norm(real_props[i, -1])), lw=1.2, alpha=0.9, zorder=3)
    lp = np.vstack([[0, 0], label[:, :2]])
    ax.plot(lp[:, 0], lp[:, 1], color=INK, lw=2, ls="--", zorder=5)
    ax.annotate("label", (lp[-1, 0], lp[-1, 1]), xytext=(6, -3), textcoords="offset points", color=INK, fontsize=9)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=BLUES), ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("real PDM score (0 = bad, 1 = good)")

    # C: the four trajectories that matter
    ax = fig.add_subplot(gs[1, :2])
    bev(ax, "C. The four trajectories that matter")
    for name, (traj, sc, ps) in key.items():
        p = np.vstack([[0, 0], traj[:, :2]])
        ax.plot(p[:, 0], p[:, 1], color=KEY[name], lw=2.2, ls="--" if name == "label" else "-",
                marker="o", ms=4, zorder=5, label=name)
    ax.legend(loc="lower right", ncol=2, frameon=False, fontsize=9)
    ax.text(0.01, 0.97, "red dot: pedestrian now · red dotted: where it walks in the next 4 s · grey: other vehicles",
            transform=ax.transAxes, va="top", fontsize=9, color=INK2)

    # D: predicted vs real score
    ax = fig.add_subplot(gs[1, 2])
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=":", zorder=1)
    ax.scatter(pred_score, real_props[:, -1], s=40, c=BLUES(norm(real_props[:, -1])), ec=SURFACE, lw=1, zorder=2)
    for name in ("picked", "best real PDMS", "closest to GT"):
        i = {"picked": i_pick, "best real PDMS": i_best, "closest to GT": i_close}[name]
        ax.scatter(pred_score[i], real_props[i, -1], s=150, fc="none", ec=KEY[name], lw=2.5, zorder=3)
        off = {"picked": (8, -14), "best real PDMS": (8, 6), "closest to GT": (8, 6)}[name]
        ax.annotate(name, (pred_score[i], real_props[i, -1]), xytext=off, textcoords="offset points",
                    color=INK, fontsize=9)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("score predicted by the head (used to pick)")
    ax.set_ylabel("real PDM score")
    ax.set_title("D. Does the head rank them right?", loc="left")
    ax.grid(color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # E: speed over time
    ax = fig.add_subplot(gs[2, 0])
    t = np.arange(1, 9) * DT
    for i in range(64):
        ax.plot(t, speeds_mph(props[i])[:8], color="#d8d7d2", lw=0.8, zorder=1)
    for name, (traj, sc, ps) in key.items():
        ax.plot(t, speeds_mph(traj)[:8], color=KEY[name], lw=2.2, ls="--" if name == "label" else "-",
                marker="o", ms=4, zorder=3, label=name)
    ax.axhline(ego_mph, color=INK2, lw=0.8, ls=":")
    ax.annotate(f"speed now {ego_mph:.0f} mph", (0.5, ego_mph), xytext=(0, 4), textcoords="offset points",
                color=INK2, fontsize=9)
    ax.set_xlabel("seconds from now"); ax.set_ylabel("speed (mph)")
    ax.set_ylim(bottom=0)
    ax.set_title("E. Speed over time (grey: the other proposals)", loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.grid(color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # F: table
    ax = fig.add_subplot(gs[2, 1:])
    ax.axis("off")
    cols = ["trajectory"] + SCORE_HEADERS + ["predicted\nscore", "closest gap\nto ped (m)", "speed at\n4 s (mph)", "failure is\nabout"]
    rows = []
    for name, (traj, sc, ps) in key.items():
        gap = closest_gap(traj, ped_track)
        rows.append([name] + [f"{v:.2f}" for v in sc] + ["—" if np.isnan(ps) else f"{ps:.2f}",
                                                         f"{gap:.1f}", f"{speeds_mph(traj)[7]:.0f}", why[name]])
    tb = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 2.6)
    tb.auto_set_column_width(list(range(len(cols))))
    for (ri, ci), cell in tb.get_celld().items():
        cell.set_edgecolor("#e4e3df")
        if ri == 0:
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_facecolor("#f0efec")
        elif ci == 0:
            cell.set_text_props(fontweight="bold", color=KEY[rows[ri - 1][0]])
    ax.set_title("F. Scores of the four trajectories (real PDM sub-scores; 1 = good)", loc="left")

    fig.subplots_adjust(top=0.94)
    fname = out / f"all_proposals_{r.token}.png"
    fig.savefig(fname, dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # numbers for all 64, for later
    df = pd.DataFrame(real_props, columns=SCORE_NAMES)
    df.insert(0, "proposal", range(64))
    df["predicted score"] = pred_score
    df["closest gap to ped (m)"] = gaps
    df["speed at 4 s (mph)"] = [speeds_mph(p)[7] for p in props]
    df["failure is about"] = culprits[:64]
    df["picked"] = df.proposal == i_pick
    df["closest to GT"] = df.proposal == i_close
    df.to_csv(out / f"all_proposals_{r.token}.csv", index=False)
    print(f"{r.token}: ego {ego_mph:.0f} mph, ped {r.ped_local_x_m:.1f} m | picked real {real_props[i_pick, -1]:.2f}"
          f" (pred {pred_score[i_pick]:.2f}), best real {real_props[i_best, -1]:.2f}, label {real_label[-1]:.2f}"
          f" | {n_hit} hit, {n_ttc} TTC fail, {n_slow} end <5 mph | min gap picked {gaps[i_pick]:.1f} m"
          f" -> {fname}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pick", type=int, nargs="+", default=[0], help="row(s) of the native sample list")
    ap.add_argument("--token", default=None, help="use this sample token instead of --pick")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = pick_samples()
    if args.token:
        val = pd.read_csv(PED_V2 / "analysis/05_selection/val_samples.csv")
        frames = pd.read_csv(PED_V2 / "analysis/02_ped_frames.csv",
                             usecols=["scene_name", "frame_idx", "ego_speed_ms", "ped_local_x_m", "ped_local_y_m"])
        rows = val[val.token == args.token].merge(frames, on=["scene_name", "frame_idx"]).head(1)
    else:
        rows = samples.iloc[args.pick]

    agent = load_agent(args.ckpt, True, True)
    for r in rows.itertuples():
        run_one(agent, r, out)


if __name__ == "__main__":
    main()
