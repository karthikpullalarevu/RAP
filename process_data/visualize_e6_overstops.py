#!/usr/bin/env python3
"""Are E6's extra stops on the held-out ped_v2 val set valid?

Takes samples where E6 (gated_pdms) stops but the label does not, and measures, from the metric cache
(real future of every agent), how close the inserted pedestrian comes to the label's path in the next 4 s:
  crosses our path : pedestrian comes within 1.2 m of the label path centre line (inside the car's width)
  near our path    : 1.2 - 3 m
  away from path   : more than 3 m
Also flags a red light on the label path. Then draws a grid of examples per group: top-down view
(ego frame, ego points up) with the label / E4 / E6 plans and the pedestrian's next 4 s, a speed plot (mph)
and the front raster the model sees.

Usage:
    python process_data/visualize_e6_overstops.py --per-group 8
"""

import argparse
import gzip
import lzma
import math
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Polygon

sys.path.insert(0, str(Path(__file__).parent.parent))

X = Path("/datahdd2/karthik/RAP/experiments")
E6 = X / "E6_pedV2yield_sub5_infonce"
E4 = X / "E4_pedV2-105k_photo50k"
VAL_CACHE = Path("/datahdd2/karthik/ped_v2/cache/val")
METRIC = Path("/datahdd/karthik/RAP/cache/train_metric_cache")
OUT = Path("/datahdd2/karthik/ped_v2/analysis/13_e6_overstops")
MPH = 2.23694
HALF_L, HALF_W = 2.588, 1.149          # ego box around the label pose (same as build_ped_cache.py)
GROUPS = ["crosses our path", "near our path", "away from path"]
C_LABEL, C_E4, C_E6, C_PED, C_AGENT, INK, MUTED = "#222222", "#2a78d6", "#1baf7a", "#eb6834", "#b5b5b5", "#222222", "#777777"


def to_local(xy, ego):
    """global (N, 2) -> ego frame (x forward, y left)"""
    c, s = math.cos(ego.rear_axle.heading), math.sin(ego.rear_axle.heading)
    d = np.asarray(xy) - [ego.rear_axle.x, ego.rear_axle.y]
    return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], -1)


def box(x, y, h):
    c, s = math.cos(h), math.sin(h)
    pts = [(HALF_L, HALF_W), (HALF_L, -HALF_W), (-HALF_L, -HALF_W), (-HALF_L, HALF_W)]
    return Polygon([(x + a * c - b * s, y + a * s + b * c) for a, b in pts])


def interp(traj, t):
    """label poses at 0.5 s (starting after now) -> pose at time t (s), from (0, 0, 0) at t = 0"""
    full = np.vstack([[0, 0, 0], traj])
    times = np.arange(len(full)) * 0.5
    return [np.interp(t, times, full[:, k]) for k in range(3)]


def scene_info(args):
    """Real future of the inserted pedestrian(s), agents and red lights, in the ego frame; distances to the label path."""
    token, sample_dir, peds = args
    try:
        mc = pickle.load(lzma.open(METRIC_PATHS[token], "rb"))
        label = np.asarray(pickle.load(gzip.open(sample_dir / "rap_target.gz"))["trajectory"])[:8]
    except Exception as e:   # noqa: BLE001
        return {"token": token, "error": str(e)}
    ego, obs = mc.ego_state, mc.observation
    path = LineString(np.vstack([[0, 0], label[:, :2]]))
    ped_xy, d_path, gap = {}, np.inf, np.inf
    red = []
    for t in range(41):                                   # 0 - 4 s at 0.1 s
        occ = obs[t]
        lx, ly, lh = interp(label, t * 0.1)
        car = box(lx, ly, lh)
        for tok in occ.tokens:
            if tok in peds:
                poly = Polygon(to_local(np.array(occ[tok].exterior.coords), ego))
                ped_xy.setdefault(tok, []).append(np.array(poly.centroid.coords[0]))
                d_path = min(d_path, poly.distance(path))
                gap = min(gap, poly.distance(car))
            elif t == 0 and tok.startswith(str(obs.red_light_token)):
                red.append(Polygon(to_local(np.array(occ[tok].exterior.coords), ego)))
    agents0 = [Polygon(to_local(np.array(obs[0][tok].exterior.coords), ego))
               for tok in obs[0].tokens if tok not in peds and not tok.startswith(str(obs.red_light_token))]
    red_on_path = any(r.distance(path) < 1.0 for r in red)
    drivable = []
    for g in mc.drivable_area_map._geometries:
        if g.distance(shapely.Point(ego.rear_axle.x, ego.rear_axle.y)) < 40:
            for gg in getattr(g, "geoms", [g]):
                drivable.append(to_local(np.array(gg.exterior.coords), ego))
    group = GROUPS[0] if d_path < 1.2 else GROUPS[1] if d_path < 3 else GROUPS[2]
    return {"token": token, "ped_to_path_m": d_path, "label_gap_m": gap, "red_light_on_path": red_on_path,
            "group": group, "ego_mph": float(np.hypot(*ego.dynamic_car_state.rear_axle_velocity_2d.array)) * MPH,
            "label": label, "ped_xy": {k: np.array(v) for k, v in ped_xy.items()}, "agents0": agents0,
            "red": red, "drivable": drivable}


def _init(paths):
    global METRIC_PATHS
    METRIC_PATHS = paths


def predict(tokens, dirs, ckpt, pick_mode):
    import torch
    from eval_ped_val import load_agent
    agent = load_agent(ckpt, True, True, pick_mode)
    out, front = {}, {}
    with torch.no_grad():
        for tok, d in zip(tokens, dirs):
            f = pickle.load(gzip.open(d / "rap_feature.gz"))
            feats = {k: v[None].cuda() for k, v in f.items()
                     if k not in ("camera_valid", "rendered_camera_feature", "frame_name")}
            out[tok] = agent.forward(feats)["trajectory"][0].float().cpu().numpy()
            front[tok] = f["camera_feature"][0].permute(1, 2, 0).numpy()
    del agent
    torch.cuda.empty_cache()
    return out, front


def speeds_mph(traj):
    full = np.vstack([[0, 0, 0], traj[:8]])
    return np.linalg.norm(np.diff(full[:, :2], axis=0), axis=-1) / 0.5 * MPH


def draw(rows, info, e6, e4, front, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.2 * n), gridspec_kw={"width_ratios": [1, 1.1, 1.6]})
    axes = np.atleast_2d(axes)
    P = lambda xy: (-np.asarray(xy)[..., 1], np.asarray(xy)[..., 0])     # ego points up, left is left
    for ax_bev, ax_v, ax_img, (_, r) in zip(axes[:, 0], axes[:, 1], axes[:, 2], rows.iterrows()):
        s = info[r.token]
        for d in s["drivable"]:
            ax_bev.fill(*P(d), color="#f1f1f1", lw=0)
        for a in s["agents0"]:
            ax_bev.fill(*P(np.array(a.exterior.coords)), color=C_AGENT, lw=0)
        for a in s["red"]:
            ax_bev.fill(*P(np.array(a.exterior.coords)), color="#d03b3b", alpha=0.6, lw=0)
        ax_bev.fill(*P(np.array(box(0, 0, 0).exterior.coords)), color="white", ec=INK, lw=1.2, zorder=3)
        for traj, c, lab, ls in [(s["label"], C_LABEL, "label", "--"), (e4[r.token], C_E4, "E4", "-"),
                                 (e6[r.token], C_E6, "E6", "-")]:
            xy = np.vstack([[0, 0], traj[:8, :2]])
            ax_bev.plot(*P(xy), ls, color=c, lw=2, marker="o", ms=3.5, label=lab, zorder=4)
        for xy in s["ped_xy"].values():
            ax_bev.plot(*P(xy), "-", color=C_PED, lw=1.5, zorder=5)
            ax_bev.plot(*P(xy[::10]), "o", color=C_PED, ms=4, zorder=5)          # every 1 s
            ax_bev.plot(*P(xy[0]), "o", color=C_PED, ms=9, mec="white", mew=1.5, zorder=6)
        ax_bev.set_xlim(-15, 15); ax_bev.set_ylim(-8, 40); ax_bev.set_aspect("equal")
        ax_bev.tick_params(labelsize=8, colors=MUTED)
        ax_bev.set_title(f"{r.token}  ·  {r.reaction}\nped to label path {s['ped_to_path_m']:.1f} m · "
                         f"label car gap {s['label_gap_m']:.1f} m" + (" · RED LIGHT" if s["red_light_on_path"] else ""),
                         fontsize=9, color=INK)
        t = np.arange(1, 9) * 0.5
        for traj, c, lab, ls in [(s["label"], C_LABEL, "label", "--"), (e4[r.token], C_E4, "E4", "-"),
                                 (e6[r.token], C_E6, "E6", "-")]:
            ax_v.plot(t, speeds_mph(traj), ls, color=c, lw=2, marker="o", ms=3.5, label=lab)
        ax_v.set_xlabel("seconds ahead", fontsize=8, color=MUTED); ax_v.set_ylabel("speed (mph)", fontsize=8, color=MUTED)
        ax_v.set_ylim(bottom=0); ax_v.grid(color="#e5e5e5", lw=0.6); ax_v.tick_params(labelsize=8, colors=MUTED)
        ax_v.set_title(f"now {s['ego_mph']:.1f} mph", fontsize=9, color=INK)
        img = front[r.token]
        img = (img - img.min()) / max(img.max() - img.min(), 1e-6)
        ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("front raster (model input)", fontsize=9, color=MUTED)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(title + "\norange = inserted pedestrian (big dot = now, small dots = every 1 s, next 4 s); grey = other agents now",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=8)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import json
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.agents.rap_dino.score_module.compute_navsim_score import INSERTED_PEDS_FILE

    r6 = pd.read_csv(E6 / "ped_val_batch3_ep15_gated_pdms/results.csv")
    r4 = pd.read_csv(E4 / "ped_val_batch3_epoch15/results.csv")[["token", "pred_stops", "no_at_fault_collisions"]]
    df = r6.merge(r4, on="token", suffixes=("", "_e4"))
    over = df[(df.pred_stops == 1) & (df.label_stops == 0)].copy()
    over["dir"] = [VAL_CACHE / f"{s}_{f:04d}" / t for s, f, t in zip(over.scene_name, over.frame_idx, over.token)]
    print(f"val samples {len(df)}, E6 stops {df.pred_stops.mean():.0%}, label stops {df.label_stops.mean():.0%}, "
          f"E6 stops where label doesn't: {len(over)}")

    peds = json.load(open(INSERTED_PEDS_FILE))
    paths = MetricCacheLoader(METRIC).metric_cache_paths
    jobs = [(t, d, set(peds.get(t.rsplit("_", 1)[0], []))) for t, d in zip(over.token, over["dir"])]
    with Pool(args.procs, initializer=_init, initargs=(paths,)) as pool:
        res = pool.map(scene_info, jobs, chunksize=8)
    info = {r["token"]: r for r in res if "error" not in r}
    print(f"read {len(info)} / {len(res)}")
    stats = pd.DataFrame([{k: r[k] for k in ("token", "ped_to_path_m", "label_gap_m", "red_light_on_path", "group", "ego_mph")}
                          for r in info.values()])
    over = over.merge(stats, on="token")
    over.drop(columns=["dir"]).to_csv(OUT / "e6_overstops.csv", index=False)

    lines = [f"E6 stops where the label keeps moving: {len(over)} of {len(df)} val samples", ""]
    g = over.groupby("group").agg(n=("token", "size"), E4_also_stops=("pred_stops_e4", "mean"),
                                  red_light=("red_light_on_path", "mean"), ped_to_path_m=("ped_to_path_m", "median"),
                                  label_gap_m=("label_gap_m", "median"), ego_mph=("ego_mph", "median"))
    g["share"] = g.n / g.n.sum()
    lines.append(g.reindex(GROUPS).round(2).to_string())
    lines += ["", "by reaction in the scene:",
              pd.crosstab(over.reaction, over.group).reindex(columns=GROUPS).to_string()]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    picks = pd.concat([over[over.group == grp].sample(min(args.per_group, (over.group == grp).sum()), random_state=args.seed)
                       for grp in GROUPS])
    dirs = dict(zip(over.token, [VAL_CACHE / f"{s}_{f:04d}" / t for s, f, t in zip(over.scene_name, over.frame_idx, over.token)]))
    e6, front = predict(list(picks.token), [dirs[t] for t in picks.token], E6 / "epoch15.ckpt", "gated_pdms")
    e4, _ = predict(list(picks.token), [dirs[t] for t in picks.token], E4 / "epoch15.ckpt", "final")
    for i, grp in enumerate(GROUPS):
        rows = picks[picks.group == grp]
        if len(rows):
            draw(rows, info, e6, e4, front, OUT / f"{i + 1}_{grp.replace(' ', '_')}.jpg",
                 f"E6 stops, label keeps moving — pedestrian {grp} ({len(over[over.group == grp])} samples)")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
