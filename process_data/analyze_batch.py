#!/usr/bin/env python3
"""Analyze an InterPlan pedestrian batch before training.

Two levels of analysis:
  A. Whole-scene overview (all frames)        -> 01_scene_summary.csv, 03_plots/A_*.png
  B. Pedestrian frames (inserted ped present) -> 02_ped_frames.csv,    03_plots/B_*.png
Plus analysis_report.json and a raster grid (04_raster_grid.png).

Usage:
    cd /datahdd/karthik/RAP
    export PYTHONPATH=/datahdd/karthik/RAP
    export NUPLAN_MAPS_ROOT=/datahdd/karthik/dataset/nuplan-maps-v1.0
    export NUPLAN_MAP_VERSION=nuplan-maps-v1.0

    python process_data/analyze_batch.py \
        --batch-root /datahdd2/karthik/ped_v2/raw \
        --output-dir /datahdd2/karthik/ped_v2/analysis \
        --workers 32

Batch labels come from <batch-root>/_batch_meta/<batch>/manifest.csv. If that folder
is missing, <batch-root>/manifest.csv is used and the batch is named after the folder.
Per-scene results are appended to <output-dir>/_partial.jsonl, so a re-run skips
scenes that are already done.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/datahdd/karthik/dataset/nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

from process_data.build_ped_cache import load_simulation_log, _patch_map_path, classify_camera

STOP_SPEED = 0.5          # m/s, below this ego counts as stopped
STUCK_WINDOW = 60         # frames (3 s at 20 Hz): stopped for the whole end of the scene = stuck
SLOW_RATIO = 0.7          # min speed < 70% of speed when ped appeared = slowed down

METRIC_COLS = ["score", "no_ego_at_fault_collisions", "time_to_collision_within_bound",
               "ego_is_comfortable", "ego_is_making_progress", "ego_progress_along_expert_route"]


# ---------------------------------------------------------------------------
# Batch metadata
# ---------------------------------------------------------------------------

def load_batch_meta(batch_root):
    """Return {scene_name: {"batch": ..., **manifest, **metrics}}."""
    meta_root = batch_root / "_batch_meta"
    if meta_root.is_dir():
        batch_dirs = [d for d in sorted(meta_root.iterdir()) if (d / "manifest.csv").exists()]
    else:
        batch_dirs = [batch_root]

    meta = {}
    for bdir in batch_dirs:
        batch = bdir.name
        with open(bdir / "manifest.csv", newline="") as f:
            for row in csv.DictReader(f):
                meta[row["scenario_name"]] = {"batch": batch, **row}
        metrics_path = bdir / "metrics_per_scene.csv"
        if metrics_path.exists():
            with open(metrics_path, newline="") as f:
                for row in csv.DictReader(f):
                    if row["scenario_name"] in meta:
                        meta[row["scenario_name"]].update({k: row[k] for k in METRIC_COLS if k in row})
    return meta


def parse_scene_config(path):
    """Pull context/scenario_type/group comment and configured ped speed/angle."""
    import yaml
    text = path.read_text()
    out = {"context": "", "scenario_type": "", "group": ""}
    m = re.search(r"context=(\S+)\s+scenario_type=(\S+)\s+group=(\S+)", text)
    if m:
        out["context"], out["scenario_type"], out["group"] = m.groups()
    try:
        cfg = yaml.safe_load(text)
        scen = next(iter(cfg["scenarios"].values()))
        peds = scen["tokens"][0].get("pedestrians", [])
        for i, p in enumerate(peds[:2], 1):
            out[f"cfg_ped{i}_speed"] = p.get("speed_mps")
            out[f"cfg_ped{i}_angle"] = p.get("crossing_angle_deg")
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Per-scene analysis
# ---------------------------------------------------------------------------

def _f(x, nd=2):
    """Round floats; NaN/inf -> '' so CSV/JSON stay clean."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return round(float(x), nd)


def analyze_scene(scene_dir, meta_row):
    scene_name = scene_dir.name
    log_files = list((scene_dir / "simulation_log").glob("*.msgpack.xz"))
    if not log_files:
        return {"scene_name": scene_name, "error": "no simulation log"}, []

    result = json.loads((scene_dir / "metadata" / "result.json").read_text())
    summary = result.get("saved_log_summary", {})
    inserted = set(summary.get("pedestrian_track_tokens", []))

    fov_path = scene_dir / "rendered" / "fov_frames.json"
    fov = set(json.loads(fov_path.read_text()).get("fov_frames", [])) if fov_path.exists() else set()
    raster_dir = scene_dir / "rendered" / "rasters"
    rasters = {int(p.name[6:]) for p in raster_dir.glob("frame_*")} if raster_dir.exists() else set()

    sim_log = load_simulation_log(log_files[0])
    data = sim_log.simulation_history.data
    n_frames = len(data)

    ego_speeds, ego_accs = [], []
    max_vehicles = max_orig_peds = 0
    ped_rows = []

    for i, sample in enumerate(data):
        ego = sample.ego_state
        speed = ego.dynamic_car_state.speed
        acc = ego.dynamic_car_state.rear_axle_acceleration_2d.x
        ego_speeds.append(speed)
        ego_accs.append(acc)

        tracks = sample.observation.tracked_objects.tracked_objects
        n_veh = sum(1 for t in tracks if t.tracked_object_type.name == "VEHICLE")
        peds = [t for t in tracks if t.tracked_object_type.name == "PEDESTRIAN"]
        ins = [p for p in peds if p.track_token in inserted]
        max_vehicles = max(max_vehicles, n_veh)
        max_orig_peds = max(max_orig_peds, len(peds) - len(ins))
        if not ins:
            continue

        # Nearest inserted ped by body-to-body gap
        foot = ego.car_footprint.geometry
        cx, cy, h = ego.center.x, ego.center.y, ego.center.heading
        gaps = [p.box.geometry.distance(foot) for p in ins]
        p = ins[int(np.argmin(gaps))]
        gap = min(gaps)

        dx, dy = p.center.x - cx, p.center.y - cy
        local_x = dx * math.cos(-h) - dy * math.sin(-h)
        local_y = dx * math.sin(-h) + dy * math.cos(-h)

        # Constant-velocity time-to-collision: gap / closing speed
        rvx = p.velocity.x - speed * math.cos(h)
        rvy = p.velocity.y - speed * math.sin(h)
        dist = math.hypot(dx, dy)
        closing = -(dx * rvx + dy * rvy) / dist if dist > 0 else 0.0
        ttc = gap / closing if closing > 0.1 else float("nan")

        ped_rows.append({
            "batch": meta_row.get("batch", ""),
            "scene_name": scene_name,
            "frame_idx": i,
            "ego_speed_ms": _f(speed),
            "ego_lon_acc_ms2": _f(acc),
            "n_inserted_peds": len(ins),
            "gap_m": _f(gap),
            "ped_side": classify_camera(local_x, local_y),
            "ped_local_x_m": _f(local_x),
            "ped_local_y_m": _f(local_y),
            "ped_speed_ms": _f(math.hypot(p.velocity.x, p.velocity.y)),
            "ttc_s": _f(ttc),
            "in_fov": int(i in fov),
            "has_raster": int(i in rasters),
        })

    ego_speeds = np.array(ego_speeds)
    ped_frames = [r["frame_idx"] for r in ped_rows]

    # Ego reaction while an inserted ped is around
    if ped_frames:
        v0 = ego_speeds[ped_frames[0]]
        vmin = ego_speeds[ped_frames].min()
        if vmin < STOP_SPEED:
            reaction = "stop"
        elif vmin < SLOW_RATIO * v0:
            reaction = "slow down"
        else:
            reaction = "keep going"
        closest = min(ped_rows, key=lambda r: r["gap_m"])
        ttcs = [r["ttc_s"] for r in ped_rows if r["ttc_s"] != ""]
    else:
        reaction, closest, ttcs = "no ped", None, []

    row = {
        "batch": meta_row.get("batch", ""),
        "scene_name": scene_name,
        "base_scene": scene_name.split("__")[0],
        "v4_status": meta_row.get("v4_status") or result.get("v4_generation", {}).get("v4_status", ""),
        "n_inserted_peds": len(inserted),
        **parse_scene_config(scene_dir / "metadata" / "scene_config.yaml"),
        "n_frames": n_frames,
        "ego_mean_speed_ms": _f(ego_speeds.mean()),
        "ego_max_speed_ms": _f(ego_speeds.max()),
        "ego_hardest_brake_ms2": _f(min(ego_accs)),
        "ego_stopped": int((ego_speeds < STOP_SPEED).any()),
        "ego_stuck_at_end": int((ego_speeds[-STUCK_WINDOW:] < STOP_SPEED).all()),
        "max_vehicles": max_vehicles,
        "max_orig_peds": max_orig_peds,
        "ego_ped_collision_frames": summary.get("ego_pedestrian_collision_frames", ""),
        "bg_ped_collision_frames": summary.get("background_pedestrian_collision_frames", ""),
        "ped_frames": len(ped_frames),
        "fov_frames": len(fov),
        "raster_frames": len(rasters),
        "min_gap_m": closest["gap_m"] if closest else "",
        "ego_speed_at_min_gap_ms": closest["ego_speed_ms"] if closest else "",
        "min_ttc_s": min(ttcs) if ttcs else "",
        "reaction": reaction,
        **{k: meta_row.get(k, "") for k in METRIC_COLS},
    }
    return row, ped_rows


def _worker(scene_dir, meta_row):
    try:
        return analyze_scene(scene_dir, meta_row)
    except Exception as e:
        return {"scene_name": scene_dir.name, "error": repr(e)}, []


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

BATCH_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # fixed categorical order
BAR_COLOR = "#2a78d6"


def _style(plt):
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#888888", "axes.grid": True, "grid.color": "#e5e5e5",
        "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold",
    })


def _num(rows, key):
    return np.array([float(r[key]) for r in rows if r.get(key) not in ("", None)])


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _bar_counts(ax, counter, title, xlabel="Scenes"):
    items = counter.most_common()
    labels = [str(k) for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    ax.barh(labels, vals, color=BAR_COLOR, height=0.7)
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v}", va="center", fontsize=9, color="#333333")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def _hist_by_batch(ax, rows, key, batches, bins, title, xlabel, ylabel="Scenes"):
    for b, c in zip(batches, BATCH_COLORS):
        vals = _num([r for r in rows if r["batch"] == b], key)
        if len(vals):
            ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=c, label=f"{b} (n={len(vals)})")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)


def make_plots(scenes, ped_rows, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    _style(plt)
    batches = sorted({r["batch"] for r in scenes})

    # ---------- A: whole-scene overview ----------
    fig, axes = plt.subplots(1, len(batches), figsize=(6 * len(batches), 3.5), squeeze=False)
    for ax, b in zip(axes[0], batches):
        _bar_counts(ax, Counter(r["v4_status"] for r in scenes if r["batch"] == b), f"Generation status — {b}")
    _save(fig, plot_dir / "A1_generation_status.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _bar_counts(axes[0], Counter(r["context"] or "unknown" for r in scenes), "Scene context")
    _bar_counts(axes[1], Counter(dict(Counter(r["scenario_type"] or "unknown" for r in scenes).most_common(15))),
                "Scenario type (top 15)")
    _save(fig, plot_dir / "A2_context_and_type.png"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _hist_by_batch(axes[0, 0], scenes, "score", batches, np.linspace(0, 1, 21), "nuPlan score per scene", "Score")
    _hist_by_batch(axes[0, 1], scenes, "ego_mean_speed_ms", batches, 30, "Ego mean speed per scene", "m/s")
    _hist_by_batch(axes[1, 0], scenes, "ego_hardest_brake_ms2", batches, 30, "Ego hardest braking per scene", "m/s² (more negative = harder)")
    _hist_by_batch(axes[1, 1], scenes, "max_vehicles", batches, 30, "Max vehicles in a frame", "Vehicles")
    fig.tight_layout(); _save(fig, plot_dir / "A3_scene_distributions.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    variants = Counter(Counter(r["base_scene"] for r in scenes).values())
    ks = sorted(variants)
    ax.bar([str(k) for k in ks], [variants[k] for k in ks], color=BAR_COLOR, width=0.7)
    ax.set_title("Variants per original NuPlan scene")
    ax.set_xlabel("Number of variants"); ax.set_ylabel("Base scenes")
    ax.grid(axis="x", visible=False)
    _save(fig, plot_dir / "A4_variants_per_base_scene.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    flags = {
        "ego–ped collision": sum(1 for r in scenes if r["ego_ped_collision_frames"] not in ("", 0, "0")),
        "car–ped collision (background)": sum(1 for r in scenes if r["bg_ped_collision_frames"] not in ("", 0, "0")),
        "ego stopped at some point": sum(r["ego_stopped"] for r in scenes),
        "ego stuck at end (last 3 s)": sum(r["ego_stuck_at_end"] for r in scenes),
        "not comfortable (nuPlan)": sum(1 for r in scenes if r["ego_is_comfortable"] in ("0.0", "0")),
    }
    _bar_counts(ax, Counter(flags), f"Scene flags (out of {len(scenes)} scenes)")
    _save(fig, plot_dir / "A5_scene_flags.png"); plt.close(fig)

    # ---------- B: pedestrian frames ----------
    blues = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
    groups = [("all batches", ped_rows)] + [(b, [r for r in ped_rows if r["batch"] == b]) for b in batches]
    fig, axes = plt.subplots(1, len(groups), figsize=(6.5 * len(groups), 5), squeeze=False)
    for ax, (name, rows) in zip(axes[0], groups):
        d, s = _num(rows, "gap_m"), _num(rows, "ego_speed_ms")
        if len(d) == 0:
            continue
        h = ax.hist2d(d, s, bins=[np.linspace(0, 60, 61), np.linspace(0, max(15, s.max()), 31)],
                      cmap=blues, norm=LogNorm(), cmin=1)
        fig.colorbar(h[3], ax=ax, label="Frames (log scale)")
        ax.set_title(f"Ego speed vs distance to ped — {name}")
        ax.set_xlabel("Gap between car and nearest inserted ped (m)")
        ax.set_ylabel("Ego speed (m/s)")
        ax.grid(False)
    fig.tight_layout(); _save(fig, plot_dir / "B1_speed_vs_distance_2d.png"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _hist_by_batch(axes[0, 0], scenes, "min_gap_m", batches, np.linspace(0, 30, 31),
                   "Closest gap to inserted ped per scene", "m")
    axes[0, 0].axvline(2, color="#555555", linestyle="--", linewidth=1)
    _hist_by_batch(axes[0, 1], ped_rows, "ped_speed_ms", batches, 30, "Ped speed (ped frames)", "m/s", "Frames")
    _hist_by_batch(axes[1, 0], scenes, "min_ttc_s", batches, np.linspace(0, 10, 41),
                   "Min time-to-collision per scene", "seconds")
    _hist_by_batch(axes[1, 1], scenes, "ego_speed_at_min_gap_ms", batches, 30,
                   "Ego speed at closest approach", "m/s")
    fig.tight_layout(); _save(fig, plot_dir / "B2_ped_distributions.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    _bar_counts(axes[0], Counter(r["reaction"] for r in scenes), "Ego reaction to ped (per scene)")
    _bar_counts(axes[1], Counter(r["ped_side"] for r in ped_rows), "Where the nearest ped is", "Frames")
    _bar_counts(axes[2], Counter(f"{r['n_inserted_peds']} ped(s) present" for r in ped_rows),
                "Inserted peds present", "Frames")
    fig.tight_layout(); _save(fig, plot_dir / "B3_reaction_side_count.png"); plt.close(fig)

    print(f"  Saved plots to {plot_dir}")


def make_raster_grid(scenes, batch_root, out_path, n=20, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = random.Random(seed)
    picks = rng.sample(scenes, min(n, len(scenes)))
    cols = 5
    fig, axes = plt.subplots(math.ceil(len(picks) / cols), cols, figsize=(4 * cols, 2.6 * math.ceil(len(picks) / cols)))
    for ax, r in zip(np.ravel(axes), picks):
        ax.axis("off")
        frames = sorted((batch_root / r["scene_name"] / "rendered" / "rasters").glob("frame_*"))
        if not frames:
            continue
        f = frames[len(frames) // 2]
        img = Image.open(f / "CAM_F0.png")
        img.thumbnail((640, 360))
        ax.imshow(img)
        ax.set_title(f"{r['scene_name'][-24:]}\n{f.name}  {r['context']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  Saved raster grid to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def build_report(scenes, ped_rows):
    def share(n):
        return f"{n} ({100 * n / max(len(scenes), 1):.1f}%)"
    gaps = _num(scenes, "min_gap_m")
    rep = {
        "total_scenes": len(scenes),
        "scenes_per_batch": dict(Counter(r["batch"] for r in scenes)),
        "base_scenes": len({r["base_scene"] for r in scenes}),
        "generation_status": dict(Counter(r["v4_status"] for r in scenes)),
        "context": dict(Counter(r["context"] for r in scenes)),
        "mean_nuplan_score": _f(_num(scenes, "score").mean()) if len(_num(scenes, "score")) else "",
        "ego_mean_speed_ms": _f(_num(scenes, "ego_mean_speed_ms").mean()),
        "scenes_ego_ped_collision": share(sum(1 for r in scenes if r["ego_ped_collision_frames"] not in ("", 0, "0"))),
        "scenes_bg_ped_collision": share(sum(1 for r in scenes if r["bg_ped_collision_frames"] not in ("", 0, "0"))),
        "scenes_ego_stuck_at_end": share(sum(r["ego_stuck_at_end"] for r in scenes)),
        "ped_frames_total": len(ped_rows),
        "ped_frames_in_fov": sum(r["in_fov"] for r in ped_rows),
        "ped_frames_with_raster": sum(r["has_raster"] for r in ped_rows),
        "scenes_min_gap_lt2m": share(int((gaps < 2).sum())),
        "scenes_min_gap_lt5m": share(int((gaps < 5).sum())),
        "median_min_gap_m": _f(np.median(gaps)) if len(gaps) else "",
        "reaction": dict(Counter(r["reaction"] for r in scenes)),
        "ped_side_frames": dict(Counter(r["ped_side"] for r in ped_rows)),
    }
    return rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Only first N scenes (quick test)")
    args = parser.parse_args()

    _patch_map_path()
    out = args.output_dir
    plot_dir = out / "03_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    partial = out / "_partial.jsonl"

    meta = load_batch_meta(args.batch_root)
    scene_dirs = sorted(d for d in args.batch_root.iterdir()
                        if d.is_dir() and not d.name.startswith("_"))
    if args.limit:
        scene_dirs = scene_dirs[:args.limit]
    print(f"Found {len(scene_dirs)} scenes ({len(meta)} in manifests)")

    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            rec = json.loads(line)
            done[rec["scene"]["scene_name"]] = rec
    todo = [d for d in scene_dirs if d.name not in done]
    print(f"  {len(done)} already done, {len(todo)} to process")

    with open(partial, "a") as pf, ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, d, meta.get(d.name, {"batch": "unknown"})) for d in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            scene, rows = fut.result()
            rec = {"scene": scene, "ped_rows": rows}
            done[scene["scene_name"]] = rec
            pf.write(json.dumps(rec) + "\n")
            pf.flush()
            if "error" in scene:
                print(f"  ERROR {scene['scene_name']}: {scene['error']}", flush=True)
            if i % 100 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] processed", flush=True)

    wanted = {d.name for d in scene_dirs}
    recs = [done[k] for k in sorted(done) if k in wanted]
    errors = [r["scene"] for r in recs if "error" in r["scene"]]
    scenes = [r["scene"] for r in recs if "error" not in r["scene"]]
    ped_rows = [row for r in recs if "error" not in r["scene"] for row in r["ped_rows"]]

    _write_csv(out / "01_scene_summary.csv", scenes)
    _write_csv(out / "02_ped_frames.csv", ped_rows)
    if errors:
        _write_csv(out / "errors.csv", errors)

    report = build_report(scenes, ped_rows)
    report["errors"] = len(errors)
    (out / "analysis_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    make_plots(scenes, ped_rows, plot_dir)
    make_raster_grid(scenes, args.batch_root, out / "04_raster_grid.png")


if __name__ == "__main__":
    main()
