#!/usr/bin/env python3
"""Evaluate a RAP checkpoint on the held-out ped_v2 (batch 3) validation set.

For every validation sample it runs the model, then scores the predicted trajectory
with the same PDM scorer used in training (via the metric cache). The label
trajectory (PDM-Closed ego from InterPlan) is scored too, as a reference.

Writes into <experiments>/<exp_name>/ped_val_batch3/:
    results.csv    one row per sample (scores, errors, speeds, scene info)
    summary.txt    overall + per reaction + per context tables
    summary.json   same numbers, machine readable

Usage:
    CUDA_VISIBLE_DEVICES=4 python eval_ped_val.py \
        --exp-name rap_resnet_with_only_ped_v1 \
        --ckpt /datahdd2/karthik/RAP/experiments/rap_resnet_with_only_ped_v1/checkpoints/last.ckpt
"""

import argparse
import json
import os
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/datahdd/karthik/dataset/nuplan-maps-v1.0")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/datahdd2/karthik/RAP/exp")
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", "/datahdd/karthik/RAP/navsim")
os.environ.setdefault("ip_head", "128.2.220.69:6379")
os.environ.setdefault("redis_password", "dummy")
os.environ.setdefault("num_nodes", "1")

EXPERIMENTS_ROOT = Path("/datahdd2/karthik/RAP/experiments")
PED_V2 = Path("/datahdd2/karthik/ped_v2")
SCORE_COLS = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
              "time_to_collision_within_bound", "comfort", "score"]
DT = 0.5
STOP_SPEED = 0.5


def load_agent(ckpt, distill, pdm):
    from navsim.agents.rap_dino.rap_agent import RAPAgent
    from navsim.agents.rap_dino.navsim_config import RAPConfig
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    config = RAPConfig()
    config.use_resnet = True
    config.trajectory_sampling = TrajectorySampling(time_horizon=5, interval_length=0.5)
    config.distill_feature = distill
    config.pdm_scorer = pdm
    config.train_metric_cache_path = "./cache/train_metric_cache"

    agent = RAPAgent(config, lr=1e-4, checkpoint_path=str(ckpt))
    agent.initialize()
    return agent.cuda().eval()


def _score(args):
    """Score [prediction, label] for one sample. Runs in a worker process."""
    from navsim.agents.rap_dino.score_module.compute_navsim_score import get_sub_score
    path, poses = args
    return get_sub_score(str(path), poses, True)[0]   # (2, 6)


def speeds(traj):
    """Speed (m/s) at each of the future poses, from xy steps (first step starts at ego = origin)."""
    xy = np.concatenate([np.zeros((len(traj), 1, 2)), traj[:, :, :2]], axis=1)
    return np.linalg.norm(np.diff(xy, axis=1), axis=-1) / DT


def table(df, by=None):
    cols = {
        "samples": ("token", "size"),
        "PDM score": ("score", "mean"),
        "label PDM": ("label_score", "mean"),
        "no collision": ("no_at_fault_collisions", "mean"),
        "TTC ok": ("time_to_collision_within_bound", "mean"),
        "progress": ("ego_progress", "mean"),
        "comfort": ("comfort", "mean"),
        "ADE 4s (m)": ("ade_4s", "mean"),
        "FDE 4s (m)": ("fde_4s", "mean"),
        "pred dist 4s (m)": ("pred_dist_4s", "mean"),
        "label dist 4s (m)": ("label_dist_4s", "mean"),
        "pred stops": ("pred_stops", "mean"),
        "label stops": ("label_stops", "mean"),
    }
    g = df.groupby(by) if by else df.groupby(lambda _: "ALL")
    return g.agg(**cols).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--val-cache", type=Path, default=PED_V2 / "cache/val")
    ap.add_argument("--val-samples", type=Path, default=PED_V2 / "analysis/05_selection/val_samples.csv")
    ap.add_argument("--no-distill", action="store_true")
    ap.add_argument("--no-pdm", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--score-procs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="", help="Suffix for the output folder, e.g. epoch12 -> ped_val_batch3_epoch12")
    args = ap.parse_args()

    from navsim.planning.training.dataset import CacheOnlyDataset

    out_dir = EXPERIMENTS_ROOT / args.exp_name / ("ped_val_batch3" + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = load_agent(args.ckpt, not args.no_distill, not args.no_pdm)
    metric_paths = agent.test_metric_cache_paths

    data = CacheOnlyDataset(cache_path=str(args.val_cache),
                            feature_builders=agent.get_feature_builders(),
                            target_builders=agent.get_target_builders())
    if args.limit:
        data = torch.utils.data.Subset(data, range(args.limit))
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    tokens, preds, labels = [], [], []
    with torch.no_grad():
        for features, targets in tqdm(loader, desc="Predicting"):
            # same input the validation step uses: the (raster) camera_feature only
            features = {k: v.cuda() for k, v in features.items()
                        if k not in ("camera_valid", "rendered_camera_feature", "frame_name")}
            pred = agent.forward(features)
            preds.append(pred["trajectory"].float().cpu().numpy())
            labels.append(targets["trajectory"].float().numpy())
            tokens += list(targets["token"])
    preds, labels = np.concatenate(preds), np.concatenate(labels)

    jobs = [(metric_paths[t], np.stack([p, l])) for t, p, l in zip(tokens, preds, labels)]
    # "spawn": forked workers deadlock once CUDA and Ray are initialised in the parent
    with mp.get_context("spawn").Pool(args.score_procs) as pool:
        scores = np.stack(list(tqdm(pool.imap(_score, jobs, chunksize=4), total=len(jobs), desc="PDM scoring")))

    err = np.linalg.norm(preds[:, :8, :2] - labels[:, :8, :2], axis=-1)
    v_pred, v_label = speeds(preds), speeds(labels)
    df = pd.DataFrame({"token": tokens})
    for i, c in enumerate(SCORE_COLS):
        df[c] = scores[:, 0, i]
    df["label_score"] = scores[:, 1, -1]
    df["label_no_collision"] = scores[:, 1, 0]
    df["ade_4s"] = err.mean(1)
    df["fde_4s"] = err[:, -1]
    df["pred_dist_4s"] = np.linalg.norm(preds[:, 7, :2], axis=-1)
    df["label_dist_4s"] = np.linalg.norm(labels[:, 7, :2], axis=-1)
    df["pred_speed_4s"] = v_pred[:, 7]
    df["label_speed_4s"] = v_label[:, 7]
    df["pred_stops"] = (v_pred[:, :8].min(1) < STOP_SPEED).astype(int)
    df["label_stops"] = (v_label[:, :8].min(1) < STOP_SPEED).astype(int)

    info = pd.read_csv(args.val_samples)
    df = df.merge(info[["token", "scene_name", "frame_idx", "reaction", "context", "base_scene"]], on="token", how="left")
    df.to_csv(out_dir / "results.csv", index=False)

    parts = [("Overall", table(df)), ("By ego reaction in the scene", table(df, "reaction")),
             ("By scene context", table(df, "context"))]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    text = f"Checkpoint: {args.ckpt}\nValidation cache: {args.val_cache} ({len(df)} samples)\n"
    for name, t in parts:
        text += f"\n### {name}\n{t.T.to_string()}\n"
    (out_dir / "summary.txt").write_text(text)
    (out_dir / "summary.json").write_text(json.dumps(
        {"checkpoint": str(args.ckpt), "samples": len(df),
         **{name: t.to_dict(orient="index") for name, t in parts}}, indent=2))
    print(text)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
