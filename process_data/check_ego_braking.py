#!/usr/bin/env python3
"""How often does the ego_status the model sees already show braking, per training cache?

ego_status (rap_features.py) is (4 frames, 11) = pose(3), velocity(2), acceleration(2), command(4); the model
uses the last frame. Counts samples with current longitudinal acceleration below -1 / -2 / -3 m/s2, and whether
the label ends stopped, to check for a "already braking -> stop" shortcut.

Usage: python process_data/check_ego_braking.py --out <txt> [--sample 5000] [--workers 16]
"""

import argparse
import gzip
import os
import pickle
import random
from multiprocessing import Pool

import numpy as np

CACHES = {
    "ped_v2 train": "/datahdd2/karthik/ped_v2/cache/train",
    "ped_v2 val": "/datahdd2/karthik/ped_v2/cache/val",
    "ped_v2 replaced originals": "/datahdd2/karthik/ped_v2/cache/train_replaced_by_yield_edit",
    "navtrain ego_50k": "/datahdd2/karthik/RAP/cache/ego_50k",
    "navtrain aug (other agents)": "/datahdd/karthik/RAP/cache/ablation/aug",
    "navtrain perturbed": "/datahdd/karthik/RAP/cache/ablation/perturbed",
}
MPH = 2.237


def sample_dirs(root):
    """Sample folders (the ones holding rap_feature.gz), at any depth up to 3 below root."""
    out = []
    for dp, dn, fn in os.walk(root, followlinks=True):   # cache folders can be symlinks
        if "rap_feature.gz" in fn:
            out.append(dp)
            dn[:] = []
        elif dp.count(os.sep) - root.count(os.sep) >= 3:
            dn[:] = []
    return out


def read(d):
    try:
        e = pickle.load(gzip.open(os.path.join(d, "rap_feature.gz")))["ego_status"][-1].numpy()
        t = pickle.load(gzip.open(os.path.join(d, "rap_target.gz")))["trajectory"]
        t = np.asarray(t)
        end_speed = float(np.linalg.norm(t[-1, :2] - t[-2, :2]) / 0.5)
        return float(e[3]), float(e[5]), end_speed
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=5000, help="Samples per cache (0 = all)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=None, help="Cache names to check (default: all)")
    args = ap.parse_args()
    random.seed(0)
    lines = []
    for name, root in CACHES.items():
        if args.only and name not in args.only:
            continue
        dirs = sample_dirs(root)
        total = len(dirs)
        if args.sample and total > args.sample:
            dirs = random.sample(dirs, args.sample)
        with Pool(args.workers) as p:
            r = np.array([x for x in p.map(read, dirs, chunksize=32) if x is not None])
        if len(r) == 0:
            lines.append(f"{name}: {total} samples found, none readable\n")
            print(lines[-1], flush=True)
            continue
        v, a, end = r[:, 0], r[:, 1], r[:, 2]
        stop = end < 0.5
        brake = a < -1.0
        lines.append(
            f"{name}: {total} samples, read {len(r)}\n"
            f"  current speed: mean {v.mean() * MPH:.1f} mph\n"
            f"  current accel: mean {a.mean():+.2f} m/s2, "
            f"< -1: {np.mean(a < -1):.1%}, < -2: {np.mean(a < -2):.1%}, < -3: {np.mean(a < -3):.1%}\n"
            f"  label ends stopped: {stop.mean():.1%}  |  if braking (< -1): {stop[brake].mean() if brake.any() else 0:.1%}"
            f"  vs not braking: {stop[~brake].mean() if (~brake).any() else 0:.1%}\n")
        print(lines[-1], flush=True)
    open(args.out, "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
