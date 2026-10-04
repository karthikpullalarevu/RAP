#!/usr/bin/env python3
"""Set the camera_valid flag in an existing ped cache, in place (no rebuild).

The ped cache holds rasters only: camera_feature is the same raster as rendered_camera_feature.
With camera_valid=True the training step treats that raster as a real photo; False sends the
sample through the raster pass only (like the augmented data). Only the flag is changed.

Each file is rewritten to a temp name and then swapped in, so an interruption cannot corrupt it.
Safe to re-run: files that already have the wanted value are skipped.

Usage:
    python process_data/set_ped_cache_camera_valid.py --value false \
        /datahdd2/karthik/ped_v2/cache/train /datahdd2/karthik/ped_v2/cache/val
"""
import argparse
import gzip
import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch


def fix(args):
    path, value = args
    try:
        with gzip.open(path, "rb") as f:
            feat = pickle.load(f)
        if bool(feat["camera_valid"]) == value:
            return "already set"
        feat["camera_valid"] = torch.tensor(value)
        tmp = path + ".tmp"
        with gzip.open(tmp, "wb", compresslevel=1) as f:
            pickle.dump(feat, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return "changed"
    except Exception as e:
        return f"error: {e!r}"[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--value", required=True, choices=["true", "false"])
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    value = args.value == "true"
    for root in args.roots:
        files = [str(t / "rap_feature.gz") for s in sorted(root.iterdir()) if s.is_dir() for t in s.iterdir()]
        counts = Counter()
        with ProcessPoolExecutor(args.workers) as ex:
            for i, r in enumerate(ex.map(fix, [(p, value) for p in files], chunksize=32), 1):
                counts[r] += 1
                if i % 5000 == 0:
                    print(f"  {root.name}: {i}/{len(files)}", flush=True)
        print(f"{root}: {len(files)} files -> {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
