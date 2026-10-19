#!/usr/bin/env python3
"""Rebuild the ped_v2 training set exactly as E4 saw it (before the lead-vehicle / ped-behind cleanup and the
yield edit), as hard links in a new folder. Nothing in the existing folders is moved or changed.

E4 set = train/ (minus the folders whose label was replaced, taken from train_replaced_by_yield_edit/ instead)
       + train_replaced_by_yield_edit/ + every train_excluded_*/ folder.

Usage: python process_data/build_e4_original_cache.py [--dry-run]
"""
import argparse
import os
import subprocess
from pathlib import Path

CACHE = Path("/datahdd2/karthik/ped_v2/cache")
OUT = CACHE / "train_original_E4"
SOURCES = ["train_replaced_by_yield_edit", "train_excluded_lead_vehicle", "train_excluded_ped_behind",
           "train_excluded_too_late", "train_excluded_side_pass"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pick = {}                                   # sample folder name -> source folder
    for src in SOURCES:
        for name in os.listdir(CACHE / src):
            assert name not in pick, f"{name} in both {pick[name]} and {src}"
            pick[name] = src
    n_from_extra = len(pick)
    for name in os.listdir(CACHE / "train"):
        pick.setdefault(name, "train")          # edited samples keep their original from train_replaced_by_yield_edit
    counts = {}
    for src in pick.values():
        counts[src] = counts.get(src, 0) + 1
    print(f"{len(pick)} samples ({n_from_extra} from moved-aside folders):", counts, flush=True)
    if args.dry_run:
        return

    OUT.mkdir(exist_ok=False)
    for i, (name, src) in enumerate(sorted(pick.items())):
        subprocess.run(["cp", "-al", str(CACHE / src / name), str(OUT / name)], check=True)   # hard links, no copy
        if i % 10000 == 0:
            print(f"  {i}/{len(pick)}", flush=True)
    print(f"done: {len(os.listdir(OUT))} samples in {OUT}")


if __name__ == "__main__":
    main()
