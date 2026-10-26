#!/usr/bin/env python3
"""Apply the yield edit (edit_yield_label.py) to the held-out ped_v2 val set, without touching the original.

Writes a new cache ped_v2/cache/val_yield_edited/ next to cache/val:
  - edited samples: rap_feature.gz hard-linked, new rap_target.gz with the edited label
  - unchanged samples: both files hard-linked
  - too_late / side_pass samples: left out (as in training)
and analysis/09_yield_edit/val/yield_edit_val.csv (outcome per sample) + val_samples_edited.csv (for eval_ped_val.py).

Usage:
    python process_data/yield_edit_val.py --workers 48
"""

import argparse
import gzip
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import process_data.edit_yield_label as Y

VAL = Y.PED / "cache" / "val"
NEW = Y.PED / "cache" / "val_yield_edited"
OUT = Y.OUT / "val"


def one(job):
    scene, anchors = job
    try:
        sc = Y.Scene(scene)
        res = []
        for r in Y.solve_scene(sc, anchors):
            lab = Y.label_poses(sc, r["anchor"], r["s"]) if r["outcome"] in ("edited", "edited_close") else None
            res.append((f"{scene}_{r['anchor']:04d}", r["outcome"], r["v0_mph"], lab))
        return res, None
    except Exception as ex:   # noqa: BLE001
        return [], f"{scene}: {ex!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    if NEW.exists():
        sys.exit(f"{NEW} already exists; not overwriting")
    OUT.mkdir(parents=True, exist_ok=True)

    names = sorted(os.listdir(VAL))                              # <scene>_<frame>
    p = pd.read_csv(Y.PED / "analysis" / "02_ped_frames.csv", usecols=["scene_name", "gap_m"])
    close = set(p.groupby("scene_name").gap_m.min().loc[lambda g: g < Y.SIDE_GAP].index)
    jobs = {}
    for n in names:
        sc_, fr = n.rsplit("_", 1)
        if sc_ in close:
            jobs.setdefault(sc_, []).append(int(fr))
    print(f"{len(names)} val samples; {sum(len(a) for a in jobs.values())} in {len(jobs)} scenes within "
          f"{Y.SIDE_GAP:g} m of the pedestrian get the edit, the rest stay unchanged", flush=True)

    outcome, labels, errs = {n: "unchanged (scene never within 3 m)" for n in names}, {}, []
    v0 = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for res, err in ex.map(one, sorted(jobs.items()), chunksize=1):
            if err:
                errs.append(err)
            for name, oc, mph, lab in res:
                outcome[name], v0[name] = oc, mph
                if lab is not None:
                    labels[name] = lab
    (OUT / "errors.txt").write_text("\n".join(errs))
    for e in errs:                                               # scenes that failed keep their original label
        print("FAILED (kept unchanged):", e)

    keep = 0
    for n in names:
        oc = outcome[n]
        if oc in ("too_late", "side_pass"):
            continue
        for tok_dir in (VAL / n).iterdir():
            dst = NEW / n / tok_dir.name
            dst.mkdir(parents=True)
            os.link(tok_dir / "rap_feature.gz", dst / "rap_feature.gz")
            if n in labels:
                old_t = pickle.load(gzip.open(tok_dir / "rap_target.gz"))
                target = {"trajectory": torch.tensor(labels[n], dtype=torch.float64), "token": old_t["token"]}
                with gzip.open(dst / "rap_target.gz", "wb", compresslevel=1) as f:
                    pickle.dump(target, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                os.link(tok_dir / "rap_target.gz", dst / "rap_target.gz")
        keep += 1

    df = pd.DataFrame({"sample": names, "outcome": [outcome[n] for n in names], "v0_mph": [v0.get(n) for n in names]})
    df.to_csv(OUT / "yield_edit_val.csv", index=False)
    vs = pd.read_csv(Y.PED / "analysis" / "05_selection" / "val_samples.csv")
    vs["sample"] = vs.scene_name + "_" + vs.frame_idx.map("{:04d}".format)
    kept = set(df[~df.outcome.isin(["too_late", "side_pass"])]["sample"])
    vs[vs["sample"].isin(kept)].drop(columns="sample").to_csv(OUT / "val_samples_edited.csv", index=False)
    print(df.outcome.value_counts().to_string())
    print(f"\nkept {keep} of {len(names)} samples in {NEW}")


if __name__ == "__main__":
    main()
