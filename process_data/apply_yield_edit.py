#!/usr/bin/env python3
"""Write the yield-edit results (edit_yield_label.py --batch) into the ped_v2 training cache. Nothing is deleted.

  edited / edited_close  original sample folder -> train_replaced_by_yield_edit/
                         new folder in train/: same rap_feature.gz (hard link), new rap_target.gz with the edited label
  too_late               sample folder -> train_excluded_too_late/
  side_pass              sample folder -> train_excluded_side_pass/
  unchanged              left alone

Undo: delete the new folders in train/ listed in yield_edit_applied.csv and move everything back.

Usage: python process_data/apply_yield_edit.py [--dry-run]
"""
import argparse
import gzip
import os
import pickle
import shutil
from pathlib import Path

import pandas as pd
import torch

CACHE = Path("/datahdd2/karthik/ped_v2/cache")
BATCH = Path("/datahdd2/karthik/ped_v2/analysis/09_yield_edit/batch")
DEST = {"edited": "train_replaced_by_yield_edit", "edited_close": "train_replaced_by_yield_edit",
        "too_late": "train_excluded_too_late", "side_pass": "train_excluded_side_pass"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = pd.read_csv(BATCH / "yield_edit_all_anchors.csv")
    d = d[d.in_train_cache & d.outcome.isin(DEST)]
    labels = pickle.load(open(BATCH / "yield_edit_labels.pkl", "rb"))
    edited = d.outcome.str.startswith("edited")
    missing = set(d[edited]["sample"]) - set(labels)
    assert not missing, f"{len(missing)} edited samples have no label"
    print(d.outcome.value_counts().to_string())
    if args.dry_run:
        return
    for name in set(DEST.values()):
        (CACHE / name).mkdir(exist_ok=True)

    done = []
    for x in d.itertuples():
        src = CACHE / "train" / x.sample
        if not src.is_dir():
            continue                                     # already applied
        dst = CACHE / DEST[x.outcome] / x.sample
        shutil.move(str(src), str(dst))
        if x.outcome.startswith("edited"):
            (tok_dir,) = [p for p in dst.iterdir() if p.is_dir()]
            new = CACHE / "train" / x.sample / tok_dir.name
            new.mkdir(parents=True)
            os.link(tok_dir / "rap_feature.gz", new / "rap_feature.gz")
            old_t = pickle.load(gzip.open(tok_dir / "rap_target.gz"))
            target = {"trajectory": torch.tensor(labels[x.sample], dtype=torch.float64), "token": old_t["token"]}
            with gzip.open(new / "rap_target.gz", "wb", compresslevel=1) as f:
                pickle.dump(target, f, protocol=pickle.HIGHEST_PROTOCOL)
        done.append((x.sample, x.outcome))
    pd.DataFrame(done, columns=["sample", "outcome"]).to_csv(BATCH / "yield_edit_applied.csv", index=False)
    print(f"applied {len(done)}; train/ now has {len(os.listdir(CACHE / 'train'))} samples")


if __name__ == "__main__":
    main()
