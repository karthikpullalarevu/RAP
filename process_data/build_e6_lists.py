#!/usr/bin/env python3
"""Lists the E6 score loss needs (ped_v2 only):

  inserted_ped_tokens.json      anchor token -> track tokens of the inserted pedestrians (the 1.5 m rule only
                                counts these, not the original nuPlan pedestrians in the scene)
  no_score_loss_tokens.txt      training sample tokens that start within NO_SCORE_GAP m of the inserted
                                pedestrian: no proposal can satisfy the 1.5 m rule there, so they skip the
                                score loss (they still train imitation)

Usage: python process_data/build_e6_lists.py
"""
import glob
import json
import os
from pathlib import Path

import pandas as pd

PED = Path("/datahdd2/karthik/ped_v2")
OUT = PED / "analysis" / "10_e6"
NO_SCORE_GAP = 2.0


def main():
    OUT.mkdir(exist_ok=True)
    man = pd.concat(pd.read_csv(m) for m in sorted(glob.glob(str(PED / "raw/_batch_meta/*/manifest.csv"))))
    lookup = {}
    for r in man.itertuples():
        f = PED / "raw" / r.scenario_name / "metadata/result.json"
        if f.exists():
            lookup[r.anchor_token] = json.loads(f.read_text())["saved_log_summary"]["pedestrian_track_tokens"]
    (OUT / "inserted_ped_tokens.json").write_text(json.dumps(lookup))

    names = os.listdir(PED / "cache/train")
    t = pd.DataFrame({"sample": names, "scene_name": [n.rsplit("_", 1)[0] for n in names],
                      "frame_idx": [int(n.rsplit("_", 1)[1]) for n in names]})
    p = pd.read_csv(PED / "analysis/02_ped_frames.csv", usecols=["scene_name", "frame_idx", "gap_m"])
    t = t.merge(p, on=["scene_name", "frame_idx"])
    close = t[t.gap_m < NO_SCORE_GAP]
    tokens = [next(os.scandir(PED / "cache/train" / s)).name for s in close["sample"]]
    (OUT / "no_score_loss_tokens.txt").write_text("\n".join(sorted(tokens)) + "\n")
    print(f"{len(lookup)} scenes with inserted pedestrians; {len(tokens)} of {len(t)} training samples start "
          f"within {NO_SCORE_GAP} m -> no score loss")


if __name__ == "__main__":
    main()
