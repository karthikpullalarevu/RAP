#!/usr/bin/env python3
"""Pick which ped_v2 frames become training / validation samples.

Reads the analysis CSVs written by analyze_batch.py and writes sample lists
for build_ped_cache.py.

Train: scenes where ego stops or slows down for the ped.
       2 Hz grid over frames with an inserted ped + raster, plus key frames
       (ped first visible, closest gap, hardest braking, ego first stopped).
Val:   every scene from held-out drives (all reactions), 2 Hz grid only.
       Whole drives are held out so no road is shared with training.

Usage:
    python process_data/select_ped_v2_samples.py \
        --analysis-dir /datahdd2/karthik/ped_v2/analysis \
        --batch-root /datahdd2/karthik/ped_v2/raw
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_ANCHOR, MAX_ANCHOR = 30, 198   # need 1.5 s history and 5 s future (see build_ped_cache.py)
GRID = 10                          # 20 Hz sim -> 2 Hz samples
STOP_SPEED = 0.5
TRAIN_REACTIONS = ["stop", "slow down"]


def pick_val_drives(scenes, val_frac, seed):
    """Random drives until ~val_frac of parent scenes, covering every context and reaction."""
    rng = np.random.default_rng(seed)
    drives = sorted(scenes.drive.unique())
    target = val_frac * scenes.base_scene.nunique()
    need_ctx, need_react = set(scenes.context.unique()), set(scenes.reaction.unique())

    picked = []
    for d in rng.permutation(drives):
        picked.append(d)
        v = scenes[scenes.drive.isin(picked)]
        if (v.base_scene.nunique() >= target and need_ctx <= set(v.context)
                and need_react <= set(v.reaction)):
            break
    return sorted(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, default=Path("/datahdd2/karthik/ped_v2/analysis"))
    ap.add_argument("--batch-root", type=Path, default=Path("/datahdd2/karthik/ped_v2/raw"))
    ap.add_argument("--val-frac", type=float, default=0.05, help="Share of parent scenes held out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.analysis_dir / "05_selection"
    out.mkdir(exist_ok=True)

    scenes = pd.read_csv(args.analysis_dir / "01_scene_summary.csv")
    scenes["drive"] = scenes.base_scene.str.replace(r"_\d+_\d+$", "", regex=True)
    manifest = pd.concat(pd.read_csv(m) for m in sorted((args.batch_root / "_batch_meta").glob("*/manifest.csv")))
    scenes = scenes.merge(manifest[["scenario_name", "anchor_token"]],
                          left_on="scene_name", right_on="scenario_name")

    frames = pd.read_csv(args.analysis_dir / "02_ped_frames.csv",
                         usecols=["scene_name", "frame_idx", "has_raster", "gap_m",
                                  "ego_speed_ms", "ego_lon_acc_ms2", "ped_side"])
    usable = frames[(frames.frame_idx >= MIN_ANCHOR) & (frames.frame_idx <= MAX_ANCHOR)
                    & (frames.has_raster == 1)]

    # --- 2 Hz grid ---
    grid = usable[usable.frame_idx % GRID == 0][["scene_name", "frame_idx"]].assign(reason="grid_2hz")

    # --- key frames (one per scene each) ---
    g = usable.groupby("scene_name")
    keys = pd.concat([
        usable.loc[g.frame_idx.idxmin(), ["scene_name", "frame_idx"]].assign(reason="ped_first_visible"),
        usable.loc[g.gap_m.idxmin(), ["scene_name", "frame_idx"]].assign(reason="closest_gap"),
        usable.loc[g.ego_lon_acc_ms2.idxmin(), ["scene_name", "frame_idx"]].assign(reason="hardest_braking"),
        usable[usable.ego_speed_ms < STOP_SPEED].groupby("scene_name", as_index=False).frame_idx.min()
            .assign(reason="ego_first_stopped"),
    ])

    val_drives = pick_val_drives(scenes, args.val_frac, args.seed)
    val_scenes = scenes[scenes.drive.isin(val_drives)]
    train_scenes = scenes[~scenes.drive.isin(val_drives) & scenes.reaction.isin(TRAIN_REACTIONS)]

    def finish(df, scene_df):
        df = df[df.scene_name.isin(scene_df.scene_name)]
        # grid rows come first, so a key frame that is already on the grid keeps reason=grid_2hz
        df = df.drop_duplicates(["scene_name", "frame_idx"], keep="first")
        df = df.merge(scene_df[["scene_name", "anchor_token", "base_scene", "batch", "reaction", "context"]],
                      on="scene_name")
        df["token"] = df.anchor_token + "_" + df.frame_idx.map("{:04d}".format)
        return df.sort_values(["scene_name", "frame_idx"])[
            ["scene_name", "frame_idx", "token", "reason", "reaction", "context", "base_scene", "batch"]]

    train = finish(pd.concat([grid, keys]), train_scenes)
    val = finish(grid, val_scenes)
    assert not set(train.base_scene) & set(val.base_scene)
    assert train.token.is_unique and val.token.is_unique

    train.to_csv(out / "train_samples.csv", index=False)
    val.to_csv(out / "val_samples.csv", index=False)
    (out / "train_tokens.txt").write_text("\n".join(train.token) + "\n")
    (out / "val_tokens.txt").write_text("\n".join(val.token) + "\n")
    (out / "val_drives.txt").write_text("\n".join(val_drives) + "\n")

    summary = {
        "seed": args.seed,
        "val_drives": len(val_drives),
        "val_parent_scenes": int(val_scenes.base_scene.nunique()),
        "val_scenes": int(val.scene_name.nunique()),
        "val_samples": len(val),
        "val_by_reaction": val.reaction.value_counts().to_dict(),
        "val_by_context": val.context.value_counts().to_dict(),
        "train_parent_scenes": int(train.base_scene.nunique()),
        "train_scenes": int(train.scene_name.nunique()),
        "train_samples": len(train),
        "train_by_reason": train.reason.value_counts().to_dict(),
        "train_by_reaction": train.reaction.value_counts().to_dict(),
        "train_by_context": train.context.value_counts().to_dict(),
    }
    (out / "selection_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote lists to {out}")


if __name__ == "__main__":
    main()
