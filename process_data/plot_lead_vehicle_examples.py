#!/usr/bin/env python3
"""Picture check for lead_vehicle_check.py: does category C really mean "a car between ego and the pedestrian"?

For a few random scenes per category, at the closest moment to the inserted pedestrian, shows
  - the top-down frame from the InterPlan video (cropped around ego; ego = yellow outline, pedestrians = red)
  - the front camera raster (what the model sees)

Usage: python process_data/plot_lead_vehicle_examples.py --per-cat C=10 A=4 --out <png>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

RAW = Path("/datahdd2/karthik/ped_v2/raw")
ANALYSIS = Path("/datahdd2/karthik/ped_v2/analysis")
NAMES = {"A": "pedestrian in our path, nothing between", "B": "pedestrian ahead, off to the side",
         "C": "another car between us and the pedestrian", "D": "pedestrian beside or behind us"}
CROP = 190          # half-size of the top-down crop, pixels around the image centre (ego)


def frames(scene, idx):
    """Top-down video frame and front raster at idx (or the nearest frame that has a raster)."""
    rdir = RAW / scene / "rendered"
    have = sorted(int(p.name[6:]) for p in (rdir / "rasters").glob("frame_*"))
    r_idx = min(have, key=lambda i: abs(i - idx))
    raster = Image.open(rdir / "rasters" / f"frame_{r_idx:04d}" / "CAM_F0.png").convert("RGB")
    vdir = next((rdir / "video").glob("*_frames"))
    bev = np.asarray(Image.open(vdir / f"frame_{idx:05d}.png").convert("RGB"))
    cy, cx = bev.shape[0] // 2 + 15, bev.shape[1] // 2 + 24        # plot area is offset slightly from image centre
    return bev[cy - CROP:cy + CROP, cx - CROP:cx + CROP], raster, r_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", nargs="+", default=["C=10", "A=4"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ANALYSIS / "06_close_pass" / "lead_vehicle_examples.png")
    args = ap.parse_args()

    r = pd.read_csv(ANALYSIS / "06_close_pass" / "closest_moment_lead_vehicle.csv")
    picks = []
    for spec in args.per_cat:
        cat, n = spec.split("=")
        picks.append(r[r.category == cat].sample(int(n), random_state=args.seed))
    picks = pd.concat(picks)

    cols = 2
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 13, rows * 4.6),
                             gridspec_kw={"width_ratios": [1, 1.78] * cols})
    for ax in axes.flat:
        ax.axis("off")
    for k, x in enumerate(picks.itertuples()):
        bev, raster, r_idx = frames(x.scene_name, int(x.frame_idx))
        a_bev, a_ras = axes[k // cols, (k % cols) * 2], axes[k // cols, (k % cols) * 2 + 1]
        a_bev.imshow(bev)
        a_ras.imshow(raster)
        lead = f"car {x.lead_gap:.1f} m ahead at {x.lead_speed_mph:.0f} mph" if not np.isnan(x.lead_gap) else "no car ahead"
        a_bev.set_title(f"{x.category}: {NAMES[x.category]}", fontsize=11, loc="left", fontweight="bold")
        a_ras.set_title(f"{x.scene_name[-30:]}  frame {x.frame_idx}" + ("" if r_idx == x.frame_idx else f" (raster {r_idx})")
                        + f"\nego {x.ego_mph:.0f} mph · ped {x.ped_x:.1f} m ahead, {x.ped_y:+.1f} m side · {lead}",
                        fontsize=10, loc="left")
    fig.suptitle("Closest moment to the pedestrian: top-down (ego = yellow outline, pedestrians = red) and front raster",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.out, dpi=70)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
