#!/usr/bin/env python3
"""Simple slides: old loss (E4) vs new loss (E6) on the camera image, 4 scenes per input type.

One figure per input (RGB, raster): 2 x 4 grid, top row with the pedestrian, bottom row the same scene without it.
Both plans are drawn on each image; the only text is a small legend. Same scenes as visualize_old_vs_new_loss.py.

Usage:
    CUDA_VISIBLE_DEVICES=0 python visualize_old_vs_new_loss_simple.py --output-dir <dir>
"""

import argparse
import glob
import json
import math
import os
import pickle
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import visualize_ped_comparison as V
import visualize_ped_gemini_bev as B
import visualize_gemini_raster_ped as G
from visualize_gemini_rgb_vs_raster import predict, make_input, box_centre
from visualize_old_vs_new_loss import MODELS, EXAMPLES, CLOSER, SURFACE, INK, draw_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})

    agents = [(n, V.load_agent(str(c), m), col) for n, c, m, col in MODELS]
    dets = {(e["token"], e["suffix"]): e["detections"] for e in json.load(open(B.LIDAR_RESULTS))}
    from navsim.common.dataloader import SceneLoader, SceneFilter
    want = {t for _, t, _, _ in EXAMPLES}
    logs, found = set(), set()
    for pkl in sorted(glob.glob(os.path.join(B.TEST_LOGS, "*.pkl"))):
        for fr in pickle.load(open(pkl, "rb")):
            if fr["token"] in want:
                logs.add(Path(pkl).stem)
                found.add(fr["token"])
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True)
    sf.log_names, sf.tokens = sorted(logs), sorted(found)
    loader = SceneLoader(sensor_blobs_path=Path(B.TEST_BLOBS), data_path=Path(B.TEST_LOGS), scene_filter=sf,
                         sensor_config=agents[0][1].get_sensor_config())

    for inp, name in (("photo", "rgb"), ("raster", "raster")):
        exs = [e for e in EXAMPLES if e[0] == inp]
        fig, axes = plt.subplots(2, len(exs), figsize=(4.6 * len(exs), 4.0))
        for col, (_, token, v, _) in enumerate(exs):
            base = loader.get_agent_input_from_token(token)
            cam = base.cameras[-1].cam_f0
            M = B.lidar2img(cam)
            if inp == "photo":
                with_ped = cv2.cvtColor(cv2.imread(os.path.join(B.GENERATED_DIR, f"{token}_front_{v}.jpg")),
                                        cv2.COLOR_BGR2RGB)
                no_ped = np.asarray(cam.image).astype(np.uint8)
            else:
                ped = B.ped_position(dets.get((token, f"front_{v}"), []), M)
                n = math.hypot(*ped)
                drawn = (ped[0] * (1 - CLOSER / n), ped[1] * (1 - CLOSER / n))
                no_ped = np.asarray(cam.rendered_image).astype(np.uint8)
                with_ped = G.draw_ped_cuboid(no_ped.copy(), M, drawn)
            for row, (img, front) in enumerate([(with_ped, with_ped), (no_ped, None)]):
                plans = [(n_, predict(a, make_input(loader, token, inp, front)), c) for n_, a, c in agents]
                draw_image(axes[row, col], img, cam, plans)
            print("done", inp, token, v, flush=True)
        fig.legend(handles=[Line2D([], [], color=c, lw=4, marker="o", ms=7, mec="white", label=n_)
                            for n_, _, _, c in MODELS],
                   loc="lower center", ncol=2, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 0.0))
        fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.11, wspace=0.02, hspace=0.03)
        fig.savefig(out / f"old_vs_new_loss_{name}.png", dpi=130)
        plt.close(fig)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
