"""
Qualitative comparison: Baseline vs Ped V1 trajectory predictions
on images with synthetically added pedestrians.

Uses SceneLoader + RAPFeatureBuilder for proper feature construction,
and direct camera projection (matching visualize_trajectories.py).
"""
import argparse
import glob
import json
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

os.environ['OPENSCENE_DATA_ROOT'] = '/datahdd2/karthik/dataset/nuplan-v1.1'
os.environ['NAVSIM_EXP_ROOT'] = '/datahdd2/karthik/RAP/exp'
os.environ['NAVSIM_DEVKIT_ROOT'] = '/datahdd/karthik/RAP/navsim'
os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'
os.environ['NUPLAN_MAPS_ROOT'] = '/datahdd/karthik/dataset/nuplan-maps-v1.0'


# Legend names for the two models (set from --baseline-label / --ped-label)
BL_LABEL, PD_LABEL = "Baseline", "Ped V1"


def load_agent(checkpoint_path, pick_mode="final", zero_ego_accel=False):
    from navsim.agents.rap_dino.rap_agent import RAPAgent
    from navsim.agents.rap_dino.navsim_config import RAPConfig
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    config = RAPConfig()
    config.use_resnet = True
    config.trajectory_sampling = TrajectorySampling(time_horizon=5, interval_length=0.5)
    config.distill_feature = True
    config.pdm_scorer = True
    config.train_metric_cache_path = './cache/train_metric_cache'
    config.pick_mode = pick_mode
    config.zero_ego_accel = zero_ego_accel     # E7/E8: model sets ego ax, ay input to 0

    agent = RAPAgent(config, lr=1e-4, checkpoint_path=checkpoint_path)
    agent.initialize()
    agent = agent.cuda()
    agent.eval()
    return agent


def predict_trajectory(agent, agent_input):
    with torch.no_grad():
        features = {}
        for builder in agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        features = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
                    for k, v in features.items()}
        pred = agent.forward(features)
        traj = pred['trajectory'].squeeze(0).cpu().numpy()[:, :2]
    return traj


def predict_with_swapped_front(agent, agent_input, new_front_path):
    """Run prediction after replacing the front camera image."""
    img = cv2.imread(new_front_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {new_front_path}")
    agent_input.cameras[-1].cam_f0.image = img
    return predict_trajectory(agent, agent_input)


def project_trajectory_to_image(traj_2d, cam):
    """Project ego-frame trajectory onto camera image (from visualize_trajectories.py)."""
    N = traj_2d.shape[0]
    points_3d = np.zeros((N, 3))
    points_3d[:, 0] = traj_2d[:, 0]
    points_3d[:, 1] = traj_2d[:, 1]
    points_3d[:, 2] = 0

    R = np.linalg.inv(cam.sensor2lidar_rotation)
    t = cam.sensor2lidar_translation
    points_cam = (R @ (points_3d - t).T).T

    valid = points_cam[:, 2] > 0.5

    K = cam.intrinsics
    points_2d = (K @ points_cam.T).T
    pixels = points_2d[:, :2] / (points_2d[:, 2:3] + 1e-6)

    return pixels, valid


def draw_trajectory_on_image(img, pixels, valid, color, label, thickness=4, marker_size=10):
    """Draw projected trajectory on the image with circles and lines."""
    h, w = img.shape[:2]
    overlay = img.copy()
    pts = []
    for i in range(len(pixels)):
        if not valid[i]:
            continue
        x, y = int(pixels[i, 0]), int(pixels[i, 1])
        if 0 <= x < w and 0 <= y < h:
            pts.append((x, y))
            cv2.circle(overlay, (x, y), marker_size, color, -1)
            cv2.circle(overlay, (x, y), marker_size + 2, (255, 255, 255), 2)
    for i in range(len(pts) - 1):
        cv2.line(overlay, pts[i], pts[i + 1], color, thickness)
    if pts:
        cv2.putText(overlay, label, (pts[-1][0] + 15, pts[-1][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    return img


def traj_kinematics(traj, dt=0.5):
    """Compute vx and ax from trajectory positions."""
    vx = np.diff(traj[:, 0]) / dt
    ax = np.diff(vx) / dt
    return vx, ax


def plot_comparison(orig_img_path, ped_img_path,
                    traj_bl_orig, traj_ped_orig,
                    traj_bl_ped, traj_ped_ped,
                    cam, token, output_path,
                    ego_vx=None, ego_ax=None):
    fig = plt.figure(figsize=(28, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.25, wspace=0.15)

    ax_img_orig = fig.add_subplot(gs[0, 0])
    ax_img_ped = fig.add_subplot(gs[0, 1])
    ax_kin_orig = fig.add_subplot(gs[1, 0])
    ax_kin_ped = fig.add_subplot(gs[1, 1])

    img_orig = cv2.cvtColor(cv2.imread(orig_img_path), cv2.COLOR_BGR2RGB)
    img_ped = cv2.cvtColor(cv2.imread(ped_img_path), cv2.COLOR_BGR2RGB)

    # Project and draw on original image
    pix_bl, val_bl = project_trajectory_to_image(traj_bl_orig, cam)
    pix_pd, val_pd = project_trajectory_to_image(traj_ped_orig, cam)
    img_orig = draw_trajectory_on_image(img_orig, pix_bl, val_bl, (30, 144, 255), BL_LABEL)
    img_orig = draw_trajectory_on_image(img_orig, pix_pd, val_pd, (255, 50, 50), PD_LABEL)

    # Project and draw on pedestrian image
    pix_bl2, val_bl2 = project_trajectory_to_image(traj_bl_ped, cam)
    pix_pd2, val_pd2 = project_trajectory_to_image(traj_ped_ped, cam)
    img_ped = draw_trajectory_on_image(img_ped, pix_bl2, val_bl2, (30, 144, 255), BL_LABEL)
    img_ped = draw_trajectory_on_image(img_ped, pix_pd2, val_pd2, (255, 50, 50), PD_LABEL)

    H, W = img_orig.shape[:2]
    for ax, img, title in [
        (ax_img_orig, img_orig, "Original Image"),
        (ax_img_ped, img_ped, "With Pedestrian"),
    ]:
        ax.imshow(img)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis("off")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    # Kinematics plots
    dt = 0.5
    t_vx = np.arange(1, len(traj_bl_orig)) * dt
    t_ax = np.arange(1, len(traj_bl_orig) - 1) * dt

    for ax_kin, traj_bl, traj_pd, subtitle in [
        (ax_kin_orig, traj_bl_orig, traj_ped_orig, "Original Image"),
        (ax_kin_ped, traj_bl_ped, traj_ped_ped, "With Pedestrian"),
    ]:
        vx_bl, ax_bl = traj_kinematics(traj_bl, dt)
        vx_pd, ax_pd = traj_kinematics(traj_pd, dt)

        ax_kin.plot(t_vx, vx_bl, 'b-o', markersize=4, label=f'{BL_LABEL} vx', linewidth=2)
        ax_kin.plot(t_vx, vx_pd, 'r-o', markersize=4, label=f'{PD_LABEL} vx', linewidth=2)

        ax2 = ax_kin.twinx()
        ax2.plot(t_ax, ax_bl, 'b--s', markersize=3, alpha=0.6, label=f'{BL_LABEL} ax', linewidth=1.5)
        ax2.plot(t_ax, ax_pd, 'r--s', markersize=3, alpha=0.6, label=f'{PD_LABEL} ax', linewidth=1.5)
        ax2.set_ylabel('ax (m/s²)', fontsize=10)
        ax2.legend(loc='upper right', fontsize=8)

        if ego_vx is not None:
            ax_kin.axhline(y=ego_vx, color='green', linestyle=':', linewidth=2, label=f'Input vx={ego_vx:.1f}')
        if ego_ax is not None:
            ax2.axhline(y=ego_ax, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Input ax={ego_ax:.1f}')
            ax2.legend(loc='upper right', fontsize=8)

        ax_kin.set_xlabel('Time (s)', fontsize=10)
        ax_kin.set_ylabel('vx (m/s)', fontsize=10)
        ax_kin.set_title(subtitle, fontsize=12)
        ax_kin.legend(loc='upper left', fontsize=8)
        ax_kin.grid(True, alpha=0.3)

    plt.suptitle(f"Token: {token}", fontsize=14)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--ped-ckpt", required=True)
    parser.add_argument("--ablation-dir",
                        default="/datahdd/karthik/sem_1/RAP_analysis_visualedits/pedestrian_ablation")
    parser.add_argument("--generated-dir",
                        default="/datahdd/karthik/sem_1/RAP_analysis_visualedits/generated_pedestrian")
    parser.add_argument("--output-dir", default="/datahdd2/karthik/RAP/qualitative_results_v3")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--baseline-pick-mode", default="final", choices=["final", "gated_pdms", "log_barrier"])
    parser.add_argument("--ped-pick-mode", default="final", choices=["final", "gated_pdms", "log_barrier"])
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--ped-label", default="Ped V1")
    args = parser.parse_args()
    global BL_LABEL, PD_LABEL
    BL_LABEL, PD_LABEL = args.baseline_label, args.ped_label

    os.makedirs(args.output_dir, exist_ok=True)

    # Load both models via RAPAgent (proper pipeline)
    print("Loading baseline agent...")
    agent_baseline = load_agent(args.baseline_ckpt, args.baseline_pick_mode)
    print("Loading ped v1 agent...")
    agent_ped = load_agent(args.ped_ckpt, args.ped_pick_mode)

    import pickle as pkl
    from navsim.common.dataloader import SceneLoader, SceneFilter

    sensor_config = agent_baseline.get_sensor_config()
    loaders = {}

    # Find which test logs contain our ablation tokens
    ablation_tokens = set()
    for d in sorted(os.listdir(args.ablation_dir)):
        if os.path.exists(os.path.join(args.ablation_dir, d, 'case.json')):
            ablation_tokens.add(d)
    print(f"Ablation tokens to find: {len(ablation_tokens)}")

    test_dir = '/datahdd2/karthik/dataset/nuplan-v1.1/navsim_logs/test'
    needed_logs = set()
    found_tokens = set()
    for pkl_file in sorted(glob.glob(os.path.join(test_dir, '*.pkl'))):
        log_name = os.path.basename(pkl_file).replace('.pkl', '')
        with open(pkl_file, 'rb') as f:
            data = pkl.load(f)
        for frame in data:
            if frame['token'] in ablation_tokens:
                needed_logs.add(log_name)
                found_tokens.add(frame['token'])

    print(f"Found {len(found_tokens)} tokens in {len(needed_logs)} test logs")

    sf_test = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True)
    sf_test.log_names = sorted(needed_logs)
    sf_test.tokens = sorted(found_tokens)
    loader_test = SceneLoader(
        sensor_blobs_path=Path('/datahdd2/karthik/dataset/nuplan-v1.1/sensor_blobs/test'),
        data_path=Path('/datahdd2/karthik/dataset/nuplan-v1.1/navsim_logs/test'),
        scene_filter=sf_test, sensor_config=sensor_config,
    )
    for t in loader_test.tokens:
        loaders[t] = loader_test

    print(f"Total available tokens: {len(loaders)}")

    # Find scenes from ablation dir
    scene_dirs = sorted(glob.glob(os.path.join(args.ablation_dir, "*")))
    if args.max_scenes:
        scene_dirs = scene_dirs[:args.max_scenes]

    for scene_dir in scene_dirs:
        token = os.path.basename(scene_dir)
        if token not in loaders:
            print(f"Skipping {token}: not in any scene loader")
            continue

        case_json = os.path.join(scene_dir, "case.json")
        if not os.path.exists(case_json):
            continue

        print(f"\nProcessing {token}...")

        try:
            # --- Original image: run both models ---
            agent_input = loaders[token].get_agent_input_from_token(token)
            orig_front = agent_input.cameras[-1].cam_f0.image
            if orig_front is None or orig_front.max() == 0:
                print(f"  No front camera image, skipping")
                continue

            # Extract input ego vx, ax from current frame's ego_status
            ego_status = agent_input.ego_statuses[-1]
            ego_vx = ego_status.ego_velocity[0]
            ego_ax = ego_status.ego_acceleration[0]

            traj_bl_orig = predict_trajectory(agent_baseline, agent_input)

            agent_input = loaders[token].get_agent_input_from_token(token)
            traj_ped_orig = predict_trajectory(agent_ped, agent_input)

            print(f"  Original — Baseline x=[{traj_bl_orig[:,0].min():.1f}, {traj_bl_orig[:,0].max():.1f}]m, "
                  f"Ped x=[{traj_ped_orig[:,0].min():.1f}, {traj_ped_orig[:,0].max():.1f}]m")

            orig_front_path = os.path.join(args.output_dir, f"_temp_orig_{token}.jpg")
            cv2.imwrite(orig_front_path, orig_front)

            # Camera object for projection
            agent_input_proj = loaders[token].get_agent_input_from_token(token)
            cam_for_proj = agent_input_proj.cameras[-1].cam_f0

            # --- Each Gemini-edited variant: run both models ---
            for variant_idx in range(3):
                gen_front = os.path.join(args.generated_dir, f"{token}_front_{variant_idx}.jpg")
                if not os.path.exists(gen_front):
                    continue

                # Baseline on pedestrian image
                agent_input = loaders[token].get_agent_input_from_token(token)
                traj_bl_ped = predict_with_swapped_front(agent_baseline, agent_input, gen_front)

                # Ped model on pedestrian image
                agent_input = loaders[token].get_agent_input_from_token(token)
                traj_ped_ped = predict_with_swapped_front(agent_ped, agent_input, gen_front)

                print(f"  v{variant_idx} Ped img — Baseline x=[{traj_bl_ped[:,0].min():.1f}, {traj_bl_ped[:,0].max():.1f}]m, "
                      f"Ped x=[{traj_ped_ped[:,0].min():.1f}, {traj_ped_ped[:,0].max():.1f}]m")

                plot_comparison(
                    orig_img_path=orig_front_path,
                    ped_img_path=gen_front,
                    traj_bl_orig=traj_bl_orig,
                    traj_ped_orig=traj_ped_orig,
                    traj_bl_ped=traj_bl_ped,
                    traj_ped_ped=traj_ped_ped,
                    cam=cam_for_proj,
                    token=token,
                    output_path=os.path.join(args.output_dir, f"{token}_v{variant_idx}.png"),
                    ego_vx=ego_vx,
                    ego_ax=ego_ax,
                )

            if os.path.exists(orig_front_path):
                os.remove(orig_front_path)

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
