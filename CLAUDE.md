# Global Rules

## System Safety
- Never run `sudo` unless I explicitly ask
- Never modify other users' files or system-wide configs
- Never delete datasets (`/datahdd/karthik/dataset/`, `/datahdd/karthik/nuplan_dataset/`)
- Don't touch `/home/karthik/.bashrc` or `/home/karthik/.zshrc` without asking first
- Don't install system packages — use conda/pip inside project environments

## Communication
- Keep answers short and simple — no jargon, no walls of text
- If I won't understand a term, explain it in plain English
- When suggesting a plan, list clear steps I can follow
- Ask before doing anything that can't be undone

## Reporting Units
- Report all speeds in **mph** (miles per hour), not m/s — tables are shown to an audience (1 m/s = 2.237 mph)
- Keep distances in metres
- For braking, give m/s² with the mph-per-second value next to it (1 m/s² = 2.237 mph/s)

## Terminology (use the same words in every plot, table and report)
- **Closest distance**: shortest distance from the car body (its box, not its centre) to the pedestrian. Say "distance", never "gap", in labels and text
- **Old loss** = original RAP score loss (BCE on the final score). **New loss** = E6 score loss (5 sub-scores + InfoNCE, 1.5 m pedestrian rule)
- **+ 0 accel** = ego acceleration input (ax, ay) set to 0 in training and testing (`zero_ego_accel=True`)
- **Original data** = ped_v2 with PDM-Closed labels (104,936 samples). **Yield-edited data** = cleaned ped_v2 with labels edited to keep 5 m from the pedestrian (61,088 samples)
- **Original trajectory / edited trajectory** = label before / after the yield edit
- **RGB** = real camera photos as input. **Raster** = rendered rasters as input
- Experiments: E4 = old loss, original data · E5 = old loss, yield-edited data · E6 = new loss, yield-edited data · E7 = E6 + 0 accel · E8 = E4 + 0 accel

## Long-Running Tasks
- If a task will take more than 5 minutes, run it inside a **tmux session** (not as a background task)
- Use `tmux new-session -d -s <name>` to create, `tmux send-keys` to run commands
- This way tasks survive disconnects and can be monitored with `tmux attach`

## Project Layout
- `/datahdd/karthik/RAP/` — main project (fresh clone of vita-epfl/RAP)
- `/datahdd/karthik/alpamayo1.5/` — Alpamayo model
- `/datahdd/karthik/dataset/` — NuPlan maps + data (don't move/delete)
- `/datahdd/karthik/nuplan_dataset/` — NuPlan dataset (don't move/delete)
- `/datahdd/karthik/sem_1/` — archived old work from last semester

## Docker
- System-wide Docker daemon (not rootless)
- Always use `-u $(id -u):$(id -g)` to avoid root-owned files
- RAP image: `karthikpullalarevu/aurora:latest`
- Alpamayo image: `karthikpullalarevu/alpamayo:latest`

## Git
- Always work on feature branches, not main
- Commit often with clear messages
- Don't force push

---

# RAP Project Rules

## What is RAP?
RAP = Rasterization Augmented Planning. It's an autonomous driving model that:
- Takes camera images as input
- Creates 3D rasterized (simplified rendered) views of the driving scene
- Uses these to train a planner that outputs driving trajectories
- Trained on NuPlan/NavSim datasets, can be finetuned on Waymo

## Environment
- Python 3.9, PyTorch 2.1, CUDA 12.1
- Conda env name: `rap` (create from environment.yml + requirements.txt)
- Docker image (if needed): `karthikpullalarevu/aurora:latest`

## Dataset Paths
- NuPlan maps: `/datahdd/karthik/dataset/nuplan-maps-v1.0`
- NuPlan data: `/datahdd/karthik/dataset/nuplan-v1.1/`
- Perturbed data: `/datahdd/karthik/dataset_perturbed/`
- Augmented data: `/datahdd/karthik/dataset_aug/`
- Cache directories go under `./cache/` inside this repo

## Environment Variables (set before running anything)
```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/datahdd/karthik/dataset/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/datahdd/karthik/RAP/exp"
export NAVSIM_DEVKIT_ROOT="/datahdd/karthik/RAP/navsim"
export OPENSCENE_DATA_ROOT="/datahdd/karthik/dataset/nuplan-v1.1"
```

## Training Steps (in order)
1. Cache ego vehicle data
2. Cache cross-agent data (uses dataset_aug)
3. Cache recovery perturbation data (uses dataset_perturbed)
4. Cache training metrics
5. Start Ray (use `ray_start` function in zshrc)
6. Run training script

## Git Rules
- Work on feature branches, never commit directly to main
- Keep experiments in separate branches (e.g., `exp/pedestrian-ablation`)
- Don't commit checkpoints, cached data, or large files

## Don't
- Don't delete or overwrite dataset files
- Don't modify files in `navsim/` unless we're forking the upstream code intentionally
- Don't install packages globally — use the conda env
