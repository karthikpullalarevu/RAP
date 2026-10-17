import os
import torch

from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import lzma
import pickle
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)
import numpy as np
import shapely
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex, EgoAreaIndex
from .train_pdm_scorer import PDMScorerConfig, PDMScorer

# metric_cache_loader = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/metric_cache"))
proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
simulator = PDMSimulator(proposal_sampling)
config = PDMScorerConfig( )
scorer = PDMScorer(proposal_sampling, config)

def get_scores(args):

    return [get_sub_score(a["token"],a["poses"],a["test"],a.get("ped_rule")) for a in args]


INSERTED_PEDS_FILE = "/datahdd2/karthik/ped_v2/analysis/10_e6/inserted_ped_tokens.json"
_inserted_peds = None


def inserted_peds(cache_path):
    """Track tokens of the inserted (InterPlan) pedestrians for a ped_v2 metric cache path, else None.

    ped_v2 samples are recognised by their token (<anchor token>_<frame>, anchor in the lookup); their metric
    caches are listed under ./cache/train_metric_cache like navtrain ones, so the path alone doesn't tell."""
    global _inserted_peds
    if _inserted_peds is None:
        import json
        with open(INSERTED_PEDS_FILE) as f:
            _inserted_peds = json.load(f)
    sample_token = os.path.basename(os.path.dirname(str(cache_path)))
    if "_" not in sample_token:
        return None
    peds = _inserted_peds.get(sample_token.rsplit("_", 1)[0])
    return set(peds) if peds else None


def ped_too_close_while_moving(simulated_states, cache_path, ped_rule):
    """(n_proposals,) bool: while moving (speed > ped_rule[2]), the proposal's car box comes within
    ped_rule[0] m of an inserted pedestrian. ped_v2 samples only (inserted pedestrians always cross our path);
    other samples (navtrain) are left to the PDM scorer. Uses the observation of the last score_proposals call."""
    peds = inserted_peds(cache_path)
    close = np.zeros(len(simulated_states), dtype=bool)
    if not peds:
        return close
    buf, v_moving = ped_rule[0], ped_rule[2]
    obs = scorer._observation
    speed = np.hypot(simulated_states[..., StateIndex.VELOCITY_X], simulated_states[..., StateIndex.VELOCITY_Y])
    for t in range(simulated_states.shape[1]):
        moving = np.where((speed[:, t] > v_moving) & ~close)[0]
        if len(moving) == 0:
            continue
        occ = obs[t]
        geoms = [occ[tok] for tok in occ.tokens if tok in peds]
        if not geoms:
            continue
        d = shapely.distance(scorer._ego_polygons[moving, t][:, None], np.array(geoms, dtype=object)[None]).min(1)
        close[moving[d < buf]] = True
    return close


def get_sub_score( metric_cache,poses,test,ped_rule=None):

    cache_path = metric_cache
    try:
        with lzma.open(metric_cache, "rb") as f:
            metric_cache = pickle.load(f)
    except (EOFError, lzma.LZMAError, pickle.UnpicklingError) as e:
        print(f"WARNING: corrupted metric cache: {metric_cache} ({e})")
        n_proposals = len(poses)
        n_poses = 40
        n_col = 2
        scores = np.zeros((n_proposals, 6))   # NC, DAC, EP, TTC, C, final (was 7 columns by mistake)
        corners = np.zeros((n_proposals, n_poses, n_col, 4, 2))
        labels = np.zeros((n_proposals, n_poses, n_col), dtype=bool)
        ego_areas = np.zeros((n_proposals, n_poses, len(EgoAreaIndex) - 1), dtype=bool)   # matches _ego_areas[:,1:,1:]
        return scores, corners, labels, ego_areas

    initial_ego_state = metric_cache.ego_state

    trajectory_states = []

    for model_trajectory in poses:
        pred_trajectory = transform_trajectory(Trajectory(model_trajectory[:8]), initial_ego_state)

        pred_states = get_trajectory_as_array(pred_trajectory, simulator.proposal_sampling,
                                              initial_ego_state.time_point)

        trajectory_states.append(pred_states)

    trajectory_states = np.stack(trajectory_states, axis=0)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)#32,41,11

    final_scores=scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.pdm_progress
    )

    num_col=2

    key_agent_corners = np.zeros([len(final_scores), scorer.proposal_sampling.num_poses ,num_col, 4, 2])
    key_agent_labels = np.zeros([len(final_scores), scorer.proposal_sampling.num_poses ,num_col],dtype=bool)
    ego_areas = scorer._ego_areas[:,1:,1:]

    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, :]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, :]
    #driving_direction_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, :  ]

    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, :]
    time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, :]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, :]


    scores=np.stack([no_at_fault_collisions,drivable_area_compliance,#driving_direction_compliance,
                     ego_progress,time_to_collision_within_bound,comfort,final_scores
                     ],axis=-1)#[:,None]

    if ped_rule is not None:   # E6: no-collision target also fails when passing a pedestrian too closely while moving
        scores = scores.copy()
        scores[ped_too_close_while_moving(simulated_states, cache_path, ped_rule), 0] = 0.0

    if not test:
        for i in range(len(scores)):
            # proposal_collided_track_ids=scorer.proposal_collided_track_ids[i]
            proposal_fault_collided_track_ids = scorer.proposal_fault_collided_track_ids[i]
            # temp_collided_track_ids=scorer.temp_collided_track_ids[i]

            if len(proposal_fault_collided_track_ids):
                col_token=proposal_fault_collided_track_ids[0]
                collision_time_idcs = int(scorer._collision_time_idcs[i])+1

                for time_idx in range(1,collision_time_idcs):
                    if  col_token in scorer._observation[time_idx].tokens:
                        key_agent_labels[i][time_idx-1,0] = True
                        key_agent_corners[i][time_idx-1,0]=np.array(scorer._observation[time_idx][col_token].boundary.xy).T[:4]

            ttc_collided_track_ids = scorer.ttc_collided_track_ids[i]

            if len(ttc_collided_track_ids):
                ttc_token=ttc_collided_track_ids[0]
                ttc_time_idcs = int(scorer._ttc_time_idcs[i])+1

                for time_idx in range(1,ttc_time_idcs):
                    if  ttc_token in scorer._observation[time_idx].tokens:
                        key_agent_labels[i][time_idx-1,1] = True
                        key_agent_corners[i][time_idx-1,1]=np.array(scorer._observation[time_idx][ttc_token].boundary.xy).T[:4]

        theta = initial_ego_state.rear_axle.heading
        origin_x = initial_ego_state.rear_axle.x
        origin_y = initial_ego_state.rear_axle.y

        c, s = np.cos(theta), np.sin(theta)
        mat = np.array([[c, -s],
                        [s, c]])

        key_agent_corners[...,0]-=origin_x
        key_agent_corners[...,1]-=origin_y

        key_agent_corners=key_agent_corners.dot(mat)

    return scores,key_agent_corners,key_agent_labels,ego_areas
