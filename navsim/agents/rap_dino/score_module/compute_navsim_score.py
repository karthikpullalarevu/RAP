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


PED_LANE_MARGIN = 1.0      # m beyond the car's sides that still counts as "in our path"
EGO_REAR_TO_FRONT = 1.461 + 2.588   # rear axle -> front bumper (m)
EGO_HALF_WIDTH = 1.1485


def ped_too_close_while_moving(simulated_states, cache_path, ped_rule):
    """(n_proposals,) bool: while moving, the proposal gets within the buffer of a pedestrian that is
    ahead of the front bumper and in or right beside our path (|lateral| < half width + PED_LANE_MARGIN).
    Pedestrians standing beside the road or next to / behind the car do not count.

    ped_rule = (buffer for ped_v2 samples, buffer for other samples, moving speed m/s).
    Uses the scorer's ego polygons and the observation of the last score_proposals call."""
    buf = ped_rule[0] if "ped_v2" in str(cache_path) else ped_rule[1]
    obs = scorer._observation
    peds = {t for t, o in obs.unique_objects.items() if o.tracked_object_type == TrackedObjectType.PEDESTRIAN}
    close = np.zeros(len(simulated_states), dtype=bool)
    if not peds:
        return close
    x, y = simulated_states[..., StateIndex.X], simulated_states[..., StateIndex.Y]
    h = simulated_states[..., StateIndex.HEADING]
    speed = np.hypot(simulated_states[..., StateIndex.VELOCITY_X], simulated_states[..., StateIndex.VELOCITY_Y])
    for t in range(simulated_states.shape[1]):
        moving = np.where((speed[:, t] > ped_rule[2]) & ~close)[0]
        if len(moving) == 0:
            continue
        occ = obs[t]
        toks = [tok for tok in occ.tokens if tok in peds]
        if not toks:
            continue
        geoms = np.array([occ[tok] for tok in toks], dtype=object)
        cxy = np.array([[g.centroid.x, g.centroid.y] for g in geoms])                       # (P, 2)
        dx = cxy[None, :, 0] - x[moving, t, None]
        dy = cxy[None, :, 1] - y[moving, t, None]
        c, s = np.cos(h[moving, t])[:, None], np.sin(h[moving, t])[:, None]
        lon, lat = dx * c + dy * s, -dx * s + dy * c                                        # (M, P) in the car frame
        in_path = (lon > EGO_REAR_TO_FRONT - 0.5) & (np.abs(lat) < EGO_HALF_WIDTH + PED_LANE_MARGIN)
        if not in_path.any():
            continue
        d = shapely.distance(scorer._ego_polygons[moving, t][:, None], geoms[None])          # (M, P)
        close[moving[((d < buf) & in_path).any(1)]] = True
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
