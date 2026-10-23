"""Check if the key-agent aux target (agent this proposal collides with / gets too close to) ever points at the
inserted ped_v2 pedestrian. For a few ped_v2 samples, score 2 proposals with the training scorer:
  keep: drive straight on at the current speed;  label: the training label.
Prints per proposal: NC, TTC, the collided / TTC agent and whether it is an inserted pedestrian, key-agent labels."""
import sys, json, random, gzip, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, "/datahdd/karthik/RAP")
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.rap_dino.score_module import compute_navsim_score as S

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
paths = MetricCacheLoader(Path("/datahdd/karthik/RAP/cache/train_metric_cache")).metric_cache_paths
peds = json.load(open(S.INSERTED_PEDS_FILE))
cands = sorted(t for t in paths if "_" in t and t.rsplit("_", 1)[0] in peds)
print(f"ped_v2 samples in train metric cache: {len(cands)} (of {len(paths)})")
random.seed(0)
train = Path("/datahdd2/karthik/ped_v2/cache/train")
done = 0
for tok in random.sample(cands, 400):
    tgt = list(train.glob(f"*/{tok}/rap_target.gz"))
    if not tgt:
        continue
    with gzip.open(tgt[0], "rb") as f:
        label = np.asarray(pickle.load(f)["trajectory"])[:8]
    mc = pickle.load(__import__("lzma").open(paths[tok], "rb"))
    v = float(np.hypot(*mc.ego_state.dynamic_car_state.rear_axle_velocity_2d.array))
    t = np.arange(1, 9) * 0.5
    keep = np.stack([v * t, 0 * t, 0 * t], -1)
    poses = np.stack([keep, label]).astype(np.float32)
    scores, corners, labels, _ = S.get_sub_score(paths[tok], poses, False, (1.5, 3.0, 0.5))
    ins = set(peds[tok.rsplit("_", 1)[0]])
    nc_pdm = S.get_sub_score(paths[tok], poses, False, None)[0][:, 0]     # without the E6 pedestrian rule
    obs = S.scorer._observation
    in_obs = sum(any(tk in obs[k].tokens for tk in ins) for k in range(S.scorer._ego_polygons.shape[1]))
    import shapely
    dmin = []
    for k in range(S.scorer._ego_polygons.shape[1]):
        g = [obs[k][tk] for tk in obs[k].tokens if tk in ins]
        dmin.append(min(shapely.distance(S.scorer._ego_polygons[0, k], gg) for gg in g) if g else np.inf)
    def kind(tid):
        for k in range(S.scorer._ego_polygons.shape[1]):
            if tid in obs[k].tokens:
                return "pedestrian?" if obs[k][tid].area < 2 else f"vehicle-size ({obs[k][tid].area:.0f} m2)"
        return "?"
    print(f"\n{tok}  ego {v*2.237:.1f} mph  inserted peds {len(ins)}")
    for i, name in enumerate(["keep", "label"]):
        col = list(S.scorer.proposal_fault_collided_track_ids[i]); ttc = list(S.scorer.ttc_collided_track_ids[i])
        tag = lambda ids: "-" if not ids else ("INSERTED PED" if ids[0] in ins else "other: " + kind(ids[0]))
        print(f"  {name:5s} NC={scores[i,0]:.1f} TTC={scores[i,3]:.1f} | collided: {tag(col):28s} ttc agent: {tag(ttc):28s}"
              f" | key-agent label steps: col={int(labels[i,:,0].sum())} ttc={int(labels[i,:,1].sum())}")
    print(f"  NC without E6 ped rule: keep={nc_pdm[0]:.1f} label={nc_pdm[1]:.1f} | inserted ped in scorer view at {in_obs}/{S.scorer._ego_polygons.shape[1]} steps,"
          f" closest gap of 'keep' car to it {min(dmin):.2f} m")
    done += 1
    if done >= N:
        break
