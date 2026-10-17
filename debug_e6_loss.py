#!/usr/bin/env python3
"""Dry run of the E6 score loss on real data, before training.

Takes real training samples (ped_v2 after the yield edit, plus normal navtrain), gets 64 real proposals per
sample from a checkpoint (E4), scores every proposal with the PDM scorer with and without the E6 pedestrian
rule (NC = 0 when within 3 m / 2.2 m of a pedestrian while moving), and computes the E6 losses
(weighted 5-head BCE + multi-positive InfoNCE on NC) for several made-up predictors:

  untrained       all logits 0 (what a fresh score head starts from)
  random          logits ~ N(0, 1)
  E4 head         the checkpoint's own outputs (its 5 sub-score outputs were never trained)
  oracle          predicts the new targets exactly (lowest possible loss)
  oracle, no rule predicts the scorer's targets but ignores the 3 m pedestrian rule

Also checks how often the training label itself fails the new NC rule (imitation vs score conflict).

Writes into --output-dir: e6_loss_overview.png, e6_loss_examples.png, e6_loss_samples.csv, e6_loss_table.csv

Usage:
    CUDA_VISIBLE_DEVICES=7 RAY_TMPDIR=/datahdd2/tmp/ray_eval python debug_e6_loss.py \
        --ckpt /datahdd2/karthik/RAP/experiments/E4_pedV2-105k_photo50k/epoch15.ckpt \
        --output-dir /datahdd2/karthik/RAP/experiments/E6_debug_loss
"""
import argparse
import glob
import gzip
import multiprocessing as mp
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PED_CACHE = Path("/datahdd2/karthik/ped_v2/cache/train")
NAV_CACHE = Path("/datahdd2/karthik/RAP/cache/ego_50k")
APPLIED = Path("/datahdd2/karthik/ped_v2/analysis/09_yield_edit/batch/yield_edit_all_anchors.csv")
RULE = (2.0, 2.0, 0.5)          # default; set with --buffer
MPH = 2.23694
HEADS = ["NC", "DAC", "EP", "TTC", "C"]


# ------------------------------------------------------------------ samples

def pick_samples(n_per_group, n_nav, seed=0):
    rng = random.Random(seed)
    d = pd.concat([pd.read_csv(APPLIED), pd.read_csv(str(APPLIED).replace(".csv", "_keyframes.csv"))])
    d = d[d.in_train_cache]
    rows = []
    for outcome, group in (("edited", "ped: 5 m kept"), ("edited_close", "ped: close stop")):
        for s in d[d.outcome == outcome].sample(n_per_group, random_state=seed)["sample"]:
            rows.append((group, PED_CACHE / s))
    edited = set(d["sample"])
    other = [n for n in os.listdir(PED_CACHE) if n not in edited]
    for s in rng.sample(other, n_per_group):
        rows.append(("ped: other", PED_CACHE / s))
    logs = sorted(os.listdir(NAV_CACHE))
    while sum(g == "navtrain" for g, _ in rows) < n_nav:
        log = NAV_CACHE / rng.choice(logs)
        rows.append(("navtrain", log / rng.choice(os.listdir(log))))
    out = []
    for group, d_ in rows:
        tok_dir = d_ if (d_ / "rap_feature.gz").exists() else next(p for p in d_.iterdir() if p.is_dir())
        out.append((group, tok_dir))
    return out


# ------------------------------------------------------------------ scoring worker

def score_worker(args):
    """Scorer targets with / without the pedestrian rule, the closest moving gap per trajectory, ped tracks."""
    import navsim.agents.rap_dino.score_module.compute_navsim_score as CNS
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    import shapely
    import lzma
    mc, poses, rule = args
    plain = CNS.get_sub_score(mc, poses, True, None)[0]
    ruled = CNS.get_sub_score(mc, poses, True, rule)[0]
    sc, obs = CNS.scorer, CNS.scorer._observation
    states = sc._states
    peds = {t for t, o in obs.unique_objects.items() if o.tracked_object_type == TrackedObjectType.PEDESTRIAN}
    speed = np.hypot(states[..., StateIndex.VELOCITY_X], states[..., StateIndex.VELOCITY_Y])
    gap = np.full(len(poses), np.inf)
    tracks = {}
    cache = pickle.load(lzma.open(mc, "rb"))
    e = cache.ego_state.rear_axle
    c, s = np.cos(-e.heading), np.sin(-e.heading)
    for t in range(states.shape[1]):
        occ = obs[t]
        toks = [k for k in occ.tokens if k in peds]
        for k in toks:
            p = occ[k].centroid
            dx, dy = p.x - e.x, p.y - e.y
            tracks.setdefault(k, []).append((dx * c - dy * s, dx * s + dy * c))
        moving = np.where(speed[:, t] > rule[2])[0]
        if toks and len(moving):
            d = shapely.distance(sc._ego_polygons[moving, t][:, None],
                                 np.array([occ[k] for k in toks], dtype=object)[None]).min(1)
            gap[moving] = np.minimum(gap[moving], d)
    return plain, ruled, gap, {k: np.array(v) for k, v in tracks.items()}


# ------------------------------------------------------------------ losses

def losses(agent_cls, cfg, logits, tgt):
    """Per-head BCE (weighted), loss A, InfoNCE for a batch. logits/tgt: (B, 64, 6)."""
    import torch.nn.functional as F
    import types
    fake = types.SimpleNamespace(_config=cfg)
    a, b = agent_cls.sub5_loss(fake, logits, tgt)
    bce = F.binary_cross_entropy_with_logits(logits[..., :5], tgt[..., :5], reduction="none").mean(dim=(0, 1))
    w = torch.tensor(cfg.sub5_weights)
    out = {f"BCE {h} (x{w[i]:g})": float(bce[i] * w[i]) for i, h in enumerate(HEADS)}
    out["loss A (sum)"] = float(a)
    out["InfoNCE"] = float(b)
    out["total score loss"] = float(a + cfg.infonce_weight * b)
    return out


def oracle_logits(t):
    return torch.logit(t.clamp(0.02, 0.98))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--per-group", type=int, default=16)
    ap.add_argument("--nav", type=int, default=16)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--buffer", type=float, default=2.0, help="moving buffer (m) for the pedestrian rule")
    args = ap.parse_args()
    rule = (args.buffer, args.buffer, 0.5)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from eval_ped_val import load_agent
    from navsim.agents.rap_dino.rap_agent import RAPAgent
    from navsim.agents.rap_dino.navsim_config import RAPConfig

    agent = load_agent(args.ckpt, True, True)
    mc_paths = agent.train_metric_cache_paths
    samples = pick_samples(args.per_group, args.nav)

    rows, jobs = [], []
    with torch.no_grad():
        for group, d in samples:
            feat = pickle.load(gzip.open(d / "rap_feature.gz"))
            target = pickle.load(gzip.open(d / "rap_target.gz"))
            token = target["token"]
            if token not in mc_paths:
                continue
            f = {k: v.unsqueeze(0).cuda() for k, v in feat.items()
                 if isinstance(v, torch.Tensor) and k not in ("camera_valid", "rendered_camera_feature")}
            o = agent.forward(f, return_score=True)
            props = o["trajectory"][0].float().cpu().numpy()            # (64, T, 3)
            logit = o["pred_logit"][0].float().cpu()                     # (64, 6)
            label = target["trajectory"].float().numpy()
            T = props.shape[1]
            rows.append(dict(group=group, token=token, dir=str(d), props=props, logit=logit, label=label[:T],
                             v0_mph=float(np.linalg.norm(feat["ego_status"][-1, 3:5])) * MPH))
            jobs.append((mc_paths[token], np.concatenate([props, label[None, :T]]), rule))
    print(f"{len(rows)} samples, scoring {len(jobs)} x 65 trajectories", flush=True)
    with mp.get_context("spawn").Pool(args.procs) as pool:
        res = pool.map(score_worker, jobs, chunksize=1)
    for r, (plain, ruled, gap, tracks) in zip(rows, res):
        r.update(plain=plain, ruled=ruled, gap=gap, tracks=tracks)

    # ---- per-sample stats
    stats = []
    for r in rows:
        p_nc, r_nc = r["plain"][:64, 0], r["ruled"][:64, 0]
        stats.append(dict(group=r["group"], token=r["token"], v0_mph=round(r["v0_mph"], 1),
                          nc_fail_scorer=int((p_nc < .5).sum()), nc_fail_rule=int((r_nc < .5).sum()),
                          flipped_by_rule=int(((p_nc > .5) & (r_nc < .5)).sum()),
                          infonce_active=bool((r_nc > .5).any() and (r_nc < .5).any()),
                          label_nc_scorer=float(r["plain"][64, 0]), label_nc_rule=float(r["ruled"][64, 0]),
                          label_moving_gap_m=round(float(r["gap"][64]), 2),
                          picked_e4=int(r["logit"][:, -1].argmax())))
    st = pd.DataFrame(stats)
    st["picked_nc_rule"] = [float(r["ruled"][s.picked_e4, 0]) for r, s in zip(rows, st.itertuples())]
    st.to_csv(out / "e6_loss_samples.csv", index=False)

    # ---- losses per predictor, ped vs navtrain
    cfg = RAPConfig()
    cfg.score_mode = "sub5"
    table = []
    torch.manual_seed(0)
    for part, sel in (("ped_v2", [r for r in rows if r["group"] != "navtrain"]),
                      ("navtrain", [r for r in rows if r["group"] == "navtrain"]), ("all", rows)):
        if not sel:
            continue
        tgt = torch.tensor(np.stack([r["ruled"][:64] for r in sel]), dtype=torch.float32)
        tgt_plain = torch.tensor(np.stack([r["plain"][:64] for r in sel]), dtype=torch.float32)
        e4 = torch.stack([r["logit"] for r in sel])
        preds = {"untrained": torch.zeros_like(tgt), "random": torch.randn_like(tgt), "E4 head": e4,
                 "oracle": oracle_logits(tgt), "oracle, no rule": oracle_logits(tgt_plain)}
        for name, lg in preds.items():
            table.append(dict(part=part, predictor=name, **losses(RAPAgent, cfg, lg, tgt)))
        old = torch.nn.functional.binary_cross_entropy_with_logits(e4[..., -1], tgt_plain[..., -1]).item()
        table.append(dict(part=part, predictor="E4 head, OLD loss (final PDMS BCE)", **{"total score loss": old}))
    tb = pd.DataFrame(table)
    tb.to_csv(out / "e6_loss_table.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(tb.round(3).to_string(index=False))
    print("\nPer group:")
    print(st.groupby("group").agg(samples=("token", "size"), v0_mph=("v0_mph", "median"),
                                  nc_fail_scorer=("nc_fail_scorer", "mean"), nc_fail_rule=("nc_fail_rule", "mean"),
                                  infonce_active=("infonce_active", "mean"),
                                  label_fails_rule=("label_nc_rule", lambda x: (x < .5).mean()),
                                  e4_pick_fails_rule=("picked_nc_rule", lambda x: (x < .5).mean())).round(2).to_string())

    draw_overview(st, tb, rows, out / "e6_loss_overview.png")
    draw_examples(rows, st, cfg, RAPAgent, out / "e6_loss_examples.png")
    print(f"Saved to {out}")


# ------------------------------------------------------------------ pictures

INK, MUTED, GRID = "#222222", "#777777", "#eeeeee"
SAFE, FAIL, FLIP, LABEL, PEDC = "#1baf7a", "#d62728", "#eb6834", "#222222", "#6250d6"
GROUPS = ["ped: 5 m kept", "ped: close stop", "ped: other", "navtrain"]


def _clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def draw_overview(st, tb, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(18, 11))

    # A: share of the 64 proposals failing NC, scorer vs rule, per group
    g = st.groupby("group").agg(scorer=("nc_fail_scorer", "mean"), rule=("nc_fail_rule", "mean")).reindex(GROUPS)
    x = np.arange(len(g))
    ax[0, 0].bar(x - 0.2, g.scorer, 0.38, color=FAIL, label="PDM scorer (hits only)")
    ax[0, 0].bar(x + 0.2, g.rule, 0.38, color=FLIP, label="with the moving rule (ped ahead in our path)")
    for i, (a, b) in enumerate(zip(g.scorer, g.rule)):
        ax[0, 0].text(i - 0.2, a + 0.5, f"{a:.1f}", ha="center", fontsize=10)
        ax[0, 0].text(i + 0.2, b + 0.5, f"{b:.1f}", ha="center", fontsize=10)
    ax[0, 0].set_xticks(x, g.index)
    ax[0, 0].set_ylabel("proposals with NC = 0 (of 64)", color=MUTED)
    ax[0, 0].set_title("A. How many of the 64 proposals count as unsafe (NC target 0)", loc="left")
    ax[0, 0].legend(frameon=False)
    _clean(ax[0, 0])

    # B: InfoNCE active share and label-fails-rule share
    h = st.groupby("group").agg(active=("infonce_active", "mean"),
                                label=("label_nc_rule", lambda v: (v < .5).mean()),
                                pick=("picked_nc_rule", lambda v: (v < .5).mean())).reindex(GROUPS) * 100
    for k, (col, c, lab) in enumerate((("active", "#2a78d6", "InfoNCE active (scene has safe + unsafe)"),
                                       ("label", LABEL, "training label itself fails the rule"),
                                       ("pick", FLIP, "E4's picked proposal fails the rule"))):
        ax[0, 1].bar(x + (k - 1) * 0.27, h[col], 0.25, color=c, label=lab)
        for i, v in enumerate(h[col]):
            ax[0, 1].text(i + (k - 1) * 0.27, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax[0, 1].set_xticks(x, h.index)
    ax[0, 1].set_ylim(0, 110)
    ax[0, 1].set_ylabel("% of samples", color=MUTED)
    ax[0, 1].set_title("B. InfoNCE coverage and label conflict", loc="left")
    ax[0, 1].legend(frameon=False, fontsize=9, loc="upper left")
    _clean(ax[0, 1])

    # C: loss values per predictor (all samples)
    t = tb[(tb.part == "all") & tb["loss A (sum)"].notna()].set_index("predictor")
    comps = [c for c in t.columns if c.startswith("BCE ")] + ["InfoNCE"]
    colors = ["#d62728", "#9a9a9a", "#bdbdbd", "#eb6834", "#d9d9d9", "#2a78d6"]
    bottom = np.zeros(len(t))
    for c, col in zip(comps, colors):
        ax[1, 0].barh(t.index, t[c], left=bottom, color=col, label=c, height=0.6, edgecolor="white", lw=1)
        bottom += t[c].values
    for i, v in enumerate(bottom):
        ax[1, 0].text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=10)
    old = tb[(tb.part == "all") & tb["loss A (sum)"].isna()]["total score loss"]
    if len(old):
        ax[1, 0].axvline(float(old.iloc[0]), color=MUTED, ls=":", lw=1)
        ax[1, 0].text(float(old.iloc[0]), len(t) - 0.4, f" old E4 score loss {float(old.iloc[0]):.2f}", fontsize=9,
                      color=MUTED)
    ax[1, 0].invert_yaxis()
    ax[1, 0].set_xlabel("loss", color=MUTED)
    ax[1, 0].set_title("C. E6 score loss by predictor (weighted BCE parts + InfoNCE)", loc="left")
    ax[1, 0].legend(frameon=False, fontsize=8.5, ncol=3, loc="lower right")
    _clean(ax[1, 0])

    # D: closest moving gap of every proposal vs its NC target
    gaps_ok, gaps_flip, gaps_hit = [], [], []
    for r in rows:
        if r["group"] == "navtrain":
            continue
        g_, p_, q_ = r["gap"][:64], r["plain"][:64, 0], r["ruled"][:64, 0]
        g_ = np.minimum(g_, 15)
        gaps_ok += list(g_[q_ > .5])
        gaps_flip += list(g_[(p_ > .5) & (q_ < .5)])
        gaps_hit += list(g_[p_ < .5])
    bins = np.arange(0, 15.5, 0.5)
    ax[1, 1].hist([gaps_hit, gaps_flip, gaps_ok], bins=bins, stacked=True, color=[FAIL, FLIP, SAFE],
                  label=["NC 0: scorer collision", "NC 0: new, rule (ped ahead in path)", "NC 1"])
    ax[1, 1].set_xlabel("closest gap to any pedestrian while moving (m, 15 = none / far); the rule only counts "
                        "peds ahead in our path", color=MUTED)
    ax[1, 1].set_ylabel("proposals (ped_v2 samples)", color=MUTED)
    ax[1, 1].set_title("D. Closest moving gap of each proposal, coloured by its NC target", loc="left")
    ax[1, 1].legend(frameon=False)
    _clean(ax[1, 1])
    fig.suptitle("E6 loss dry run on real training samples (E4 proposals)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def draw_examples(rows, st, cfg, agent_cls, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # most informative: ped samples with the most proposals flipped by the rule, one per group if possible
    st = st.assign(i=range(len(st)))
    picks = []
    for g in GROUPS:
        s = st[st.group == g].sort_values("flipped_by_rule", ascending=False)
        if len(s):
            picks.append(int(s.i.iloc[0]))
    fig, axes = plt.subplots(1, len(picks), figsize=(6 * len(picks), 8.5))
    for ax, i in zip(np.atleast_1d(axes), picks):
        r = rows[i]
        p_nc, q_nc = r["plain"][:64, 0], r["ruled"][:64, 0]
        for k, tr in enumerate(r["props"]):
            col = FAIL if p_nc[k] < .5 else (FLIP if q_nc[k] < .5 else SAFE)
            path_ = np.vstack([[0, 0], tr[:, :2]])
            ax.plot(-path_[:, 1], path_[:, 0], color=col, lw=1, alpha=0.7)
        lab = np.vstack([[0, 0], r["label"][:, :2]])
        ax.plot(-lab[:, 1], lab[:, 0], "--", color=LABEL, lw=2.2,
                label=f"label (NC {'1' if r['ruled'][64, 0] > .5 else '0'} with rule)")
        for tr in r["tracks"].values():
            ax.plot(-tr[:, 1], tr[:, 0], ":", color=PEDC, lw=1.5)
            ax.plot(-tr[0, 1], tr[0, 0], "o", color=PEDC, ms=8)
        ax.plot([], [], color=SAFE, label="NC 1")
        ax.plot([], [], color=FLIP, label="NC 0 by moving rule")
        ax.plot([], [], color=FAIL, label="NC 0 by scorer")
        ax.plot([], [], ":o", color=PEDC, label="pedestrian (now, next 4 s)")
        tgt = torch.tensor(r["ruled"][None, :64], dtype=torch.float32)
        le4 = losses(agent_cls, cfg, r["logit"][None], tgt)
        lor = losses(agent_cls, cfg, oracle_logits(tgt), tgt)
        ax.set_title(f"{r['group']} · ego {r['v0_mph']:.0f} mph · label gap {st.label_moving_gap_m.iloc[i]:.1f} m\n"
                     f"flipped by rule {st.flipped_by_rule.iloc[i]} · loss A E4 {le4['loss A (sum)']:.2f} / oracle "
                     f"{lor['loss A (sum)']:.2f} · InfoNCE E4 {le4['InfoNCE']:.2f}", fontsize=10.5, loc="left")
        ax.set_aspect("equal")
        lim = max(15, np.abs(r["props"][..., :2]).max() + 2)
        ax.set_xlim(-lim / 2, lim / 2)
        ax.set_ylim(-3, lim)
        ax.set_xlabel("left  ←  m  →  right", color=MUTED)
        ax.set_ylabel("m ahead", color=MUTED)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        _clean(ax)
    fig.suptitle("Examples: the 64 E4 proposals coloured by their new NC target", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
