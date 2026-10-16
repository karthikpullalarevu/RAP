#!/usr/bin/env python3
"""Rewrite ped_v2 labels so ego yields: the inserted pedestrian never comes closer than GAP m to the car body.

Same path as PDM-Closed drove; only the speed along it changes. Solved separately from every 2 Hz
training anchor t_a, starting at PDM-Closed's real state there (what the model's inputs show):

  profile  = keep the current speed for a while, smooth (minimum-jerk) slow-down to v_min,
             hold v_min, then a smooth pull-away. v_min = 0 is a full stop.
  rule     = at every 20 Hz step of the next HORIZON s, the car box at its edited position is
             >= GAP m from every inserted pedestrian (real future positions from the log).
  choice   = gentlest that works: lowest peak braking (2.0 .. 4.0 m/s2), then highest v_min,
             then latest braking start, then shortest wait.

Outcome per anchor:
  unchanged  PDM-Closed's own drive already keeps the gap
  edited     a profile with peak braking <= A_HARD keeps the gap -> new label
  too_late   none does (or the pedestrian is already within GAP m) -> drop the sample
  side_pass  the pedestrian never enters our lane in this scene, and the label comes within SIDE_GAP m -> drop

Usage:
  python process_data/edit_yield_label.py --clips       # pictures for the 6 clips in 08_ped_beside_path_clips
  python process_data/edit_yield_label.py --scene <scene_name>
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/datahdd/karthik/dataset/nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
from process_data.build_ped_cache import load_simulation_log, _patch_map_path, global_to_local

PED = Path("/datahdd2/karthik/ped_v2")
RAW = PED / "raw"
OUT = PED / "analysis" / "09_yield_edit"
HZ, MPH = 20, 2.23694
HALF_LEN, HALF_WID = 2.588, 1.1485           # ego box around its centre (labels are centre poses)
LANE_MARGIN = 0.3                            # our lane = car width + this each side
GAP = 5.0                                    # m, required body-to-body gap
SIDE_GAP = 3.0                               # m, side-pass samples closer than this are dropped
A_LEVELS = [2.0, 2.5, 3.0, 3.5, 4.0]         # m/s2 peak braking tried in order; 2.0 = natural, 4.0 = hard limit
A_PULL_AWAY = 2.0                            # m/s2 peak acceleration when moving off
SMOOTH_PEAK = 1.875                          # peak / average rate of a minimum-jerk speed change
HORIZON = 8.0                                # s checked after the anchor (5 s label + 3 s so it doesn't end in a trap)
LABEL_STEPS, STEP = 10, 10                   # 10 poses every 0.5 s (10 frames at 20 Hz)
MIN_ANCHOR, MAX_ANCHOR = 30, 198
DS = 0.1                                     # m, path grid for the gap table


def smoothstep(tau):
    tau = np.clip(tau, 0, 1)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


# ---------------------------------------------------------------- scene loading / geometry

def load(scene):
    _patch_map_path()
    d = RAW / scene
    inserted = set(json.loads((d / "metadata/result.json").read_text())["saved_log_summary"]["pedestrian_track_tokens"])
    data = load_simulation_log(next((d / "simulation_log").glob("*.msgpack.xz"))).simulation_history.data
    ego = np.array([[s.ego_state.center.x, s.ego_state.center.y, s.ego_state.center.heading,
                     s.ego_state.dynamic_car_state.speed] for s in data])
    peds = {}                                     # token -> (n, 3) x, y, radius; nan where absent
    for i, s in enumerate(data):
        for o in s.observation.tracked_objects.tracked_objects:
            if o.track_token in inserted:
                a = peds.setdefault(o.track_token, np.full((len(data), 3), np.nan))
                a[i] = o.center.x, o.center.y, max(o.box.length, o.box.width) / 2
    return ego, peds


class Path1D:
    """The driven centre path as arc length -> pose (extended straight past its end)."""

    def __init__(self, xy, heading_end):
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        pts = xy[np.r_[True, seg > 1e-3]]
        pts = np.vstack([pts, pts[-1] + 150 * np.array([math.cos(heading_end), math.sin(heading_end)])])
        self.xy = pts
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        d = np.diff(pts, axis=0)
        self.seg_h = np.arctan2(d[:, 1], d[:, 0])

    def pose(self, s):
        s = np.asarray(s, float)
        x = np.interp(s, self.s, self.xy[:, 0])
        y = np.interp(s, self.s, self.xy[:, 1])
        i = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.seg_h) - 1)
        return x, y, self.seg_h[i]

    def project(self, p):
        a, b = self.xy[:-1], self.xy[1:]
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
        q = a + t[:, None] * ab
        k = int(np.argmin(np.linalg.norm(p - q, axis=1)))
        side = np.sign(ab[k, 0] * (p[1] - a[k, 1]) - ab[k, 1] * (p[0] - a[k, 0]))
        return self.s[k] + t[k] * np.linalg.norm(ab[k]), side * float(np.linalg.norm(p - q[k]))


def box_gap(x, y, h, px, py, pr):
    """Body-to-body gap between ego boxes (x, y, h) and round pedestrians (px, py, pr); broadcasts."""
    c, s = np.cos(h), np.sin(h)
    dx, dy = px - x, py - y
    lx, ly = dx * c + dy * s, -dx * s + dy * c
    return np.hypot(np.maximum(np.abs(lx) - HALF_LEN, 0), np.maximum(np.abs(ly) - HALF_WID, 0)) - pr


class Scene:
    def __init__(self, name):
        self.name = name
        self.ego, peds = load(name)
        self.n = len(self.ego)
        self.path = Path1D(self.ego[:, :2], self.ego[-1, 2])
        s = np.array([self.path.project(p)[0] for p in self.ego[:, :2]])
        self.s_pdm = np.maximum.accumulate(s)
        self.v_pdm = self.ego[:, 3]
        self.peds = np.stack(list(peds.values())) if peds else np.full((0, self.n, 3), np.nan)   # (P, n, 3)

        # gap table: G[t, k] = smallest gap from the car at path position k*DS to any pedestrian at time t
        self.sgrid = np.arange(0, self.s_pdm[-1] + 60, DS)
        gx, gy, gh = self.path.pose(self.sgrid)
        G = np.full((self.n, len(self.sgrid)), np.inf)
        for tr in self.peds:
            ok = ~np.isnan(tr[:, 0])
            g = box_gap(gx[None], gy[None], gh[None], tr[ok, 0, None], tr[ok, 1, None], tr[ok, 2, None])
            G[ok] = np.minimum(G[ok], g)
        self.G = G

        # does any pedestrian enter our lane (ahead of us) at any time?
        lane = HALF_WID + LANE_MARGIN
        self.enters_lane = False
        for tr in self.peds:
            for i in np.where(~np.isnan(tr[:, 0]))[0][::5]:
                sp, off = self.path.project(tr[i, :2])
                if abs(off) < lane + tr[i, 2] and sp > self.s_pdm[i]:
                    self.enters_lane = True
                    break
            if self.enters_lane:
                break

    def gaps(self, t0, s):
        """Gap at times t0, t0+1, ... for positions s (array)."""
        k = np.clip(np.round(s / DS).astype(int), 0, len(self.sgrid) - 1)
        return self.G[t0 + np.arange(len(s)), k]


# ---------------------------------------------------------------- speed profiles

def profile(s0, v0, n, tau_d, a_p, v_min, hold, v_tgt):
    """Distance and speed at 20 Hz for n+1 steps: cruise tau_d s, smooth slow-down to v_min at peak a_p,
    hold v_min for `hold` s (inf = never go), smooth pull-away to v_tgt."""
    t = np.arange(n + 1) / HZ
    T_d = SMOOTH_PEAK * (v0 - v_min) / a_p if v0 > v_min else 0.0
    v = np.where(t < tau_d, v0, v0 - (v0 - v_min) * smoothstep((t - tau_d) / max(T_d, 1e-6)))
    if np.isfinite(hold) and v_tgt > v_min:
        t_go = tau_d + T_d + hold
        T_a = SMOOTH_PEAK * (v_tgt - v_min) / A_PULL_AWAY
        v = np.where(t >= t_go, v_min + (v_tgt - v_min) * smoothstep((t - t_go) / T_a), v)
    s = s0 + np.r_[0, np.cumsum((v[1:] + v[:-1]) / 2 / HZ)]
    return s, v


def last_true(grid, ok):
    """Largest value in an increasing grid with ok(value) True, assuming ok is True..True False..False."""
    lo, hi = 0, len(grid) - 1
    if not ok(grid[0]):
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(grid[mid]):
            lo = mid
        else:
            hi = mid - 1
    return grid[lo]


def solve_anchor(sc, ta):
    """Edited 20 Hz (s, v) from anchor ta, its outcome and parameters."""
    n = min(int(HORIZON * HZ), sc.n - 1 - ta)
    s0, v0 = sc.s_pdm[ta], sc.v_pdm[ta]
    s_pdm = sc.s_pdm[ta:ta + n + 1]
    lab = slice(0, LABEL_STEPS * STEP + 1)
    g_pdm = sc.gaps(ta, s_pdm)
    res = dict(anchor=ta, v0_mph=v0 * MPH, pdm_min_gap=float(g_pdm[lab].min()), a_peak=np.nan, v_min_mph=np.nan,
               brake_after_s=np.nan, wait_s=np.nan)
    if not sc.enters_lane:
        res.update(outcome="side_pass" if g_pdm[lab].min() < SIDE_GAP else "unchanged", s=s_pdm,
                   v=sc.v_pdm[ta:ta + n + 1])
        return res
    if g_pdm.min() >= GAP:
        res.update(outcome="unchanged", s=s_pdm, v=sc.v_pdm[ta:ta + n + 1])
        return res
    v_tgt = max(v0, sc.v_pdm[ta:ta + n + 1].max())
    ok = lambda *p: sc.gaps(ta, profile(s0, v0, n, *p, v_tgt)[0]).min() >= GAP

    vgrid = np.arange(0, v0 + 1e-9, 0.25) if v0 > 0.25 else np.array([0.0])
    tgrid = np.arange(0, n / HZ + 1e-9, 0.1)
    for a_p in A_LEVELS:
        if not ok(0.0, a_p, 0.0, np.inf):                    # brake now, stop and never go: safest with this a_p
            continue
        v_min = last_true(vgrid, lambda vm: ok(0.0, a_p, vm, np.inf))
        tau_d = last_true(tgrid, lambda td: ok(td, a_p, v_min, np.inf))
        # shortest wait: ok is False..False True..True in hold, so search on the reversed condition
        holds = tgrid
        k = last_true(holds, lambda h: not ok(tau_d, a_p, v_min, h))
        hold = holds[0] if k is None else (holds[np.searchsorted(holds, k) + 1] if k < holds[-1] else np.inf)
        s, v = profile(s0, v0, n, tau_d, a_p, v_min, hold, v_tgt)
        res.update(outcome="edited", s=s, v=v, a_peak=a_p if v0 - v_min > 0.05 else 0.0, v_min_mph=v_min * MPH,
                   brake_after_s=tau_d, wait_s=hold)
        return res
    res.update(outcome="too_late", s=None, v=None)
    return res


def label_poses(sc, ta, s):
    """10 label poses (local x, y, heading) at +0.5 .. +5.0 s from edited distances s (20 Hz from ta)."""
    idx = np.arange(1, LABEL_STEPS + 1) * STEP
    x, y, h = sc.path.pose(s[idx])
    return global_to_local(np.column_stack([x, y]), h, *sc.ego[ta, :3])


def solve_scene(sc):
    anchors = range(MIN_ANCHOR, min(MAX_ANCHOR, sc.n - 1 - LABEL_STEPS * STEP) + 1, 10)
    return [solve_anchor(sc, ta) for ta in anchors]


# ---------------------------------------------------------------- picture

def plot(sc, rows, out, title_extra=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    PDM, EDIT, PEDC, INK, MUTED = "#9a9a9a", "#2a78d6", "#eb6834", "#222222", "#777777"
    OUTC = {"unchanged": "#1baf7a", "edited": EDIT, "too_late": "#d62728", "side_pass": "#a05fd0"}
    t = np.arange(sc.n) / HZ
    fig = plt.figure(figsize=(21, 12))
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 1.7], height_ratios=[1, 1, 1, 0.35])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_v, ax_g, ax_a, ax_o = (fig.add_subplot(gs[k, 1]) for k in range(4))

    # --- map: world rotated so the car starts pointing up
    ox, oy, oh = sc.ego[MIN_ANCHOR, :3]
    c, s_ = math.cos(-oh), math.sin(-oh)
    L = lambda x, y: (-((np.asarray(x) - ox) * s_ + (np.asarray(y) - oy) * c),
                      (np.asarray(x) - ox) * c - (np.asarray(y) - oy) * s_)
    ax_map.plot(*L(sc.ego[:, 0], sc.ego[:, 1]), color=PDM, lw=6, alpha=0.35, solid_capstyle="round")
    for tr in sc.peds:
        ok = ~np.isnan(tr[:, 0])
        ax_map.plot(*L(tr[ok, 0], tr[ok, 1]), color=PEDC, lw=1.5, ls=":")
        for i in np.where(ok)[0][::HZ]:
            ax_map.add_patch(Circle(L(tr[i, 0], tr[i, 1]), GAP, fc="none", ec=PEDC, lw=0.4, alpha=0.25))
            ax_map.add_patch(Circle(L(tr[i, 0], tr[i, 1]), tr[i, 2], fc=PEDC, ec="white", lw=0.5))
            ax_map.annotate(f"{t[i]:.0f}s", L(tr[i, 0], tr[i, 1]), xytext=(5, 3), textcoords="offset points",
                            fontsize=7.5, color=PEDC)
    for r in rows:
        ta = r["anchor"]
        ax_map.plot(*L(sc.ego[ta, 0], sc.ego[ta, 1]), "o", ms=6, color=OUTC[r["outcome"]], mec="white", zorder=5)
        if r["outcome"] == "edited":
            x, y, _ = sc.path.pose(r["s"][LABEL_STEPS * STEP])
            ax_map.plot(*L(x, y), "s", ms=5, color=EDIT, mec="white", zorder=6)
    ax_map.set_aspect("equal")
    ax_map.set_xlim(-25, 25)
    ys = np.r_[L(sc.ego[:, 0], sc.ego[:, 1])[1]]
    ax_map.set_ylim(-8, min(ys.max() + 5, 90))
    ax_map.grid(color="#eeeeee")
    ax_map.set_title("From above (car starts pointing up)\nwide grey = PDM-Closed path · circles = sample spots "
                     "(colour = outcome)\nblue squares = where each edited label ends after 5 s · orange = pedestrian, "
                     "faint ring = 5 m", fontsize=10.5, loc="left")

    # --- speed / gap / acceleration: PDM-Closed whole scene + each anchor's edited 5 s label
    ax_v.plot(t, sc.v_pdm * MPH, color=PDM, lw=2.5)
    g_full = sc.gaps(0, sc.s_pdm)
    ax_g.plot(t, g_full, color=PDM, lw=2.5)
    ax_a.plot(t, np.gradient(sc.v_pdm) * HZ, color=PDM, lw=1.2)
    for r in rows:
        if r["outcome"] != "edited":
            continue
        ta, m = r["anchor"], LABEL_STEPS * STEP + 1
        tt = t[ta] + np.arange(m) / HZ
        ax_v.plot(tt, r["v"][:m] * MPH, color=EDIT, lw=1.3, alpha=0.8)
        ax_g.plot(tt, sc.gaps(ta, r["s"][:m]), color=EDIT, lw=1.3, alpha=0.8)
        ax_a.plot(tt, np.gradient(r["v"][:m]) * HZ, color=EDIT, lw=1.3, alpha=0.8)
    ax_v.set_ylabel("speed (mph)", color=MUTED)
    ax_v.set_ylim(bottom=0)
    ax_v.set_title("Speed: grey = PDM-Closed (current labels), blue = edited 5 s label from each sample", loc="left",
                   fontsize=11.5)
    ax_g.axhline(GAP, color=INK, lw=0.9, ls="--")
    ax_g.set_ylim(0, 25)
    ax_g.set_ylabel("gap to pedestrian (m)", color=MUTED)
    ed = [r for r in rows if r["outcome"] == "edited"]
    worst = min((sc.gaps(r["anchor"], r["s"][:LABEL_STEPS * STEP + 1]).min() for r in ed), default=np.nan)
    ax_g.set_title(f"Gap, car body to pedestrian (dashed = {GAP:g} m) · PDM-Closed closest {np.nanmin(g_full):.1f} m"
                   f" · edited labels closest {worst:.1f} m", loc="left", fontsize=11.5)
    ax_a.axhline(0, color=MUTED, lw=0.6)
    ax_a.set_ylim(-5, 3)
    ax_a.set_ylabel("acceleration (m/s²)", color=MUTED)
    peak = max((r["a_peak"] for r in ed), default=0)
    ax_a.set_title(f"Acceleration (negative = braking) · edited peak braking {peak:.1f} m/s² ({peak * MPH:.1f} mph/s)",
                   loc="left", fontsize=11.5)

    # --- outcome strip
    for r in rows:
        ax_o.bar(t[r["anchor"]], 1, width=0.4, color=OUTC[r["outcome"]])
    ax_o.set_yticks([])
    ax_o.set_xlabel("seconds into the scene", color=MUTED)
    counts = pd.Series([r["outcome"] for r in rows]).value_counts().to_dict()
    ax_o.set_title("Training samples: " + " · ".join(f"{k} {counts.get(k, 0)}" for k in OUTC), loc="left",
                   fontsize=11.5)
    ax_o.legend(handles=[Line2D([], [], color=c, lw=6, label=k) for k, c in OUTC.items()], ncol=4, frameon=False,
                loc="upper right", bbox_to_anchor=(1, 1.9), fontsize=10)
    for ax in (ax_v, ax_g, ax_a, ax_o):
        ax.set_xlim(0, t[-1])
        ax.grid(color="#eeeeee")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"{sc.name}{title_extra}\nYield edit: keep every pedestrian ≥ {GAP:g} m from the car body · braking "
                 f"{A_LEVELS[0]:g}–{A_LEVELS[-1]:g} m/s² · pedestrian {'enters' if sc.enters_lane else 'never enters'}"
                 f" our lane", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=75)
    plt.close(fig)


def run(scene, tag=""):
    sc = Scene(scene)
    rows = solve_scene(sc)
    for r in rows:
        if r["outcome"] == "edited":
            p = label_poses(sc, r["anchor"], r["s"])
            r["label_end_m"] = float(np.hypot(*p[-1, :2]))
            r["label_gap_min"] = float(sc.gaps(r["anchor"], r["s"][:LABEL_STEPS * STEP + 1]).min())
    plot(sc, rows, OUT / f"{tag}yield_edit_{scene}.png", f"  ({tag.rstrip('_')})" if tag else "")
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("s", "v")} for r in rows])
    df.insert(0, "scene", scene)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--clips", action="store_true", help="the 6 clips in 08_ped_beside_path_clips/clips.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.clips:
        c = pd.read_csv(PED / "analysis" / "08_ped_beside_path_clips" / "clips.csv")
        jobs = [(x.scene, f"clip{x.clip[:2]}_") for x in c.itertuples()]
    if args.scene:
        jobs.append((args.scene, ""))
    pd.set_option("display.width", 250)
    for scene, tag in jobs:
        df = run(scene, tag)
        df.to_csv(OUT / f"{tag}yield_edit_{scene}.csv", index=False)
        print(f"\n== {tag}{scene}")
        cols = ["anchor", "v0_mph", "pdm_min_gap", "outcome", "a_peak", "v_min_mph", "brake_after_s", "wait_s",
                "label_gap_min"]
        print(df[[c for c in cols if c in df]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
