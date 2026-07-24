"""V11 T1: Cong-targeted SA (Simulated Annealing) polish.

Greedy swap_refine (script6) accepts only downhill moves, so once we hit a
local minimum it can't escape — the trace for ibm06 V10 shows 14+ refine_iters
each shaving < 1e-3 off the proxy. This adds Metropolis acceptance with an
exponentially cooled temperature, biased toward macros sitting in the top-5%
congestion cells.

Key differences vs script6.swap_refine:
  • Metropolis: accept worsening moves with probability exp(-Δ/T)
  • Hot-macro bias: with probability `hot_prob`, pick m1 from the set of
    macros whose footprint touches a top-`hot_pct` congestion cell.
  • Exponential cooling: T_t = T_start · (T_end/T_start)^(t/T_max)
  • Returns BEST-seen pos, not last-accepted — outer-verify safe.

Caller is expected to legalize the result if any swap could create overlap
(this routine rejects any swap that introduces overlap, so output is already
legal as long as input was).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from script4 import FastProxy, _congestion_grids


def _hot_macros(pos_aug, sizes, n_hard, movable, fp: FastProxy,
                hot_pct=0.05, radius_cells=2.0) -> np.ndarray:
    """Indices of hard movable macros whose footprint touches a top-`hot_pct`
    congestion cell. Returns int64 array, possibly empty."""
    pin_pos = pos_aug[fp.owner_idx] + fp.offset_xy
    V, H = _congestion_grids(
        pin_pos, fp.mask, fp.net_weights,
        fp.cell_w, fp.cell_h, fp.grid_col, fp.grid_row,
        fp.grid_v_routes, fp.grid_h_routes, fp.smooth_range,
        pos_aug[:n_hard], sizes[:n_hard],
        fp.vrouting_alloc, fp.hrouting_alloc,
    )
    vh = V + H
    n_total = vh.size
    cnt = max(1, int(math.floor(n_total * hot_pct)))
    thresh = np.partition(vh, -cnt)[-cnt]
    gr, gc = fp.grid_row, fp.grid_col
    hot_mask = (vh >= thresh).reshape(gr, gc)
    cell_w = fp.cell_w
    cell_h = fp.cell_h
    pad_x = radius_cells * cell_w
    pad_y = radius_cells * cell_h
    out = []
    for m in range(n_hard):
        if not movable[m]:
            continue
        mw = sizes[m, 0]
        mh = sizes[m, 1]
        mx = pos_aug[m, 0]
        my = pos_aug[m, 1]
        x_min = mx - mw * 0.5 - pad_x
        x_max = mx + mw * 0.5 + pad_x
        y_min = my - mh * 0.5 - pad_y
        y_max = my + mh * 0.5 + pad_y
        bl_c = max(0, int(math.floor(x_min / cell_w)))
        ur_c = min(gc - 1, int(math.floor(x_max / cell_w)))
        bl_r = max(0, int(math.floor(y_min / cell_h)))
        ur_r = min(gr - 1, int(math.floor(y_max / cell_h)))
        if bl_c > ur_c or bl_r > ur_r:
            continue
        if hot_mask[bl_r:ur_r + 1, bl_c:ur_c + 1].any():
            out.append(m)
    return np.array(out, dtype=np.int64)


def _macro_overlaps(idx, pos, sizes, n_hard, margin):
    hw_i = sizes[idx, 0] * 0.5
    hh_i = sizes[idx, 1] * 0.5
    xi = pos[idx, 0]
    yi = pos[idx, 1]
    hw = sizes[:n_hard, 0] * 0.5
    hh = sizes[:n_hard, 1] * 0.5
    dx = np.abs(pos[:n_hard, 0] - xi)
    dy = np.abs(pos[:n_hard, 1] - yi)
    overlap = (dx < hw_i + hw + margin) & (dy < hh_i + hh + margin)
    overlap[idx] = False
    return bool(overlap.any())


def _full_proxy(fp: FastProxy, pos_aug):
    wl = fp.hpwl_normalized(pos_aug)
    d = fp.density_cost(pos_aug)
    chalf = fp.congestion_cost(pos_aug)
    return wl + d + chalf


def cong_sa_polish(placement: torch.Tensor, benchmark: Benchmark, plc, *,
                   n_attempts=10000, area_min_ratio=0.5, area_max_ratio=2.0,
                   margin=0.005, seed=42, max_pins_per_net=None,
                   hot_prob=0.7, hot_pct=0.05, radius_cells=2.0,
                   T_start_frac=0.005, T_end=1e-5,
                   refresh_hot_every=2000, time_cap=None, verbose=False):
    """Metropolis-SA polish with hot-macro bias. Returns the BEST pos seen.

    Args
    ----
    placement       : input [num_macros, 2] tensor.
    n_attempts      : hard cap on swap attempts.
    area_min_ratio,
    area_max_ratio  : same area-similarity filter as swap_refine.
    margin          : overlap test margin.
    hot_prob        : with this probability, m1 is drawn from the hot set.
    hot_pct         : cells in top X% of (V+H) are hot.
    radius_cells    : macro footprint expansion when assigning to hot cells.
    T_start_frac    : T_start = T_start_frac * init_proxy (so cooling is
                      scale-invariant; ~0.5% of init is typical SA).
    T_end           : final temperature.
    refresh_hot_every : recompute hot-macro set every N attempts to track
                      shifts in congestion. Cheaper than every attempt.
    time_cap        : hard wallclock cap in seconds.
    """
    rng = np.random.default_rng(seed)
    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros
    n_ports = benchmark.port_positions.shape[0]
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    mov_idx = np.flatnonzero(movable)
    if mov_idx.size < 2:
        return placement.clone()

    pos_aug = np.empty((n_macros + n_ports, 2), dtype=np.float64)
    pos_aug[:n_macros] = placement[:n_macros].cpu().numpy().astype(np.float64)
    if n_ports:
        pos_aug[n_macros:] = benchmark.port_positions.numpy().astype(np.float64)

    fp = FastProxy(benchmark, plc=plc, max_pins_per_net=max_pins_per_net)
    init = _full_proxy(fp, pos_aug)
    if verbose:
        print(f"  [cong-SA] init proxy={init:.6f}", flush=True)
    cur = init
    best = init
    best_pos = pos_aug.copy()

    areas = sizes[:n_hard, 0] * sizes[:n_hard, 1]
    accepts = 0
    metropolis_accepts = 0
    overlap_rej = 0
    score_rej = 0

    T_start = max(T_start_frac * init, 1e-4)
    log_T_ratio = math.log(max(T_end / T_start, 1e-12))

    hot_set = _hot_macros(pos_aug, sizes, n_hard, movable, fp,
                          hot_pct=hot_pct, radius_cells=radius_cells)
    if hot_set.size < 2:
        hot_set = mov_idx  # fall back to all movable

    t0 = time.time()
    for it in range(n_attempts):
        if time_cap is not None and time.time() - t0 > time_cap:
            break
        # Exponential cool: T = T_start * exp(log_ratio * t/N)
        T = T_start * math.exp(log_T_ratio * (it / max(n_attempts - 1, 1)))

        # Refresh hot macros periodically (congestion shifts as we swap).
        if it > 0 and it % refresh_hot_every == 0:
            try:
                hs = _hot_macros(best_pos, sizes, n_hard, movable, fp,
                                 hot_pct=hot_pct, radius_cells=radius_cells)
                if hs.size >= 2:
                    hot_set = hs
            except Exception:
                pass

        # Pick m1: with prob hot_prob from hot set, else uniform from movable.
        if hot_set.size >= 2 and rng.random() < hot_prob:
            m1 = int(rng.choice(hot_set))
        else:
            m1 = int(rng.choice(mov_idx))
        a1 = areas[m1]
        am = areas[mov_idx]
        cand = ((am > a1 * area_min_ratio)
                & (am < a1 * area_max_ratio)
                & (mov_idx != m1))
        if not cand.any():
            continue
        m2 = int(rng.choice(mov_idx[cand]))

        old1 = pos_aug[m1].copy()
        old2 = pos_aug[m2].copy()
        pos_aug[m1] = old2
        pos_aug[m2] = old1
        if (_macro_overlaps(m1, pos_aug, sizes, n_hard, margin)
                or _macro_overlaps(m2, pos_aug, sizes, n_hard, margin)):
            pos_aug[m1] = old1
            pos_aug[m2] = old2
            overlap_rej += 1
            continue

        new = _full_proxy(fp, pos_aug)
        delta = new - cur
        if delta < -1e-9:
            cur = new
            accepts += 1
            if cur < best:
                best = cur
                best_pos = pos_aug.copy()
        else:
            # Metropolis: accept with exp(-delta/T)
            if T > 1e-12 and rng.random() < math.exp(-delta / T):
                cur = new
                metropolis_accepts += 1
            else:
                pos_aug[m1] = old1
                pos_aug[m2] = old2
                score_rej += 1

    if verbose:
        print(f"  [cong-SA] greedy={accepts} metro={metropolis_accepts} "
              f"ovr_rej={overlap_rej} score_rej={score_rej} "
              f"{init:.6f} -> {best:.6f} "
              f"({100*(best-init)/init:+.2f}%) t={time.time()-t0:.0f}s",
              flush=True)
    out = placement.clone()
    out[:n_macros] = torch.from_numpy(best_pos[:n_macros].astype(np.float32))
    return out
