from __future__ import annotations

import time
import numpy as np
import torch

from macro_place.benchmark import Benchmark
from script4 import FastProxy


def _full_proxy(fp: FastProxy, pos_aug):
    wl = fp.hpwl_normalized(pos_aug)
    d = fp.density_cost(pos_aug)
    chalf = fp.congestion_cost(pos_aug)
    return wl + d + chalf


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


def swap_refine(placement: torch.Tensor, benchmark: Benchmark, plc, *,
                n_attempts=2000, area_min_ratio=0.5, area_max_ratio=2.0,
                margin=0.005, seed=42, max_pins_per_net=None, verbose=False,
                time_cap=None):
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
        print(f"  [swap] init proxy={init:.6f}", flush=True)
    cur = init
    best = init
    best_pos = pos_aug.copy()

    areas = sizes[:n_hard, 0] * sizes[:n_hard, 1]
    accepts = 0
    overlap_rej = 0
    score_rej = 0
    t0 = time.time()
    for it in range(n_attempts):
        if time_cap is not None and time.time() - t0 > time_cap:
            break
        m1 = int(rng.choice(mov_idx))
        a1 = areas[m1]
        am = areas[mov_idx]
        cand = (am > a1 * area_min_ratio) & (am < a1 * area_max_ratio) & (mov_idx != m1)
        if not cand.any():
            continue
        m2 = int(rng.choice(mov_idx[cand]))

        old1 = pos_aug[m1].copy()
        old2 = pos_aug[m2].copy()
        pos_aug[m1] = old2
        pos_aug[m2] = old1
        if _macro_overlaps(m1, pos_aug, sizes, n_hard, margin) or \
                _macro_overlaps(m2, pos_aug, sizes, n_hard, margin):
            pos_aug[m1] = old1
            pos_aug[m2] = old2
            overlap_rej += 1
            continue

        new = _full_proxy(fp, pos_aug)
        if new < cur - 1e-9:
            cur = new
            accepts += 1
            if cur < best:
                best = cur
                best_pos = pos_aug.copy()
        else:
            pos_aug[m1] = old1
            pos_aug[m2] = old2
            score_rej += 1

    if verbose:
        print(f"  [swap] accepts={accepts} ovr_rej={overlap_rej} "
              f"score_rej={score_rej} {init:.6f} -> {best:.6f} "
              f"({100*(best-init)/init:+.2f}%) t={time.time()-t0:.0f}s",
              flush=True)
    out = placement.clone()
    out[:n_macros] = torch.from_numpy(best_pos[:n_macros].astype(np.float32))
    return out
