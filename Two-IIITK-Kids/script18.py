"""V13-rev2 Opt 3 — Direct hot-cell attack.

V13's hotspot_perturb (script9) randomly jitters macros near hot cells with
Gaussian noise — the typical outcome is a +0.025 proxy spike that CD then
fails to recover within its ~30s budget. graph_grad's direct_congestion_attack
takes the opposite approach: a PRINCIPLED grid sweep around each hot-cell
macro, scored with the bit-exact oracle, accepting only strict improvements.

Algorithm (inspired by graph_grad/placer.py:direct_congestion_attack):
  1. Compute the current FastProxy congestion grids (V + H smoothed, plus
     macro_route additive).
  2. Identify the top `n_top_cells` (default 16) hottest cells — these
     dominate the top-5% mean that the proxy's congestion cost measures.
  3. For each hot cell, find HARD movable macros whose footprint touches
     it (radius_cells expansion). Score them by total touched-cell heat.
  4. Iterate macros in priority order (highest stress first). For each:
       a. 5x5 grid sweep around current position (radius = sweep_radius_frac
          * canvas).
       b. For each candidate cell: legality check (no overlap), then score
          via compute_proxy_cost.
       c. Move to best position if it strictly improves the proxy.
  5. Outer caller wraps the whole thing through _record so worst case is
     a no-op.

Why this beats hotspot_perturb: random jitter sometimes lands on a good
spot but usually doesn't, and the post-jitter CD can't always recover.
Grid sweep is deterministic — it finds the best candidate within radius
in one shot. Trades wider exploration for tighter convergence.

Cost: ~80s budget for 30 macros × 25 grid cells × 1-2s oracle proxy.
Targets cong-dominated benches (ibm06, ibm12, ibm17, ibm18) where
congestion is the limiting metric.
"""
from __future__ import annotations

import math
import time
import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from script4 import FastProxy, _congestion_grids


def _slide_legal(pos_aug, i, nx, ny, sizes, n_hard, margin=0.005):
    """Return True if placing macro i at (nx, ny) doesn't overlap any
    other hard macro."""
    hw_i = sizes[i, 0] * 0.5
    hh_i = sizes[i, 1] * 0.5
    hw = sizes[:n_hard, 0] * 0.5
    hh = sizes[:n_hard, 1] * 0.5
    dx = np.abs(pos_aug[:n_hard, 0] - nx)
    dy = np.abs(pos_aug[:n_hard, 1] - ny)
    overlap = (dx < hw_i + hw + margin) & (dy < hh_i + hh + margin)
    overlap[i] = False
    return not overlap.any()


def hot_cell_attack(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    n_top_cells: int = 16,
    radius_cells: float = 2.0,
    sweep_steps: int = 5,
    sweep_radius_frac: float = 0.06,
    max_macros: int = 30,
    max_pins_per_net: int = None,
    seed: int = 42,
    time_cap: float = 100.0,
    verbose: bool = False,
) -> torch.Tensor:
    """Direct hot-cell attack with bit-exact oracle scoring.

    Returns the best-cost placement found (or input if nothing improved).
    Outer-verify safe — caller wraps result through _record.
    """
    n_hard = int(benchmark.num_hard_macros)
    n_macros = int(benchmark.num_macros)
    n_ports = int(benchmark.port_positions.shape[0])
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    movable = benchmark.get_movable_mask()[:n_hard].cpu().numpy()

    pos_aug = np.empty((n_macros + n_ports, 2), dtype=np.float64)
    pos_aug[:n_macros] = placement[:n_macros].cpu().numpy().astype(np.float64)
    if n_ports:
        pos_aug[n_macros:] = benchmark.port_positions.cpu().numpy().astype(np.float64)

    # Initial proxy via the oracle.
    try:
        cur_c = compute_proxy_cost(placement, benchmark, plc)
        cur_proxy = float(cur_c["proxy_cost"])
        if int(cur_c["overlap_count"]) > 0:
            return placement.clone()
    except Exception:
        return placement.clone()
    best_proxy = cur_proxy
    best_pos_aug = pos_aug.copy()

    # Identify hot cells via FastProxy's congestion grids.
    try:
        fp = FastProxy(benchmark, plc=plc, max_pins_per_net=max_pins_per_net)
        pin_pos = pos_aug[fp.owner_idx] + fp.offset_xy
        V_grid, H_grid = _congestion_grids(
            pin_pos, fp.mask, fp.net_weights,
            fp.cell_w, fp.cell_h, fp.grid_col, fp.grid_row,
            fp.grid_v_routes, fp.grid_h_routes, fp.smooth_range,
            pos_aug[:n_hard], fp.sizes[:n_hard],
            fp.vrouting_alloc, fp.hrouting_alloc,
        )
        vh = (V_grid + H_grid).reshape(fp.grid_row, fp.grid_col)
    except Exception:
        return placement.clone()

    # Top-N hot cells.
    n_cells_total = vh.size
    k_top = min(n_top_cells, max(1, int(n_cells_total * 0.05)))
    flat_idx = np.argpartition(-vh.ravel(), k_top - 1)[:k_top]
    hot_rows = (flat_idx // fp.grid_col).astype(int)
    hot_cols = (flat_idx % fp.grid_col).astype(int)
    hot_heat = vh.ravel()[flat_idx]
    hot_cells = list(zip(hot_rows.tolist(), hot_cols.tolist(), hot_heat.tolist()))

    # Score each hard movable macro by sum of touching-hot-cell heat.
    # A macro "touches" a hot cell if its footprint (expanded by
    # radius_cells) intersects the cell.
    pad_x = radius_cells * fp.cell_w
    pad_y = radius_cells * fp.cell_h
    macro_stress = {}
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
        bl_c = max(0, int(math.floor(x_min / fp.cell_w)))
        ur_c = min(fp.grid_col - 1, int(math.floor(x_max / fp.cell_w)))
        bl_r = max(0, int(math.floor(y_min / fp.cell_h)))
        ur_r = min(fp.grid_row - 1, int(math.floor(y_max / fp.cell_h)))
        s = 0.0
        for (r, c, h) in hot_cells:
            if bl_r <= r <= ur_r and bl_c <= c <= ur_c:
                s += h
        if s > 0:
            macro_stress[m] = s

    if not macro_stress:
        return placement.clone()

    macros_sorted = sorted(macro_stress.items(), key=lambda kv: -kv[1])[:max_macros]

    # Per-macro grid sweep with oracle scoring.
    t0 = time.time()
    accepts = 0
    rx = sweep_radius_frac * cw
    ry = sweep_radius_frac * ch
    # Pre-build sweep offsets (skip 0,0 — that's the current position).
    sweep_dx = np.linspace(-rx, rx, sweep_steps)
    sweep_dy = np.linspace(-ry, ry, sweep_steps)

    for m, _stress in macros_sorted:
        if time.time() - t0 > time_cap:
            break
        ox = pos_aug[m, 0]
        oy = pos_aug[m, 1]
        hw = sizes[m, 0] * 0.5
        hh = sizes[m, 1] * 0.5
        best_local_xy = None
        best_local_proxy = best_proxy
        for dx in sweep_dx:
            if time.time() - t0 > time_cap:
                break
            for dy in sweep_dy:
                nx = max(hw, min(cw - hw, ox + dx))
                ny = max(hh, min(ch - hh, oy + dy))
                if abs(nx - ox) < 1e-6 and abs(ny - oy) < 1e-6:
                    continue
                # Legality: no overlap with any other hard macro
                if not _slide_legal(pos_aug, m, nx, ny, sizes, n_hard):
                    continue
                # Tentative move
                pos_aug[m, 0] = nx
                pos_aug[m, 1] = ny
                try:
                    c = compute_proxy_cost(
                        torch.from_numpy(pos_aug[:n_macros].astype(np.float32)),
                        benchmark, plc,
                    )
                    p_new = float(c["proxy_cost"])
                    ovr_new = int(c["overlap_count"])
                except Exception:
                    pos_aug[m, 0] = ox
                    pos_aug[m, 1] = oy
                    continue
                if ovr_new == 0 and p_new < best_local_proxy - 1e-9:
                    best_local_proxy = p_new
                    best_local_xy = (nx, ny)
                # Restore for next probe
                pos_aug[m, 0] = ox
                pos_aug[m, 1] = oy

        if best_local_xy is not None and best_local_proxy < best_proxy - 1e-9:
            pos_aug[m, 0] = best_local_xy[0]
            pos_aug[m, 1] = best_local_xy[1]
            best_proxy = best_local_proxy
            best_pos_aug = pos_aug.copy()
            accepts += 1

    if verbose:
        print(
            f"  [hot-cell-attack] {accepts}/{len(macros_sorted)} macros improved "
            f"({cur_proxy:.4f} -> {best_proxy:.4f}) t={time.time() - t0:.0f}s",
            flush=True,
        )

    out = placement.clone()
    out[:n_macros] = torch.from_numpy(best_pos_aug[:n_macros].astype(np.float32))
    return out
