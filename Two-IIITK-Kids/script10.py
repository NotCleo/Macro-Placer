"""LNS (Large Neighborhood Search) targeting congestion hotspots.

Algorithm:
  1. Compute congestion grid V + H via numba (TILOS-accurate).
  2. Find top-percentile hotspot cells (e.g., top 5%).
  3. Identify movable hard macros within a radius of any hotspot cell.
     These are the "destroyed" subset.
  4. Temporarily fix all OTHER macros via benchmark.macro_fixed.
  5. Run v14 with strong noise injection on this constrained problem —
     the focused subset will get significantly perturbed and re-converge,
     potentially escaping the local minimum that left the hotspot.
  6. Restore benchmark.macro_fixed.
  7. Run a global CD pass to clean up any minor disturbances.

Naive LNS (random-region destroy + uniform-random repair) was tried earlier
and was worse than CD. The differences here:
  - Destroy targets the actual cost driver (highest C cells)
  - Repair uses v14's smooth-proxy gradient (proxy-aligned), not random
"""
from __future__ import annotations

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from script12 import NumpyProxy
from script11 import _compute_congestion_njit


def find_hotspot_macros(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    top_pct: float = 0.02,
    radius_cells: float = 2.0,
    max_pins_per_net: int = None,
    max_select_frac: float = 0.20,    # cap selection at this fraction of movable
    min_select: int = 5,              # don't select fewer than this
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Identify movable hard macros nearest to top-pct hotspot cells.

    Selection is capped at `max_select_frac * n_movable` to ensure the
    destroy is targeted (not a broad relax). If radius would yield too many,
    we keep the K closest by min-distance to any hotspot cell.

    Returns (selected_indices, hotspot_cell_mask, info_dict).
    """
    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros
    n_ports = benchmark.port_positions.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    fp = NumpyProxy(benchmark, plc=plc, max_pins_per_net=max_pins_per_net)

    pos_aug = np.zeros((n_macros + n_ports, 2), dtype=np.float64)
    pos_aug[:n_macros] = placement[:n_macros].cpu().numpy().astype(np.float64)
    if n_ports:
        pos_aug[n_macros:] = benchmark.port_positions.numpy().astype(np.float64)

    pin_pos = pos_aug[fp.owner_idx] + fp.offset_xy
    V, H = _compute_congestion_njit(
        pin_pos, fp.mask, fp.net_weights,
        fp.cell_w, fp.cell_h,
        fp.grid_col, fp.grid_row,
        fp.grid_v_routes, fp.grid_h_routes,
        fp.smooth_range,
        pos_aug[:n_hard], fp.sizes[:n_hard],
        fp.vrouting_alloc, fp.hrouting_alloc,
    )
    grid = (V + H).reshape(fp.grid_row, fp.grid_col)
    n_cells = grid.size
    n_top = max(int(n_cells * top_pct), 1)
    flat_sorted = np.sort(grid.flatten())[::-1]
    threshold = flat_sorted[n_top - 1]
    hotspot_mask = grid >= threshold

    # Hotspot cell centers in micron.
    hot_rows, hot_cols = np.where(hotspot_mask)
    hot_y = (hot_rows + 0.5) * fp.cell_h
    hot_x = (hot_cols + 0.5) * fp.cell_w

    # For each movable hard macro, find min distance to any hotspot cell.
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    macro_pos = pos_aug[:n_hard]
    radius = radius_cells * max(fp.cell_w, fp.cell_h)

    # Compute distances vectorized: [n_hard, n_hot]
    dx = macro_pos[:, 0:1] - hot_x[None, :]
    dy = macro_pos[:, 1:2] - hot_y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = dist.min(axis=1)
    # Soft selection: within radius AND movable.
    in_radius = (min_dist <= radius) & movable
    candidate_idx = np.flatnonzero(in_radius)

    # Cap selection at max_select_frac of movable.
    max_n = max(min_select, int(max_select_frac * movable.sum()))
    if candidate_idx.size > max_n:
        # Keep K closest to hotspot.
        sub_dist = min_dist[candidate_idx]
        keep = np.argsort(sub_dist)[:max_n]
        selected_idx = candidate_idx[keep]
    else:
        selected_idx = candidate_idx

    info = {
        "n_hotspot_cells": int(hotspot_mask.sum()),
        "n_total_cells": int(n_cells),
        "threshold_cong": float(threshold),
        "max_cong": float(grid.max()),
        "n_selected_macros": int(selected_idx.size),
        "n_in_radius": int(in_radius.sum()),
        "n_movable": int(movable.sum()),
        "radius_um": float(radius),
    }
    return selected_idx, hotspot_mask, info


def lns_cong_repair(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    v14_runner,                      # callable: (b, plc, init_pos, seed, **kw) -> placement
    cd_runner,                       # callable: (p, b, plc, seed) -> placement
    seed: int = 42,
    top_pct: float = 0.05,
    radius_cells: float = 3.0,
    noise_start: float = 0.5,
    n_steps: int = 500,
    max_pins_per_net: int = None,
    verbose: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Run LNS-cong: identify hotspot macros, repair via constrained v14,
    then global CD cleanup.

    `v14_runner` should be the runner's run_v14() — we pass it pre-built so
    we can keep this module independent of the runner's specifics.
    `cd_runner` should be the runner's run_cd().

    Returns (refined_placement, info_dict).
    """
    selected_idx, hotspot_mask, info = find_hotspot_macros(
        placement, benchmark, plc,
        top_pct=top_pct, radius_cells=radius_cells,
        max_pins_per_net=max_pins_per_net,
    )
    if verbose:
        print(f"  [lns] hotspot: {info['n_hotspot_cells']}/{info['n_total_cells']} "
              f"cells, max C={info['max_cong']:.4f}, threshold={info['threshold_cong']:.4f}",
              flush=True)
        print(f"  [lns] selected {info['n_selected_macros']} of {info['n_movable']} "
              f"movable macros (radius={info['radius_um']:.2f}um)", flush=True)

    if info['n_selected_macros'] == 0:
        if verbose:
            print(f"  [lns] no macros in hotspot region — skipping", flush=True)
        return placement, info

    # Save and override macro_fixed.
    n_hard = benchmark.num_hard_macros
    saved_fixed = benchmark.macro_fixed.clone()
    new_fixed = saved_fixed.clone()
    # Fix everything except the selected hard macros.
    new_fixed[:n_hard] = True
    new_fixed[selected_idx] = False
    benchmark.macro_fixed = new_fixed

    try:
        # Constrained v14 with strong noise on the selected subset.
        if verbose:
            print(f"  [lns] running constrained v14 (noise={noise_start}, "
                  f"n={n_steps})...", flush=True)
        p_refined = v14_runner(
            benchmark, plc, placement, seed,
            n_steps=n_steps, noise_start=noise_start,
            max_pins_per_net=max_pins_per_net,
        )
    finally:
        benchmark.macro_fixed = saved_fixed

    # Global CD cleanup.
    if verbose:
        print(f"  [lns] global CD cleanup...", flush=True)
    p_final = cd_runner(p_refined, benchmark, plc, seed,
                        max_pins_per_net=max_pins_per_net)
    return p_final, info
