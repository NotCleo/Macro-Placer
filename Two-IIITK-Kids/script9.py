"""Targeted hotspot perturbation — small jitter applied only to macros
sitting near the top congestion cells.

Replaces the AMRUT2 `basin_hop` strategy which jittered every movable
macro by sigma ~ 2.5% of canvas. That uniform shake-up destroyed too
much structure (proxy jumped 30%+) and CD could never recover within
the remaining budget.

This one:
  1. Re-uses the existing TILOS-faithful numba congestion grids from
     proxy_fast (no new kernel).
  2. Finds the top `top_pct` of (V+H) cells by smoothed value.
  3. Picks hard movable macros whose footprint intersects any hot cell
     (with `radius_cells` slack), capped at `max_select` and
     `select_frac * |movable|`.
  4. Applies a small Gaussian jitter (sigma = `sigma_frac` * canvas) to
     ONLY those macros. Everything else stays put.

Caller is expected to follow up with the existing legalize + cd_refine
pipeline — outer-verify keeps it loss-safe.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from macro_place.benchmark import Benchmark
from script4 import FastProxy, _congestion_grids


def _hot_cells(V_flat, H_flat, top_pct, gr, gc):
    """Return boolean mask [gr, gc] of cells in the top top_pct% of V+H."""
    vh = V_flat + H_flat
    n_total = vh.size
    cnt = max(1, int(math.floor(n_total * top_pct)))
    thresh = np.partition(vh, -cnt)[-cnt]
    mask_flat = vh >= thresh
    return mask_flat.reshape(gr, gc)


def _macros_in_hot_region(pos_aug, sizes, n_hard, movable_mask,
                          hot_mask, cell_w, cell_h, gc, gr,
                          radius_cells):
    """Return indices of hard movable macros whose footprint (expanded by
    `radius_cells`) intersects any hot cell. Order is irrelevant."""
    out = []
    pad_x = radius_cells * cell_w
    pad_y = radius_cells * cell_h
    for m in range(n_hard):
        if not movable_mask[m]:
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


def hotspot_perturb(placement: torch.Tensor, benchmark: Benchmark, plc,
                    *,
                    top_pct: float = 0.05,
                    radius_cells: float = 2.0,
                    select_frac: float = 0.12,
                    min_select: int = 4,
                    max_select: int = 32,
                    sigma_frac: float = 0.04,
                    max_pins_per_net: int | None = None,
                    seed: int = 42) -> tuple[torch.Tensor, dict]:
    """Returns (perturbed_placement, info_dict).

    `info_dict` includes:
      - 'n_selected': number of macros perturbed
      - 'n_hot_cells': number of hot cells found
      - 'sigma_frac': effective sigma fraction used

    Note: this does NOT re-legalize. Caller must legalize before scoring.
    """
    rng = np.random.default_rng(seed)
    n_hard = int(benchmark.num_hard_macros)
    n_macros = int(benchmark.num_macros)
    n_ports = int(benchmark.port_positions.shape[0])
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    movable = benchmark.get_movable_mask()[:n_hard].numpy()

    pos_aug = np.empty((n_macros + n_ports, 2), dtype=np.float64)
    pos_aug[:n_macros] = placement[:n_macros].detach().cpu().numpy().astype(np.float64)
    if n_ports:
        pos_aug[n_macros:] = benchmark.port_positions.numpy().astype(np.float64)

    fp = FastProxy(benchmark, plc=plc, max_pins_per_net=max_pins_per_net)
    pin_pos = pos_aug[fp.owner_idx] + fp.offset_xy
    V, H = _congestion_grids(
        pin_pos, fp.mask, fp.net_weights,
        fp.cell_w, fp.cell_h, fp.grid_col, fp.grid_row,
        fp.grid_v_routes, fp.grid_h_routes, fp.smooth_range,
        pos_aug[:n_hard], fp.sizes[:n_hard],
        fp.vrouting_alloc, fp.hrouting_alloc,
    )

    hot_mask = _hot_cells(V, H, top_pct, fp.grid_row, fp.grid_col)
    n_hot = int(hot_mask.sum())

    cands = _macros_in_hot_region(
        pos_aug, sizes, n_hard, movable, hot_mask,
        fp.cell_w, fp.cell_h, fp.grid_col, fp.grid_row,
        radius_cells,
    )

    n_movable = int(movable.sum())
    n_target = int(min(max_select, max(min_select, math.floor(select_frac * n_movable))))
    if cands.size > n_target:
        sel = rng.choice(cands, size=n_target, replace=False)
    else:
        sel = cands

    out = placement.clone().cpu()
    if sel.size > 0:
        jx = rng.normal(0.0, sigma_frac * cw, sel.size).astype(np.float32)
        jy = rng.normal(0.0, sigma_frac * ch, sel.size).astype(np.float32)
        # Keep centers inside canvas; legalizer will resolve any residual overlap.
        for k, m in enumerate(sel):
            hw = float(sizes[m, 0]) * 0.5
            hh = float(sizes[m, 1]) * 0.5
            nx = float(out[m, 0]) + float(jx[k])
            ny = float(out[m, 1]) + float(jy[k])
            nx = max(hw, min(cw - hw, nx))
            ny = max(hh, min(ch - hh, ny))
            out[m, 0] = nx
            out[m, 1] = ny

    info = {
        "n_selected": int(sel.size),
        "n_hot_cells": n_hot,
        "sigma_frac": sigma_frac,
        "n_cands": int(cands.size),
    }
    return out, info
