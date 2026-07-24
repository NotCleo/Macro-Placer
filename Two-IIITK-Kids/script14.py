"""Signal-driven perturb-LNS for V8 (extracted from v14_final/test_signal_lns.py).

Each hotspot macro moves AWAY from its neighboring cell with the highest
congestion (V+H). Uses the actual congestion grid as signal to direct
perturbation, rather than random Gaussian noise.

Algorithm:
  1. Compute current V+H congestion grid (via numba).
  2. Identify hotspot macros (those near top-pct C cells).
  3. For each hotspot macro M:
     a. Find M's current cell (col, row).
     b. Examine 8 neighbor cells.
     c. Find the neighbor with HIGHEST V+H congestion (the "source of pressure").
     d. Move M AWAY from that direction by `step_um` micron.
     e. Clamp to canvas bounds.
  4. Caller re-converges globally via GD + CD + swap + CD + cd_cong.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from script12 import NumpyProxy
from script11 import _compute_congestion_njit
from script10 import find_hotspot_macros


def signal_perturb(p, b, plc, step_um, max_pins=None,
                   top_pct=0.05, radius_cells=2.0,
                   max_select_frac=0.20, seed=42):
    """For each hotspot macro, move AWAY from highest-C neighboring cell.

    Returns (perturbed_placement, info_dict). Caller is responsible for
    legalization and re-convergence — this function only perturbs and clamps.
    """
    n_hard = b.num_hard_macros
    n_macros = b.num_macros
    n_ports = b.port_positions.shape[0]
    cw = float(b.canvas_width)
    ch = float(b.canvas_height)

    # Build pos_aug
    pos_aug = np.zeros((n_macros + n_ports, 2), dtype=np.float64)
    pos_aug[:n_macros] = p[:n_macros].cpu().numpy().astype(np.float64)
    if n_ports:
        pos_aug[n_macros:] = b.port_positions.numpy().astype(np.float64)

    # Compute V+H grid via numba
    fp = NumpyProxy(b, plc=plc, max_pins_per_net=max_pins)
    pin_pos = pos_aug[fp.owner_idx] + fp.offset_xy
    V, H = _compute_congestion_njit(
        pin_pos, fp.mask, fp.net_weights,
        fp.cell_w, fp.cell_h, fp.grid_col, fp.grid_row,
        fp.grid_v_routes, fp.grid_h_routes, fp.smooth_range,
        pos_aug[:n_hard], fp.sizes[:n_hard],
        fp.vrouting_alloc, fp.hrouting_alloc,
    )
    grid = (V + H).reshape(fp.grid_row, fp.grid_col)

    # Find hotspot macros (uses same NumpyProxy + numba kernel internally)
    selected_idx, hotspot_mask, info = find_hotspot_macros(
        p, b, plc, top_pct=top_pct, radius_cells=radius_cells,
        max_pins_per_net=max_pins, max_select_frac=max_select_frac,
    )

    if info['n_selected_macros'] == 0:
        return p.clone(), info

    # For each hotspot macro, find max-C neighbor and move away
    sizes = b.macro_sizes.numpy()
    cell_w = fp.cell_w
    cell_h = fp.cell_h
    G_w = fp.grid_col
    G_h = fp.grid_row
    out = p.clone()
    out_np = out.numpy()
    n_actually_moved = 0
    for m in selected_idx:
        cx = int(np.floor(out_np[m, 0] / cell_w))
        cy = int(np.floor(out_np[m, 1] / cell_h))
        cx = max(0, min(G_w - 1, cx))
        cy = max(0, min(G_h - 1, cy))
        max_c = -np.inf
        max_dxdy = (0, 0)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                ncx, ncy = cx + dx, cy + dy
                if 0 <= ncx < G_w and 0 <= ncy < G_h:
                    c_val = float(grid[ncy, ncx])
                    if c_val > max_c:
                        max_c = c_val
                        max_dxdy = (dx, dy)
        if max_dxdy != (0, 0):
            mag = np.sqrt(max_dxdy[0] ** 2 + max_dxdy[1] ** 2)
            ux = -max_dxdy[0] / mag
            uy = -max_dxdy[1] / mag
            new_x = out_np[m, 0] + ux * step_um
            new_y = out_np[m, 1] + uy * step_um
            half_w = sizes[m, 0] / 2.0
            half_h = sizes[m, 1] / 2.0
            new_x = max(half_w, min(cw - half_w, new_x))
            new_y = max(half_h, min(ch - half_h, new_y))
            out_np[m, 0] = new_x
            out_np[m, 1] = new_y
            n_actually_moved += 1

    info["n_actually_moved"] = n_actually_moved
    return torch.from_numpy(out_np), info
