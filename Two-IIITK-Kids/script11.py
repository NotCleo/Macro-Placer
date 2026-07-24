"""Numba-accelerated congestion routing for the proxy.

Mirrors the pure-numpy logic in proxy_numpy.py:
  - Net routing: 2-pin / 3-pin / split (>3-pin) using L/T routing
  - Macro routing over grid cells (with PARTIAL adjustments)
  - V / H smoothing (kernel sum)
  - abu(V+H, 0.05) — top 5% mean

Pure-numpy version on ibm01: ~122 ms.
Numba version target:        ~5-15 ms (8-25x speedup).

The function `compute_congestion_numba` takes all the flat arrays and returns
1D V, H congestion arrays of length grid_row*grid_col. The downstream
`congestion_cost` wrapper computes top-5% mean.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _route_2pin_into(V, H, grid_col, src_y, src_x, snk_y, snk_x, weight):
    """Route a 2-pin net (source -> sink) into V, H congestion arrays."""
    col_min = src_x if src_x < snk_x else snk_x
    col_max = src_x if src_x > snk_x else snk_x
    row_min = src_y if src_y < snk_y else snk_y
    row_max = src_y if src_y > snk_y else snk_y
    # H routing on source row from col_min..col_max (excl)
    if col_max > col_min:
        base = src_y * grid_col
        for c in range(col_min, col_max):
            H[base + c] += weight
    # V routing on sink col from row_min..row_max (excl)
    if row_max > row_min:
        for r in range(row_min, row_max):
            V[r * grid_col + snk_x] += weight


@njit(cache=True, fastmath=False)
def _route_3pin_into(V, H, grid_col, gys, gxs, weight):
    """3-pin net routing — gys, gxs are length-3 arrays of cell coords.

    Mirrors TILOS __three_pin_net_routing dispatch on (sorted by (col, row)).
    """
    # Sort indices by (col, row).
    order = np.empty(3, dtype=np.int64)
    used = np.zeros(3, dtype=np.bool_)
    for i in range(3):
        best = -1
        for j in range(3):
            if used[j]:
                continue
            if best == -1 or (gxs[j] < gxs[best]) or (gxs[j] == gxs[best] and gys[j] < gys[best]):
                best = j
        order[i] = best
        used[best] = True
    y1 = gys[order[0]]; x1 = gxs[order[0]]
    y2 = gys[order[1]]; x2 = gxs[order[1]]
    y3 = gys[order[2]]; x3 = gxs[order[2]]

    if x1 < x2 and x2 < x3 and (min(y1, y3) < y2) and (max(y1, y3) > y2):
        # L routing
        for c in range(x1, x2):
            H[y1 * grid_col + c] += weight
        for c in range(x2, x3):
            H[y2 * grid_col + c] += weight
        for r in range(min(y1, y2), max(y1, y2)):
            V[r * grid_col + x2] += weight
        for r in range(min(y2, y3), max(y2, y3)):
            V[r * grid_col + x3] += weight
    elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
        for c in range(x1, x2):
            H[y1 * grid_col + c] += weight
        for r in range(y1, max(y2, y3)):
            V[r * grid_col + x2] += weight
    elif y2 == y3:
        for c in range(x1, x2):
            H[y1 * grid_col + c] += weight
        for c in range(x2, x3):
            H[y2 * grid_col + c] += weight
        for r in range(min(y2, y1), max(y2, y1)):
            V[r * grid_col + x2] += weight
    else:
        # T routing — sort by (row, col).
        # New order: by y ascending then x.
        used2 = np.zeros(3, dtype=np.bool_)
        order2 = np.empty(3, dtype=np.int64)
        for i in range(3):
            best = -1
            for j in range(3):
                if used2[j]:
                    continue
                if best == -1 or (gys[j] < gys[best]) or (gys[j] == gys[best] and gxs[j] < gxs[best]):
                    best = j
            order2[i] = best
            used2[best] = True
        ty1 = gys[order2[0]]; tx1 = gxs[order2[0]]
        ty2 = gys[order2[1]]; tx2 = gxs[order2[1]]
        ty3 = gys[order2[2]]; tx3 = gxs[order2[2]]
        xmin = tx1 if tx1 < tx2 else tx2
        if tx3 < xmin: xmin = tx3
        xmax = tx1 if tx1 > tx2 else tx2
        if tx3 > xmax: xmax = tx3
        for c in range(xmin, xmax):
            H[ty2 * grid_col + c] += weight
        for r in range(min(ty1, ty2), max(ty1, ty2)):
            V[r * grid_col + tx1] += weight
        for r in range(min(ty2, ty3), max(ty2, ty3)):
            V[r * grid_col + tx3] += weight


@njit(cache=True, fastmath=False)
def _macro_route_into(Vm, Hm, gc, gr,
                      mx, my, mw, mh,
                      cw_cell, ch_cell,
                      vroutes_alloc, hroutes_alloc):
    x_min = mx - mw * 0.5
    x_max = mx + mw * 0.5
    y_min = my - mh * 0.5
    y_max = my + mh * 0.5
    ur_col = int(math.floor(x_max / cw_cell))
    bl_col = int(math.floor(x_min / cw_cell))
    ur_row = int(math.floor(y_max / ch_cell))
    bl_row = int(math.floor(y_min / ch_cell))
    if ur_row < 0 or ur_col < 0:
        return
    if bl_row > gr - 1 or bl_col > gc - 1:
        return
    if bl_row < 0: bl_row = 0
    if bl_col < 0: bl_col = 0
    if ur_row > gr - 1: ur_row = gr - 1
    if ur_col > gc - 1: ur_col = gc - 1

    if_PARTIAL_VERT = False
    if_PARTIAL_HOR = False
    for r in range(bl_row, ur_row + 1):
        cy_min = r * ch_cell; cy_max = (r + 1) * ch_cell
        for c in range(bl_col, ur_col + 1):
            cx_min = c * cw_cell; cx_max = (c + 1) * cw_cell
            x_dist = (cx_max if cx_max < x_max else x_max) - (cx_min if cx_min > x_min else x_min)
            y_dist = (cy_max if cy_max < y_max else y_max) - (cy_min if cy_min > y_min else y_min)
            if x_dist <= 0 or y_dist <= 0:
                continue
            if ur_row != bl_row:
                if (r == bl_row and abs(y_dist - ch_cell) > 1e-5) or (r == ur_row and abs(y_dist - ch_cell) > 1e-5):
                    if_PARTIAL_VERT = True
            if ur_col != bl_col:
                if (c == bl_col and abs(x_dist - cw_cell) > 1e-5) or (c == ur_col and abs(x_dist - cw_cell) > 1e-5):
                    if_PARTIAL_HOR = True
            Vm[r * gc + c] += x_dist * vroutes_alloc
            Hm[r * gc + c] += y_dist * hroutes_alloc
    if if_PARTIAL_VERT:
        r = ur_row
        cy_min = r * ch_cell; cy_max = (r + 1) * ch_cell
        for c in range(bl_col, ur_col + 1):
            cx_min = c * cw_cell; cx_max = (c + 1) * cw_cell
            x_dist = (cx_max if cx_max < x_max else x_max) - (cx_min if cx_min > x_min else x_min)
            Vm[r * gc + c] -= x_dist * vroutes_alloc
    if if_PARTIAL_HOR:
        c = ur_col
        cx_min = c * cw_cell; cx_max = (c + 1) * cw_cell
        for r in range(bl_row, ur_row + 1):
            cy_min = r * ch_cell; cy_max = (r + 1) * ch_cell
            y_dist = (cy_max if cy_max < y_max else y_max) - (cy_min if cy_min > y_min else y_min)
            Hm[r * gc + c] -= y_dist * hroutes_alloc


@njit(cache=True, fastmath=False)
def _smooth_v_into(out, V_grid, smooth_range, gr, gc):
    for row in range(gr):
        for col in range(gc):
            lp = col - smooth_range
            if lp < 0: lp = 0
            rp = col + smooth_range
            if rp > gc - 1: rp = gc - 1
            gcell_cnt = rp - lp + 1
            val = V_grid[row, col] / gcell_cnt
            for ptr in range(lp, rp + 1):
                out[row, ptr] += val


@njit(cache=True, fastmath=False)
def _smooth_h_into(out, H_grid, smooth_range, gr, gc):
    for row in range(gr):
        for col in range(gc):
            lp = row - smooth_range
            if lp < 0: lp = 0
            up = row + smooth_range
            if up > gr - 1: up = gr - 1
            gcell_cnt = up - lp + 1
            val = H_grid[row, col] / gcell_cnt
            for ptr in range(lp, up + 1):
                out[ptr, col] += val


@njit(cache=True, fastmath=False)
def _compute_congestion_njit(
    pin_pos,           # [N_nets, P_max, 2] float64
    mask,              # [N_nets, P_max] bool
    net_weights,       # [N_nets] float64
    cell_w, cell_h,
    grid_col, grid_row,
    grid_v_routes, grid_h_routes,
    smooth_range,
    hard_pos,          # [n_hard, 2] float64
    hard_sizes,        # [n_hard, 2] float64
    vroutes_alloc, hroutes_alloc,
):
    """Full numba congestion computation. Returns (V, H) flat arrays."""
    n_nets, p_max = mask.shape
    n_cells = grid_col * grid_row

    V = np.zeros(n_cells, dtype=np.float64)
    H = np.zeros(n_cells, dtype=np.float64)
    Vm = np.zeros(n_cells, dtype=np.float64)
    Hm = np.zeros(n_cells, dtype=np.float64)

    # Per-net buffers — allocate in worst-case size of p_max.
    gys = np.empty(p_max, dtype=np.int64)
    gxs = np.empty(p_max, dtype=np.int64)
    # Track unique gcells via a bitset-ish dedup: hash to (grid_col*grid_row).
    # Simpler: just track first-seen by linear scan in a small array.

    for net_i in range(n_nets):
        w = net_weights[net_i]
        # Source pin = first unmasked pin.
        src_idx = -1
        for k in range(p_max):
            if mask[net_i, k]:
                src_idx = k
                break
        if src_idx < 0:
            continue
        src_x = pin_pos[net_i, src_idx, 0]
        src_y = pin_pos[net_i, src_idx, 1]
        src_col = int(math.floor(src_x / cell_w))
        src_row = int(math.floor(src_y / cell_h))
        # Clamp to valid grid range (pin position can land exactly on canvas
        # boundary -> floor gives grid_col / grid_row, which is OOB and causes
        # numba memory corruption / segfault).
        if src_col < 0: src_col = 0
        elif src_col > grid_col - 1: src_col = grid_col - 1
        if src_row < 0: src_row = 0
        elif src_row > grid_row - 1: src_row = grid_row - 1

        # Collect unique gcells across all valid pins.
        n_uniq = 0
        for k in range(p_max):
            if not mask[net_i, k]:
                continue
            x = pin_pos[net_i, k, 0]
            y = pin_pos[net_i, k, 1]
            c = int(math.floor(x / cell_w))
            r = int(math.floor(y / cell_h))
            if c < 0: c = 0
            elif c > grid_col - 1: c = grid_col - 1
            if r < 0: r = 0
            elif r > grid_row - 1: r = grid_row - 1
            # Linear-scan dedup.
            seen = False
            for u in range(n_uniq):
                if gys[u] == r and gxs[u] == c:
                    seen = True
                    break
            if not seen:
                gys[n_uniq] = r
                gxs[n_uniq] = c
                n_uniq += 1

        if n_uniq == 2:
            # gcells = the two unique. Determine sink.
            if gys[0] == src_row and gxs[0] == src_col:
                snk_y = gys[1]; snk_x = gxs[1]
            else:
                snk_y = gys[0]; snk_x = gxs[0]
            _route_2pin_into(V, H, grid_col, src_row, src_col, snk_y, snk_x, w)
        elif n_uniq == 3:
            _route_3pin_into(V, H, grid_col, gys[:3], gxs[:3], w)
        elif n_uniq > 3:
            # Split: each non-source pairs with source as 2-pin.
            for u in range(n_uniq):
                if gys[u] == src_row and gxs[u] == src_col:
                    continue
                _route_2pin_into(V, H, grid_col, src_row, src_col, gys[u], gxs[u], w)

    # Macro routing.
    n_hard = hard_pos.shape[0]
    for mi in range(n_hard):
        _macro_route_into(Vm, Hm, grid_col, grid_row,
                           hard_pos[mi, 0], hard_pos[mi, 1],
                           hard_sizes[mi, 0], hard_sizes[mi, 1],
                           cell_w, cell_h,
                           vroutes_alloc, hroutes_alloc)

    # Normalize.
    for i in range(n_cells):
        V[i] /= grid_v_routes
        H[i] /= grid_h_routes
        Vm[i] /= grid_v_routes
        Hm[i] /= grid_h_routes

    # Smoothing.
    V_grid = V.reshape((grid_row, grid_col))
    H_grid = H.reshape((grid_row, grid_col))
    V_sm = np.zeros((grid_row, grid_col), dtype=np.float64)
    H_sm = np.zeros((grid_row, grid_col), dtype=np.float64)
    _smooth_v_into(V_sm, V_grid, smooth_range, grid_row, grid_col)
    _smooth_h_into(H_sm, H_grid, smooth_range, grid_row, grid_col)
    V_out = V_sm.flatten()
    H_out = H_sm.flatten()
    for i in range(n_cells):
        V_out[i] += Vm[i]
        H_out[i] += Hm[i]
    return V_out, H_out


def congestion_cost_numba(proxy, positions_aug, pin_pos=None):
    """Wrapper computing congestion via Numba kernel and applying abu(0.05).

    If `pin_pos` is provided (precomputed [N, P, 2] float64), it is used
    directly — caller is responsible for keeping it consistent with
    positions_aug. This avoids the per-call O(N*P) fancy-indexing rebuild,
    the dominant cost for high-net-count benchmarks (~67% on ibm14).
    """
    if pin_pos is None:
        pin_pos = positions_aug[proxy.owner_idx] + proxy.offset_xy
    n_hard = proxy.n_hard
    # NOTE: positions_aug and pin_pos are float64 by construction — no astype
    # copy needed. proxy.sizes is also float64 already (set in NumpyProxy.__init__).
    V, H = _compute_congestion_njit(
        pin_pos,
        proxy.mask,
        proxy.net_weights,
        proxy.cell_w, proxy.cell_h,
        proxy.grid_col, proxy.grid_row,
        proxy.grid_v_routes, proxy.grid_h_routes,
        proxy.smooth_range,
        positions_aug[:n_hard],
        proxy.sizes[:n_hard],
        proxy.vrouting_alloc, proxy.hrouting_alloc,
    )
    all_vh = np.concatenate([V, H])
    n_total = all_vh.size
    cnt = int(math.floor(n_total * 0.05))
    if cnt == 0:
        return float(all_vh.max())
    sorted_desc = np.sort(all_vh)[::-1]
    return float(np.mean(sorted_desc[:cnt]))


# ----------------------------------------------------------------------------
# Validation against pure-numpy version on ibm01.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import os, sys, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from v6_legalize import V6LegalizePlacer
    from proxy_numpy import NumpyProxy

    bench = "ibm01"
    iccad = f"external/MacroPlacement/Testcases/ICCAD04/{bench}"
    print(f"=== proxy_numba validation on {bench} ===", flush=True)
    b, plc = load_benchmark_from_dir(iccad)

    placement = V6LegalizePlacer().place(b)
    real = compute_proxy_cost(placement, b, plc)
    print(f"  TILOS C = {real['congestion_cost']:.6f}", flush=True)

    fp = NumpyProxy(b, plc=plc)
    n_macros = b.num_macros
    n_ports = b.port_positions.shape[0]
    pos_aug = np.empty((n_macros + n_ports, 2), dtype=np.float64)
    pos_aug[:n_macros] = placement.numpy().astype(np.float64)[:n_macros]
    if n_ports:
        pos_aug[n_macros:] = b.port_positions.numpy().astype(np.float64)

    # Pure numpy reference.
    t0 = time.time()
    c_np = fp.congestion_cost(pos_aug)
    t_np = time.time() - t0
    print(f"  pure-numpy C = {c_np:.6f}  t={t_np*1000:.1f}ms", flush=True)

    # Numba (with JIT warm-up).
    print(f"  warming up numba JIT...", flush=True)
    _ = congestion_cost_numba(fp, pos_aug)
    t0 = time.time()
    c_nb = congestion_cost_numba(fp, pos_aug)
    t_nb = time.time() - t0
    print(f"  numba C    = {c_nb:.6f}  t={t_nb*1000:.1f}ms", flush=True)
    print(f"  diff (numba vs numpy): {abs(c_nb - c_np):.2e}  "
          f"(rel {100*abs(c_nb - c_np)/max(c_np, 1e-9):.6f}%)", flush=True)
    print(f"  diff (numba vs TILOS): {abs(c_nb - real['congestion_cost']):.2e}  "
          f"(rel {100*abs(c_nb - real['congestion_cost'])/max(real['congestion_cost'], 1e-9):.4f}%)",
          flush=True)
    print(f"  speedup numpy/numba: {t_np/max(t_nb, 1e-9):.1f}x", flush=True)

    # Run 5 more times to measure steady-state.
    times = []
    for _ in range(5):
        t0 = time.time()
        c_nb = congestion_cost_numba(fp, pos_aug)
        times.append(time.time() - t0)
    print(f"  steady-state numba: {1000*sum(times)/len(times):.2f}ms avg, "
          f"{1000*min(times):.2f}ms min", flush=True)
