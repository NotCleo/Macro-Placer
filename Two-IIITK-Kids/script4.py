from __future__ import annotations

import math
import numpy as np
from numba import njit

from macro_place.benchmark import Benchmark


@njit(cache=True, fastmath=False)
def _route_2pin(V, H, gcol, sy, sx, ty, tx, w):
    col_min = sx if sx < tx else tx
    col_max = sx if sx > tx else tx
    row_min = sy if sy < ty else ty
    row_max = sy if sy > ty else ty
    if col_max > col_min:
        base = sy * gcol
        for c in range(col_min, col_max):
            H[base + c] += w
    if row_max > row_min:
        for r in range(row_min, row_max):
            V[r * gcol + tx] += w


@njit(cache=True, fastmath=False)
def _route_3pin(V, H, gcol, gys, gxs, w):
    # Sort by (col, row).
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
    y1, x1 = gys[order[0]], gxs[order[0]]
    y2, x2 = gys[order[1]], gxs[order[1]]
    y3, x3 = gys[order[2]], gxs[order[2]]
    if x1 < x2 and x2 < x3 and (min(y1, y3) < y2) and (max(y1, y3) > y2):
        for c in range(x1, x2):
            H[y1 * gcol + c] += w
        for c in range(x2, x3):
            H[y2 * gcol + c] += w
        for r in range(min(y1, y2), max(y1, y2)):
            V[r * gcol + x2] += w
        for r in range(min(y2, y3), max(y2, y3)):
            V[r * gcol + x3] += w
    elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
        for c in range(x1, x2):
            H[y1 * gcol + c] += w
        for r in range(y1, max(y2, y3)):
            V[r * gcol + x2] += w
    elif y2 == y3:
        for c in range(x1, x2):
            H[y1 * gcol + c] += w
        for c in range(x2, x3):
            H[y2 * gcol + c] += w
        for r in range(min(y2, y1), max(y2, y1)):
            V[r * gcol + x2] += w
    else:
        # T-routing: sort by (row, col).
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
        ty1, tx1 = gys[order2[0]], gxs[order2[0]]
        ty2, tx2 = gys[order2[1]], gxs[order2[1]]
        ty3, tx3 = gys[order2[2]], gxs[order2[2]]
        xmin = tx1
        if tx2 < xmin:
            xmin = tx2
        if tx3 < xmin:
            xmin = tx3
        xmax = tx1
        if tx2 > xmax:
            xmax = tx2
        if tx3 > xmax:
            xmax = tx3
        for c in range(xmin, xmax):
            H[ty2 * gcol + c] += w
        for r in range(min(ty1, ty2), max(ty1, ty2)):
            V[r * gcol + tx1] += w
        for r in range(min(ty2, ty3), max(ty2, ty3)):
            V[r * gcol + tx3] += w


@njit(cache=True, fastmath=False)
def _route_one_net(V, H, gc, gr, pin_pos_net, mask_net, w, cw_c, ch_c):
    """Route a single net into the V/H grids with weight w.

    V18: extracted verbatim from the per-net body of _congestion_grids so
    the full evaluator and the incremental maintenance share one source of
    truth. All route kernels are linear in w, so calling this with -w
    exactly removes a net's prior contribution.
    """
    p_max = mask_net.shape[0]
    src_idx = -1
    for k in range(p_max):
        if mask_net[k]:
            src_idx = k
            break
    if src_idx < 0:
        return
    src_c = int(math.floor(pin_pos_net[src_idx, 0] / cw_c))
    src_r = int(math.floor(pin_pos_net[src_idx, 1] / ch_c))
    if src_c < 0:
        src_c = 0
    elif src_c > gc - 1:
        src_c = gc - 1
    if src_r < 0:
        src_r = 0
    elif src_r > gr - 1:
        src_r = gr - 1

    gys = np.empty(p_max, dtype=np.int64)
    gxs = np.empty(p_max, dtype=np.int64)
    n_uniq = 0
    for k in range(p_max):
        if not mask_net[k]:
            continue
        c = int(math.floor(pin_pos_net[k, 0] / cw_c))
        r = int(math.floor(pin_pos_net[k, 1] / ch_c))
        if c < 0:
            c = 0
        elif c > gc - 1:
            c = gc - 1
        if r < 0:
            r = 0
        elif r > gr - 1:
            r = gr - 1
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
        if gys[0] == src_r and gxs[0] == src_c:
            ty = gys[1]
            tx = gxs[1]
        else:
            ty = gys[0]
            tx = gxs[0]
        _route_2pin(V, H, gc, src_r, src_c, ty, tx, w)
    elif n_uniq == 3:
        _route_3pin(V, H, gc, gys[:3], gxs[:3], w)
    elif n_uniq > 3:
        for u in range(n_uniq):
            if gys[u] == src_r and gxs[u] == src_c:
                continue
            _route_2pin(V, H, gc, src_r, src_c, gys[u], gxs[u], w)


@njit(cache=True, fastmath=False)
def _batch_route_nets(V, H, gc, gr, pin_pos, mask, net_w, net_ids, sign,
                      cw_c, ch_c):
    """Route a set of nets (by id) into V/H with weight sign*net_w[ni].

    V18: one python→numba boundary crossing per try_move/revert instead of
    one per net.
    """
    for j in range(net_ids.shape[0]):
        ni = net_ids[j]
        _route_one_net(V, H, gc, gr, pin_pos[ni], mask[ni],
                       sign * net_w[ni], cw_c, ch_c)


@njit(cache=True, fastmath=False)
def _macro_route(Vm, Hm, gc, gr, mx, my, mw, mh, cw_c, ch_c,
                 v_alloc, h_alloc):
    x_min = mx - mw * 0.5
    x_max = mx + mw * 0.5
    y_min = my - mh * 0.5
    y_max = my + mh * 0.5
    ur_col = int(math.floor(x_max / cw_c))
    bl_col = int(math.floor(x_min / cw_c))
    ur_row = int(math.floor(y_max / ch_c))
    bl_row = int(math.floor(y_min / ch_c))
    if ur_row < 0 or ur_col < 0:
        return
    if bl_row > gr - 1 or bl_col > gc - 1:
        return
    if bl_row < 0:
        bl_row = 0
    if bl_col < 0:
        bl_col = 0
    if ur_row > gr - 1:
        ur_row = gr - 1
    if ur_col > gc - 1:
        ur_col = gc - 1

    pv = False
    ph = False
    for r in range(bl_row, ur_row + 1):
        cy_min = r * ch_c
        cy_max = (r + 1) * ch_c
        for c in range(bl_col, ur_col + 1):
            cx_min = c * cw_c
            cx_max = (c + 1) * cw_c
            x_d = (cx_max if cx_max < x_max else x_max) - (cx_min if cx_min > x_min else x_min)
            y_d = (cy_max if cy_max < y_max else y_max) - (cy_min if cy_min > y_min else y_min)
            if x_d <= 0 or y_d <= 0:
                continue
            if ur_row != bl_row:
                if (r == bl_row and abs(y_d - ch_c) > 1e-5) or (r == ur_row and abs(y_d - ch_c) > 1e-5):
                    pv = True
            if ur_col != bl_col:
                if (c == bl_col and abs(x_d - cw_c) > 1e-5) or (c == ur_col and abs(x_d - cw_c) > 1e-5):
                    ph = True
            Vm[r * gc + c] += x_d * v_alloc
            Hm[r * gc + c] += y_d * h_alloc
    if pv:
        r = ur_row
        for c in range(bl_col, ur_col + 1):
            cx_min = c * cw_c
            cx_max = (c + 1) * cw_c
            x_d = (cx_max if cx_max < x_max else x_max) - (cx_min if cx_min > x_min else x_min)
            Vm[r * gc + c] -= x_d * v_alloc
    if ph:
        c = ur_col
        for r in range(bl_row, ur_row + 1):
            cy_min = r * ch_c
            cy_max = (r + 1) * ch_c
            y_d = (cy_max if cy_max < y_max else y_max) - (cy_min if cy_min > y_min else y_min)
            Hm[r * gc + c] -= y_d * h_alloc


@njit(cache=True, fastmath=False)
def _batch_macro_route(Vm, Hm, gc, gr, hard_pos, hard_sizes,
                       cw_c, ch_c, v_alloc, h_alloc):
    """Route all hard-macro blockages into Vm/Hm. V18: shared by the full
    evaluator and the incremental rebuild."""
    n_hard = hard_pos.shape[0]
    for mi in range(n_hard):
        _macro_route(Vm, Hm, gc, gr,
                     hard_pos[mi, 0], hard_pos[mi, 1],
                     hard_sizes[mi, 0], hard_sizes[mi, 1],
                     cw_c, ch_c, v_alloc, h_alloc)


@njit(cache=True, fastmath=False)
def _combine_cong_grids(Vraw, Hraw, Vm, Hm, gc, gr,
                        grid_v_routes, grid_h_routes, sr):
    """Normalize + smooth the persistent raw grids into the final
    (V_out ++ H_out) congestion vector.

    V18: mirrors the normalize/smooth/add-macro tail of _congestion_grids
    exactly (divide-then-smooth; macro grids added unsmoothed) but works
    from the incrementally-maintained raw grids — O(grid cells), no net
    rerouting.
    """
    n_cells = gc * gr
    Vg = np.empty((gr, gc), dtype=np.float64)
    Hg = np.empty((gr, gc), dtype=np.float64)
    for r in range(gr):
        for c in range(gc):
            i = r * gc + c
            Vg[r, c] = Vraw[i] / grid_v_routes
            Hg[r, c] = Hraw[i] / grid_h_routes
    Vs = np.zeros((gr, gc), dtype=np.float64)
    Hs = np.zeros((gr, gc), dtype=np.float64)
    _smooth_v(Vs, Vg, sr, gr, gc)
    _smooth_h(Hs, Hg, sr, gr, gc)
    all_vh = np.empty(2 * n_cells, dtype=np.float64)
    for r in range(gr):
        for c in range(gc):
            i = r * gc + c
            all_vh[i] = Vs[r, c] + Vm[i] / grid_v_routes
            all_vh[n_cells + i] = Hs[r, c] + Hm[i] / grid_h_routes
    return all_vh


@njit(cache=True, fastmath=False)
def _smooth_v(out, V, sr, gr, gc):
    for r in range(gr):
        for c in range(gc):
            lp = c - sr
            if lp < 0:
                lp = 0
            rp = c + sr
            if rp > gc - 1:
                rp = gc - 1
            cnt = rp - lp + 1
            val = V[r, c] / cnt
            for p in range(lp, rp + 1):
                out[r, p] += val


@njit(cache=True, fastmath=False)
def _smooth_h(out, H, sr, gr, gc):
    for r in range(gr):
        for c in range(gc):
            lp = r - sr
            if lp < 0:
                lp = 0
            up = r + sr
            if up > gr - 1:
                up = gr - 1
            cnt = up - lp + 1
            val = H[r, c] / cnt
            for p in range(lp, up + 1):
                out[p, c] += val


@njit(cache=True, fastmath=False)
def _congestion_grids(pin_pos, mask, net_w, cw_c, ch_c, gc, gr,
                      grid_v_routes, grid_h_routes, sr,
                      hard_pos, hard_sizes, v_alloc, h_alloc):
    n_nets, p_max = mask.shape
    n_cells = gc * gr
    V = np.zeros(n_cells, dtype=np.float64)
    H = np.zeros(n_cells, dtype=np.float64)
    Vm = np.zeros(n_cells, dtype=np.float64)
    Hm = np.zeros(n_cells, dtype=np.float64)

    # V18: per-net body extracted into _route_one_net (shared with the
    # incremental maintenance path in IncrementalProxy).
    for net_i in range(n_nets):
        _route_one_net(V, H, gc, gr, pin_pos[net_i], mask[net_i],
                       net_w[net_i], cw_c, ch_c)

    _batch_macro_route(Vm, Hm, gc, gr, hard_pos, hard_sizes,
                       cw_c, ch_c, v_alloc, h_alloc)

    for i in range(n_cells):
        V[i] /= grid_v_routes
        H[i] /= grid_h_routes
        Vm[i] /= grid_v_routes
        Hm[i] /= grid_h_routes

    Vg = V.reshape((gr, gc))
    Hg = H.reshape((gr, gc))
    Vs = np.zeros((gr, gc), dtype=np.float64)
    Hs = np.zeros((gr, gc), dtype=np.float64)
    _smooth_v(Vs, Vg, sr, gr, gc)
    _smooth_h(Hs, Hg, sr, gr, gc)
    V_out = Vs.flatten()
    H_out = Hs.flatten()
    for i in range(n_cells):
        V_out[i] += Vm[i]
        H_out[i] += Hm[i]
    return V_out, H_out


# ── FastProxy: full WL + D + C evaluator ───────────────────────────────
class FastProxy:
    def __init__(self, benchmark: Benchmark, plc=None, max_pins_per_net=None):
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.n_macros = int(benchmark.num_macros)
        self.n_hard = int(benchmark.num_hard_macros)
        self.n_ports = int(benchmark.port_positions.shape[0])
        self.sizes = benchmark.macro_sizes.numpy().astype(np.float64).copy()
        self.grid_col = int(benchmark.grid_cols)
        self.grid_row = int(benchmark.grid_rows)
        self.cell_w = self.cw / self.grid_col
        self.cell_h = self.ch / self.grid_row

        if plc is not None:
            self.hroutes_per_micron = float(getattr(plc, "hroutes_per_micron", benchmark.hroutes_per_micron))
            self.vroutes_per_micron = float(getattr(plc, "vroutes_per_micron", benchmark.vroutes_per_micron))
            self.smooth_range = int(getattr(plc, "smooth_range", 2))
            self.hrouting_alloc = float(getattr(plc, "hrouting_alloc", 0.0))
            self.vrouting_alloc = float(getattr(plc, "vrouting_alloc", 0.0))
        else:
            self.hroutes_per_micron = float(benchmark.hroutes_per_micron)
            self.vroutes_per_micron = float(benchmark.vroutes_per_micron)
            self.smooth_range = 2
            self.hrouting_alloc = 0.0
            self.vrouting_alloc = 0.0
        self.grid_v_routes = self.cell_w * self.vroutes_per_micron
        self.grid_h_routes = self.cell_h * self.hroutes_per_micron
        self._max_pins = max_pins_per_net

        nets_pin = []
        for net in benchmark.net_pin_nodes:
            if net.shape[0] >= 2:
                nets_pin.append(net.long().numpy().astype(np.int64))
        if not nets_pin:
            self.num_nets_used = 0
            self.owner_idx = np.zeros((0, 0), dtype=np.int64)
            self.offset_xy = np.zeros((0, 0, 2), dtype=np.float64)
            self.mask = np.zeros((0, 0), dtype=bool)
            self.net_weights = np.zeros(0, dtype=np.float64)
        else:
            true_max = max(int(n.shape[0]) for n in nets_pin)
            if self._max_pins is not None and true_max > self._max_pins:
                cap = int(self._max_pins)
            else:
                cap = true_max
            self.num_nets_used = len(nets_pin)
            owner = np.zeros((self.num_nets_used, cap), dtype=np.int64)
            offx = np.zeros((self.num_nets_used, cap, 2), dtype=np.float64)
            mk = np.zeros((self.num_nets_used, cap), dtype=bool)
            pin_off = benchmark.macro_pin_offsets
            for i, net in enumerate(nets_pin):
                k = min(int(net.shape[0]), cap)
                owner[i, :k] = net[:k, 0]
                for j in range(k):
                    o = int(net[j, 0])
                    slot = int(net[j, 1])
                    if o < self.n_hard and slot < pin_off[o].shape[0]:
                        offx[i, j, 0] = float(pin_off[o][slot, 0])
                        offx[i, j, 1] = float(pin_off[o][slot, 1])
                mk[i, :k] = True
            self.owner_idx = owner
            self.offset_xy = offx
            self.mask = mk

            self.net_weights = np.ones(self.num_nets_used, dtype=np.float64)
            if plc is not None:
                drivers = list(plc.nets.keys())
                if len(drivers) == self.num_nets_used:
                    for i, drv in enumerate(drivers):
                        d_idx = plc.mod_name_to_indices.get(drv)
                        if d_idx is None:
                            continue
                        d_pin = plc.modules_w_pins[d_idx]
                        if hasattr(d_pin, "get_weight"):
                            self.net_weights[i] = float(d_pin.get_weight())

        self.net_count = float(self.net_weights.sum()) if self.num_nets_used else 1.0
        if self.net_count <= 0:
            self.net_count = 1.0

    # WL --------------------------------------------------------------
    def hpwl_total(self, pos_aug):
        if self.num_nets_used == 0:
            return 0.0
        pp = pos_aug[self.owner_idx] + self.offset_xy
        m = self.mask
        px_hi = np.where(m, pp[..., 0], -1e30)
        py_hi = np.where(m, pp[..., 1], -1e30)
        px_lo = np.where(m, pp[..., 0], 1e30)
        py_lo = np.where(m, pp[..., 1], 1e30)
        x_span = px_hi.max(axis=1) - px_lo.min(axis=1)
        y_span = py_hi.max(axis=1) - py_lo.min(axis=1)
        return float(((x_span + y_span) * self.net_weights).sum())

    def hpwl_normalized(self, pos_aug):
        return self.hpwl_total(pos_aug) / (self.net_count * (self.cw + self.ch))

    # Density ---------------------------------------------------------
    def density_cost(self, pos_aug):
        cw_c = self.cell_w
        ch_c = self.cell_h
        cw = self.cw
        ch = self.ch
        gc = self.grid_col
        gr = self.grid_row
        grid = np.zeros((gr, gc), dtype=np.float64)
        for m in range(self.n_macros):
            mw = self.sizes[m, 0]
            mh = self.sizes[m, 1]
            mx = pos_aug[m, 0]
            my = pos_aug[m, 1]
            x_min = mx - mw * 0.5
            x_max = mx + mw * 0.5
            y_min = my - mh * 0.5
            y_max = my + mh * 0.5
            if x_max <= 0 or y_max <= 0 or x_min >= cw or y_min >= ch:
                continue
            bl_c = max(0, int(math.floor(x_min / cw_c)))
            ur_c = min(gc - 1, int(math.floor(x_max / cw_c)))
            bl_r = max(0, int(math.floor(y_min / ch_c)))
            ur_r = min(gr - 1, int(math.floor(y_max / ch_c)))
            for r in range(bl_r, ur_r + 1):
                cy_min = r * ch_c
                cy_max = (r + 1) * ch_c
                oy = min(cy_max, y_max) - max(cy_min, y_min)
                if oy <= 0:
                    continue
                for c in range(bl_c, ur_c + 1):
                    cx_min = c * cw_c
                    cx_max = (c + 1) * cw_c
                    ox = min(cx_max, x_max) - max(cx_min, x_min)
                    if ox <= 0:
                        continue
                    grid[r, c] += ox * oy
        grid /= (cw_c * ch_c)
        flat = grid.flatten()
        occ = flat[flat != 0.0]
        if occ.size == 0:
            return 0.0
        n_total = grid.size
        if n_total < 10:
            return 0.5 * float(occ.mean())
        cnt = int(math.floor(n_total * 0.1))
        sd = np.sort(occ)[::-1]
        if cnt == 0 or cnt > sd.size:
            return 0.5 * float(sd.mean())
        return 0.5 * float(sd[:cnt].mean())

    # Congestion ------------------------------------------------------
    def congestion_cost(self, pos_aug, pin_pos=None):
        if pin_pos is None:
            pin_pos = pos_aug[self.owner_idx] + self.offset_xy
        V, H = _congestion_grids(
            pin_pos, self.mask, self.net_weights,
            self.cell_w, self.cell_h, self.grid_col, self.grid_row,
            self.grid_v_routes, self.grid_h_routes, self.smooth_range,
            pos_aug[:self.n_hard], self.sizes[:self.n_hard],
            self.vrouting_alloc, self.hrouting_alloc,
        )
        all_vh = np.concatenate([V, H])
        n_total = all_vh.size
        cnt = int(math.floor(n_total * 0.05))
        if cnt == 0:
            return float(all_vh.max())
        sd = np.sort(all_vh)[::-1]
        return 0.5 * float(sd[:cnt].mean())  # 0.5 prefactor baked in for caller

    def proxy_full(self, pos_aug, pin_pos=None):
        wl = self.hpwl_normalized(pos_aug)
        d = self.density_cost(pos_aug)
        c_half = self.congestion_cost(pos_aug, pin_pos=pin_pos)  # already *0.5
        return wl + d + c_half, wl, d, 2.0 * c_half  # also return raw cong for logs


# ── IncrementalProxy: O(macro degree) try/commit/revert ──────────────
class IncrementalProxy:
    def __init__(self, benchmark: Benchmark, plc=None):
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.n_macros = int(benchmark.num_macros)
        self.n_hard = int(benchmark.num_hard_macros)
        self.n_ports = int(benchmark.port_positions.shape[0])
        self.sizes = benchmark.macro_sizes.numpy().astype(np.float64).copy()
        self.grid_col = int(benchmark.grid_cols)
        self.grid_row = int(benchmark.grid_rows)
        self.cell_w = self.cw / self.grid_col
        self.cell_h = self.ch / self.grid_row
        self.cell_area = self.cell_w * self.cell_h

        self.net_pin_xy = []
        self.net_pin_offsets = []
        self.net_pin_owner = []
        self.net_weights = []
        self.net_bbox_total = []
        self.macro_to_pins = [[] for _ in range(self.n_macros + self.n_ports)]

        self.pos_aug = np.zeros((self.n_macros + self.n_ports, 2), dtype=np.float64)
        self.pos_aug[:self.n_macros] = benchmark.macro_positions.numpy().astype(np.float64)[:self.n_macros]
        if self.n_ports:
            self.pos_aug[self.n_macros:] = benchmark.port_positions.numpy().astype(np.float64)

        pin_off = benchmark.macro_pin_offsets
        n_used = 0
        for net in benchmark.net_pin_nodes:
            if net.shape[0] < 2:
                continue
            arr = net.long().numpy().astype(np.int64)
            P = int(arr.shape[0])
            owners = arr[:, 0].copy()
            offs = np.zeros((P, 2), dtype=np.float64)
            for p in range(P):
                o = int(arr[p, 0])
                slot = int(arr[p, 1])
                if o < self.n_hard and slot < pin_off[o].shape[0]:
                    offs[p, 0] = float(pin_off[o][slot, 0])
                    offs[p, 1] = float(pin_off[o][slot, 1])
            pin_xy = self.pos_aug[owners] + offs
            self.net_pin_xy.append(pin_xy)
            self.net_pin_offsets.append(offs)
            self.net_pin_owner.append(owners)
            self.net_weights.append(1.0)
            bbox = (float(pin_xy[:, 0].max() - pin_xy[:, 0].min())
                    + float(pin_xy[:, 1].max() - pin_xy[:, 1].min()))
            self.net_bbox_total.append(bbox)
            for p in range(P):
                self.macro_to_pins[int(owners[p])].append((n_used, p))
            n_used += 1
        self.num_nets_used = n_used

        if plc is not None and n_used:
            drivers = list(plc.nets.keys())
            if len(drivers) == n_used:
                for i, drv in enumerate(drivers):
                    d_idx = plc.mod_name_to_indices.get(drv)
                    if d_idx is None:
                        continue
                    d_pin = plc.modules_w_pins[d_idx]
                    if hasattr(d_pin, "get_weight"):
                        self.net_weights[i] = float(d_pin.get_weight())

        self.net_count = float(sum(self.net_weights)) if n_used else 1.0
        if self.net_count <= 0:
            self.net_count = 1.0
        self.total_hpwl_weighted = 0.0
        for i in range(n_used):
            self.total_hpwl_weighted += self.net_weights[i] * self.net_bbox_total[i]

        self.grid_occ = np.zeros((self.grid_row, self.grid_col), dtype=np.float64)
        self.macro_footprint = [None] * self.n_macros
        for m in range(self.n_macros):
            self._add_density(m, self.pos_aug[m, 0], self.pos_aug[m, 1])
        self._pending = None

        # V13-rev4: dense pin-pos cache + plc congestion config so
        # proxy_full_cached() can call _congestion_grids in O(n_nets * pins)
        # without rebuilding pin_pos from pos_aug + offsets each call.
        # Maintained incrementally by try_move/revert_move (~O(macro_degree)
        # per move). On medium benches this drops full-proxy from ~30-50ms
        # (FastProxy.proxy_full) to ~3-8ms.
        if self.num_nets_used > 0:
            max_pins_d = max(o.shape[0] for o in self.net_pin_owner)
        else:
            max_pins_d = 0
        self._dense_pin_pos = np.zeros(
            (self.num_nets_used, max_pins_d, 2), dtype=np.float64
        )
        self._dense_mask = np.zeros(
            (self.num_nets_used, max_pins_d), dtype=np.bool_
        )
        for ni in range(self.num_nets_used):
            P = int(self.net_pin_xy[ni].shape[0])
            self._dense_pin_pos[ni, :P] = self.net_pin_xy[ni]
            self._dense_mask[ni, :P] = True
        self._net_weights_arr = np.array(self.net_weights, dtype=np.float64)
        if plc is not None:
            self._hroutes_per_um = float(getattr(plc, "hroutes_per_micron",
                                                  benchmark.hroutes_per_micron))
            self._vroutes_per_um = float(getattr(plc, "vroutes_per_micron",
                                                  benchmark.vroutes_per_micron))
            self._smooth_range = int(getattr(plc, "smooth_range", 2))
            self._hrouting_alloc = float(getattr(plc, "hrouting_alloc", 0.0))
            self._vrouting_alloc = float(getattr(plc, "vrouting_alloc", 0.0))
        else:
            self._hroutes_per_um = float(benchmark.hroutes_per_micron)
            self._vroutes_per_um = float(benchmark.vroutes_per_micron)
            self._smooth_range = 2
            self._hrouting_alloc = 0.0
            self._vrouting_alloc = 0.0
        self._grid_v_routes = self.cell_w * self._vroutes_per_um
        self._grid_h_routes = self.cell_h * self._hroutes_per_um

        # ── V18: persistent raw congestion grids, maintained incrementally ──
        # proxy_full_cached() used to reroute ALL nets (28k-46k on HUGE) via
        # _congestion_grids on every call — the LAHC inner-loop bottleneck
        # (~3-8ms/iter). The route kernels are linear in their weight args,
        # so a move only needs to re-route the moved macro's own nets
        # (subtract old contribution, add new) — O(deg) instead of O(n_nets).
        # The final normalize/smooth/top-5% works on the small grid
        # (O(grid cells)) and stays per-eval.
        self.macro_to_nets = []
        for m in range(self.n_macros + self.n_ports):
            uniq = sorted({ni for (ni, _pi) in self.macro_to_pins[m]})
            self.macro_to_nets.append(np.array(uniq, dtype=np.int64))
        self._all_net_ids = np.arange(self.num_nets_used, dtype=np.int64)
        self._cong_ops = 0  # incremental ops since last full rebuild
        self._rebuild_cong()

    def _rebuild_cong(self):
        """Full rebuild of the raw congestion grids from current state.

        Costs ≈ one pre-V18 proxy_full_cached call; used at init, on
        reset_to_positions, and periodically to wash out float drift from
        repeated subtract/add cycles.
        """
        n_cells = self.grid_row * self.grid_col
        self._Vraw = np.zeros(n_cells, dtype=np.float64)
        self._Hraw = np.zeros(n_cells, dtype=np.float64)
        self._Vm = np.zeros(n_cells, dtype=np.float64)
        self._Hm = np.zeros(n_cells, dtype=np.float64)
        if self.num_nets_used:
            _batch_route_nets(self._Vraw, self._Hraw,
                              self.grid_col, self.grid_row,
                              self._dense_pin_pos, self._dense_mask,
                              self._net_weights_arr, self._all_net_ids, 1.0,
                              self.cell_w, self.cell_h)
        if self.n_hard:
            _batch_macro_route(self._Vm, self._Hm,
                               self.grid_col, self.grid_row,
                               self.pos_aug[:self.n_hard],
                               self.sizes[:self.n_hard],
                               self.cell_w, self.cell_h,
                               self._vrouting_alloc, self._hrouting_alloc)
        self._cong_ops = 0

    def hpwl_cost(self):
        return self.total_hpwl_weighted / (self.net_count * (self.cw + self.ch))

    def density_cost(self):
        flat = self.grid_occ.flatten() / self.cell_area
        occ = flat[flat != 0.0]
        if occ.size == 0:
            return 0.0
        n_total = flat.size
        if n_total < 10:
            return 0.5 * float(occ.mean())
        cnt = int(math.floor(n_total * 0.1))
        sd = np.sort(occ)[::-1]
        if cnt == 0 or cnt > sd.size:
            return 0.5 * float(sd.mean())
        return 0.5 * float(sd[:cnt].mean())

    def partial_proxy_no_cong(self):
        return self.hpwl_cost() + 0.5 * self.density_cost()

    def _footprint(self, mw, mh, mx, my):
        x_min = mx - mw * 0.5
        x_max = mx + mw * 0.5
        y_min = my - mh * 0.5
        y_max = my + mh * 0.5
        if x_max <= 0 or y_max <= 0 or x_min >= self.cw or y_min >= self.ch:
            return []
        bl_c = max(0, int(math.floor(x_min / self.cell_w)))
        ur_c = min(self.grid_col - 1, int(math.floor(x_max / self.cell_w)))
        bl_r = max(0, int(math.floor(y_min / self.cell_h)))
        ur_r = min(self.grid_row - 1, int(math.floor(y_max / self.cell_h)))
        out = []
        for r in range(bl_r, ur_r + 1):
            cy_min = r * self.cell_h
            cy_max = (r + 1) * self.cell_h
            oy = min(cy_max, y_max) - max(cy_min, y_min)
            if oy <= 0:
                continue
            for c in range(bl_c, ur_c + 1):
                cx_min = c * self.cell_w
                cx_max = (c + 1) * self.cell_w
                ox = min(cx_max, x_max) - max(cx_min, x_min)
                if ox <= 0:
                    continue
                out.append((r, c, ox * oy))
        return out

    def _add_density(self, m, mx, my):
        fp = self._footprint(self.sizes[m, 0], self.sizes[m, 1], mx, my)
        for r, c, a in fp:
            self.grid_occ[r, c] += a
        self.macro_footprint[m] = fp

    def try_move(self, m, new_x, new_y):
        assert self._pending is None, "previous move not finalized"
        old_x = float(self.pos_aug[m, 0])
        old_y = float(self.pos_aug[m, 1])
        old_fp = self.macro_footprint[m]
        # V18: remove this macro's net routes (and, if hard, its blockage)
        # from the persistent raw grids while the pin cache still holds the
        # OLD positions. The matching re-add happens after the pin updates.
        _nets_m = self.macro_to_nets[m]
        if _nets_m.size:
            _batch_route_nets(self._Vraw, self._Hraw,
                              self.grid_col, self.grid_row,
                              self._dense_pin_pos, self._dense_mask,
                              self._net_weights_arr, _nets_m, -1.0,
                              self.cell_w, self.cell_h)
        if m < self.n_hard:
            _macro_route(self._Vm, self._Hm, self.grid_col, self.grid_row,
                         old_x, old_y, self.sizes[m, 0], self.sizes[m, 1],
                         self.cell_w, self.cell_h,
                         -self._vrouting_alloc, -self._hrouting_alloc)
        affected = []
        delta = 0.0
        for (ni, pi) in self.macro_to_pins[m]:
            old_pin_xy = self.net_pin_xy[ni][pi].copy()
            old_bbox = self.net_bbox_total[ni]
            ox = self.net_pin_offsets[ni][pi, 0]
            oy = self.net_pin_offsets[ni][pi, 1]
            new_px = new_x + ox
            new_py = new_y + oy
            self.net_pin_xy[ni][pi, 0] = new_px
            self.net_pin_xy[ni][pi, 1] = new_py
            # V13-rev4: keep dense pin_pos cache in sync.
            self._dense_pin_pos[ni, pi, 0] = new_px
            self._dense_pin_pos[ni, pi, 1] = new_py
            xy = self.net_pin_xy[ni]
            new_bbox = (float(xy[:, 0].max() - xy[:, 0].min())
                        + float(xy[:, 1].max() - xy[:, 1].min()))
            self.net_bbox_total[ni] = new_bbox
            delta += self.net_weights[ni] * (new_bbox - old_bbox)
            affected.append((ni, pi, old_pin_xy, old_bbox))
        self.total_hpwl_weighted += delta
        if old_fp is not None:
            for r, c, a in old_fp:
                self.grid_occ[r, c] -= a
        new_fp = self._footprint(self.sizes[m, 0], self.sizes[m, 1], new_x, new_y)
        for r, c, a in new_fp:
            self.grid_occ[r, c] += a
        self.macro_footprint[m] = new_fp
        self.pos_aug[m, 0] = new_x
        self.pos_aug[m, 1] = new_y
        # V18: re-add net routes with the NEW pin positions + new blockage.
        if _nets_m.size:
            _batch_route_nets(self._Vraw, self._Hraw,
                              self.grid_col, self.grid_row,
                              self._dense_pin_pos, self._dense_mask,
                              self._net_weights_arr, _nets_m, 1.0,
                              self.cell_w, self.cell_h)
        if m < self.n_hard:
            _macro_route(self._Vm, self._Hm, self.grid_col, self.grid_row,
                         new_x, new_y, self.sizes[m, 0], self.sizes[m, 1],
                         self.cell_w, self.cell_h,
                         self._vrouting_alloc, self._hrouting_alloc)
        self._cong_ops += 1
        self._pending = {"m": m, "ox": old_x, "oy": old_y,
                         "affected": affected, "delta": delta, "old_fp": old_fp}

    def commit_move(self):
        self._pending = None
        # V18: wash out float drift from the subtract/add cycles with a
        # periodic full rebuild (~ cost of ONE pre-V18 eval, amortized ≈ 0).
        if self._cong_ops >= 16384:
            self._rebuild_cong()

    def revert_move(self):
        pm = self._pending
        if pm is None:
            return
        m = pm["m"]
        # V18: remove the routes added with the NEW pin positions (the pin
        # cache still holds them) and the blockage at the new position; the
        # matching re-add with the restored OLD state happens at the end.
        _nets_m = self.macro_to_nets[m]
        if _nets_m.size:
            _batch_route_nets(self._Vraw, self._Hraw,
                              self.grid_col, self.grid_row,
                              self._dense_pin_pos, self._dense_mask,
                              self._net_weights_arr, _nets_m, -1.0,
                              self.cell_w, self.cell_h)
        if m < self.n_hard:
            _macro_route(self._Vm, self._Hm, self.grid_col, self.grid_row,
                         float(self.pos_aug[m, 0]), float(self.pos_aug[m, 1]),
                         self.sizes[m, 0], self.sizes[m, 1],
                         self.cell_w, self.cell_h,
                         -self._vrouting_alloc, -self._hrouting_alloc)
        # Reverse order so for macros with 2+ pins on same net, the
        # earliest saved bbox (= the true pre-try value) is the final value.
        for (ni, pi, old_xy, old_bbox) in reversed(pm["affected"]):
            self.net_pin_xy[ni][pi, 0] = old_xy[0]
            self.net_pin_xy[ni][pi, 1] = old_xy[1]
            # V13-rev4: keep dense pin_pos cache in sync on rollback.
            self._dense_pin_pos[ni, pi, 0] = old_xy[0]
            self._dense_pin_pos[ni, pi, 1] = old_xy[1]
            self.net_bbox_total[ni] = old_bbox
        self.total_hpwl_weighted -= pm["delta"]
        new_fp = self.macro_footprint[m]
        for r, c, a in new_fp:
            self.grid_occ[r, c] -= a
        old_fp = pm["old_fp"]
        if old_fp is not None:
            for r, c, a in old_fp:
                self.grid_occ[r, c] += a
        self.macro_footprint[m] = old_fp
        self.pos_aug[m, 0] = pm["ox"]
        self.pos_aug[m, 1] = pm["oy"]
        # V18: re-add net routes with the restored OLD pin positions + the
        # blockage at the old position.
        if _nets_m.size:
            _batch_route_nets(self._Vraw, self._Hraw,
                              self.grid_col, self.grid_row,
                              self._dense_pin_pos, self._dense_mask,
                              self._net_weights_arr, _nets_m, 1.0,
                              self.cell_w, self.cell_h)
        if m < self.n_hard:
            _macro_route(self._Vm, self._Hm, self.grid_col, self.grid_row,
                         pm["ox"], pm["oy"],
                         self.sizes[m, 0], self.sizes[m, 1],
                         self.cell_w, self.cell_h,
                         self._vrouting_alloc, self._hrouting_alloc)
        self._cong_ops += 1
        if self._cong_ops >= 16384:
            self._rebuild_cong()
        self._pending = None

    # ── V13-rev4: fast full proxy using cached state ────────────────────
    def proxy_full_cached(self):
        """Full proxy (WL + 0.5*D + 0.5*C) using incremental state.

        Mirrors FastProxy.proxy_full's return shape: (total, wl, d_half,
        c_raw). The 0.5 factors are baked into d_half and into the
        congestion contribution inside `total`.

        Returns a tuple instead of building a dict — LAHC's inner loop
        calls this thousands of times so the dict overhead matters.

        V18: congestion now comes from the incrementally-maintained raw
        grids (O(grid cells) per eval) instead of rerouting all nets
        (O(n_nets), 28k-46k on HUGE benches — the old ~3-8ms/iter cost).
        """
        wl = self.hpwl_cost()
        d_half = self.density_cost()
        if self.num_nets_used == 0:
            return wl + d_half, wl, d_half, 0.0
        all_vh = _combine_cong_grids(
            self._Vraw, self._Hraw, self._Vm, self._Hm,
            self.grid_col, self.grid_row,
            self._grid_v_routes, self._grid_h_routes, self._smooth_range,
        )
        n_total = all_vh.size
        cnt = int(math.floor(n_total * 0.05))
        if cnt == 0:
            c_raw = float(all_vh.max())
        else:
            # Top-cnt mean via partial selection — no full sort needed.
            c_raw = float(np.partition(all_vh, n_total - cnt)[n_total - cnt:].mean())
        c_half = 0.5 * c_raw
        return wl + d_half + c_half, wl, d_half, c_raw

    def reset_to_positions(self, pos_macros: np.ndarray) -> None:
        """Restore state to a fresh macro-positions snapshot.

        Used by LAHC drift-reset: when the search has wandered too far from
        best, we snap the entire IncrementalProxy state back to best_pos
        in one bulk operation (O(n_nets + n_macros)) instead of millions
        of individual try_move calls.

        Assumes `pos_macros.shape == (n_macros, 2)`. Port positions are
        untouched.
        """
        self.pos_aug[:self.n_macros] = pos_macros
        # Rebuild WL state.
        self.total_hpwl_weighted = 0.0
        for ni in range(self.num_nets_used):
            owners = self.net_pin_owner[ni]
            offs = self.net_pin_offsets[ni]
            pin_xy = self.pos_aug[owners] + offs
            self.net_pin_xy[ni] = pin_xy
            P = int(pin_xy.shape[0])
            self._dense_pin_pos[ni, :P] = pin_xy
            bbox = (float(pin_xy[:, 0].max() - pin_xy[:, 0].min())
                    + float(pin_xy[:, 1].max() - pin_xy[:, 1].min()))
            self.net_bbox_total[ni] = bbox
            self.total_hpwl_weighted += self.net_weights[ni] * bbox
        # Rebuild density grid.
        self.grid_occ.fill(0.0)
        for m in range(self.n_macros):
            self._add_density(m, self.pos_aug[m, 0], self.pos_aug[m, 1])
        # V18: rebuild the raw congestion grids from the fresh snapshot
        # (also washes out any accumulated float drift).
        self._rebuild_cong()
        self._pending = None
