"""Pure numpy port of the TILOS PlacementCost proxy formula.

Goal: a fast, faithful re-implementation of the proxy that we can call
millions of times during real-proxy CD/LNS without the Python overhead
of the TILOS reference implementation.

Components:

  HPWL (wirelength_cost):
    For each net, compute pin-level bounding box span:
        bbox_hpwl = (max(px) - min(px)) + (max(py) - min(py))
    Sum across nets, then normalize by:
        net_count * (canvas_w + canvas_h)
    Pin position = owner_center + pin_offset (zero offset for soft macros
    and ports, real offset for hard macros).

  Density (density_cost):
    For each macro, compute exact rectangle-overlap area against every
    grid cell its bbox touches; accumulate into grid_occupied.
    grid_density = grid_occupied / (cell_w * cell_h).
    cost = 0.5 * mean(top 10% occupied cells).

  Congestion (congestion_cost):
    [DEFERRED — TILOS routing logic is complex (L/T routing, smoothing).
     For an initial CD POC we assume congestion is approximately stable
     under small moves and can be evaluated less frequently with TILOS.]
    For now we expose `compute_partial_proxy(...)` returning HPWL+Density,
    and `compute_full_proxy_from_components(...)` to combine with a
    TILOS-evaluated congestion term.

NumPy-only first; @njit decoration can be applied once correctness is
verified on ibm01.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from macro_place.benchmark import Benchmark


# -----------------------------------------------------------------------------
# Static structures we precompute once per benchmark.
# -----------------------------------------------------------------------------


class NumpyProxy:
    """Pre-extracted netlist tensors + canvas info, suitable for fast
    HPWL and density evaluation.

    Build once per benchmark; then call `hpwl(positions_aug)`,
    `density(positions_aug)`, `proxy_partial(positions_aug)` as needed.

    `positions_aug` is a [n_macros + n_ports, 2] numpy float64 array of
    macro centers (rows [0, n_macros)) and port positions (rows
    [n_macros, n_macros + n_ports)).
    """

    def __init__(self, benchmark: Benchmark, grid_col: int = None,
                 grid_row: int = None, plc=None,
                 max_pins_per_net: int = None):
        # max_pins_per_net: optional cap to bound the [N, max_pins, 2] tensor
        # that pin_pos = positions_aug[owner_idx] + offset_xy allocates every
        # congestion call. ibm14 has one 84-pin net that pads everything to
        # 521MB per call; capping at e.g. 24 keeps it at 144MB and gives a
        # huge speedup. The dropped pins on outlier nets contribute negligibly
        # to L-shape routing (they're typically clock/scan-chain).
        self._max_pins_per_net = max_pins_per_net
        # `plc`: optional TILOS PlacementCost — used to source per-net weights
        # (driver pin .get_weight()). Without it we assume weight=1 per net,
        # which is wrong when the netlist has weighted nets (some IBM ones do).
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.n_macros = int(benchmark.num_macros)
        self.n_hard = int(benchmark.num_hard_macros)
        self.n_ports = int(benchmark.port_positions.shape[0])

        # Macro sizes ([n_macros, 2]).
        self.sizes = benchmark.macro_sizes.numpy().astype(np.float64).copy()

        # Grid dims: TILOS uses set_placement_grid(col, row); benchmark stores
        # grid_cols and grid_rows.
        self.grid_col = int(grid_col if grid_col is not None
                            else benchmark.grid_cols)
        self.grid_row = int(grid_row if grid_row is not None
                            else benchmark.grid_rows)
        self.cell_w = self.cw / self.grid_col
        self.cell_h = self.ch / self.grid_row

        # Routing parameters (sourced from plc if provided, else from benchmark).
        if plc is not None:
            self.hroutes_per_micron = float(getattr(plc, "hroutes_per_micron",
                                                       benchmark.hroutes_per_micron))
            self.vroutes_per_micron = float(getattr(plc, "vroutes_per_micron",
                                                       benchmark.vroutes_per_micron))
            self.smooth_range = int(getattr(plc, "smooth_range", 2))
            self.hrouting_alloc = float(getattr(plc, "hrouting_alloc", 0.0))
            self.vrouting_alloc = float(getattr(plc, "vrouting_alloc", 0.0))
        else:
            self.hroutes_per_micron = float(benchmark.hroutes_per_micron)
            self.vroutes_per_micron = float(benchmark.vroutes_per_micron)
            self.smooth_range = 2
            self.hrouting_alloc = 0.0
            self.vrouting_alloc = 0.0
        # Per-cell routing capacity (TILOS: grid_v_routes = grid_width * vroutes_per_micron).
        self.grid_v_routes = self.cell_w * self.vroutes_per_micron
        self.grid_h_routes = self.cell_h * self.hroutes_per_micron

        # ---- Pin-level net structure (padded) -------------------------
        # We need owner_idx, offset_xy, mask, plus per-net weights (TILOS
        # weights nets by driver pin .get_weight()).
        nets_pin: list = []
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
            true_max_pins = max(int(n.shape[0]) for n in nets_pin)
            if (self._max_pins_per_net is not None
                and true_max_pins > self._max_pins_per_net):
                max_pins = int(self._max_pins_per_net)
            else:
                max_pins = true_max_pins
            self.num_nets_used = len(nets_pin)
            owner_idx = np.zeros((self.num_nets_used, max_pins),
                                  dtype=np.int64)
            offset_xy = np.zeros((self.num_nets_used, max_pins, 2),
                                  dtype=np.float64)
            mask = np.zeros((self.num_nets_used, max_pins), dtype=bool)
            pin_offsets = benchmark.macro_pin_offsets
            for i, net in enumerate(nets_pin):
                k = int(net.shape[0])
                k_capped = min(k, max_pins)
                owner_idx[i, :k_capped] = net[:k_capped, 0]
                for j in range(k_capped):
                    owner = int(net[j, 0])
                    pin_slot = int(net[j, 1])
                    if owner < self.n_hard and pin_slot < pin_offsets[owner].shape[0]:
                        offset_xy[i, j, 0] = float(pin_offsets[owner][pin_slot, 0])
                        offset_xy[i, j, 1] = float(pin_offsets[owner][pin_slot, 1])
                mask[i, :k_capped] = True
            self.owner_idx = owner_idx
            self.offset_xy = offset_xy
            self.mask = mask

            # Per-net weights. Source from plc if provided (driver pin's
            # weight). Loader's net order is `plc.nets.items()` (driver-keyed)
            # filtered by nodes_in_net non-empty + at least one valid pin.
            # We retrieve weights in the same order so they zip correctly.
            self.net_weights = np.ones(self.num_nets_used, dtype=np.float64)
            if plc is not None:
                # Re-walk plc.nets in the SAME order the loader did, applying
                # the same skip rule (only nets that produced pins_in_net
                # non-empty become an entry). Simplest: iterate plc.nets in
                # order; each one corresponds 1:1 with our nets_pin entry.
                drivers = list(plc.nets.keys())
                if len(drivers) == self.num_nets_used:
                    for i, driver in enumerate(drivers):
                        d_idx = plc.mod_name_to_indices.get(driver)
                        if d_idx is None:
                            continue
                        d_pin = plc.modules_w_pins[d_idx]
                        if hasattr(d_pin, "get_weight"):
                            self.net_weights[i] = float(d_pin.get_weight())
                else:
                    # Mismatch: the loader filtered some nets out (none with
                    # >= 2 valid pins). Fall back to weight=1; will be slightly
                    # off if those filtered nets had nontrivial weights.
                    pass

        # net_count for HPWL normalization. TILOS uses self.net_cnt which is
        # the SUM OF NET WEIGHTS (counts each net by its weight). Weighted
        # nets (weight > 1) inflate this past simple net count.
        # Use sum of our captured weights — matches plc.net_cnt exactly when
        # we sourced weights from plc.
        self.net_count = float(self.net_weights.sum()) if self.num_nets_used else 1.0
        if self.net_count <= 0:
            self.net_count = 1.0

    # ------------------------------------------------------------------
    # HPWL
    # ------------------------------------------------------------------
    def hpwl_total(self, positions_aug: np.ndarray) -> float:
        """Pin-level WEIGHTED total HPWL.

            sum_i [ weight_i * ((xmax-xmin) + (ymax-ymin)) ]

        positions_aug: [n_macros + n_ports, 2]
        """
        if self.num_nets_used == 0:
            return 0.0
        # pin_pos = positions_aug[owner_idx] + offset_xy   [N, P, 2]
        pin_pos = positions_aug[self.owner_idx] + self.offset_xy
        # Mask out invalid pin slots by replacing with +/- inf so they don't
        # contribute to max/min.
        NEG_INF = -1e30
        POS_INF = 1e30
        m = self.mask
        px_max = np.where(m, pin_pos[..., 0], NEG_INF)
        py_max = np.where(m, pin_pos[..., 1], NEG_INF)
        px_min = np.where(m, pin_pos[..., 0], POS_INF)
        py_min = np.where(m, pin_pos[..., 1], POS_INF)
        x_span = px_max.max(axis=1) - px_min.min(axis=1)
        y_span = py_max.max(axis=1) - py_min.min(axis=1)
        per_net = (x_span + y_span)
        return float((per_net * self.net_weights).sum())

    def hpwl_normalized(self, positions_aug: np.ndarray) -> float:
        """TILOS get_cost: total_hpwl / (net_cnt * (cw + ch))."""
        return self.hpwl_total(positions_aug) / (self.net_count * (self.cw + self.ch))

    # ------------------------------------------------------------------
    # Density
    # ------------------------------------------------------------------
    def density_grid(self, positions_macros: np.ndarray) -> np.ndarray:
        """Per-cell density (occupied area / cell area) using EXACT
        rectangle-rectangle overlap (no bilinear smoothing).

        positions_macros: [n_macros, 2] of (x, y) macro centers.
        Returns: [grid_row, grid_col] density grid.
        """
        gc = self.grid_col
        gr = self.grid_row
        cw_cell = self.cell_w
        ch_cell = self.cell_h
        cell_area = cw_cell * ch_cell
        cw, ch = self.cw, self.ch

        grid_occupied = np.zeros((gr, gc), dtype=np.float64)

        sizes = self.sizes
        for m in range(self.n_macros):
            mw = sizes[m, 0]
            mh = sizes[m, 1]
            mx = positions_macros[m, 0]
            my = positions_macros[m, 1]
            x_min = mx - mw * 0.5
            x_max = mx + mw * 0.5
            y_min = my - mh * 0.5
            y_max = my + mh * 0.5
            # Skip OOB macros (TILOS skips when ur_row/ur_col out of bound).
            if x_max <= 0 or y_max <= 0 or x_min >= cw or y_min >= ch:
                continue

            # Cell range that the bbox touches.
            bl_col = max(0, int(math.floor(x_min / cw_cell)))
            ur_col = min(gc - 1, int(math.floor(x_max / cw_cell)))
            bl_row = max(0, int(math.floor(y_min / ch_cell)))
            ur_row = min(gr - 1, int(math.floor(y_max / ch_cell)))

            for r in range(bl_row, ur_row + 1):
                cell_y_min = r * ch_cell
                cell_y_max = (r + 1) * ch_cell
                oy = min(cell_y_max, y_max) - max(cell_y_min, y_min)
                if oy <= 0:
                    continue
                for c in range(bl_col, ur_col + 1):
                    cell_x_min = c * cw_cell
                    cell_x_max = (c + 1) * cw_cell
                    ox = min(cell_x_max, x_max) - max(cell_x_min, x_min)
                    if ox <= 0:
                        continue
                    grid_occupied[r, c] += ox * oy

        return grid_occupied / cell_area

    def density_cost(self, positions_macros: np.ndarray) -> float:
        """0.5 * mean(top 10% occupied cells), matching TILOS."""
        grid = self.density_grid(positions_macros)
        flat = grid.flatten()
        occupied = flat[flat != 0.0]
        if occupied.size == 0:
            return 0.0
        n_total = grid.size
        density_cnt = int(math.floor(n_total * 0.1))
        if n_total < 10:
            return 0.5 * float(np.mean(occupied))
        sorted_desc = np.sort(occupied)[::-1]
        k = min(density_cnt, sorted_desc.size)
        if k == 0:
            return 0.0
        return 0.5 * float(np.mean(sorted_desc[:density_cnt])
                           if density_cnt <= sorted_desc.size
                           else np.mean(sorted_desc))

    # ------------------------------------------------------------------
    # Congestion (port of TILOS get_routing + abu)
    # ------------------------------------------------------------------
    def congestion_grids(self, positions_aug: np.ndarray):
        """Compute (V_routing_cong, H_routing_cong) full proxy after net
        routing + macro routing + smoothing.

        Returns: (V, H) each shape (grid_row, grid_col) numpy float64.
        """
        gc = self.grid_col
        gr = self.grid_row
        cw_cell = self.cell_w
        ch_cell = self.cell_h
        smooth_range = self.smooth_range
        vroutes_alloc = self.vrouting_alloc
        hroutes_alloc = self.hrouting_alloc
        grid_v_routes = self.grid_v_routes
        grid_h_routes = self.grid_h_routes

        V = np.zeros(gr * gc, dtype=np.float64)
        H = np.zeros(gr * gc, dtype=np.float64)
        Vm = np.zeros(gr * gc, dtype=np.float64)
        Hm = np.zeros(gr * gc, dtype=np.float64)

        # 1) Iterate nets and route them.
        # owner_idx: [N, P], offset_xy: [N, P, 2], mask: [N, P], net_weights: [N]
        if self.num_nets_used:
            owner = self.owner_idx
            mask = self.mask
            # Pin positions = positions_aug[owner] + offset_xy.
            pin_pos = positions_aug[owner] + self.offset_xy   # [N, P, 2]
            for net_i in range(self.num_nets_used):
                m = mask[net_i]
                if not m.any():
                    continue
                w = float(self.net_weights[net_i])
                xs = pin_pos[net_i, m, 0]
                ys = pin_pos[net_i, m, 1]
                # Convert to gcell coords.
                cols = np.floor(xs / cw_cell).astype(np.int64)
                rows = np.floor(ys / ch_cell).astype(np.int64)
                # Unique gcells.
                gcells = list({(int(rows[k]), int(cols[k]))
                               for k in range(rows.size)})

                # Source gcell = first pin's gcell (TILOS uses driver pin which
                # is the first entry in our owner_idx tensor — preserved by the
                # loader's `[driver] + sinks` order).
                source = (int(rows[0]), int(cols[0]))

                if len(gcells) == 2:
                    self._two_pin_route(V, H, gc, source, gcells, w)
                elif len(gcells) == 3:
                    self._three_pin_route(V, H, gc, gcells, w)
                elif len(gcells) > 3:
                    # split: each non-source pin pairs with source as a 2-pin net
                    for sink in gcells:
                        if sink != source:
                            self._two_pin_route(V, H, gc, source, [source, sink], w)

        # 2) Iterate hard macros and add macro routing cong (Vm, Hm).
        for mi in range(self.n_hard):
            mw = self.sizes[mi, 0]
            mh = self.sizes[mi, 1]
            mx = positions_aug[mi, 0]
            my = positions_aug[mi, 1]
            self._macro_route(Vm, Hm, gc, gr, mx, my, mw, mh,
                              cw_cell, ch_cell, vroutes_alloc, hroutes_alloc)

        # 3) Normalize by grid routing capacity.
        V /= grid_v_routes
        H /= grid_h_routes
        Vm /= grid_v_routes
        Hm /= grid_h_routes

        # 4) Smooth V and H (kernel sum over smooth_range cells along route axis).
        V = self._smooth_v(V.reshape(gr, gc), smooth_range).flatten()
        H = self._smooth_h(H.reshape(gr, gc), smooth_range).flatten()

        # 5) Add macro contributions.
        V += Vm
        H += Hm

        return V.reshape(gr, gc), H.reshape(gr, gc)

    @staticmethod
    def _two_pin_route(V, H, grid_col, source, gcells, weight):
        # gcells is list of 2 unique cells; sink = the one != source.
        sink = gcells[0] if gcells[0] != source else gcells[1]
        sy, sx = source
        ty, tx = sink
        col_min = min(sx, tx); col_max = max(sx, tx)
        row_min = min(sy, ty); row_max = max(sy, ty)
        # H routing on source row from col_min..col_max (excl)
        if col_max > col_min:
            base = sy * grid_col
            H[base + col_min : base + col_max] += weight
        # V routing on sink col from row_min..row_max (excl)
        if row_max > row_min:
            for r in range(row_min, row_max):
                V[r * grid_col + tx] += weight

    @staticmethod
    def _three_pin_route(V, H, grid_col, gcells, weight):
        # Sort by (col, row) ascending.
        sorted_g = sorted(gcells, key=lambda x: (x[1], x[0]))
        y1, x1 = sorted_g[0]
        y2, x2 = sorted_g[1]
        y3, x3 = sorted_g[2]
        if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
            # L routing
            NumpyProxy._l_routing(V, H, grid_col, sorted_g, weight)
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            for col in range(x1, x2):
                H[y1 * grid_col + col] += weight
            for row in range(y1, max(y2, y3)):
                V[row * grid_col + x2] += weight
        elif y2 == y3:
            for col in range(x1, x2):
                H[y1 * grid_col + col] += weight
            for col in range(x2, x3):
                H[y2 * grid_col + col] += weight
            for row in range(min(y2, y1), max(y2, y1)):
                V[row * grid_col + x2] += weight
        else:
            # T routing
            NumpyProxy._t_routing(V, H, grid_col, sorted_g, weight)

    @staticmethod
    def _l_routing(V, H, grid_col, sorted_g, weight):
        y1, x1 = sorted_g[0]
        y2, x2 = sorted_g[1]
        y3, x3 = sorted_g[2]
        for col in range(x1, x2):
            H[y1 * grid_col + col] += weight
        for col in range(x2, x3):
            H[y2 * grid_col + col] += weight
        for row in range(min(y1, y2), max(y1, y2)):
            V[row * grid_col + x2] += weight
        for row in range(min(y2, y3), max(y2, y3)):
            V[row * grid_col + x3] += weight

    @staticmethod
    def _t_routing(V, H, grid_col, gcells, weight):
        # T routing: TILOS sorts by (row, col) ascending.
        sorted_g = sorted(gcells)
        y1, x1 = sorted_g[0]
        y2, x2 = sorted_g[1]
        y3, x3 = sorted_g[2]
        xmin = min(x1, x2, x3)
        xmax = max(x1, x2, x3)
        for col in range(xmin, xmax):
            H[y2 * grid_col + col] += weight
        for row in range(min(y1, y2), max(y1, y2)):
            V[row * grid_col + x1] += weight
        for row in range(min(y2, y3), max(y2, y3)):
            V[row * grid_col + x3] += weight

    @staticmethod
    def _macro_route(Vm, Hm, gc, gr, mx, my, mw, mh,
                      cw_cell, ch_cell, vroutes_alloc, hroutes_alloc):
        # macro bbox.
        x_min = mx - mw * 0.5
        x_max = mx + mw * 0.5
        y_min = my - mh * 0.5
        y_max = my + mh * 0.5
        # Skip OOB.
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
                x_dist = min(cx_max, x_max) - max(cx_min, x_min)
                y_dist = min(cy_max, y_max) - max(cy_min, y_min)
                if x_dist <= 0 or y_dist <= 0:
                    continue
                if ur_row != bl_row:
                    if (r == bl_row and abs(y_dist - ch_cell) > 1e-5) or \
                       (r == ur_row and abs(y_dist - ch_cell) > 1e-5):
                        if_PARTIAL_VERT = True
                if ur_col != bl_col:
                    if (c == bl_col and abs(x_dist - cw_cell) > 1e-5) or \
                       (c == ur_col and abs(x_dist - cw_cell) > 1e-5):
                        if_PARTIAL_HOR = True
                Vm[r * gc + c] += x_dist * vroutes_alloc
                Hm[r * gc + c] += y_dist * hroutes_alloc
        # PARTIAL adjustments — TILOS subtracts the contribution of the
        # final row (vert) and final col (hor).
        if if_PARTIAL_VERT:
            r = ur_row
            cy_min = r * ch_cell; cy_max = (r + 1) * ch_cell
            for c in range(bl_col, ur_col + 1):
                cx_min = c * cw_cell; cx_max = (c + 1) * cw_cell
                x_dist = min(cx_max, x_max) - max(cx_min, x_min)
                Vm[r * gc + c] -= x_dist * vroutes_alloc
        if if_PARTIAL_HOR:
            c = ur_col
            cx_min = c * cw_cell; cx_max = (c + 1) * cw_cell
            for r in range(bl_row, ur_row + 1):
                cy_min = r * ch_cell; cy_max = (r + 1) * ch_cell
                y_dist = min(cy_max, y_max) - max(cy_min, y_min)
                Hm[r * gc + c] -= y_dist * hroutes_alloc

    @staticmethod
    def _smooth_v(V_grid, smooth_range):
        # TILOS V smoothing: each cell's value is averaged across [col-r, col+r]
        # of the SAME row. Implemented by `temp[ptr] += val/gcell_cnt` for ptr
        # in [lp, rp] for each (row, col).
        gr, gc = V_grid.shape
        out = np.zeros_like(V_grid)
        for row in range(gr):
            for col in range(gc):
                lp = max(0, col - smooth_range)
                rp = min(gc - 1, col + smooth_range)
                gcell_cnt = rp - lp + 1
                val = V_grid[row, col] / gcell_cnt
                out[row, lp : rp + 1] += val
        return out

    @staticmethod
    def _smooth_h(H_grid, smooth_range):
        # TILOS H smoothing: each cell's value is averaged across [row-r, row+r]
        # of the SAME col.
        gr, gc = H_grid.shape
        out = np.zeros_like(H_grid)
        for row in range(gr):
            for col in range(gc):
                lp = max(0, row - smooth_range)
                up = min(gr - 1, row + smooth_range)
                gcell_cnt = up - lp + 1
                val = H_grid[row, col] / gcell_cnt
                out[lp : up + 1, col] += val
        return out

    def congestion_cost(self, positions_aug: np.ndarray) -> float:
        """abu(V + H, 0.05) — top 5% mean of V + H congestion across all
        (cell, direction) pairs.

        Uses the Numba-jitted backend by default (~30x faster). Set
        `use_numba=False` on the instance to fall back to pure numpy
        (used for testing).
        """
        if getattr(self, "use_numba", True):
            try:
                from proxy_numba import congestion_cost_numba
                return congestion_cost_numba(self, positions_aug)
            except Exception:
                # Fall back to numpy if numba isn't usable.
                pass
        V, H = self.congestion_grids(positions_aug)
        all_vh = np.concatenate([V.flatten(), H.flatten()])
        n_total = all_vh.size
        cnt = int(math.floor(n_total * 0.05))
        if cnt == 0:
            return float(all_vh.max())
        sorted_desc = np.sort(all_vh)[::-1]
        return float(np.mean(sorted_desc[:cnt]))

    # ------------------------------------------------------------------
    # Partial proxy (HPWL + density only — congestion deferred)
    # ------------------------------------------------------------------
    def proxy_partial(self, positions_aug: np.ndarray) -> dict:
        """Compute (wl_cost, density_cost) and partial proxy with the same
        coefficients as TILOS (1.0 * WL + 0.5 * Density).

        Returns dict {wirelength_cost, density_cost, partial_proxy}.
        """
        wl = self.hpwl_normalized(positions_aug)
        d = self.density_cost(positions_aug[:self.n_macros])
        return {
            "wirelength_cost": wl,
            "density_cost": d,
            # Note: 'partial_proxy' DROPS congestion. Real proxy is
            # WL + 0.5 D + 0.5 C. Use this when you have a fixed C estimate.
            "partial_proxy_no_cong": wl + 0.5 * d,
        }


# -----------------------------------------------------------------------------
# Quick standalone test against TILOS on ibm01.
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import os, sys, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from v6_legalize import V6LegalizePlacer
    import torch

    bench = "ibm01"
    iccad = f"external/MacroPlacement/Testcases/ICCAD04/{bench}"
    print(f"=== proxy_numpy validation on {bench} ===", flush=True)
    b, plc = load_benchmark_from_dir(iccad)
    print(f"  hard={b.num_hard_macros}  total={b.num_macros}  nets={b.num_nets}  "
          f"canvas={b.canvas_width:.1f}x{b.canvas_height:.1f}  "
          f"grid={b.grid_cols}x{b.grid_rows}", flush=True)

    # Use V6 legalize as the test placement.
    placement = V6LegalizePlacer().place(b)

    # TILOS reference.
    t0 = time.time()
    real = compute_proxy_cost(placement, b, plc)
    t_tilos = time.time() - t0
    print(f"\n  TILOS proxy: {real['proxy_cost']:.6f}  "
          f"WL={real['wirelength_cost']:.6f}  "
          f"D={real['density_cost']:.6f}  "
          f"C={real['congestion_cost']:.6f}  "
          f"t={t_tilos*1000:.1f}ms", flush=True)

    # Numpy port.
    fp = NumpyProxy(b, plc=plc)
    print(f"  numpy net_count={fp.net_count} (TILOS net_cnt={plc.net_cnt})",
          flush=True)
    nontrivial_w = int((fp.net_weights > 1.0).sum())
    print(f"  numpy nets with weight>1: {nontrivial_w}", flush=True)
    n_macros = b.num_macros
    n_ports = b.port_positions.shape[0]
    pos_aug = np.empty((n_macros + n_ports, 2), dtype=np.float64)
    pos_aug[:n_macros] = placement.numpy().astype(np.float64)[:n_macros]
    if n_ports:
        pos_aug[n_macros:] = b.port_positions.numpy().astype(np.float64)

    # WL + Density.
    t0 = time.time()
    parts = fp.proxy_partial(pos_aug)
    t_part = time.time() - t0
    print(f"  numpy WL+D : partial={parts['partial_proxy_no_cong']:.6f}  "
          f"WL={parts['wirelength_cost']:.6f}  "
          f"D={parts['density_cost']:.6f}  "
          f"t={t_part*1000:.1f}ms", flush=True)

    # Congestion.
    t0 = time.time()
    cong = fp.congestion_cost(pos_aug)
    t_cong = time.time() - t0
    print(f"  numpy C    : {cong:.6f}  t={t_cong*1000:.1f}ms", flush=True)

    # Full numpy proxy.
    full_proxy = parts['wirelength_cost'] + 0.5 * parts['density_cost'] + 0.5 * cong
    print(f"  numpy full proxy: {full_proxy:.6f}", flush=True)
    print(f"  TILOS full proxy: {real['proxy_cost']:.6f}", flush=True)

    # Diffs.
    wl_err = abs(parts['wirelength_cost'] - real['wirelength_cost'])
    wl_pct = 100.0 * wl_err / max(abs(real['wirelength_cost']), 1e-9)
    d_err = abs(parts['density_cost'] - real['density_cost'])
    d_pct = 100.0 * d_err / max(abs(real['density_cost']), 1e-9)
    c_err = abs(cong - real['congestion_cost'])
    c_pct = 100.0 * c_err / max(abs(real['congestion_cost']), 1e-9)
    full_err = abs(full_proxy - real['proxy_cost'])
    full_pct = 100.0 * full_err / max(abs(real['proxy_cost']), 1e-9)
    print(f"\n  WL    diff abs={wl_err:.2e}  rel={wl_pct:.4f}%", flush=True)
    print(f"  D     diff abs={d_err:.2e}  rel={d_pct:.4f}%", flush=True)
    print(f"  C     diff abs={c_err:.2e}  rel={c_pct:.4f}%", flush=True)
    print(f"  PROXY diff abs={full_err:.2e}  rel={full_pct:.4f}%", flush=True)
    t_np_total = t_part + t_cong
    print(f"\n  speedup (TILOS / numpy_total): {t_tilos / max(t_np_total, 1e-9):.1f}x",
          flush=True)
