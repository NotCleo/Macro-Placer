"""Smooth differentiable analog of the TILOS PlacementCost proxy.

The TILOS proxy = 1.0 * WL + 0.5 * Density + 0.5 * Congestion is
piecewise-constant in macro coordinates (every variable runs through
floor(...) of pin/macro position). Gradient methods cannot attack it
directly. This module provides a SMOOTH approximation whose gradient
points in the same descent direction as the TILOS proxy, suitable as
a loss for v14's analytical placer.

Components:

1. smooth_wl(positions_aug, owner_idx, offset_xy, mask, gamma_wl,
              net_weights, net_count, cw, ch) -> scalar
   - WA-HPWL via LSE with per-net weights, normalized by
     net_count * (cw + ch). Exact as gamma_wl -> infinity.

2. smooth_density_grid(macros_pos, sizes, G_w, G_h, cw, ch) -> [G_h, G_w]
   - Per-cell area-overlap density (occupancy ratio). EXACT match to
     TILOS density grid (smooth in macro position via relu-clipped
     overlap dimensions).

3. smooth_density_topk(grid, p_density) -> scalar
   - Power-mean of grid cells: (mean(grid^p))^(1/p). Approximates
     top-K mean as p -> infinity; smooth gradient flows through all
     cells with weight ~ cell^p.

4. smooth_net_routing(positions_aug, ..., G_w, G_h, cw, ch,
                       hroutes_per_micron, vroutes_per_micron,
                       gamma_bbox, softness_factor)
                       -> (V_net [G_h, G_w], H_net [G_h, G_w])
   - For each net: bbox via WA-HPWL LSE (gamma_bbox), demand
     pin_count / bbox_area * net_weight uniformly distributed inside
     bbox via sigmoid-smoothed indicator, split equally between V and
     H, normalized by grid_v_routes / grid_h_routes.

5. smooth_macro_blockage(positions_hard, sizes_hard, G_w, G_h, cw, ch,
                          hrouting_alloc, vrouting_alloc,
                          hroutes_per_micron, vroutes_per_micron)
                          -> (V_macro [G_h, G_w], H_macro [G_h, G_w])
   - For each hard macro and cell, compute area-overlap dimensions
     (x_dist, y_dist), V demand = x_dist * (y_dist / cell_h) *
     vrouting_alloc per cell, similarly H. Normalized by
     grid_v_routes / grid_h_routes. Exact match to TILOS macro
     routing (modulo the partial-overlap subtraction quirk, which
     has negligible magnitude).

6. smooth_5tap(grid, smooth_range, axis) -> [G_h, G_w]
   - 1D rectangular average filter along the routing axis.
     Matches TILOS's __smooth_routing_cong exactly.

7. smooth_congestion(positions_aug, sizes, owner_idx, offset_xy, mask,
                      net_weights, G_w, G_h, cw, ch,
                      hroutes_per_micron, vroutes_per_micron,
                      hrouting_alloc, vrouting_alloc, n_hard,
                      smooth_range, p_cong) -> scalar
   - Combine: 5tap(net_routing) + macro_blockage, then power-mean of
     top-K via combined V+H concatenation.

8. smooth_proxy_components(positions_aug, ...) ->
       (wl, density_cost, congestion_cost) tensor scalars
   - Returns the three components separately so caller can apply
     custom weighting or annealing.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# WL
# ---------------------------------------------------------------------------


def smooth_wl(
    positions_aug: torch.Tensor,    # [num_macros + num_ports, 2]
    owner_idx: torch.Tensor,        # [num_nets, max_pins] long
    offset_xy: torch.Tensor,        # [num_nets, max_pins, 2]
    mask: torch.Tensor,             # [num_nets, max_pins] bool
    gamma_wl: float,
    net_weights: torch.Tensor,      # [num_nets]
    net_count: float,               # sum of weights (TILOS net_cnt)
    cw: float, ch: float,
) -> torch.Tensor:
    """Pin-level WA-HPWL with per-net weights. Returns scalar matching
    TILOS get_cost = sum_w * (xspan + yspan) / (net_count * (cw + ch))."""
    pin_pos = positions_aug[owner_idx] + offset_xy   # [N, P, 2]
    NEG_INF = -1e20
    POS_INF = 1e20
    px = torch.where(mask, pin_pos[..., 0],
                       pin_pos[..., 0].new_full((), NEG_INF))
    py = torch.where(mask, pin_pos[..., 1],
                       pin_pos[..., 1].new_full((), NEG_INF))
    pxn = torch.where(mask, pin_pos[..., 0],
                        pin_pos[..., 0].new_full((), POS_INF))
    pyn = torch.where(mask, pin_pos[..., 1],
                        pin_pos[..., 1].new_full((), POS_INF))
    x_max = torch.logsumexp(gamma_wl * px, dim=1) / gamma_wl
    x_min = -torch.logsumexp(-gamma_wl * pxn, dim=1) / gamma_wl
    y_max = torch.logsumexp(gamma_wl * py, dim=1) / gamma_wl
    y_min = -torch.logsumexp(-gamma_wl * pyn, dim=1) / gamma_wl
    per_net = (x_max - x_min) + (y_max - y_min)
    return (per_net * net_weights).sum() / (net_count * (cw + ch))


# ---------------------------------------------------------------------------
# Density grid (exact match to TILOS density grid; smooth in positions)
# ---------------------------------------------------------------------------


def smooth_density_grid(
    macros_pos: torch.Tensor,       # [N, 2]
    sizes: torch.Tensor,            # [N, 2]
    G_w: int, G_h: int,
    cw: float, ch: float,
) -> torch.Tensor:
    """Per-cell density (area_overlap / cell_area). Smooth in macros_pos
    via relu-clipped exact area overlap. Matches TILOS get_grid_cells_density
    exactly."""
    cell_w = cw / G_w
    cell_h = ch / G_h
    half_w = sizes[:, 0:1] / 2.0
    half_h = sizes[:, 1:2] / 2.0
    half_W = cell_w / 2.0
    half_H = cell_h / 2.0
    cells_x = (torch.arange(G_w, device=macros_pos.device,
                              dtype=macros_pos.dtype) + 0.5) * cell_w
    cells_y = (torch.arange(G_h, device=macros_pos.device,
                              dtype=macros_pos.dtype) + 0.5) * cell_h
    m_x_lo = macros_pos[:, 0:1] - half_w
    m_x_hi = macros_pos[:, 0:1] + half_w
    c_x_lo = cells_x[None, :] - half_W
    c_x_hi = cells_x[None, :] + half_W
    ox = torch.relu(torch.minimum(m_x_hi, c_x_hi)
                     - torch.maximum(m_x_lo, c_x_lo))   # [N, G_w]
    m_y_lo = macros_pos[:, 1:2] - half_h
    m_y_hi = macros_pos[:, 1:2] + half_h
    c_y_lo = cells_y[None, :] - half_H
    c_y_hi = cells_y[None, :] + half_H
    oy = torch.relu(torch.minimum(m_y_hi, c_y_hi)
                     - torch.maximum(m_y_lo, c_y_lo))   # [N, G_h]
    grid_area_overlap = torch.einsum("ni,nj->ij", oy, ox)   # [G_h, G_w]
    return grid_area_overlap / (cell_w * cell_h)


# ---------------------------------------------------------------------------
# Density top-K (smooth via power-mean)
# ---------------------------------------------------------------------------


def electrostatic_density_loss(rho: torch.Tensor) -> torch.Tensor:
    """DREAMPlace/ePlace-style electrostatic potential energy of a 2D
    density grid. Solves Poisson's equation in Fourier space with periodic
    boundary conditions:

        rho_centered = rho - mean(rho)
        rho_fft  = FFT2(rho_centered)
        phi_fft  = rho_fft / |k|^2     (zero at DC)
        phi      = IFFT2(phi_fft).real
        energy   = sum(rho_centered * phi)

    Energy is non-negative, minimized to zero when rho is uniform. Acts as
    a spreading prior: like-charges (high-density regions) repel each other
    via the electrostatic field. Used in DREAMPlace as the primary density
    objective; here we add it as an OPTIONAL extra term alongside the TILOS-
    aligned smooth_density_topk."""
    import math as _math
    G_h, G_w = rho.shape
    rho_c = rho - rho.mean()
    rho_fft = torch.fft.fft2(rho_c)
    kx = torch.fft.fftfreq(G_w, d=1.0, device=rho.device).to(rho.dtype)
    ky = torch.fft.fftfreq(G_h, d=1.0, device=rho.device).to(rho.dtype)
    Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
    K2 = (2 * _math.pi) ** 2 * (Kx ** 2 + Ky ** 2)
    K2_safe = K2.clone()
    K2_safe[0, 0] = 1.0
    phi_fft = rho_fft / K2_safe
    phi_fft[0, 0] = 0
    phi = torch.fft.ifft2(phi_fft).real
    return (rho_c * phi).sum()


def smooth_density_topk(
    grid: torch.Tensor,             # [G_h, G_w]
    p_density: float = 10.0,
) -> torch.Tensor:
    """Power-mean of all cells = (mean(rho^p))^(1/p). Approximates
    top-K mean of rho values; smooth gradient flows everywhere (~weight
    proportional to cell^p)."""
    flat = grid.flatten()
    # Add small epsilon to avoid 0^p indeterminate gradient
    flat_clamped = flat.clamp(min=1e-12)
    return (flat_clamped.pow(p_density).mean()).pow(1.0 / p_density)


# ---------------------------------------------------------------------------
# Net routing (smooth bbox-RUDY)
# ---------------------------------------------------------------------------


def smooth_net_routing(
    positions_aug: torch.Tensor,
    owner_idx: torch.Tensor,
    offset_xy: torch.Tensor,
    mask: torch.Tensor,
    net_weights: torch.Tensor,
    G_w: int, G_h: int,
    cw: float, ch: float,
    hroutes_per_micron: float,
    vroutes_per_micron: float,
    gamma_bbox: float = 8.0,
    softness_factor: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RUDY net routing demand with sigmoid-smoothed bbox indicator.
    Returns (V_net, H_net) each [G_h, G_w] in proxy-routing units
    (demand / capacity).

    Note: TILOS uses L/T-shape Steiner skeleton, NOT bbox-uniform RUDY.
    RUDY is a structural approximation that overestimates demand
    spread per net but is smooth-differentiable and tractable. The
    gradient direction is qualitatively correct (push macros to reduce
    bbox span -> reduce routing demand).
    """
    pin_pos = positions_aug[owner_idx] + offset_xy
    NEG_INF = -1e20; POS_INF = 1e20
    px = torch.where(mask, pin_pos[..., 0],
                      pin_pos[..., 0].new_full((), NEG_INF))
    py = torch.where(mask, pin_pos[..., 1],
                      pin_pos[..., 1].new_full((), NEG_INF))
    pxn = torch.where(mask, pin_pos[..., 0],
                       pin_pos[..., 0].new_full((), POS_INF))
    pyn = torch.where(mask, pin_pos[..., 1],
                       pin_pos[..., 1].new_full((), POS_INF))
    x_max = torch.logsumexp(gamma_bbox * px, dim=1) / gamma_bbox
    x_min = -torch.logsumexp(-gamma_bbox * pxn, dim=1) / gamma_bbox
    y_max = torch.logsumexp(gamma_bbox * py, dim=1) / gamma_bbox
    y_min = -torch.logsumexp(-gamma_bbox * pyn, dim=1) / gamma_bbox

    # Pin count per net (clamped >= 2 since we filtered nets to >=2 pins)
    pin_count = mask.sum(dim=1).to(positions_aug.dtype).clamp(min=2.0)

    # bbox area
    bbox_w = (x_max - x_min).clamp(min=1e-3)
    bbox_h = (y_max - y_min).clamp(min=1e-3)

    # RUDY demand per unit area inside the bbox = weight * pin_count / bbox_area.
    # When integrated over the bbox area (sum over cells), total demand =
    # weight * pin_count.
    demand_per_area = net_weights * pin_count / (bbox_w * bbox_h)  # [N]

    # Sigmoid soft indicator for "cell inside bbox".
    cell_w = cw / G_w
    cell_h = ch / G_h
    softness = max(cell_w, cell_h) * softness_factor
    cells_x = (torch.arange(G_w, device=positions_aug.device,
                              dtype=positions_aug.dtype) + 0.5) * cell_w
    cells_y = (torch.arange(G_h, device=positions_aug.device,
                              dtype=positions_aug.dtype) + 0.5) * cell_h

    in_x = torch.sigmoid((cells_x - x_min[:, None]) / softness) * \
           torch.sigmoid((x_max[:, None] - cells_x) / softness)   # [N, G_w]
    in_y = torch.sigmoid((cells_y - y_min[:, None]) / softness) * \
           torch.sigmoid((y_max[:, None] - cells_y) / softness)   # [N, G_h]

    # Demand per cell (μm⁻²) = sum over nets (in_y * in_x * demand_per_area).
    weighted_y = in_y * demand_per_area[:, None]
    rudy_grid = torch.einsum("ni,nj->ij", weighted_y, in_x)   # [G_h, G_w]

    # rudy_grid is demand-per-unit-area. Multiply by cell_area to get demand
    # per cell.
    rudy_grid_cell = rudy_grid * (cell_w * cell_h)

    # TILOS V_routing_cong is normalized by grid_v_routes = cell_w * vroutes_per_micron.
    # H by grid_h_routes = cell_h * hroutes_per_micron.
    # We split the RUDY demand equally between V and H since RUDY doesn't
    # distinguish.
    grid_v_routes = cell_w * vroutes_per_micron
    grid_h_routes = cell_h * hroutes_per_micron
    V_net = (rudy_grid_cell * 0.5) / max(grid_v_routes, 1e-9)
    H_net = (rudy_grid_cell * 0.5) / max(grid_h_routes, 1e-9)
    return V_net, H_net


# ---------------------------------------------------------------------------
# L-shape routing (smooth analog of TILOS's __two_pin_net_routing applied
# star-decomposed to multi-pin nets)
# ---------------------------------------------------------------------------


def smooth_l_route(
    positions_aug: torch.Tensor,
    owner_idx: torch.Tensor,
    offset_xy: torch.Tensor,
    mask: torch.Tensor,
    net_weights: torch.Tensor,
    G_w: int, G_h: int,
    cw: float, ch: float,
    hroutes_per_micron: float,
    vroutes_per_micron: float,
    softness_factor: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth L-shape routing demand. Each net is star-decomposed:
    pin 0 is the source (driver), pins 1..P-1 are sinks. Each (src, sink)
    pair contributes:
      H demand at row y_src on cols from min(x_src, x_sink) to max(...)
      V demand at col x_sink on rows from min(y_src, y_sink) to max(...)

    All indicators sigmoid-smoothed for differentiability. Net weight
    multiplies all per-pair contributions for a net.

    Returns (V_route, H_route) [G_h, G_w] in proxy units (demand /
    capacity per cell).
    """
    pin_pos = positions_aug[owner_idx] + offset_xy   # [N, P, 2]
    N, P, _ = pin_pos.shape
    if P < 2:
        return (torch.zeros(G_h, G_w, dtype=positions_aug.dtype,
                              device=positions_aug.device),
                torch.zeros(G_h, G_w, dtype=positions_aug.dtype,
                              device=positions_aug.device))

    cell_w = cw / G_w
    cell_h = ch / G_h
    softness = max(cell_w, cell_h) * softness_factor

    cells_x = (torch.arange(G_w, device=positions_aug.device,
                              dtype=positions_aug.dtype) + 0.5) * cell_w
    cells_y = (torch.arange(G_h, device=positions_aug.device,
                              dtype=positions_aug.dtype) + 0.5) * cell_h

    src_x = pin_pos[:, 0, 0]   # [N]
    src_y = pin_pos[:, 0, 1]

    # ---- 3-pin Steiner H-row correction --------------------------------------
    # TILOS routes 3-pin nets via T-Steiner: H demand goes at MEDIAN y row, not
    # driver y row. For 2-pin and >3-pin nets, current star routing (H at
    # driver row) matches TILOS exactly. Only 3-pin diverges.
    #
    # For 3-pin nets, median_y = sum(ys) - max(ys) - min(ys). Differentiable
    # via min/max (gradient flows to whichever pin is the extremum). For other
    # nets, this expression is meaningless but we use is_3pin mask to apply it
    # only where it matters.
    pin_y_all = pin_pos[:, :, 1]                        # [N, P]
    mask_f = mask.to(positions_aug.dtype)
    pin_count = mask_f.sum(dim=1)                       # [N]
    is_3pin = ((pin_count > 2.5) & (pin_count < 3.5)).to(positions_aug.dtype)
    # max/min over valid pins only (mask invalid to ±inf-ish).
    big = float(ch * 100.0)
    y_for_max = torch.where(mask, pin_y_all, pin_y_all.new_full((), -big))
    y_for_min = torch.where(mask, pin_y_all, pin_y_all.new_full((), big))
    sum_y = (pin_y_all * mask_f).sum(dim=1)
    max_y = y_for_max.max(dim=1).values
    min_y = y_for_min.min(dim=1).values
    median_y_3pin = sum_y - max_y - min_y               # [N], only valid for is_3pin
    # peak_y: median_y for 3-pin nets, src_y otherwise.
    peak_y = is_3pin * median_y_3pin + (1.0 - is_3pin) * src_y

    # peak_row[N, G_h] = soft indicator that cell row equals peak_y row.
    # (sigmoid edges of width "1 cell ± softness".)
    peak_row = torch.sigmoid((cells_y[None, :]
                                - (peak_y[:, None] - cell_h * 0.5)) / softness) * \
               torch.sigmoid(((peak_y[:, None] + cell_h * 0.5)
                                - cells_y[None, :]) / softness)

    # Sink data: [N, P-1].
    sink_x = pin_pos[:, 1:, 0]
    sink_y = pin_pos[:, 1:, 1]
    valid_sink = mask[:, 1:].to(positions_aug.dtype)   # [N, P-1] in {0, 1}

    x_min_p = torch.minimum(src_x[:, None], sink_x)   # [N, P-1]
    x_max_p = torch.maximum(src_x[:, None], sink_x)
    y_min_p = torch.minimum(src_y[:, None], sink_y)
    y_max_p = torch.maximum(src_y[:, None], sink_y)

    # ----- H demand -----
    # between_cols[N, P-1, G_w] = soft indicator(x_min <= cell_x <= x_max)
    between_cols = torch.sigmoid((cells_x[None, None, :]
                                     - x_min_p[:, :, None]) / softness) * \
                   torch.sigmoid((x_max_p[:, :, None]
                                     - cells_x[None, None, :]) / softness)
    # Mask invalid pairs and multiply weights.
    bc_valid = between_cols * valid_sink[:, :, None]
    # Sum over sinks (star decomposition, correct for 2-pin and >3-pin):
    # [N, G_w] = sum_k bc_valid[n, k, w] * weight[n]
    bc_summed_star = bc_valid.sum(dim=1) * net_weights[:, None]   # [N, G_w]

    # 3-pin Steiner H profile: single line from net's xmin to xmax (no double-
    # counting from star). For 3-pin nets we replace bc_summed_star with this
    # single-line profile to match TILOS T-routing exactly. For other nets we
    # keep the star sum.
    x_for_max = torch.where(mask, pin_pos[:, :, 0],
                              pin_pos[:, :, 0].new_full((), -big))
    x_for_min = torch.where(mask, pin_pos[:, :, 0],
                              pin_pos[:, :, 0].new_full((), big))
    xmax_net = x_for_max.max(dim=1).values     # [N]
    xmin_net = x_for_min.min(dim=1).values     # [N]
    between_xspan = torch.sigmoid((cells_x[None, :] - xmin_net[:, None]) / softness) * \
                    torch.sigmoid((xmax_net[:, None] - cells_x[None, :]) / softness)
    bc_summed_3pin = between_xspan * net_weights[:, None]   # [N, G_w]
    bc_summed = (is_3pin[:, None] * bc_summed_3pin
                 + (1.0 - is_3pin)[:, None] * bc_summed_star)

    # H[h, w] = sum_n peak_row[n, h] * bc_summed[n, w]
    H_route = torch.einsum("nh,nw->hw", peak_row, bc_summed)

    # ----- V demand -----
    # peak_col[N, P-1, G_w] = soft indicator(cell_x == sink_x)
    peak_col = torch.sigmoid((cells_x[None, None, :]
                                - (sink_x[:, :, None] - cell_w * 0.5)) / softness) * \
               torch.sigmoid(((sink_x[:, :, None] + cell_w * 0.5)
                                - cells_x[None, None, :]) / softness)
    # between_rows[N, P-1, G_h] = soft indicator(y_min <= cell_y <= y_max)
    between_rows = torch.sigmoid((cells_y[None, None, :]
                                     - y_min_p[:, :, None]) / softness) * \
                   torch.sigmoid((y_max_p[:, :, None]
                                     - cells_y[None, None, :]) / softness)
    # Multiply by net weights and valid mask.
    br_w = between_rows * valid_sink[:, :, None] * net_weights[:, None, None]
    # V[h, w] = sum_{n, k} br_w[n, k, h] * peak_col[n, k, w]
    V_route = torch.einsum("nkh,nkw->hw", br_w, peak_col)

    # Normalize by routing capacity.
    grid_v_routes = cell_w * vroutes_per_micron
    grid_h_routes = cell_h * hroutes_per_micron
    V_route = V_route / max(grid_v_routes, 1e-9)
    H_route = H_route / max(grid_h_routes, 1e-9)
    return V_route, H_route


# ---------------------------------------------------------------------------
# Macro blockage
# ---------------------------------------------------------------------------


def smooth_macro_blockage(
    positions_hard: torch.Tensor,   # [n_hard, 2]
    sizes_hard: torch.Tensor,       # [n_hard, 2]
    G_w: int, G_h: int,
    cw: float, ch: float,
    hrouting_alloc: float,
    vrouting_alloc: float,
    hroutes_per_micron: float,
    vroutes_per_micron: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-cell V/H macro routing demand. Matches TILOS macro routing
    (modulo partial-overlap subtraction quirk).

    For each (macro, cell):
      x_dist = horizontal extent of overlap (in [0, cell_w])
      y_dist = vertical extent of overlap   (in [0, cell_h])
      V demand += x_dist * (y_dist / cell_h) * vrouting_alloc
        (the y_dist/cell_h factor approximates "macro overlaps row r"
         indicator: 1 when fully covering, 0 when not at all, linear in
         partial overlap)
      H demand += y_dist * (x_dist / cell_w) * hrouting_alloc
    Normalized by grid_v_routes / grid_h_routes.
    """
    cell_w = cw / G_w
    cell_h = ch / G_h
    half_w = sizes_hard[:, 0:1] / 2.0
    half_h = sizes_hard[:, 1:2] / 2.0
    half_W = cell_w / 2.0
    half_H = cell_h / 2.0
    cells_x = (torch.arange(G_w, device=positions_hard.device,
                              dtype=positions_hard.dtype) + 0.5) * cell_w
    cells_y = (torch.arange(G_h, device=positions_hard.device,
                              dtype=positions_hard.dtype) + 0.5) * cell_h
    m_x_lo = positions_hard[:, 0:1] - half_w
    m_x_hi = positions_hard[:, 0:1] + half_w
    c_x_lo = cells_x[None, :] - half_W
    c_x_hi = cells_x[None, :] + half_W
    x_dist = torch.relu(torch.minimum(m_x_hi, c_x_hi)
                          - torch.maximum(m_x_lo, c_x_lo))   # [N, G_w]
    m_y_lo = positions_hard[:, 1:2] - half_h
    m_y_hi = positions_hard[:, 1:2] + half_h
    c_y_lo = cells_y[None, :] - half_H
    c_y_hi = cells_y[None, :] + half_H
    y_dist = torch.relu(torch.minimum(m_y_hi, c_y_hi)
                          - torch.maximum(m_y_lo, c_y_lo))   # [N, G_h]

    # Per-cell V demand = sum_macros x_dist[m, c] * (y_dist[m, r] / cell_h)
    # = einsum on (y_dist, x_dist)
    v_indicator_y = y_dist / cell_h   # [N, G_h], in [0, 1]
    h_indicator_x = x_dist / cell_w   # [N, G_w], in [0, 1]
    V_macro_demand = torch.einsum("ni,nj->ij", v_indicator_y, x_dist) \
                       * vrouting_alloc   # [G_h, G_w]
    H_macro_demand = torch.einsum("ni,nj->ij", y_dist, h_indicator_x) \
                       * hrouting_alloc

    grid_v_routes = cell_w * vroutes_per_micron
    grid_h_routes = cell_h * hroutes_per_micron
    V_macro = V_macro_demand / max(grid_v_routes, 1e-9)
    H_macro = H_macro_demand / max(grid_h_routes, 1e-9)
    return V_macro, H_macro


# ---------------------------------------------------------------------------
# 5-tap smoothing
# ---------------------------------------------------------------------------


def smooth_5tap(
    grid: torch.Tensor,        # [G_h, G_w]
    smooth_range: int,
    axis: str,                 # "v" or "h"
) -> torch.Tensor:
    """Match TILOS __smooth_routing_cong:
       For V (axis='v'): per (row, col), distribute val/(rp-lp+1) over
       cells [col-smooth_range, col+smooth_range] in the SAME row.
       Edge handling: clamp to [0, grid_col-1].

       For H (axis='h'): same but along rows (col fixed).

    Implementation: TILOS does this destructively via temp grid.
    Equivalent to a non-uniform-weight 1D filter (because edges
    don't divide by the same gcell_cnt as middle cells).

    Faithful port: build the smoothing as scatter-add with proper
    per-cell weights. For interior cells far from edges, this is
    exactly a (2*smooth_range+1)-tap mean filter. Near edges, the
    weights are different.
    """
    if smooth_range <= 0:
        return grid

    G_h, G_w = grid.shape
    out = torch.zeros_like(grid)

    if axis == "v":
        # For each (row, col): val = grid[row, col] / gcell_cnt(col).
        # Add val to out[row, lp..rp].
        # gcell_cnt(col) = (rp - lp + 1) where lp = max(0, col-r), rp = min(G_w-1, col+r).
        cols = torch.arange(G_w, device=grid.device)
        lp = torch.clamp(cols - smooth_range, min=0)
        rp = torch.clamp(cols + smooth_range, max=G_w - 1)
        gcell_cnt = (rp - lp + 1).to(grid.dtype)   # [G_w]

        val = grid / gcell_cnt[None, :]   # [G_h, G_w]

        # For each col c, add val[:, c] to out[:, lp[c]..rp[c]].
        # Vectorize: for each offset in [-r, r], for each c, contribute
        # val[:, c] to out[:, c+offset] if in bounds.
        for offset in range(-smooth_range, smooth_range + 1):
            src_cols = cols
            dst_cols = src_cols + offset
            valid = (dst_cols >= 0) & (dst_cols < G_w)
            src_valid = src_cols[valid]
            dst_valid = dst_cols[valid]
            out[:, dst_valid] = out[:, dst_valid] + val[:, src_valid]
        return out

    elif axis == "h":
        rows = torch.arange(G_h, device=grid.device)
        lp = torch.clamp(rows - smooth_range, min=0)
        up = torch.clamp(rows + smooth_range, max=G_h - 1)
        gcell_cnt = (up - lp + 1).to(grid.dtype)   # [G_h]
        val = grid / gcell_cnt[:, None]   # [G_h, G_w]
        for offset in range(-smooth_range, smooth_range + 1):
            src_rows = rows
            dst_rows = src_rows + offset
            valid = (dst_rows >= 0) & (dst_rows < G_h)
            src_valid = src_rows[valid]
            dst_valid = dst_rows[valid]
            out[dst_valid, :] = out[dst_valid, :] + val[src_valid, :]
        return out

    else:
        raise ValueError(f"axis must be 'v' or 'h', got {axis}")


# ---------------------------------------------------------------------------
# Congestion top-K (combined V+H, power-mean for top 5%)
# ---------------------------------------------------------------------------


def smooth_congestion_topk(
    V: torch.Tensor,           # [G_h, G_w]
    H: torch.Tensor,           # [G_h, G_w]
    p_cong: float = 16.0,
) -> torch.Tensor:
    """Power-mean of concat(V.flat, H.flat). Approximates top-5% mean
    of TILOS abu(V+H, 0.05)."""
    combined = torch.cat([V.flatten(), H.flatten()], dim=0)
    # epsilon to avoid 0^p
    combined_clamped = combined.clamp(min=1e-12)
    return (combined_clamped.pow(p_cong).mean()).pow(1.0 / p_cong)


# ---------------------------------------------------------------------------
# Full smooth proxy
# ---------------------------------------------------------------------------


def smooth_proxy_components(
    positions_aug: torch.Tensor,
    sizes: torch.Tensor,           # [n_hard + n_soft, 2]
    owner_idx: torch.Tensor,
    offset_xy: torch.Tensor,
    mask: torch.Tensor,
    net_weights: torch.Tensor,
    net_count: float,
    G_w: int, G_h: int,
    cw: float, ch: float,
    hroutes_per_micron: float,
    vroutes_per_micron: float,
    hrouting_alloc: float,
    vrouting_alloc: float,
    n_hard: int,
    n_macros: int,
    gamma_wl: float = 6.0,
    gamma_bbox: float = 8.0,
    smooth_range: int = 2,
    p_density: float = 10.0,
    p_cong: float = 16.0,
    softness_factor: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (wl, density_cost, congestion_cost) tensors. The full
    smooth proxy is wl + density_cost + congestion_cost (the 0.5 factors
    are baked in via the 0.5 prefactor on density_cost and
    congestion_cost — see below)."""
    # WL
    wl = smooth_wl(positions_aug, owner_idx, offset_xy, mask,
                    gamma_wl, net_weights, net_count, cw, ch)

    # Density: build grid over hard + soft macros, take power-mean.
    macros_pos = positions_aug[:n_macros]
    grid = smooth_density_grid(macros_pos, sizes[:n_macros],
                                  G_w, G_h, cw, ch)
    density_topk = smooth_density_topk(grid, p_density)
    density_cost = 0.5 * density_topk

    # Congestion:
    # 1. Net routing — L-shape (proxy-aligned). Star-decomposes multi-pin
    #    nets to (src, sink) pairs; each pair contributes H demand at
    #    src row + V demand at sink col.
    V_net, H_net = smooth_l_route(
        positions_aug, owner_idx, offset_xy, mask, net_weights,
        G_w, G_h, cw, ch, hroutes_per_micron, vroutes_per_micron,
        softness_factor=softness_factor,
    )
    # 2. 5-tap smooth on net routing (TILOS smooths net but not macro)
    V_net_sm = smooth_5tap(V_net, smooth_range, axis="v")
    H_net_sm = smooth_5tap(H_net, smooth_range, axis="h")
    # 3. Macro blockage
    V_macro, H_macro = smooth_macro_blockage(
        positions_aug[:n_hard], sizes[:n_hard],
        G_w, G_h, cw, ch,
        hrouting_alloc, vrouting_alloc,
        hroutes_per_micron, vroutes_per_micron,
    )
    # 4. Combine: smoothed net + macro
    V_total = V_net_sm + V_macro
    H_total = H_net_sm + H_macro
    # 5. Top-K via power-mean over combined V+H
    cong_topk = smooth_congestion_topk(V_total, H_total, p_cong)
    cong_cost = 0.5 * cong_topk

    return wl, density_cost, cong_cost


def smooth_proxy_total(
    positions_aug: torch.Tensor,
    sizes: torch.Tensor,
    owner_idx: torch.Tensor,
    offset_xy: torch.Tensor,
    mask: torch.Tensor,
    net_weights: torch.Tensor,
    net_count: float,
    G_w: int, G_h: int,
    cw: float, ch: float,
    hroutes_per_micron: float,
    vroutes_per_micron: float,
    hrouting_alloc: float,
    vrouting_alloc: float,
    n_hard: int,
    n_macros: int,
    **kwargs,
) -> torch.Tensor:
    """Convenience: returns wl + density_cost + congestion_cost as a
    single scalar."""
    wl, d, c = smooth_proxy_components(
        positions_aug, sizes, owner_idx, offset_xy, mask, net_weights,
        net_count, G_w, G_h, cw, ch, hroutes_per_micron, vroutes_per_micron,
        hrouting_alloc, vrouting_alloc, n_hard, n_macros, **kwargs,
    )
    return wl + d + c
