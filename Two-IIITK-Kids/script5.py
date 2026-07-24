from __future__ import annotations

import math
import torch
import torch.nn.functional as F


_NEG_INF = -1e20
_POS_INF = 1e20


# ── WL ─────────────────────────────────────────────────────────────────
def smooth_wl(positions_aug, owner_idx, offset_xy, mask, gamma,
              net_weights, net_count, canvas_w, canvas_h):
    """Pin-level weighted WA-HPWL via LSE. Sharper at high gamma."""
    pin_xy = positions_aug[owner_idx] + offset_xy  # [N, P, 2]
    px = torch.where(mask, pin_xy[..., 0], pin_xy[..., 0].new_full((), _NEG_INF))
    py = torch.where(mask, pin_xy[..., 1], pin_xy[..., 1].new_full((), _NEG_INF))
    pxn = torch.where(mask, pin_xy[..., 0], pin_xy[..., 0].new_full((), _POS_INF))
    pyn = torch.where(mask, pin_xy[..., 1], pin_xy[..., 1].new_full((), _POS_INF))
    x_hi = torch.logsumexp(gamma * px, dim=1) / gamma
    x_lo = -torch.logsumexp(-gamma * pxn, dim=1) / gamma
    y_hi = torch.logsumexp(gamma * py, dim=1) / gamma
    y_lo = -torch.logsumexp(-gamma * pyn, dim=1) / gamma
    per_net = (x_hi - x_lo) + (y_hi - y_lo)
    return (per_net * net_weights).sum() / (net_count * (canvas_w + canvas_h))


# ── density ────────────────────────────────────────────────────────────
def density_grid(macros_pos, sizes, grid_w, grid_h, canvas_w, canvas_h):
    """Per-cell occupancy ratio. Smooth in pos via clipped rectangle overlap."""
    cell_w = canvas_w / grid_w
    cell_h = canvas_h / grid_h
    hw = sizes[:, 0:1] / 2
    hh = sizes[:, 1:2] / 2
    HW = cell_w / 2
    HH = cell_h / 2
    cx = (torch.arange(grid_w, device=macros_pos.device, dtype=macros_pos.dtype) + 0.5) * cell_w
    cy = (torch.arange(grid_h, device=macros_pos.device, dtype=macros_pos.dtype) + 0.5) * cell_h
    mx_lo = macros_pos[:, 0:1] - hw
    mx_hi = macros_pos[:, 0:1] + hw
    cx_lo = cx[None, :] - HW
    cx_hi = cx[None, :] + HW
    ox = torch.relu(torch.minimum(mx_hi, cx_hi) - torch.maximum(mx_lo, cx_lo))
    my_lo = macros_pos[:, 1:2] - hh
    my_hi = macros_pos[:, 1:2] + hh
    cy_lo = cy[None, :] - HH
    cy_hi = cy[None, :] + HH
    oy = torch.relu(torch.minimum(my_hi, cy_hi) - torch.maximum(my_lo, cy_lo))
    return torch.einsum("ni,nj->ij", oy, ox) / (cell_w * cell_h)


def density_topk(grid, p=10.0):
    flat = grid.flatten().clamp(min=1e-12)
    return (flat.pow(p).mean()).pow(1.0 / p)


# ── congestion ─────────────────────────────────────────────────────────
def net_l_routing(positions_aug, owner_idx, offset_xy, mask, net_weights,
                  grid_w, grid_h, canvas_w, canvas_h,
                  hroutes_per_um, vroutes_per_um, softness=0.5):
    """Smooth analog of TILOS net L-routing (star decomposition for >3-pin,
    explicit Steiner-T row for 3-pin nets).
    Returns (V_route, H_route) [grid_h, grid_w] normalized by routing capacity.
    """
    pin_xy = positions_aug[owner_idx] + offset_xy
    N, P, _ = pin_xy.shape
    cw = canvas_w / grid_w
    ch = canvas_h / grid_h
    s = max(cw, ch) * softness
    cx = (torch.arange(grid_w, device=positions_aug.device, dtype=positions_aug.dtype) + 0.5) * cw
    cy = (torch.arange(grid_h, device=positions_aug.device, dtype=positions_aug.dtype) + 0.5) * ch

    src_x = pin_xy[:, 0, 0]
    src_y = pin_xy[:, 0, 1]

    # 3-pin median-y handling (T-Steiner row).
    pin_y = pin_xy[:, :, 1]
    pin_x = pin_xy[:, :, 0]
    mask_f = mask.to(positions_aug.dtype)
    pin_cnt = mask_f.sum(dim=1)
    is_3 = ((pin_cnt > 2.5) & (pin_cnt < 3.5)).to(positions_aug.dtype)
    big = float(canvas_h * 100.0)
    y_for_max = torch.where(mask, pin_y, pin_y.new_full((), -big))
    y_for_min = torch.where(mask, pin_y, pin_y.new_full((), big))
    sum_y = (pin_y * mask_f).sum(dim=1)
    max_y = y_for_max.max(dim=1).values
    min_y = y_for_min.min(dim=1).values
    median_y = sum_y - max_y - min_y
    peak_y = is_3 * median_y + (1.0 - is_3) * src_y

    peak_row = (torch.sigmoid((cy[None, :] - (peak_y[:, None] - ch * 0.5)) / s)
                * torch.sigmoid(((peak_y[:, None] + ch * 0.5) - cy[None, :]) / s))

    # Sink data
    sink_x = pin_xy[:, 1:, 0]
    sink_y = pin_xy[:, 1:, 1]
    sink_v = mask[:, 1:].to(positions_aug.dtype)

    x_min_p = torch.minimum(src_x[:, None], sink_x)
    x_max_p = torch.maximum(src_x[:, None], sink_x)
    y_min_p = torch.minimum(src_y[:, None], sink_y)
    y_max_p = torch.maximum(src_y[:, None], sink_y)

    # H demand: star (default) and 3-pin Steiner span
    h_between = (torch.sigmoid((cx[None, None, :] - x_min_p[:, :, None]) / s)
                 * torch.sigmoid((x_max_p[:, :, None] - cx[None, None, :]) / s))
    h_star = (h_between * sink_v[:, :, None]).sum(dim=1) * net_weights[:, None]

    x_for_max = torch.where(mask, pin_x, pin_x.new_full((), -big))
    x_for_min = torch.where(mask, pin_x, pin_x.new_full((), big))
    xmax_net = x_for_max.max(dim=1).values
    xmin_net = x_for_min.min(dim=1).values
    h_span = (torch.sigmoid((cx[None, :] - xmin_net[:, None]) / s)
              * torch.sigmoid((xmax_net[:, None] - cx[None, :]) / s))
    h_3pin = h_span * net_weights[:, None]
    h_per_net = is_3[:, None] * h_3pin + (1.0 - is_3)[:, None] * h_star

    H_route = torch.einsum("nh,nw->hw", peak_row, h_per_net)

    # V demand
    peak_col = (torch.sigmoid((cx[None, None, :] - (sink_x[:, :, None] - cw * 0.5)) / s)
                * torch.sigmoid(((sink_x[:, :, None] + cw * 0.5) - cx[None, None, :]) / s))
    v_between = (torch.sigmoid((cy[None, None, :] - y_min_p[:, :, None]) / s)
                 * torch.sigmoid((y_max_p[:, :, None] - cy[None, None, :]) / s))
    v_w = v_between * sink_v[:, :, None] * net_weights[:, None, None]
    V_route = torch.einsum("nkh,nkw->hw", v_w, peak_col)

    V_route = V_route / max(cw * vroutes_per_um, 1e-9)
    H_route = H_route / max(ch * hroutes_per_um, 1e-9)
    return V_route, H_route


def macro_blockage(hard_pos, hard_sizes, grid_w, grid_h, canvas_w, canvas_h,
                   hrouting_alloc, vrouting_alloc,
                   hroutes_per_um, vroutes_per_um):
    """Smooth per-cell macro routing demand. Matches TILOS macro routing."""
    cw = canvas_w / grid_w
    ch = canvas_h / grid_h
    hw = hard_sizes[:, 0:1] / 2
    hh = hard_sizes[:, 1:2] / 2
    HW = cw / 2
    HH = ch / 2
    cx = (torch.arange(grid_w, device=hard_pos.device, dtype=hard_pos.dtype) + 0.5) * cw
    cy = (torch.arange(grid_h, device=hard_pos.device, dtype=hard_pos.dtype) + 0.5) * ch
    mx_lo = hard_pos[:, 0:1] - hw
    mx_hi = hard_pos[:, 0:1] + hw
    cx_lo = cx[None, :] - HW
    cx_hi = cx[None, :] + HW
    x_d = torch.relu(torch.minimum(mx_hi, cx_hi) - torch.maximum(mx_lo, cx_lo))
    my_lo = hard_pos[:, 1:2] - hh
    my_hi = hard_pos[:, 1:2] + hh
    cy_lo = cy[None, :] - HH
    cy_hi = cy[None, :] + HH
    y_d = torch.relu(torch.minimum(my_hi, cy_hi) - torch.maximum(my_lo, cy_lo))
    v_ind = y_d / ch  # [N, grid_h]
    h_ind = x_d / cw  # [N, grid_w]
    V = torch.einsum("ni,nj->ij", v_ind, x_d) * vrouting_alloc
    H = torch.einsum("ni,nj->ij", y_d, h_ind) * hrouting_alloc
    V = V / max(cw * vroutes_per_um, 1e-9)
    H = H / max(ch * hroutes_per_um, 1e-9)
    return V, H


def smooth_avg_1d(grid, smooth_range, axis):
    """1D rectangular average filter (matches TILOS smoothing).
    axis 'v': average per-row along columns. axis 'h': per-column along rows."""
    if smooth_range <= 0:
        return grid
    gh, gw = grid.shape
    out = torch.zeros_like(grid)
    if axis == "v":
        idx = torch.arange(gw, device=grid.device)
        lp = torch.clamp(idx - smooth_range, min=0)
        rp = torch.clamp(idx + smooth_range, max=gw - 1)
        cnt = (rp - lp + 1).to(grid.dtype)
        val = grid / cnt[None, :]
        for off in range(-smooth_range, smooth_range + 1):
            src = idx
            dst = src + off
            mk = (dst >= 0) & (dst < gw)
            out[:, dst[mk]] = out[:, dst[mk]] + val[:, src[mk]]
        return out
    elif axis == "h":
        idx = torch.arange(gh, device=grid.device)
        lp = torch.clamp(idx - smooth_range, min=0)
        up = torch.clamp(idx + smooth_range, max=gh - 1)
        cnt = (up - lp + 1).to(grid.dtype)
        val = grid / cnt[:, None]
        for off in range(-smooth_range, smooth_range + 1):
            src = idx
            dst = src + off
            mk = (dst >= 0) & (dst < gh)
            out[dst[mk], :] = out[dst[mk], :] + val[src[mk], :]
        return out
    raise ValueError(axis)


def congestion_topk(V, H, p=16.0):
    combined = torch.cat([V.flatten(), H.flatten()], dim=0).clamp(min=1e-12)
    return (combined.pow(p).mean()).pow(1.0 / p)


def full_components(positions_aug, sizes, owner_idx, offset_xy, mask,
                    net_weights, net_count,
                    grid_w, grid_h, canvas_w, canvas_h,
                    hroutes_per_um, vroutes_per_um,
                    hrouting_alloc, vrouting_alloc,
                    n_hard, n_macros,
                    gamma_wl=6.0, smooth_range=2,
                    p_density=10.0, p_cong=16.0, softness=0.5):
    """Return (wl, density_cost, congestion_cost) for backprop.
    The 0.5 factors on density/congestion are baked in here, so the
    summed loss matches TILOS proxy: WL + 0.5*D + 0.5*C."""
    wl = smooth_wl(positions_aug, owner_idx, offset_xy, mask,
                   gamma_wl, net_weights, net_count, canvas_w, canvas_h)
    macros_pos = positions_aug[:n_macros]
    grid = density_grid(macros_pos, sizes[:n_macros], grid_w, grid_h,
                        canvas_w, canvas_h)
    d_cost = 0.5 * density_topk(grid, p_density)

    V_net, H_net = net_l_routing(
        positions_aug, owner_idx, offset_xy, mask, net_weights,
        grid_w, grid_h, canvas_w, canvas_h,
        hroutes_per_um, vroutes_per_um, softness=softness,
    )
    V_net = smooth_avg_1d(V_net, smooth_range, axis="v")
    H_net = smooth_avg_1d(H_net, smooth_range, axis="h")
    V_m, H_m = macro_blockage(
        positions_aug[:n_hard], sizes[:n_hard],
        grid_w, grid_h, canvas_w, canvas_h,
        hrouting_alloc, vrouting_alloc,
        hroutes_per_um, vroutes_per_um,
    )
    c_cost = 0.5 * congestion_topk(V_net + V_m, H_net + H_m, p_cong)
    return wl, d_cost, c_cost
