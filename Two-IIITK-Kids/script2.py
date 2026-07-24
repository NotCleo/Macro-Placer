from __future__ import annotations

import math
import time

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from script5 import full_components


def _build_pin_tensors(benchmark: Benchmark, device, max_pins_per_net=None):
    n_macros = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    nets_pin = []
    for net in benchmark.net_pin_nodes:
        if net.shape[0] >= 2:
            nets_pin.append(net.long())
    if not nets_pin:
        return None
    true_max = max(int(n.shape[0]) for n in nets_pin)
    if max_pins_per_net is not None and true_max > max_pins_per_net:
        cap = int(max_pins_per_net)
    else:
        cap = true_max
    N = len(nets_pin)
    owner = torch.zeros(N, cap, dtype=torch.long, device=device)
    offset = torch.zeros(N, cap, 2, device=device)
    mask = torch.zeros(N, cap, dtype=torch.bool, device=device)
    pin_off = benchmark.macro_pin_offsets
    for i, net in enumerate(nets_pin):
        k = min(int(net.shape[0]), cap)
        owner[i, :k] = net[:k, 0].long().to(device)
        for j in range(k):
            o = int(net[j, 0])
            slot = int(net[j, 1])
            if o < n_hard and slot < pin_off[o].shape[0]:
                offset[i, j, 0] = float(pin_off[o][slot, 0])
                offset[i, j, 1] = float(pin_off[o][slot, 1])
        mask[i, :k] = True
    return owner, offset, mask


def _build_net_weights(benchmark, plc, device):
    drivers = list(plc.nets.keys())
    base = []
    for d in drivers:
        idx = plc.mod_name_to_indices.get(d)
        w = 1.0
        if idx is not None:
            pin = plc.modules_w_pins[idx]
            if hasattr(pin, "get_weight"):
                w = float(pin.get_weight())
        base.append(w)
    kept = []
    for i, net in enumerate(benchmark.net_pin_nodes):
        if net.shape[0] >= 2:
            kept.append(base[i] if i < len(base) else 1.0)
    return torch.tensor(kept, dtype=torch.float32, device=device)


def _gamma_at(step, n_steps, g_start, g_mid, g_end, phase1_frac=0.5):
    t = step / max(n_steps - 1, 1)
    if t < phase1_frac:
        u = t / phase1_frac
        return g_start + u * (g_mid - g_start)
    u = (t - phase1_frac) / max(1.0 - phase1_frac, 1e-9)
    return g_mid * (g_end / g_mid) ** u


def _lr_at(step, n_steps, lr_start, lr_end, warmup=20):
    if step < warmup:
        return lr_start * (step + 1) / warmup
    t = (step - warmup) / max(n_steps - warmup - 1, 1)
    return lr_start + t * (lr_end - lr_start)


def run_gd(benchmark: Benchmark, plc, init_pos: torch.Tensor, *,
           n_steps=1000, lr_start=0.2, lr_end=0.02,
           gamma_start=0.5, gamma_mid=2.0, gamma_end=8.0,
           p_density=10.0, p_cong=16.0, softness=0.5,
           wl_weight=1.0, density_weight=1.0, cong_weight=1.0,
           noise_start=0.0, noise_every=50, noise_frac=0.7,
           max_pins_per_net=None, seed=42, device=None, verbose=False,
           plateau_window=80, plateau_rel_tol=1e-4, plateau_min_frac=0.80,
           net_weight_multiplier=None):
    """Run Adam GD on the smooth-proxy surrogate. Returns the trainable
    [num_macros, 2] tensor on CPU (legalized by the caller).

    Plateau early-stop: after `plateau_min_frac * n_steps` (i.e. after the
    gamma anneal has reached its sharp tail), compare the mean loss over the
    last `plateau_window` steps to the previous window. If the relative
    change is below `plateau_rel_tol`, stop early. Cuts 10-30% of GD time on
    converged training without quality loss.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros
    sizes_all = benchmark.macro_sizes.to(device).float()
    half_w = sizes_all[:, 0] / 2.0
    half_h = sizes_all[:, 1] / 2.0

    pin_data = _build_pin_tensors(benchmark, device, max_pins_per_net=max_pins_per_net)
    if pin_data is None:
        return init_pos.clone().cpu()
    owner_idx, offset_xy, mask = pin_data
    net_weights = _build_net_weights(benchmark, plc, device)
    # V10 (A2): optional per-net multiplier — boosts WL importance of
    # selected nets (typically the top-20% longest by initial bbox).
    # graph_grad's gp.py uses dynamic reweight every 60 steps with α=3.0;
    # our static version applies the same idea precomputed in placer.py.
    if net_weight_multiplier is not None:
        try:
            nwm = net_weight_multiplier.to(device).float()
            if nwm.shape[0] == net_weights.shape[0]:
                net_weights = net_weights * nwm
        except Exception:
            pass
    net_count = float(net_weights.sum())

    G_w = int(plc.grid_col)
    G_h = int(plc.grid_row)
    h_per = float(plc.hroutes_per_micron)
    v_per = float(plc.vroutes_per_micron)
    h_alloc = float(plc.hrouting_alloc)
    v_alloc = float(plc.vrouting_alloc)
    smooth_range = int(plc.smooth_range)

    init = init_pos.to(device).float().clone()
    init[:, 0].clamp_(half_w, cw - half_w)
    init[:, 1].clamp_(half_h, ch - half_h)
    movable = benchmark.get_movable_mask().to(device)
    port_pos = benchmark.port_positions.to(device).float()
    fixed_init = init.clone()
    param = torch.nn.Parameter(init.clone())
    opt = torch.optim.Adam([param], lr=lr_start)

    if verbose:
        print(f"[gd] device={device} n_steps={n_steps} "
              f"gamma {gamma_start}->{gamma_end}", flush=True)

    t0 = time.time()
    loss_hist = []
    plateau_min_step = int(n_steps * plateau_min_frac)
    last_step = n_steps
    for step in range(n_steps):
        for pg in opt.param_groups:
            pg["lr"] = _lr_at(step, n_steps, lr_start, lr_end)
        gamma = _gamma_at(step, n_steps, gamma_start, gamma_mid, gamma_end)

        opt.zero_grad()
        positions_aug = torch.cat([param, port_pos], dim=0)
        wl, d_c, c_c = full_components(
            positions_aug, sizes_all, owner_idx, offset_xy, mask,
            net_weights, net_count, G_w, G_h, cw, ch,
            h_per, v_per, h_alloc, v_alloc,
            int(n_hard), int(n_macros),
            gamma_wl=gamma, smooth_range=smooth_range,
            p_density=p_density, p_cong=p_cong, softness=softness,
        )
        loss = wl_weight * wl + density_weight * d_c + cong_weight * c_c
        loss.backward()
        opt.step()

        with torch.no_grad():
            if (noise_start > 0 and step % noise_every == 0
                    and step < n_steps * noise_frac):
                decay = 1.0 - step / max(n_steps * noise_frac, 1.0)
                sigma = noise_start * decay
                if sigma > 0:
                    param.add_(torch.randn_like(param) * sigma)
            param[:, 0].clamp_(half_w, cw - half_w)
            param[:, 1].clamp_(half_h, ch - half_h)
            if (~movable).any():
                param[~movable] = fixed_init[~movable]

        # Plateau detection — only after gamma has reached its sharp tail
        # AND after a full noise-injection cycle has settled. Compare the
        # last `plateau_window` mean loss to the previous window.
        loss_hist.append(float(loss.detach()))
        if (step >= plateau_min_step
                and step >= 2 * plateau_window
                and step >= n_steps * noise_frac + plateau_window):
            recent = sum(loss_hist[-plateau_window:]) / plateau_window
            older = sum(loss_hist[-2 * plateau_window:-plateau_window]) / plateau_window
            if abs(recent - older) / max(abs(older), 1e-9) < plateau_rel_tol:
                last_step = step + 1
                break

    if verbose:
        print(f"[gd] {last_step} steps in {time.time() - t0:.1f}s "
              f"(early-stop: {last_step < n_steps})", flush=True)
    return param.detach().cpu().clone()
