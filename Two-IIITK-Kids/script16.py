"""V11 T3: Longest-net joint reposition.

For the top-K nets by current HPWL bbox, the per-net wirelength contribution
is concentrated in a small number of pins. Standard CD / swap moves act on
ONE macro at a time, so the score change from any individual move is small
even when the net is grossly mis-routed. This routine identifies the top-K
longest nets and proposes COMPACTING all hard movable macros on each net
toward the net's pin centroid, then legalizes and accepts via the outer
proxy gate.

The motion is gentle (default `compact_frac=0.5`, half-way toward centroid)
so legalization usually succeeds with minimal cascade. Per-net trial is
independent; failures don't poison successes since we restore the snapshot
between trials.

Outer-verify safe — caller is expected to pass the result through
`_record` / `_legalize_full`.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost

from script3 import (
    legalize as legalize_np,
    _list_violators,
    _force_displace_grid,
)


def _legal_in_place(pos_np, sizes, movable, cw, ch, seed=0):
    legal = legalize_np(pos_np[:sizes.shape[0]].copy(), sizes, movable, cw, ch, seed=seed)
    if _list_violators(legal, sizes, movable):
        legal = _force_displace_grid(legal, sizes, movable, cw, ch,
                                     n_steps=160, max_sweeps=6)
    return legal


def _net_bbox_lens(benchmark: Benchmark, pos_np):
    """Return [n_nets] of L1 bbox length for each net at the given placement.
    Skips nets with <2 pins or unresolved pin owners."""
    n_macros = benchmark.num_macros
    n_hard = benchmark.num_hard_macros
    port_pos = benchmark.port_positions.cpu().numpy().astype(np.float64)
    n_ports = port_pos.shape[0]
    pin_offsets = benchmark.macro_pin_offsets
    nets = benchmark.net_pin_nodes
    out = np.zeros(len(nets), dtype=np.float64)
    for i, net in enumerate(nets):
        pn = net.cpu().numpy().astype(np.int64) if hasattr(net, "cpu") else np.asarray(net, dtype=np.int64)
        if pn.shape[0] < 2:
            continue
        xs, ys = [], []
        for k in range(pn.shape[0]):
            o, s = int(pn[k, 0]), int(pn[k, 1])
            if o < n_macros:
                px = float(pos_np[o, 0])
                py = float(pos_np[o, 1])
                if (o < n_hard and pin_offsets and o < len(pin_offsets)
                        and pin_offsets[o] is not None
                        and pin_offsets[o].shape[0] > s):
                    px += float(pin_offsets[o][s, 0])
                    py += float(pin_offsets[o][s, 1])
                xs.append(px); ys.append(py)
            else:
                pi = o - n_macros
                if 0 <= pi < n_ports:
                    xs.append(float(port_pos[pi, 0]))
                    ys.append(float(port_pos[pi, 1]))
        if xs:
            out[i] = (max(xs) - min(xs)) + (max(ys) - min(ys))
    return out


def longest_net_repair(placement: torch.Tensor, benchmark: Benchmark, plc, *,
                       top_k=5, compact_frac=0.5, seed=42,
                       time_cap=None, verbose=False):
    """For each of top-K longest nets, try compacting its hard movable
    macros toward the net centroid by `compact_frac`. Accept per-net if
    the resulting overall proxy is lower; restore otherwise. Returns the
    placement at the end of the trials (callers should pass through
    `_record` for outer-verify safety).
    """
    rng = np.random.default_rng(seed)
    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros
    n_ports = benchmark.port_positions.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes_all = benchmark.macro_sizes.numpy().astype(np.float64)
    sizes = sizes_all[:n_hard]
    movable_full = benchmark.get_movable_mask().cpu().numpy()
    movable_hard = movable_full[:n_hard]

    pos_np = placement.cpu().numpy().astype(np.float64).copy()
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5

    # Score current.
    cur_t = torch.from_numpy(pos_np.astype(np.float32))
    try:
        cur_c = compute_proxy_cost(cur_t, benchmark, plc)
        cur_proxy = float(cur_c["proxy_cost"])
        cur_ovr = int(cur_c["overlap_count"])
        if cur_ovr > 0:
            if verbose:
                print(f"  [net-repair] input has {cur_ovr} overlaps — skipping",
                      flush=True)
            return placement.clone()
    except Exception:
        return placement.clone()

    # Rank nets by current bbox length.
    lens = _net_bbox_lens(benchmark, pos_np)
    if lens.size == 0 or lens.max() <= 0:
        return placement.clone()
    order = np.argsort(lens)[::-1]
    top = order[:top_k]

    nets = benchmark.net_pin_nodes
    pin_offsets = benchmark.macro_pin_offsets

    best_pos = pos_np.copy()
    best_proxy = cur_proxy
    t0 = time.time()
    attempts = 0
    accepts = 0

    for net_id in top:
        if time_cap is not None and time.time() - t0 > time_cap:
            break
        attempts += 1
        net = nets[int(net_id)]
        pn = net.cpu().numpy().astype(np.int64) if hasattr(net, "cpu") else np.asarray(net, dtype=np.int64)
        if pn.shape[0] < 2:
            continue
        # Gather pin coords + which (hard movable) macros are on this net.
        pin_xs, pin_ys = [], []
        macros_on_net = []
        for k in range(pn.shape[0]):
            o, s = int(pn[k, 0]), int(pn[k, 1])
            if o < n_macros:
                px = float(pos_np[o, 0])
                py = float(pos_np[o, 1])
                if (o < n_hard and pin_offsets and o < len(pin_offsets)
                        and pin_offsets[o] is not None
                        and pin_offsets[o].shape[0] > s):
                    px += float(pin_offsets[o][s, 0])
                    py += float(pin_offsets[o][s, 1])
                pin_xs.append(px); pin_ys.append(py)
                if o < n_hard and movable_hard[o]:
                    macros_on_net.append(o)
            else:
                pi = o - n_macros
                if 0 <= pi < n_ports:
                    pin_xs.append(float(benchmark.port_positions[pi, 0]))
                    pin_ys.append(float(benchmark.port_positions[pi, 1]))
        if not pin_xs or not macros_on_net:
            continue
        cx = sum(pin_xs) / len(pin_xs)
        cy = sum(pin_ys) / len(pin_ys)

        # Snapshot the positions of macros on this net so we can revert.
        macros_on_net = list(set(macros_on_net))
        snap = {m: pos_np[m].copy() for m in macros_on_net}

        # Apply compact move: position = old + compact_frac * (centroid - old).
        for m in macros_on_net:
            ox, oy = pos_np[m]
            new_x = ox + compact_frac * (cx - ox)
            new_y = oy + compact_frac * (cy - oy)
            # Clamp inside canvas with half-extent margin.
            new_x = min(max(new_x, half_w[m]), cw - half_w[m])
            new_y = min(max(new_y, half_h[m]), ch - half_h[m])
            pos_np[m, 0] = new_x
            pos_np[m, 1] = new_y

        # Legalize and score.
        legal = _legal_in_place(pos_np, sizes, movable_hard, cw, ch,
                                seed=int(rng.integers(0, 1 << 30)))
        trial = pos_np.copy()
        trial[:n_hard] = legal
        try:
            tc = compute_proxy_cost(torch.from_numpy(trial.astype(np.float32)),
                                    benchmark, plc)
            t_proxy = float(tc["proxy_cost"])
            t_ovr = int(tc["overlap_count"])
        except Exception:
            for m, p in snap.items():
                pos_np[m] = p
            continue

        if t_ovr == 0 and t_proxy < best_proxy - 1e-9:
            best_pos = trial.copy()
            best_proxy = t_proxy
            pos_np = trial.copy()
            accepts += 1
        else:
            for m, p in snap.items():
                pos_np[m] = p

    if verbose:
        print(f"  [net-repair] {accepts}/{attempts} nets improved "
              f"({cur_proxy:.6f} -> {best_proxy:.6f}) t={time.time()-t0:.0f}s",
              flush=True)
    out = placement.clone()
    out[:n_macros] = torch.from_numpy(best_pos[:n_macros].astype(np.float32))
    return out
