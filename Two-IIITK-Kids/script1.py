from __future__ import annotations

import time
import numpy as np
import torch

from macro_place.benchmark import Benchmark
from script4 import FastProxy, IncrementalProxy


def _no_overlap(idx, pos, sizes, n_hard, margin=0.005):
    hw_i = sizes[idx, 0] * 0.5
    hh_i = sizes[idx, 1] * 0.5
    xi = pos[idx, 0]
    yi = pos[idx, 1]
    hw = sizes[:n_hard, 0] * 0.5
    hh = sizes[:n_hard, 1] * 0.5
    dx = np.abs(pos[:n_hard, 0] - xi)
    dy = np.abs(pos[:n_hard, 1] - yi)
    sep_x = hw_i + hw + margin
    sep_y = hh_i + hh + margin
    overlap = (dx < sep_x) & (dy < sep_y)
    overlap[idx] = False
    return not overlap.any()


def cd_refine(placement: torch.Tensor, benchmark: Benchmark, plc, *,
              K_trials=8, max_rounds=12,
              sigma_fracs=(0.003, 0.01, 0.03, 0.08, 0.15, 0.30),
              sigma_weights=(0.30, 0.25, 0.20, 0.15, 0.07, 0.03),
              early_stop_accept_pct=0.005, early_stop_consec=3,
              score_mode="partial", seed=42, max_pins_per_net=None,
              verbose=False, time_cap=None,
              order_by_congestion=False):
    """Run CD on hard movable macros. Returns the best placement seen.

    V13-rev2 Opt 2: `order_by_congestion=True` orders macros each round by
    their congestion-stress contribution (sum of cong over their nets'
    bboxes), highest first, instead of random. On dense benches, 80% of
    CD improvements come from 20% of macros (those whose nets cross hot
    cells). Priority ordering hits high-impact macros first, so the same
    `max_rounds` budget reaches deeper convergence.
    Inspired by graph_grad/placer.py:_macro_priority.
    """
    rng = np.random.default_rng(seed)

    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros
    n_ports = benchmark.port_positions.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    mov_idx = np.flatnonzero(movable)

    benchmark.macro_positions = placement[:n_macros].clone()
    inc = IncrementalProxy(benchmark, plc=plc)
    fp = FastProxy(benchmark, plc=plc, max_pins_per_net=max_pins_per_net)
    pos_aug = inc.pos_aug

    # V13-rev2 Opt 2: precompute owner-to-nets mapping if congestion
    # ordering is requested. One-time cost, reused across rounds.
    _owner_to_nets: dict = {}
    if order_by_congestion:
        try:
            for ni in range(inc.num_nets_used):
                owners_arr = inc.net_pin_owner[ni]
                for k in range(int(owners_arr.shape[0])):
                    o = int(owners_arr[k])
                    if o < n_hard and movable[o]:
                        _owner_to_nets.setdefault(o, []).append(ni)
        except Exception:
            order_by_congestion = False  # fallback to random on any error

    # Pin position cache for congestion (avoids per-call O(N*P) rebuild).
    fp_pin_pos = (pos_aug[fp.owner_idx] + fp.offset_xy).astype(np.float64, copy=False)
    fp_cap = fp.mask.shape[1]
    macro_to_fp = [None] * (n_macros + n_ports)
    for m in range(n_macros + n_ports):
        entries = inc.macro_to_pins[m] if m < len(inc.macro_to_pins) else []
        macro_to_fp[m] = [(ni, pi) for (ni, pi) in entries if pi < fp_cap]

    def _refresh_fp(macro):
        new_x = pos_aug[macro, 0]
        new_y = pos_aug[macro, 1]
        saved = []
        for (ni, pi) in macro_to_fp[macro]:
            saved.append((ni, pi, fp_pin_pos[ni, pi, 0], fp_pin_pos[ni, pi, 1]))
            fp_pin_pos[ni, pi, 0] = new_x + fp.offset_xy[ni, pi, 0]
            fp_pin_pos[ni, pi, 1] = new_y + fp.offset_xy[ni, pi, 1]
        return saved

    def _restore_fp(saved):
        for (ni, pi, ox, oy) in saved:
            fp_pin_pos[ni, pi, 0] = ox
            fp_pin_pos[ni, pi, 1] = oy

    init_partial = inc.partial_proxy_no_cong()
    init_chalf = fp.congestion_cost(pos_aug, pin_pos=fp_pin_pos)  # already *0.5
    init_full = init_partial + init_chalf
    if verbose:
        print(f"  [cd] init proxy={init_full:.6f}", flush=True)
    best_full = init_full
    best_pos = pos_aug[:n_macros].copy()

    sw = np.asarray(sigma_weights, dtype=np.float64)
    sw = sw / sw.sum()
    sf = np.asarray(sigma_fracs, dtype=np.float64)

    consec_low = 0
    n_accepts_total = 0
    t0 = time.time()
    for rnd in range(max_rounds):
        rnd_t0 = time.time()
        rnd_accepts = 0
        # V13-rev2 Opt 2: priority order on odd rounds, random on even.
        # Alternation keeps the diversity that random ordering provides
        # (rare-but-useful low-priority moves) while front-loading
        # high-impact macros. Also re-computes priority each round so it
        # adapts to the current congestion grid (hot cells shift as we
        # accept moves).
        if order_by_congestion and (rnd % 2 == 0):
            try:
                pin_pos = (pos_aug[fp.owner_idx] + fp.offset_xy).astype(np.float64, copy=False)
                # Reuse fp's congestion machinery to score each cell once.
                from script4 import _congestion_grids
                V_grid, H_grid = _congestion_grids(
                    pin_pos, fp.mask, fp.net_weights,
                    fp.cell_w, fp.cell_h, fp.grid_col, fp.grid_row,
                    fp.grid_v_routes, fp.grid_h_routes, fp.smooth_range,
                    pos_aug[:n_hard], fp.sizes[:n_hard],
                    fp.vrouting_alloc, fp.hrouting_alloc,
                )
                vh = (V_grid + H_grid).reshape(fp.grid_row, fp.grid_col)
                scores = np.zeros(mov_idx.shape[0], dtype=np.float64)
                for ix, m in enumerate(mov_idx):
                    nets_for_m = _owner_to_nets.get(int(m), ())
                    s = 0.0
                    for ni in nets_for_m:
                        xs = inc.net_pin_xy[ni][:, 0]
                        ys = inc.net_pin_xy[ni][:, 1]
                        x0 = max(0, int(xs.min() / fp.cell_w))
                        x1 = min(fp.grid_col - 1, int(xs.max() / fp.cell_w))
                        y0 = max(0, int(ys.min() / fp.cell_h))
                        y1 = min(fp.grid_row - 1, int(ys.max() / fp.cell_h))
                        s += float(vh[y0:y1 + 1, x0:x1 + 1].sum())
                    scores[ix] = s
                # Descending stress, deterministic tie-break via rng
                jitter = rng.random(scores.shape[0]) * 1e-12
                order_idx = np.argsort(-(scores + jitter))
                order = mov_idx[order_idx]
            except Exception:
                order = rng.permutation(mov_idx)
        else:
            order = rng.permutation(mov_idx)
        for macro in order:
            if time_cap is not None and time.time() - t0 > time_cap:
                break
            cx = float(pos_aug[macro, 0])
            cy = float(pos_aug[macro, 1])
            hw = sizes[macro, 0] * 0.5
            hh = sizes[macro, 1] * 0.5
            sel = rng.choice(len(sf), size=K_trials, p=sw)
            sigma_x = sf[sel] * cw
            sigma_y = sf[sel] * ch
            jx = rng.normal(0.0, 1.0, K_trials) * sigma_x
            jy = rng.normal(0.0, 1.0, K_trials) * sigma_y
            tx = np.clip(cx + jx, hw, cw - hw)
            ty = np.clip(cy + jy, hh, ch - hh)

            best_score = float("inf")
            best_xy = None
            for k in range(K_trials):
                inc.try_move(macro, float(tx[k]), float(ty[k]))
                if not _no_overlap(macro, pos_aug, sizes, n_hard):
                    inc.revert_move()
                    continue
                if score_mode == "cong_only":
                    saved = _refresh_fp(macro)
                    score = fp.congestion_cost(pos_aug, pin_pos=fp_pin_pos)
                    _restore_fp(saved)
                elif score_mode == "full":
                    saved = _refresh_fp(macro)
                    chalf = fp.congestion_cost(pos_aug, pin_pos=fp_pin_pos)
                    _restore_fp(saved)
                    score = inc.partial_proxy_no_cong() + chalf
                else:
                    score = inc.partial_proxy_no_cong()
                inc.revert_move()
                if score < best_score - 1e-9:
                    best_score = score
                    best_xy = (float(tx[k]), float(ty[k]))

            if best_xy is None:
                continue

            if score_mode == "full":
                if best_score < best_full - 1e-9:
                    inc.try_move(macro, best_xy[0], best_xy[1])
                    inc.commit_move()
                    _refresh_fp(macro)
                    best_full = best_score
                    best_pos = pos_aug[:n_macros].copy()
                    rnd_accepts += 1
            else:
                inc.try_move(macro, best_xy[0], best_xy[1])
                saved = _refresh_fp(macro)
                chalf = fp.congestion_cost(pos_aug, pin_pos=fp_pin_pos)
                full_p = inc.partial_proxy_no_cong() + chalf
                if full_p < best_full - 1e-9:
                    inc.commit_move()
                    best_full = full_p
                    best_pos = pos_aug[:n_macros].copy()
                    rnd_accepts += 1
                else:
                    _restore_fp(saved)
                    inc.revert_move()
        n_accepts_total += rnd_accepts
        accept_pct = rnd_accepts / max(len(mov_idx), 1)
        if verbose:
            print(f"  [cd] r{rnd+1}/{max_rounds} accept={rnd_accepts}/{len(mov_idx)} "
                  f"({100*accept_pct:.1f}%) proxy={best_full:.6f} "
                  f"t={time.time() - rnd_t0:.1f}s", flush=True)
        if time_cap is not None and time.time() - t0 > time_cap:
            break
        if accept_pct < early_stop_accept_pct:
            consec_low += 1
            if consec_low >= early_stop_consec:
                break
        else:
            consec_low = 0

    if verbose:
        print(f"  [cd] done {n_accepts_total} accepts "
              f"{init_full:.6f} -> {best_full:.6f} "
              f"({100*(best_full-init_full)/init_full:+.2f}%) in {time.time()-t0:.1f}s",
              flush=True)
    out = placement.clone()
    out[:n_macros] = torch.from_numpy(best_pos.astype(np.float32))
    return out
