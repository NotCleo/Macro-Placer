"""V13-rev4 — Late Acceptance Hill Climbing (LAHC) polish.

LAHC (Burke & Bykov, 2008) accepts a move iff cand < cur OR cand < history[i
mod L], where `history` is a queue of the last L past-cost values. This lets
the search slide past micro-plateaus that V13's strict-improvement refine
loop can't cross, with NO temperature parameter and NO random acceptance of
strictly-worse moves.

Idea credit: graph_grad/placer2.py:lahc_polish.

History of this file's bugs (so we don't repeat them):

  v3:  Partial proxy (WL+0.5*D, no cong) in queue, full proxy only every
       200 iters. On cong-dominated benches (ibm08 cong ~ 65%), the
       position drifted partial-good full-bad; 692s of LAHC produced
       zero improvement.
  v3b: Switched to full proxy via FastProxy.proxy_full(pos), but pos had
       shape (n_macros, 2) — FastProxy expects pos_aug shape
       (n_macros + n_ports, 2). Crashed immediately with IndexError.
  v3c: Fixed pos_aug shape. Used FastProxy.proxy_full() which has a Python
       loop in density_cost; ~30-50ms/iter on medium benches gave only
       4000-7000 iters in 220s — too few accepts to push the basin.
  v4-a: First v4 used a 3x3 spatial-grid neighborhood for hard-macro
       overlap check (`_no_overlap_local`). On ibm08 with nbr_cells=17,
       cell size ~0.059·cw while macros span 1.5-2 cells; pairs with
       centers 1.6-1.8 cells apart overlapped but the 3x3 missed them
       because the neighbor's CELL was 2 cells away. Result: 13
       hard-macro overlaps leaked through, invalidating the placement
       (proxy dropped 1.0865→1.0465 but the harness rejected it).

This v4 uses `IncrementalProxy.proxy_full_cached()` (script4.py) for the
inner loop and a vectorized O(n_hard) brute-force overlap check that is
geometry-correct regardless of macro size distribution. A final legality
sweep verifies best_pos before returning — if a bug ever sneaks an
overlap through again, we return the original input placement instead
of corrupting the pipeline output.

The drift-reset trigger (cur > best * ratio) snaps state back to best_pos
via `inc.reset_to_positions(best_pos)` — one O(n_nets + n_macros) call
instead of replaying every accept.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from script4 import IncrementalProxy


# ─────────────────────────────────────────────────────────────────────────
# Vectorized overlap checks (geometry-correct, no spatial-grid bugs)
# ─────────────────────────────────────────────────────────────────────────


def _proposes_overlap(pos_aug: np.ndarray, n_hard: int,
                      half_w: np.ndarray, half_h: np.ndarray,
                      i: int, nx: float, ny: float,
                      tol: float = 1e-9) -> bool:
    """True iff macro `i` at proposed (nx, ny) would overlap any other
    hard macro by more than `tol` (numerical noise tolerance) on BOTH axes.

    Vectorized — costs ~3-5us for n_hard up to ~1500. Cheaper than the
    proxy_full_cached call that follows, so checking every proposal is
    free in practice.
    """
    if n_hard <= 1:
        return False
    hw_i = float(half_w[i])
    hh_i = float(half_h[i])
    cx = pos_aug[:n_hard, 0]
    cy = pos_aug[:n_hard, 1]
    sep_x = np.abs(cx - nx) - (half_w[:n_hard] + hw_i)
    sep_y = np.abs(cy - ny) - (half_h[:n_hard] + hh_i)
    overlap = (sep_x < -tol) & (sep_y < -tol)
    overlap[i] = False
    return bool(overlap.any())


def _count_overlap_pairs(pos_aug: np.ndarray, n_hard: int,
                         half_w: np.ndarray, half_h: np.ndarray,
                         tol: float = 1e-9) -> int:
    """Count pairs of hard macros that overlap by more than `tol`.

    Used as a final safety check before returning best_pos. O(n_hard^2)
    but called once — for n_hard up to 1500 that's ~2M ops, ~5ms.
    """
    if n_hard <= 1:
        return 0
    cx = pos_aug[:n_hard, 0]
    cy = pos_aug[:n_hard, 1]
    hw = half_w[:n_hard]
    hh = half_h[:n_hard]
    sep_x = np.abs(cx[:, None] - cx[None, :]) - (hw[:, None] + hw[None, :])
    sep_y = np.abs(cy[:, None] - cy[None, :]) - (hh[:, None] + hh[None, :])
    overlap = (sep_x < -tol) & (sep_y < -tol)
    np.fill_diagonal(overlap, False)
    return int(overlap.sum() // 2)


def _soft_centroid(macro_to_pins, net_pin_xy, net_weights,
                   i: int):
    """Net-peer centroid using IncrementalProxy's cached `net_pin_xy`.

    For each net touching macro i, take the mean of the OTHER pins'
    world positions. The cached pin_xy already includes port world
    positions, so we don't need to handle port indices separately.
    """
    cx = 0.0
    cy = 0.0
    w_tot = 0.0
    pins = macro_to_pins[i]
    for (ni, pi) in pins:
        xy = net_pin_xy[ni]
        P = xy.shape[0]
        if P < 2:
            continue
        sum_x = float(xy[:, 0].sum()) - float(xy[pi, 0])
        sum_y = float(xy[:, 1].sum()) - float(xy[pi, 1])
        w_n = net_weights[ni]
        cx += w_n * sum_x / (P - 1)
        cy += w_n * sum_y / (P - 1)
        w_tot += w_n
    if w_tot > 0:
        return cx / w_tot, cy / w_tot
    return None


def _move_radius_scale(t_elapsed: float, t_total: float,
                       lo: float = 0.4, hi: float = 1.0) -> float:
    """Cosine decay from `hi` to `lo` over [0, t_total].

    V13-rev5 edit F: LAHC explores broadly early (radius_scale ≈ hi),
    finetunes late (radius_scale ≈ lo). Small radius at end is the
    classic SA cooling pattern — local moves dominate as the basin
    is nailed down.
    """
    if t_total <= 0:
        return hi
    frac = min(1.0, max(0.0, t_elapsed / t_total))
    return lo + (hi - lo) * 0.5 * (1.0 + math.cos(math.pi * frac))


def _do_ils_kick(inc, rng, n_hard: int, n_macros: int,
                 cw: float, ch: float,
                 half_w: np.ndarray, half_h: np.ndarray,
                 movable_soft: np.ndarray, movable_hard: np.ndarray,
                 best_pos: np.ndarray,
                 rx_soft: float, ry_soft: float,
                 rx_hard: float, ry_hard: float,
                 soft_kick_frac: float = 0.08,
                 hard_kick_frac: float = 0.04,
                 kick_radius_mult: float = 3.0) -> float:
    """ILS kick — perturb a random subset of macros from `best_pos`.

    V13-rev5 edit E: LAHC alone gets stuck in narrow basins at deep
    convergence (ibm08 V13-rev4: LAHC eventually plateaus after ~150s
    of inner loop). The kick reseeds from best with a coarse jitter
    on ~5% of macros, opening up a new neighborhood that LAHC then
    polishes. graph_grad/placer2.py uses a similar perturbation
    pattern. Returns the proxy after the kick.

    Hard moves still respect _proposes_overlap so legality is
    preserved. Soft moves are unconstrained (can stack — that's
    soft-macro semantics).
    """
    # Restore to best.
    inc.reset_to_positions(best_pos)

    # Soft jitters.
    if movable_soft.size > 0:
        n_kick = max(1, int(soft_kick_frac * movable_soft.size))
        idxs = rng.choice(movable_soft, size=min(n_kick, movable_soft.size),
                          replace=False)
        kx = kick_radius_mult * rx_soft
        ky = kick_radius_mult * ry_soft
        for i in idxs:
            i = int(i)
            ox = float(inc.pos_aug[i, 0])
            oy = float(inc.pos_aug[i, 1])
            nx = ox + float(rng.uniform(-kx, kx))
            ny = oy + float(rng.uniform(-ky, ky))
            if nx < half_w[i]:
                nx = half_w[i]
            elif nx > cw - half_w[i]:
                nx = cw - half_w[i]
            if ny < half_h[i]:
                ny = half_h[i]
            elif ny > ch - half_h[i]:
                ny = ch - half_h[i]
            inc.try_move(i, nx, ny)
            inc.commit_move()

    # Hard jitters with overlap check.
    if movable_hard.size > 0:
        n_kick = max(0, int(hard_kick_frac * movable_hard.size))
        if n_kick > 0:
            idxs = rng.choice(movable_hard,
                              size=min(n_kick, movable_hard.size),
                              replace=False)
            kx = kick_radius_mult * rx_hard
            ky = kick_radius_mult * ry_hard
            for i in idxs:
                i = int(i)
                ox = float(inc.pos_aug[i, 0])
                oy = float(inc.pos_aug[i, 1])
                nx = ox + float(rng.uniform(-kx, kx))
                ny = oy + float(rng.uniform(-ky, ky))
                if nx < half_w[i]:
                    nx = half_w[i]
                elif nx > cw - half_w[i]:
                    nx = cw - half_w[i]
                if ny < half_h[i]:
                    ny = half_h[i]
                elif ny > ch - half_h[i]:
                    ny = ch - half_h[i]
                if _proposes_overlap(inc.pos_aug, n_hard, half_w, half_h,
                                     i, nx, ny):
                    continue
                inc.try_move(i, nx, ny)
                inc.commit_move()

    return float(inc.proxy_full_cached()[0])


# ─────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────


def lahc_polish(placement: torch.Tensor, benchmark: Benchmark, plc,
                list_len: int = 80,
                time_cap: float = 300.0,
                hard_move_frac: float = 0.025,
                soft_move_frac: float = 0.015,
                soft_prob: float = 0.55,
                soft_centroid_prob: float = 0.40,
                drift_reset_ratio: float = 1.002,
                drift_check_every: int = 100,
                ils_kick_interval: float = 50.0,
                ils_kick_radius_mult: float = 3.0,
                ils_min_remaining: float = 30.0,
                radius_lo: float = 0.4,
                radius_hi: float = 1.0,
                seed: int = 42,
                verbose: bool = False) -> torch.Tensor:
    """LAHC polish using IncrementalProxy.proxy_full_cached() in the inner loop.

    Args:
        placement: [num_macros, 2] tensor with current macro positions.
        list_len: LAHC history queue length.
        time_cap: budget in seconds.
        hard_move_frac: hard-macro jitter radius as fraction of canvas dim.
        soft_move_frac: soft-macro jitter radius.
        soft_prob: probability of choosing a soft action vs hard.
        soft_centroid_prob: within a soft action, probability of
            net-centroid pull vs random jitter.
        drift_reset_ratio: if cur > best * this ratio, reset state to
            best position. Prevents the partial-good full-bad drift
            that broke v3 LAHC.
        drift_check_every: drift-check cadence (iters).
        ils_kick_interval: seconds without improving best before
            triggering an ILS kick (V13-rev5 edit E). Set <=0 to disable.
        ils_kick_radius_mult: kick move-radius as multiple of normal
            (~3.0 means a kick moves macros 3× further than a normal
            propose, opening a new neighborhood).
        ils_min_remaining: don't kick if less than this many seconds of
            budget remain (avoids kicking right before time-out, which
            leaves the perturbed state uncommitted to best).
        radius_lo, radius_hi: cosine decay endpoints for move radius
            (V13-rev5 edit F). Multiplied into the per-iter rx/ry.
        seed: RNG seed.
        verbose: print periodic progress.

    Returns: [num_macros, 2] tensor — best (by full proxy) position seen.
        If a bug ever introduces overlaps in best_pos despite the
        per-move check, the original input placement is returned
        instead (so the harness still sees a legal placement).
    """
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    n_hard = int(benchmark.num_hard_macros)
    n_macros = int(benchmark.num_macros)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5
    movable = benchmark.get_movable_mask().numpy().astype(bool)

    placement_np = placement.detach().cpu().numpy().astype(np.float64)
    if placement_np.shape != (n_macros, 2):
        raise ValueError(
            f"placement has shape {placement_np.shape}, expected ({n_macros}, 2)"
        )

    # Build IncrementalProxy and seed it from the input placement.
    inc = IncrementalProxy(benchmark, plc)
    inc.reset_to_positions(placement_np)

    # Initial proxy.
    full_t = inc.proxy_full_cached()
    cur = float(full_t[0])
    if not math.isfinite(cur):
        raise ValueError(f"initial proxy not finite: {cur}")
    best = cur
    best_pos = inc.pos_aug[:n_macros].copy()
    initial = cur
    # Verify the input placement is legal — if not, every move would fail
    # the overlap check and we'd just spin. Use the input as the baseline
    # tolerance: any overlap above this is "introduced by LAHC" and rejected.
    input_overlap = _count_overlap_pairs(inc.pos_aug, n_hard, half_w, half_h)
    if input_overlap > 0:
        # The input is already invalid; skip LAHC and return it unchanged.
        if verbose:
            print(
                f"  [LAHC] input has {input_overlap} hard-macro overlaps — skipping",
                flush=True,
            )
        return placement.clone()

    history = [cur] * int(list_len)

    rx_hard = hard_move_frac * cw
    ry_hard = hard_move_frac * ch
    rx_soft = soft_move_frac * cw
    ry_soft = soft_move_frac * ch

    movable_hard = np.where(movable[:n_hard])[0]
    movable_soft = np.where(movable[n_hard:])[0] + n_hard
    has_soft = movable_soft.size > 0
    has_hard = movable_hard.size > 0
    if not has_soft and not has_hard:
        return torch.from_numpy(best_pos).float()
    if not has_soft:
        soft_prob = 0.0
    if not has_hard:
        soft_prob = 1.0

    t0 = time.time()
    it = 0
    accepts = 0
    resets = 0
    hard_rejected_overlap = 0
    kicks = 0
    last_log = t0
    # V13-rev5 edit E: track when `best` last improved. If stalled for
    # `ils_kick_interval` seconds and budget allows, do an ILS kick.
    last_best_improve_t = t0
    # After a kick, `cur` is intentionally worse than `best`, so drift-reset
    # would immediately snap it back and undo the kick. Suspend drift-reset
    # until this iter index is reached.
    drift_suspend_until_it = -1

    while time.time() - t0 < time_cap:
        now = time.time()
        elapsed = now - t0
        remaining = time_cap - elapsed

        if verbose and now - last_log > 30.0:
            rate = it / max(elapsed, 1e-3)
            print(
                f"  [LAHC] t={elapsed:.0f}s it={it} ({rate:.0f}/s) "
                f"acc={accepts} resets={resets} kicks={kicks} "
                f"ovr_rej={hard_rejected_overlap} "
                f"cur={cur:.4f} best={best:.4f}",
                flush=True,
            )
            last_log = now

        # V13-rev5 edit E: ILS kick on best-stall.
        # Only fire if (a) interval is enabled, (b) stalled for long enough,
        # (c) enough budget remains for the kick to be polished back.
        if (ils_kick_interval > 0
                and (now - last_best_improve_t) > ils_kick_interval
                and remaining > ils_min_remaining):
            cur = _do_ils_kick(
                inc, rng, n_hard, n_macros, cw, ch, half_w, half_h,
                movable_soft, movable_hard, best_pos,
                rx_soft, ry_soft, rx_hard, ry_hard,
                soft_kick_frac=0.08, hard_kick_frac=0.04,
                kick_radius_mult=ils_kick_radius_mult,
            )
            history = [cur] * int(list_len)
            last_best_improve_t = now  # restart stall clock
            # Suspend drift-reset for the next ~500 iters so LAHC can
            # explore from the kicked state without snapping back.
            drift_suspend_until_it = it + 500
            kicks += 1
            it += 1
            continue

        # Drift-reset — bulk restore via inc.reset_to_positions().
        # Suspended right after a kick (see drift_suspend_until_it above).
        if (it > 0 and it % int(drift_check_every) == 0
                and it >= drift_suspend_until_it):
            if cur > best * drift_reset_ratio:
                inc.reset_to_positions(best_pos)
                cur = best
                history = [cur] * int(list_len)
                resets += 1

        # V13-rev5 edit F: cosine-decay move radius from 1.0 → radius_lo.
        radius_scale = _move_radius_scale(elapsed, time_cap, radius_lo, radius_hi)
        rx_hard_eff = rx_hard * radius_scale
        ry_hard_eff = ry_hard * radius_scale
        rx_soft_eff = rx_soft * radius_scale
        ry_soft_eff = ry_soft * radius_scale

        action_soft = (rng.random() < soft_prob)

        if action_soft and movable_soft.size:
            i = int(movable_soft[int(rng.integers(0, movable_soft.size))])
            ox = float(inc.pos_aug[i, 0])
            oy = float(inc.pos_aug[i, 1])
            if rng.random() < soft_centroid_prob:
                tgt = _soft_centroid(inc.macro_to_pins, inc.net_pin_xy,
                                     inc.net_weights, i)
                if tgt is None:
                    nx = ox + float(rng.uniform(-rx_soft_eff, rx_soft_eff))
                    ny = oy + float(rng.uniform(-ry_soft_eff, ry_soft_eff))
                else:
                    cx, cy = tgt
                    f = float(rng.uniform(0.05, 0.4))
                    nx = ox + f * (cx - ox)
                    ny = oy + f * (cy - oy)
            else:
                nx = ox + float(rng.uniform(-rx_soft_eff, rx_soft_eff))
                ny = oy + float(rng.uniform(-ry_soft_eff, ry_soft_eff))
            if nx < half_w[i]:
                nx = half_w[i]
            elif nx > cw - half_w[i]:
                nx = cw - half_w[i]
            if ny < half_h[i]:
                ny = half_h[i]
            elif ny > ch - half_h[i]:
                ny = ch - half_h[i]
            inc.try_move(i, nx, ny)
            full_t = inc.proxy_full_cached()
            cand = float(full_t[0])
            idx_h = it % int(list_len)
            if cand < cur or cand < history[idx_h]:
                inc.commit_move()
                history[idx_h] = cand
                cur = cand
                accepts += 1
                if cand < best:
                    best = cand
                    best_pos = inc.pos_aug[:n_macros].copy()
                    last_best_improve_t = time.time()
            else:
                inc.revert_move()
        elif movable_hard.size:
            i = int(movable_hard[int(rng.integers(0, movable_hard.size))])
            ox = float(inc.pos_aug[i, 0])
            oy = float(inc.pos_aug[i, 1])
            nx = ox + float(rng.uniform(-rx_hard_eff, rx_hard_eff))
            ny = oy + float(rng.uniform(-ry_hard_eff, ry_hard_eff))
            if nx < half_w[i]:
                nx = half_w[i]
            elif nx > cw - half_w[i]:
                nx = cw - half_w[i]
            if ny < half_h[i]:
                ny = half_h[i]
            elif ny > ch - half_h[i]:
                ny = ch - half_h[i]
            # Vectorized geometry-correct overlap check against all
            # other hard macros — replaces the buggy 3x3 spatial-grid
            # check in v4-a that missed pairs with stride > 1 cell.
            if _proposes_overlap(inc.pos_aug, n_hard, half_w, half_h, i, nx, ny):
                hard_rejected_overlap += 1
                it += 1
                continue
            inc.try_move(i, nx, ny)
            full_t = inc.proxy_full_cached()
            cand = float(full_t[0])
            idx_h = it % int(list_len)
            if cand < cur or cand < history[idx_h]:
                inc.commit_move()
                history[idx_h] = cand
                cur = cand
                accepts += 1
                if cand < best:
                    best = cand
                    best_pos = inc.pos_aug[:n_macros].copy()
                    last_best_improve_t = time.time()
            else:
                inc.revert_move()

        it += 1

    # Final safety net: brute-force verify best_pos has no introduced
    # hard-macro overlaps. If a bug ever bypasses _proposes_overlap, we
    # fall back to the input placement so the pipeline isn't corrupted.
    aug_check = inc.pos_aug.copy()
    aug_check[:n_macros] = best_pos
    final_overlap = _count_overlap_pairs(aug_check, n_hard, half_w, half_h)
    if final_overlap > 0:
        if verbose:
            print(
                f"  [LAHC] FINAL CHECK FAILED: {final_overlap} hard-macro "
                f"overlap pairs in best_pos — returning input unchanged",
                flush=True,
            )
        return placement.clone()

    if verbose:
        gain = initial - best
        print(
            f"  [LAHC] done t={time.time() - t0:.0f}s it={it} "
            f"acc={accepts} resets={resets} kicks={kicks} "
            f"ovr_rej={hard_rejected_overlap} "
            f"initial={initial:.4f} best={best:.4f} gain={gain:.4f}",
            flush=True,
        )

    return torch.from_numpy(best_pos).float()
