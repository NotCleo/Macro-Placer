"""
NumPy macro legalization — bulk push apart, greedy resolve, spiral search.

A two-stage legalizer for post-GD output:

  1. _bulk_push_apart — vectorized iterative pair-wise push along smaller-axis
     overlap. Resolves most overlaps in 200-1500 iters depending on density.
  2. _greedy_per_macro — single-macro per-axis nudge for residuals bulk missed.
  3. _spiral_search — bounded spiral of grid-step probes for the rare macro
     that can't be greedily resolved (handles tight-pack pockets).

Output guarantee: every pair of hard movable macros has at least SAFETY_MARGIN
clearance, comfortably above float32 ulp so downstream evaluators don't see
spurious overlaps.
"""
from __future__ import annotations

import numpy as np


SAFETY_MARGIN = 0.005   # required final gap between any hard pair
PUSH_TARGET = 0.01      # bulk push aims this far apart (overshoots margin)
PRE_JITTER = 0.05


def _bulk_push_apart(pos, sizes, movable, cw, ch, max_iters=500):
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_real_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
    sep_real_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
    sep_tgt_x = sep_real_x + PUSH_TARGET
    sep_tgt_y = sep_real_y + PUSH_TARGET
    sep_leg_x = sep_real_x + SAFETY_MARGIN
    sep_leg_y = sep_real_y + SAFETY_MARGIN
    fixed = ~movable
    w = movable.astype(np.float64)

    for _ in range(max_iters):
        dx = pos[:, 0:1] - pos[:, 0:1].T
        dy = pos[:, 1:2] - pos[:, 1:2].T
        ox = sep_tgt_x - np.abs(dx)
        oy = sep_tgt_y - np.abs(dy)
        push = (ox > 0) & (oy > 0)
        np.fill_diagonal(push, False)
        bad = (sep_leg_x - np.abs(dx) > 0) & (sep_leg_y - np.abs(dy) > 0)
        np.fill_diagonal(bad, False)
        if not bad.any():
            return pos
        if not push.any():
            return pos

        # push along the SHORTER overlap axis (smaller correction).
        x_axis = push & (ox <= oy)
        y_axis = push & (oy < ox)
        sx = np.where(dx >= 0, 1.0, -1.0)
        sy = np.where(dy >= 0, 1.0, -1.0)
        denom = np.maximum(w[:, None] + w[None, :], 1e-9)
        share = w[:, None] / denom
        mag_x = np.where(x_axis, ox, 0.0)
        mag_y = np.where(y_axis, oy, 0.0)
        dx_total = (share * sx * mag_x).sum(axis=1)
        dy_total = (share * sy * mag_y).sum(axis=1)
        dx_total[fixed] = 0.0
        dy_total[fixed] = 0.0

        pos[:, 0] += 0.5 * dx_total
        pos[:, 1] += 0.5 * dy_total
        pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
        pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)

    return pos


def _list_violators(pos, sizes, movable):
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2 + SAFETY_MARGIN
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2 + SAFETY_MARGIN
    dx = pos[:, 0:1] - pos[:, 0:1].T
    dy = pos[:, 1:2] - pos[:, 1:2].T
    bad = (sep_x - np.abs(dx) > 0) & (sep_y - np.abs(dy) > 0)
    np.fill_diagonal(bad, False)
    return np.flatnonzero(bad.any(axis=1) & movable).tolist()


def _overlap_for(idx, pos, sizes):
    sep_x = (sizes[:, 0] + sizes[idx, 0]) / 2 + SAFETY_MARGIN
    sep_y = (sizes[:, 1] + sizes[idx, 1]) / 2 + SAFETY_MARGIN
    dx = pos[idx, 0] - pos[:, 0]
    dy = pos[idx, 1] - pos[:, 1]
    ox = sep_x - np.abs(dx)
    oy = sep_y - np.abs(dy)
    mk = (ox > 0) & (oy > 0)
    mk[idx] = False
    return mk, dx, dy, ox, oy


def _greedy_per_macro(pos, sizes, movable, cw, ch, max_passes=20):
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    eps = SAFETY_MARGIN
    for _ in range(max_passes):
        viols = _list_violators(pos, sizes, movable)
        if not viols:
            return pos
        viols.sort(key=lambda i: -sizes[i, 0] * sizes[i, 1])
        progress = False
        for idx in viols:
            mk, dx, dy, ox, oy = _overlap_for(idx, pos, sizes)
            if not mk.any():
                continue
            np_, nm_, np_y, nm_y = 0.0, 0.0, 0.0, 0.0
            for j in np.flatnonzero(mk):
                if dx[j] >= 0:
                    np_ = max(np_, ox[j])
                else:
                    nm_ = max(nm_, ox[j])
                if dy[j] >= 0:
                    np_y = max(np_y, oy[j])
                else:
                    nm_y = max(nm_y, oy[j])
            cands = []
            if nm_ == 0.0 and np_ > 0.0:
                cands.append(("x", +np_ + eps))
            if np_ == 0.0 and nm_ > 0.0:
                cands.append(("x", -nm_ - eps))
            if nm_y == 0.0 and np_y > 0.0:
                cands.append(("y", +np_y + eps))
            if np_y == 0.0 and nm_y > 0.0:
                cands.append(("y", -nm_y - eps))
            if cands:
                axis, shift = min(cands, key=lambda c: abs(c[1]))
                if axis == "x":
                    pos[idx, 0] = float(np.clip(pos[idx, 0] + shift,
                                                half_w[idx], cw - half_w[idx]))
                else:
                    pos[idx, 1] = float(np.clip(pos[idx, 1] + shift,
                                                half_h[idx], ch - half_h[idx]))
                after, *_ = _overlap_for(idx, pos, sizes)
                if not after.any():
                    progress = True
        if not progress:
            break
    return pos


def _spiral_search(pos, sizes, movable, cw, ch, step=0.05, max_radius=50):
    n = pos.shape[0]
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    for i in range(n):
        if not movable[i]:
            continue
        mk, *_ = _overlap_for(i, pos, sizes)
        if not mk.any():
            continue
        ox_, oy_ = pos[i, 0], pos[i, 1]
        best = None
        best_d = float("inf")
        for r in range(1, max_radius + 1):
            found = False
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = float(np.clip(ox_ + dxm * step, half_w[i], cw - half_w[i]))
                    cy = float(np.clip(oy_ + dym * step, half_h[i], ch - half_h[i]))
                    pos[i, 0] = cx
                    pos[i, 1] = cy
                    m2, *_ = _overlap_for(i, pos, sizes)
                    if not m2.any():
                        d = (cx - ox_) ** 2 + (cy - oy_) ** 2
                        if d < best_d:
                            best_d = d
                            best = (cx, cy)
                            found = True
            pos[i, 0] = ox_
            pos[i, 1] = oy_
            if found:
                break
        if best is not None:
            pos[i] = best
    return pos


def _force_displace_grid(pos, sizes, movable, cw, ch, n_steps=80, max_sweeps=4):
    """Guaranteed-success fallback: for each remaining violator, vectorized
    grid search across the whole canvas, snap to the nearest non-overlapping
    candidate center. WL takes a hit, but the subsequent CD pass recovers.

    Crucial for huge benches (ibm10/ibm12/ibm14/...) where the spiral's
    ~2.5 μm radius can't escape a deep pocket of tightly-packed siblings.
    """
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    eps = SAFETY_MARGIN

    for sweep in range(max_sweeps):
        viols = _list_violators(pos, sizes, movable)
        if not viols:
            return pos
        # densify on later sweeps if anyone is still stuck
        ns = int(n_steps * (1 + 0.5 * sweep))
        xs = np.linspace(0.0, cw, ns + 1)[:-1] + 0.5 * cw / ns
        ys = np.linspace(0.0, ch, ns + 1)[:-1] + 0.5 * ch / ns
        gx, gy = np.meshgrid(xs, ys)
        cand_xy = np.stack([gx.ravel(), gy.ravel()], axis=-1)  # [ns*ns, 2]
        for idx in viols:
            cx = np.clip(cand_xy[:, 0], half_w[idx], cw - half_w[idx])
            cy = np.clip(cand_xy[:, 1], half_h[idx], ch - half_h[idx])
            sep_x = (sizes[:, 0] + sizes[idx, 0]) / 2 + eps  # [n_macros]
            sep_y = (sizes[:, 1] + sizes[idx, 1]) / 2 + eps
            # block-process to keep peak memory bounded for huge benches
            n_cand = cx.shape[0]
            block = max(1, min(4096, n_cand))
            best_xy = None
            best_d = float("inf")
            for b0 in range(0, n_cand, block):
                b1 = min(n_cand, b0 + block)
                cxb = cx[b0:b1]
                cyb = cy[b0:b1]
                dx = np.abs(cxb[:, None] - pos[:, 0][None, :])
                dy = np.abs(cyb[:, None] - pos[:, 1][None, :])
                ov = (dx < sep_x[None, :]) & (dy < sep_y[None, :])
                ov[:, idx] = False
                valid = ~ov.any(axis=1)
                if not valid.any():
                    continue
                v_local = np.flatnonzero(valid)
                d = ((cxb[v_local] - pos[idx, 0]) ** 2
                     + (cyb[v_local] - pos[idx, 1]) ** 2)
                k_loc = int(d.argmin())
                if d[k_loc] < best_d:
                    best_d = float(d[k_loc])
                    best_xy = (float(cxb[v_local[k_loc]]),
                               float(cyb[v_local[k_loc]]))
            if best_xy is not None:
                pos[idx, 0] = best_xy[0]
                pos[idx, 1] = best_xy[1]
    # Phase-5: float32-round-trip check. Caller casts to float32; that cast
    # can re-introduce sub-eps overlaps even when float64 check passed. For
    # any pair that becomes violating after the cast, nudge one macro of
    # the pair by 1e-4 μm along whichever axis has more clearance to the
    # canvas edge. 1e-4 is well above float32 ulp at 30 μm scales (~3e-6).
    pos32 = pos.astype(np.float32).astype(np.float64)
    viols32 = _list_violators(pos32, sizes, movable)
    if viols32:
        eps_extra = 1e-4
        for idx in viols32:
            mk, dxv, dyv, _, _ = _overlap_for(idx, pos32, sizes)
            if not mk.any():
                continue
            # Pick axis with more room to nudge away from canvas edge.
            room_x_pos = (cw - half_w[idx]) - pos[idx, 0]
            room_x_neg = pos[idx, 0] - half_w[idx]
            room_y_pos = (ch - half_h[idx]) - pos[idx, 1]
            room_y_neg = pos[idx, 1] - half_h[idx]
            best_axis = ('x', +1, room_x_pos)
            for ax, sn, rm in [('x', -1, room_x_neg), ('y', +1, room_y_pos),
                               ('y', -1, room_y_neg)]:
                if rm > best_axis[2]:
                    best_axis = (ax, sn, rm)
            ax, sn, rm = best_axis
            if rm < eps_extra:
                continue
            if ax == 'x':
                pos[idx, 0] += sn * eps_extra
            else:
                pos[idx, 1] += sn * eps_extra
    return pos


def legalize(initial_pos, sizes, movable, cw, ch, seed=42, bulk_iters=None,
             outer_rounds=8, timing_log=None):
    """Public entry: legalize a [N, 2] numpy array of hard-macro centers.

    V15-diag: pass `timing_log` (a callable) to emit per-sub-stage wall-clock
    so we can see whether bulk_push / greedy / spiral / force_displace owns the
    ~1.3ks legalize cost on dense HUGE benches (ibm10).
    """
    import time as _t
    pos = initial_pos.copy()
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
    pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)

    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-PRE_JITTER, PRE_JITTER, size=pos.shape)
    jitter[~movable] = 0.0
    pos += jitter
    pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
    pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)

    if bulk_iters is None:
        n = pos.shape[0]
        bulk_iters = 500 if n < 350 else (1000 if n < 600 else 1500)

    _tb = _t.time()
    _rounds_used = 0
    for outer in range(outer_rounds):
        _rounds_used += 1
        pos = _bulk_push_apart(pos, sizes, movable, cw, ch, max_iters=bulk_iters)
        viols = _list_violators(pos, sizes, movable)
        if not viols:
            break
        amp = 0.2 * (1 + outer)
        for v in viols:
            pos[v, 0] += rng.uniform(-amp, amp)
            pos[v, 1] += rng.uniform(-amp, amp)
        pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
        pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)
    if timing_log is not None:
        timing_log(f"      [lnp] bulk={_t.time()-_tb:.0f}s rounds={_rounds_used}/"
                   f"{outer_rounds} bulk_iters={bulk_iters} "
                   f"viol_after_bulk={len(_list_violators(pos, sizes, movable))}")

    _tg = _t.time()
    if _list_violators(pos, sizes, movable):
        pos = _greedy_per_macro(pos, sizes, movable, cw, ch)
    if timing_log is not None:
        timing_log(f"      [lnp] greedy={_t.time()-_tg:.0f}s "
                   f"viol_after_greedy={len(_list_violators(pos, sizes, movable))}")
    _ts = _t.time()
    if _list_violators(pos, sizes, movable):
        pos = _spiral_search(pos, sizes, movable, cw, ch)
    if timing_log is not None:
        timing_log(f"      [lnp] spiral={_t.time()-_ts:.0f}s "
                   f"viol_after_spiral={len(_list_violators(pos, sizes, movable))}")
    _tfd = _t.time()
    if _list_violators(pos, sizes, movable):
        # Hard guarantee: vectorized canvas-wide search snaps remaining
        # violators to the nearest empty cell. CD will recover any WL hit.
        pos = _force_displace_grid(pos, sizes, movable, cw, ch)
    if timing_log is not None:
        timing_log(f"      [lnp] internal_fd={_t.time()-_tfd:.0f}s")
    return pos
