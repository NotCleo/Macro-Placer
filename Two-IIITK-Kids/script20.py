"""V13-rev7 — Laplacian soft-resolve.

Idea credit: vmallela_v7/_soft_laplacian (Tsay & Kuh 1991 clique-model
quadratic HPWL). The math is independently re-implemented against
V13's IncrementalProxy, not copied from the competitor's IncrementalEvaluator.

# What it does

For SOFT macros (no overlap, ~zero footprint, contribute only to HPWL
+ routing), the HPWL surrogate sum_n w_n * (max - min) is convex in
soft positions when hard positions are held fixed. The clique-model
quadratic surrogate

    HPWL_quad(x) = sum_n (w_n / (k_n - 1)) * sum_{(i,j) in net n} |p_i - p_j|^2

has a closed-form global minimum via the Laplacian linear system:

    L_ff @ x_f = -L_fc @ x_c + (port contribution)

where:
  - L is the netlist clique-model Laplacian (sparse, symmetric, PSD)
  - x_f = free (soft) macro centers; x_c = constrained (hard) macro centers
  - port pins are treated as fixed boundary points (contribute to RHS)

Solved with scipy.sparse.linalg.cg (conjugate gradient) in O(nnz)
per CG iter, converges in tens of iters for our sparsity.

# Why not bulk-apply the Laplacian solution

The closed-form solution is the HPWL-quadratic minimum but it ignores
density and congestion. On ICCAD04 soft macros (small non-zero
footprint) the solution can collapse softs into hot density cells,
making proxy WORSE. The fix is per-soft line search with full-proxy
gating: each soft moves toward its target only if the full proxy
improves. By construction this cannot regress.

# Expected gain

Per the competitor's README: -0.0005 to -0.005 per bench, ~30s wall.
Validated empirically on their 17-bench sweep.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from macro_place.benchmark import Benchmark
from script4 import IncrementalProxy


def _build_clique_laplacian(
    benchmark: Benchmark,
    plc,
    hard_pos: np.ndarray,
    port_pos: np.ndarray,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Build the soft-restricted clique-model Laplacian.

    Returns:
        L_ff: (n_soft, n_soft) symmetric PSD sparse matrix.
        b_x, b_y: (n_soft,) right-hand-side vectors.

    Math notes:
      Hard macros and ports are treated as fixed boundary points.
      Pin offsets are applied to hard macro positions to get the true
      pin world coordinates; soft macros have no per-pin offsets in V13
      (single pin at macro center), so soft positions are the macro
      centers themselves.

      For each net of k pins with weight w_n, the clique pair weight is
      w_pair = w_n / (k - 1) (so total clique edge weight = w_n * k / 2,
      matching standard Tsay-Kuh derivation).

      For each pair (i, j) on the net:
        - if both soft → off-diagonal L entry (-w_pair) + diag (+w_pair each)
        - if one soft, other fixed → diag (+w_pair) + RHS (+w_pair * fixed_pos)
        - if both fixed → constant, drops out
    """
    n_macros = int(benchmark.num_macros)
    n_hard = int(benchmark.num_hard_macros)
    n_soft = n_macros - n_hard
    if n_soft <= 0:
        return sp.csr_matrix((0, 0), dtype=np.float64), \
               np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    pin_off = benchmark.macro_pin_offsets

    # Driver weights (mirrors IncrementalProxy.__init__)
    drivers = list(plc.nets.keys()) if plc is not None else []

    # Walk nets, accumulate L (sparse coords) and b.
    L_rows: list[int] = []
    L_cols: list[int] = []
    L_vals: list[float] = []
    L_diag = np.zeros(n_soft, dtype=np.float64)
    b_x = np.zeros(n_soft, dtype=np.float64)
    b_y = np.zeros(n_soft, dtype=np.float64)

    net_index_used = 0
    for net_pin_nodes_idx, net in enumerate(benchmark.net_pin_nodes):
        if net.shape[0] < 2:
            continue
        # Net weight matches IncrementalProxy logic.
        if (plc is not None
                and net_index_used < len(drivers)):
            drv = drivers[net_index_used]
            d_idx = plc.mod_name_to_indices.get(drv)
            if d_idx is not None:
                d_pin = plc.modules_w_pins[d_idx]
                w_n = float(d_pin.get_weight()) if hasattr(d_pin, "get_weight") else 1.0
            else:
                w_n = 1.0
        else:
            w_n = 1.0
        net_index_used += 1

        soft_idxs: list[int] = []
        fixed_x: list[float] = []
        fixed_y: list[float] = []
        net_np = net.long().numpy().astype(np.int64)
        for p in range(net_np.shape[0]):
            owner = int(net_np[p, 0])
            slot = int(net_np[p, 1])
            if 0 <= owner < n_hard:
                # Hard macro pin: world = hard center + pin offset.
                fx = float(hard_pos[owner, 0])
                fy = float(hard_pos[owner, 1])
                if slot < pin_off[owner].shape[0]:
                    fx += float(pin_off[owner][slot, 0])
                    fy += float(pin_off[owner][slot, 1])
                fixed_x.append(fx)
                fixed_y.append(fy)
            elif n_hard <= owner < n_macros:
                # Soft macro: variable (single pin at center).
                soft_idxs.append(owner - n_hard)
            else:
                # Port: fixed at port world position.
                port_idx = owner - n_macros
                if 0 <= port_idx < port_pos.shape[0]:
                    fixed_x.append(float(port_pos[port_idx, 0]))
                    fixed_y.append(float(port_pos[port_idx, 1]))

        k = len(soft_idxs) + len(fixed_x)
        if k < 2:
            continue
        w_pair = w_n / float(k - 1)

        # Soft-soft pairs: off-diagonal + diagonal.
        n_s = len(soft_idxs)
        for i in range(n_s):
            si = soft_idxs[i]
            for j in range(i + 1, n_s):
                sj = soft_idxs[j]
                if si == sj:
                    continue
                L_rows.append(si); L_cols.append(sj); L_vals.append(-w_pair)
                L_rows.append(sj); L_cols.append(si); L_vals.append(-w_pair)
                L_diag[si] += w_pair
                L_diag[sj] += w_pair

        # Soft-fixed pairs: diag + RHS.
        if fixed_x and soft_idxs:
            n_fix = len(fixed_x)
            sum_fx = float(sum(fixed_x))
            sum_fy = float(sum(fixed_y))
            for si in soft_idxs:
                L_diag[si] += w_pair * n_fix
                b_x[si] += w_pair * sum_fx
                b_y[si] += w_pair * sum_fy

    # Build sparse matrix in one COO assembly + diagonal add.
    if L_rows:
        L = sp.coo_matrix(
            (L_vals, (L_rows, L_cols)),
            shape=(n_soft, n_soft), dtype=np.float64,
        ).tocsr()
    else:
        L = sp.csr_matrix((n_soft, n_soft), dtype=np.float64)
    L = L + sp.diags(L_diag, 0, shape=(n_soft, n_soft), format="csr")
    return L, b_x, b_y


def laplacian_soft_resolve(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    # V14: finer alpha ladder. The original (1.0, 0.5, 0.25, 0.1, 0.05)
    # misses moves whose optimal fraction sits between 0.05 and 0.10 (small
    # softs near hot density cells where full Laplacian target would worsen
    # density). Adding 0.7, 0.35, 0.15, 0.025 fills the gaps; line-search
    # exits on first improving alpha, so the overhead is paid only on
    # softs that didn't move at the coarser steps.
    alphas: tuple = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.025),
    cg_rtol: float = 1e-6,
    cg_maxiter: int = 200,
    time_cap: float = 60.0,
    min_target_distance: float = 1e-3,
    seed: int = 42,
    verbose: bool = False,
) -> torch.Tensor:
    """Solve the Laplacian for soft positions, then line-search each soft.

    Args:
        placement: [num_macros, 2] tensor with current macro positions.
        benchmark: V13 Benchmark.
        plc: optional .plc loader for net weights.
        alphas: line-search fractions (descending). First strictly-
            improving move is taken; if none, soft stays put.
        cg_rtol, cg_maxiter: CG convergence params.
        time_cap: total budget in seconds.
        min_target_distance: skip softs whose target is within this
            fraction of canvas (saves time on no-op moves).
        seed: RNG seed (unused in core solve; reserved).
        verbose: print progress.

    Returns: [num_macros, 2] tensor with refined soft positions (hard
        positions unchanged). On any failure, returns the input
        placement unchanged.
    """
    t0 = time.time()
    n_macros = int(benchmark.num_macros)
    n_hard = int(benchmark.num_hard_macros)
    n_soft = n_macros - n_hard
    if n_soft <= 0:
        return placement.clone()

    placement_np = placement.detach().cpu().numpy().astype(np.float64)
    if placement_np.shape != (n_macros, 2):
        raise ValueError(
            f"placement shape {placement_np.shape}, expected ({n_macros}, 2)"
        )

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    port_pos = benchmark.port_positions.numpy().astype(np.float64)
    hard_pos = placement_np[:n_hard]

    # Build the Laplacian + RHS for soft positions.
    L, b_x, b_y = _build_clique_laplacian(benchmark, plc, hard_pos, port_pos)
    build_t = time.time() - t0

    if L.shape[0] == 0:
        return placement.clone()

    # Use current soft positions as warm-start.
    x0 = placement_np[n_hard:, 0].copy()
    y0 = placement_np[n_hard:, 1].copy()

    # CG solves on x and y axes (independent).
    try:
        x_sol, info_x = spla.cg(L, b_x, x0=x0, rtol=cg_rtol, maxiter=cg_maxiter)
        y_sol, info_y = spla.cg(L, b_y, x0=y0, rtol=cg_rtol, maxiter=cg_maxiter)
    except Exception as e:
        if verbose:
            print(f"  [laplacian] CG failed: {e}", flush=True)
        return placement.clone()
    solve_t = time.time() - t0 - build_t

    # Clip to canvas (softs allowed anywhere in [0, cw] × [0, ch]).
    np.clip(x_sol, 0.0, cw, out=x_sol)
    np.clip(y_sol, 0.0, ch, out=y_sol)

    # Per-soft line search using IncrementalProxy.
    inc = IncrementalProxy(benchmark, plc)
    inc.reset_to_positions(placement_np)
    current_cost = float(inc.proxy_full_cached()[0])
    initial_cost = current_cost

    min_dx = min_target_distance * cw
    min_dy = min_target_distance * ch
    n_moved = 0
    n_skipped_close = 0
    n_softs_visited = 0

    # Permute soft order so a partial pass (time cap hit) doesn't bias
    # toward low-index softs.
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    soft_order = rng.permutation(n_soft)

    for soft_idx in soft_order:
        if time.time() - t0 > time_cap:
            break
        n_softs_visited += 1
        m = n_hard + int(soft_idx)
        ox = float(inc.pos_aug[m, 0])
        oy = float(inc.pos_aug[m, 1])
        tx = float(x_sol[soft_idx])
        ty = float(y_sol[soft_idx])
        if abs(tx - ox) < min_dx and abs(ty - oy) < min_dy:
            n_skipped_close += 1
            continue
        moved_this = False
        for alpha in alphas:
            nx = ox + alpha * (tx - ox)
            ny = oy + alpha * (ty - oy)
            if nx < 0.0:
                nx = 0.0
            elif nx > cw:
                nx = cw
            if ny < 0.0:
                ny = 0.0
            elif ny > ch:
                ny = ch
            inc.try_move(m, nx, ny)
            new_cost = float(inc.proxy_full_cached()[0])
            if new_cost < current_cost - 1e-7:
                inc.commit_move()
                current_cost = new_cost
                n_moved += 1
                moved_this = True
                break
            inc.revert_move()
        # If no alpha improved, the soft stays at its original position.

    if verbose:
        gain = initial_cost - current_cost
        total_t = time.time() - t0
        print(
            f"  [laplacian] build={build_t:.1f}s cg={solve_t:.1f}s "
            f"ls={total_t - build_t - solve_t:.1f}s "
            f"moved={n_moved}/{n_softs_visited} (skip_close={n_skipped_close}) "
            f"cost {initial_cost:.4f} -> {current_cost:.4f} (Δ {gain:.4f})",
            flush=True,
        )

    return torch.from_numpy(inc.pos_aug[:n_macros].copy()).float()
