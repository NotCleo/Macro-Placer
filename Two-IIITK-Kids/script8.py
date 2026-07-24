"""Hessian smallest-eigenvalue escape: find flattest direction at v14
converged point, perturb along it, re-run v14+CD to find a new basin.

Inspired by vmallela's "saddle escape" technique. Their pipeline finds
saddles (negative eigenvalues) — ours hits true local minima (per HANDOFF),
so we use the smallest POSITIVE eigenvalue direction instead. This is the
"flattest" direction: small proxy increase but large position change, which
might reach a basin the gradient descent can't see from the current point.

Algorithm:
  1. Build smooth proxy as a callable F(pos) -> scalar
  2. Compute Hessian d^2F/dpos^2 via torch autograd
  3. Eigendecomposition; pick eigenvector for smallest eigenvalue
  4. Perturb positions: pos_new = pos + alpha * v_min
  5. Re-run v14 + CD from pos_new

For 1500-dim Hessian (ibm17), scipy eigh takes ~30s.
"""
from __future__ import annotations

import os, sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from macro_place.benchmark import Benchmark
from script2 import _build_pin_tensors, _build_net_weights
from script13 import smooth_proxy_components


class SmoothProxyCallable:
    """Callable F(positions_hard) -> scalar smooth proxy. Used for Hessian
    computation. Soft macros and ports are held at their input positions."""

    def __init__(self, benchmark: Benchmark, plc, device,
                 gamma_wl: float = 8.0,
                 max_pins_per_net: int | None = None):
        self.b = benchmark
        self.plc = plc
        self.device = device
        self.gamma_wl = gamma_wl

        n_macros = benchmark.num_macros
        n_hard = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        self.cw = cw; self.ch = ch
        self.n_hard = n_hard
        self.n_macros = n_macros
        self.sizes_all = benchmark.macro_sizes.to(device).float()
        self.port_pos = benchmark.port_positions.to(device).float()

        pin_data = _build_pin_tensors(benchmark, device)
        owner_idx, offset_xy, mask = pin_data
        if max_pins_per_net is not None:
            P_full = mask.shape[1]
            cap = int(max_pins_per_net)
            if P_full > cap:
                owner_idx = owner_idx[:, :cap].contiguous()
                offset_xy = offset_xy[:, :cap].contiguous()
                mask = mask[:, :cap].contiguous()
        self.owner_idx = owner_idx
        self.offset_xy = offset_xy
        self.mask = mask
        self.net_weights = _build_net_weights(benchmark, plc, device)
        self.net_count = float(self.net_weights.sum())

        self.G_w = int(plc.grid_col)
        self.G_h = int(plc.grid_row)
        self.hroutes_per_micron = float(plc.hroutes_per_micron)
        self.vroutes_per_micron = float(plc.vroutes_per_micron)
        self.hrouting_alloc = float(plc.hrouting_alloc)
        self.vrouting_alloc = float(plc.vrouting_alloc)
        self.smooth_range = int(plc.smooth_range)

        # Soft positions are kept fixed during Hessian computation.
        # Soft macros sit at indices [n_hard, n_macros) in macro_positions.
        soft = benchmark.macro_positions[n_hard:n_macros].to(device).float()
        self.soft_pos = soft

    def __call__(self, hard_pos: torch.Tensor) -> torch.Tensor:
        """hard_pos: [n_hard, 2] tensor. Returns scalar smooth proxy."""
        positions_aug = torch.cat([hard_pos, self.soft_pos, self.port_pos], dim=0)
        wl, d_cost, c_cost = smooth_proxy_components(
            positions_aug, self.sizes_all,
            self.owner_idx, self.offset_xy, self.mask,
            self.net_weights, self.net_count,
            self.G_w, self.G_h, self.cw, self.ch,
            self.hroutes_per_micron, self.vroutes_per_micron,
            self.hrouting_alloc, self.vrouting_alloc,
            self.n_hard, self.n_macros,
            gamma_wl=self.gamma_wl, gamma_bbox=8.0,
            smooth_range=self.smooth_range,
            p_density=10.0, p_cong=16.0, softness_factor=0.5,
        )
        return wl + d_cost + c_cost


def smallest_eigvec_hvp(callable_fn, hard_pos_init: torch.Tensor,
                          verbose: bool = True, top_k: int = 1,
                          max_iter: int = 50, tol: float = 1e-4):
    """HVP-based smallest eigvec(s) via Lanczos (scipy.sparse.linalg.eigsh).

    Replaces the O(N^3) full-Hessian + eigh path with O(N * iterations).
    Each HVP is one extra backward pass through `callable_fn` — for ibm12
    (N_hard = 423, dim = 846) this is ~25× faster than building the full
    846x846 Hessian (which costs ~846 backward passes).

    Algorithm:
      1. forward + first-backward to get gradient (with create_graph=True)
      2. each HVP = grad(grad @ v, pos) — 1 more backward pass
      3. Lanczos via scipy eigsh(LinearOperator, k=top_k, which='SA')

    Args:
        callable_fn: F(hard_pos[n_hard,2]) -> scalar
        hard_pos_init: starting position
        top_k: how many smallest-algebraic eigenpairs to extract
        max_iter: per-eigenpair Lanczos iterations
    Returns:
        same shape as smallest_eigvec() — (eigval, eigvec) if top_k=1, else lists
    """
    n_hard = hard_pos_init.shape[0]
    dim = n_hard * 2
    device = hard_pos_init.device

    pos = hard_pos_init.clone().detach().requires_grad_(True)
    if verbose:
        print(f"  [hvp] dim={dim}, computing grad (first backward)...", flush=True)
    t0 = time.time()
    loss = callable_fn(pos)
    grad = torch.autograd.grad(loss, pos, create_graph=True)[0]
    if verbose:
        print(f"  [hvp] first backward in {time.time()-t0:.1f}s", flush=True)

    hvp_count = [0]
    def hvp_matvec(v_np):
        hvp_count[0] += 1
        v = torch.from_numpy(np.ascontiguousarray(v_np, dtype=np.float32)).to(device).view(n_hard, 2)
        # Hv = d(grad . v)/dpos. Use retain_graph so we can call HVP many times.
        Hv = torch.autograd.grad(grad, pos, grad_outputs=v, retain_graph=True)[0]
        return Hv.detach().flatten().cpu().numpy().astype(np.float32)

    from scipy.sparse.linalg import LinearOperator, eigsh
    Lo = LinearOperator((dim, dim), matvec=hvp_matvec, dtype=np.float32)
    t1 = time.time()
    if verbose:
        print(f"  [hvp] Lanczos eigsh (top_k={top_k}, max_iter={max_iter})...", flush=True)
    # 'SA' = smallest algebraic (most-negative or near-zero). v0 makes deterministic.
    eigvals, eigvecs = eigsh(Lo, k=top_k, which='SA',
                              maxiter=max_iter * max(top_k, 4),
                              tol=tol)
    if verbose:
        print(f"  [hvp] eigsh in {time.time()-t1:.1f}s, {hvp_count[0]} HVPs, "
              f"eigvals: {eigvals.tolist()}", flush=True)
    # Sort ascending (eigsh doesn't guarantee order for 'SA')
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    if top_k == 1:
        return float(eigvals[0]), eigvecs[:, 0].reshape(n_hard, 2)
    return ([float(v) for v in eigvals],
            [eigvecs[:, k].reshape(n_hard, 2) for k in range(top_k)])


def smallest_eigvec(callable_fn, hard_pos_init: torch.Tensor,
                    verbose: bool = True, top_k: int = 1, use_hvp: bool = True):
    """Compute top-K smallest eigenvalues / eigenvectors of Hessian of
    callable_fn at hard_pos_init.

    If use_hvp=True (default): O(N * iters) HVP-based Lanczos. Much faster
    on large benchmarks. Fall back to full Hessian if use_hvp=False.

    If top_k=1, returns (smallest_eigval, eigvec [n_hard, 2]) for backward
    compat. If top_k>1, returns (eigvals [k] list, eigvecs [k, n_hard, 2]
    list).
    """
    if use_hvp:
        return smallest_eigvec_hvp(callable_fn, hard_pos_init,
                                     verbose=verbose, top_k=top_k)
    n_hard = hard_pos_init.shape[0]
    dim = n_hard * 2
    if verbose:
        print(f"  [hess] computing Hessian ({dim} x {dim} = {dim*dim/1e6:.1f}M entries)...",
              flush=True)
    t0 = time.time()
    pos = hard_pos_init.clone().detach()
    # torch's hessian wants a function on a flat tensor.
    def f_flat(flat_pos):
        return callable_fn(flat_pos.view(n_hard, 2))
    hess = torch.autograd.functional.hessian(f_flat, pos.flatten())
    # hess shape: [dim, dim]
    if verbose:
        print(f"  [hess] hessian done in {time.time()-t0:.1f}s. "
              f"max={hess.abs().max().item():.3e}", flush=True)

    t1 = time.time()
    # Symmetrize defensively (autograd hessian is symmetric in theory).
    hess_np = hess.cpu().detach().numpy()
    hess_sym = 0.5 * (hess_np + hess_np.T)
    from scipy.linalg import eigh
    eigvals, eigvecs = eigh(hess_sym)
    if verbose:
        print(f"  [hess] eigh done in {time.time()-t1:.1f}s. "
              f"smallest {min(5, top_k+2)} eigvals: {eigvals[:min(5, top_k+2)]}",
              flush=True)

    if top_k == 1:
        min_eig = float(eigvals[0])
        min_vec = eigvecs[:, 0].reshape(n_hard, 2)
        return min_eig, min_vec
    else:
        eigval_list = [float(eigvals[i]) for i in range(top_k)]
        eigvec_list = [eigvecs[:, i].reshape(n_hard, 2) for i in range(top_k)]
        return eigval_list, eigvec_list


def hessian_multi_escape(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    alpha: float = 2.0,
    top_k: int = 3,
    max_pins_per_net: int | None = None,
    verbose: bool = True,
) -> tuple[list, dict]:
    """Generate K candidate placements, each perturbed along a different
    eigenvector of the smooth-proxy Hessian (the top-K smallest eigvecs).

    The smallest eigenvector is most negative-curvature (or flattest); higher
    eigvecs are also low-curvature directions. Each gives a different escape
    direction; the caller re-converges all K and picks the best.

    Returns (list of K perturbed placements, info dict with eigvals+norms).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros

    callable_fn = SmoothProxyCallable(benchmark, plc, device,
                                       max_pins_per_net=max_pins_per_net)
    hard_pos = placement[:n_hard].to(device).float().clone()

    eigvals, eigvecs = smallest_eigvec(callable_fn, hard_pos,
                                         verbose=verbose, top_k=top_k)

    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    half_w = (benchmark.macro_sizes[:n_hard, 0] / 2.0).to(device).float()
    half_h = (benchmark.macro_sizes[:n_hard, 1] / 2.0).to(device).float()

    perturbed_placements = []
    pert_norms = []
    for k in range(top_k):
        eig_vec = eigvecs[k].copy()
        eig_vec[~movable] = 0.0
        norm = np.linalg.norm(eig_vec)
        if norm > 1e-12:
            eig_vec = eig_vec / norm
        pert = torch.from_numpy(eig_vec).to(device).float() * alpha
        new_hard_pos = hard_pos + pert
        new_hard_pos[:, 0] = torch.clamp(new_hard_pos[:, 0], half_w,
                                           float(benchmark.canvas_width) - half_w)
        new_hard_pos[:, 1] = torch.clamp(new_hard_pos[:, 1], half_h,
                                           float(benchmark.canvas_height) - half_h)
        out = placement.clone()
        out[:n_hard] = new_hard_pos.cpu()
        perturbed_placements.append(out)
        pert_norms.append(float(np.linalg.norm(pert.cpu().numpy())))
        if verbose:
            print(f"  [hess-multi] eigvec {k}: eigval={eigvals[k]:.5f}  "
                  f"pert_norm={pert_norms[-1]:.2f}um", flush=True)

    info = {
        "eigvals": [float(e) for e in eigvals],
        "alpha": float(alpha),
        "top_k": int(top_k),
        "pert_norms_um": pert_norms,
    }
    return perturbed_placements, info


def hessian_eigvec_escape(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    alpha: float = 1.0,
    max_pins_per_net: int | None = None,
    verbose: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Compute smallest-eig direction of smooth proxy Hessian and return
    perturbed placement (only movable hard macros perturbed)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros

    callable_fn = SmoothProxyCallable(benchmark, plc, device,
                                       max_pins_per_net=max_pins_per_net)

    hard_pos = placement[:n_hard].to(device).float().clone()

    min_eig, min_vec = smallest_eigvec(callable_fn, hard_pos, verbose=verbose)

    # Restrict perturbation to movable hard macros.
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    min_vec_masked = min_vec.copy()
    min_vec_masked[~movable] = 0.0
    # Renormalize to unit vector after masking
    norm = np.linalg.norm(min_vec_masked)
    if norm > 1e-12:
        min_vec_masked = min_vec_masked / norm

    # Perturb (alpha is the magnitude of position shift in micron units).
    pert = torch.from_numpy(min_vec_masked).to(device).float() * alpha
    new_hard_pos = hard_pos + pert

    # Clamp to canvas
    half_w = (benchmark.macro_sizes[:n_hard, 0] / 2.0).to(device).float()
    half_h = (benchmark.macro_sizes[:n_hard, 1] / 2.0).to(device).float()
    new_hard_pos[:, 0] = torch.clamp(new_hard_pos[:, 0], half_w,
                                      float(benchmark.canvas_width) - half_w)
    new_hard_pos[:, 1] = torch.clamp(new_hard_pos[:, 1], half_h,
                                      float(benchmark.canvas_height) - half_h)

    out = placement.clone()
    out[:n_hard] = new_hard_pos.cpu()

    info = {
        "smallest_eig": float(min_eig),
        "alpha": float(alpha),
        "pert_norm_um": float(np.linalg.norm(pert.cpu().numpy())),
        "max_per_macro_shift_um": float(np.abs(pert.cpu().numpy()).max()),
    }
    if verbose:
        print(f"  [hess] perturbed by alpha={alpha}, "
              f"norm={info['pert_norm_um']:.2f}um, "
              f"max_per_macro={info['max_per_macro_shift_um']:.3f}um", flush=True)
    return out, info
