"""V13-rev8 — Multi-alpha saddle escape sweep.

Idea credit: vmallela_v7/_hessian_escape (Crippen-Snyder 1971,
Henkelman-Jonsson 2000 dimer method, Nesterov-Polyak 2006 cubic
regularization). Math implementation reuses V13's existing
`smallest_eigvec_hvp` from script8 and `smooth_proxy_components`
from script13 — only the multi-alpha sweep + reconvergence wrapper
is new.

# What this does (vs the existing script8 hess_escape)

V13's script8 already does the right kind of eigendecomposition:
PyTorch double-backward Hvp + scipy.sparse.linalg.eigsh with
which='SA' (smallest algebraic). But the placer.py wrapper calls
it ONCE with alpha=2.0 — a single shot in a single direction at a
single magnitude.

vmallela's Phase 3 saddle escape tries MULTIPLE step sizes
(±0.02, ±0.05 * canvas_diag) in PARALLEL (mp.Pool), reconverges
each via the full pipeline, and keeps best. The sign of an
eigvec is mathematically ambiguous (Hv = λv ⇒ H(-v) = λ(-v));
trying both signs costs nothing and one direction may land in
a strictly better basin.

This module does that multi-alpha sweep sequentially (no mp.Pool
to keep things simple — V13's reconverge is fast enough). For
each candidate:
  1. Perturb hard pos by α · v_min (with sign).
  2. Clamp to canvas.
  3. Legalize → short GD → short CD.
  4. Score with full TILOS proxy.
Returns the lowest-scoring candidate placement.

# Why it can outperform a single-alpha shot

For a given v_min:
  - α small → tiny perturbation → reconverges to SAME basin → wasted.
  - α large → blows out of basin entirely → reconverges to some other
    basin which MAY be better, may be worse.
  - α "just right" → crosses one saddle ridge → reconverges to the
    NEIGHBORING basin. The right α is bench-dependent.

Sweeping 4-6 alphas covers the spectrum from "tiny refinement" to
"full basin hop", and the best wins. _record-gated by the caller.

# Cost

  - Hvp+Lanczos: ~30-80s on ibm10-ibm17 (script8 measurements).
  - Per-candidate reconverge: ~30-60s (short GD 300 steps + CD).
  - Total for 6 candidates: 30+6×45 ≈ 300s. Caller gates on
    remaining > 400s.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


def saddle_escape_multi(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    legalize_fn,                 # callable(pos, benchmark, seed) -> pos
    run_gd_fn,                   # callable(benchmark, plc, pos, **kw) -> pos
    cd_refine_fn,                # callable(pos, benchmark, plc, **kw) -> pos
    alphas: tuple = (1.0, 2.5, 5.0, 10.0),  # in MICRON units (per-axis)
    try_both_signs: bool = True,
    n_gd_steps: int = 300,
    cd_max_rounds: int = 8,
    max_pins_per_net: int | None = None,
    cong_weight: float = 3.0,
    seed: int = 42,
    time_cap: float = 320.0,
    verbose: bool = False,
) -> tuple[torch.Tensor, float] | tuple[None, None]:
    """Try multiple alpha magnitudes along v_min (smallest-algebraic
    eigvec of smooth-proxy Hessian); reconverge each; return the BEST
    candidate placement + its proxy.

    Returns (best_placement, best_proxy) or (None, None) if no candidate
    was produced (eigsh failed / time_cap exceeded before reconverge).
    The caller MUST wrap the return through `_record` to gate any
    regression vs the current best.
    """
    t_start = time.time()
    n_hard = int(benchmark.num_hard_macros)
    n_macros = int(benchmark.num_macros)
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Compute v_min via script8's HVP + Lanczos helper. ──────────
    try:
        from script8 import SmoothProxyCallable, smallest_eigvec_hvp
    except Exception as e:
        if verbose:
            print(f"  [saddle21] script8 import failed: {e}", flush=True)
        return None, None

    try:
        callable_fn = SmoothProxyCallable(
            benchmark, plc, device, max_pins_per_net=max_pins_per_net,
        )
        hard_pos = placement[:n_hard].to(device).float().clone()
        lam_min, v_min = smallest_eigvec_hvp(
            callable_fn, hard_pos, verbose=False, top_k=1,
            max_iter=50, tol=1e-4,
        )
    except Exception as e:
        if verbose:
            print(f"  [saddle21] eigsh/HVP failed: {e}", flush=True)
        return None, None

    # Mask perturbation to movable hard macros only.
    movable = benchmark.get_movable_mask()[:n_hard].numpy()
    v_min_masked = v_min.copy()
    v_min_masked[~movable] = 0.0
    v_norm = float(np.linalg.norm(v_min_masked))
    if v_norm < 1e-12:
        if verbose:
            print(f"  [saddle21] degenerate eigvec (||v||={v_norm:.2e})",
                  flush=True)
        return None, None
    v_unit = v_min_masked / v_norm
    if verbose:
        print(
            f"  [saddle21] λ_min={lam_min:.4f}  ||v_min||={v_norm:.4f}  "
            f"setup={time.time() - t_start:.1f}s",
            flush=True,
        )

    # Half-extents for canvas clamping (numpy for vector ops).
    half_w = (benchmark.macro_sizes[:n_hard, 0] / 2.0).cpu().numpy().astype(np.float64)
    half_h = (benchmark.macro_sizes[:n_hard, 1] / 2.0).cpu().numpy().astype(np.float64)
    hard_pos_np = placement[:n_hard].cpu().numpy().astype(np.float64)

    # Build candidate sweep: (+α, -α) for each α in alphas.
    candidates: list[tuple[float, np.ndarray]] = []
    for alpha in alphas:
        signs = (1.0, -1.0) if try_both_signs else (1.0,)
        for s in signs:
            shift = s * alpha * v_unit  # shape (n_hard, 2) in micron units
            new_hard = hard_pos_np + shift
            # Clamp to canvas; movable mask zeroes shift for fixed macros so
            # they don't move.
            new_hard[:, 0] = np.clip(new_hard[:, 0], half_w, cw - half_w)
            new_hard[:, 1] = np.clip(new_hard[:, 1], half_h, ch - half_h)
            candidates.append((s * alpha, new_hard))

    # ── 2. Reconverge each candidate; track best by full proxy. ───────
    best_pos: torch.Tensor | None = None
    best_proxy = float("inf")
    pl_full = placement.clone()
    n_done = 0
    for i, (signed_alpha, pert_hard_np) in enumerate(candidates):
        if time.time() - t_start > time_cap:
            if verbose:
                print(
                    f"  [saddle21] time cap hit after {n_done}/{len(candidates)} "
                    f"candidates",
                    flush=True,
                )
            break
        pos_cand = pl_full.clone()
        pos_cand[:n_hard] = torch.from_numpy(pert_hard_np).float()
        # Legalize the perturbation (may have caused overlaps).
        try:
            pos_legal = legalize_fn(pos_cand, benchmark, seed=seed + i)
        except Exception as e:
            if verbose:
                print(f"  [saddle21] cand α={signed_alpha:+.1f} legalize "
                      f"failed: {e}", flush=True)
            continue
        # Short re-converging GD to settle into the new basin.
        try:
            pos_g = run_gd_fn(
                benchmark, plc, pos_legal, device=device,
                n_steps=n_gd_steps, lr_start=0.05, lr_end=0.005,
                gamma_start=3.0, gamma_mid=5.0, gamma_end=8.0,
                cong_weight=cong_weight, noise_start=0.0,
                max_pins_per_net=max_pins_per_net, seed=seed + i,
                verbose=False,
            )
            pos_g = legalize_fn(pos_g, benchmark, seed=seed + i)
        except Exception as e:
            if verbose:
                print(f"  [saddle21] cand α={signed_alpha:+.1f} GD failed: {e}",
                      flush=True)
            continue
        # Short CD polish to capture the basin floor.
        remaining = time_cap - (time.time() - t_start)
        if remaining > 30:
            try:
                pos_g = cd_refine_fn(
                    pos_g, benchmark, plc,
                    max_rounds=cd_max_rounds, seed=seed + i,
                    max_pins_per_net=max_pins_per_net, verbose=False,
                    time_cap=max(15.0, min(remaining - 15, 40.0)),
                )
            except Exception as e:
                if verbose:
                    print(f"  [saddle21] cand α={signed_alpha:+.1f} CD failed: {e}",
                          flush=True)
        # Score with full TILOS proxy. Reject overlaps.
        try:
            costs = compute_proxy_cost(pos_g, benchmark, plc)
            ovr = int(costs.get("overlap_count", 0))
            pc = float(costs["proxy_cost"])
        except Exception as e:
            if verbose:
                print(f"  [saddle21] cand α={signed_alpha:+.1f} score failed: {e}",
                      flush=True)
            continue
        if ovr == 0 and pc < best_proxy:
            best_proxy = pc
            best_pos = pos_g.clone()
        n_done += 1
        if verbose:
            print(
                f"  [saddle21] cand {n_done}/{len(candidates)} "
                f"α={signed_alpha:+.1f}um → proxy={pc:.4f} ovr={ovr} "
                f"{'★' if (ovr == 0 and pc == best_proxy) else ''}",
                flush=True,
            )

    if best_pos is None:
        return None, None
    if verbose:
        total_t = time.time() - t_start
        print(
            f"  [saddle21] DONE in {total_t:.1f}s — best proxy={best_proxy:.4f} "
            f"from {n_done} reconverged candidates",
            flush=True,
        )
    return best_pos, best_proxy
