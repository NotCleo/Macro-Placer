"""V13-rev2 Opt 1 — Soft-macro centroid pull.

V13's pipeline only moves soft macros during cycle 1/2/3 GD. After cd_cong,
every polishing stage (cd_final, cd_cong, T1 SA, refine_iter, polish loop)
targets HARD macros only. Soft macros stay frozen at their cycle-3 GD
positions.

Soft macros are 70-90% of total macros and contribute to wirelength and
density (they are stdcell-cluster abstractions whose optimal position is
the centroid of their net peers). Leaving them frozen wastes the biggest
single degree of freedom in the proxy cost.

Algorithm (inspired by graph_grad/placer.py:_soft_centroid_target):
  1. Rank soft macros by sum of their nets' current bbox lengths.
  2. For the top `max_macros` (default 200), compute the weighted centroid
     of their net partners' pin positions (weight = 1/(net_fanout-1) so
     dense nets don't dominate).
  3. Propose moves at fractions f ∈ {0.25, 0.5} toward the centroid.
  4. Score via IncrementalProxy's partial_proxy_no_cong() — O(macro degree)
     instead of O(all nets), so 200 macros × 2 fractions × O(degree) ≈ 30-60s.
  5. Accept if partial proxy strictly improves.
  6. Caller wraps the result through _record (outer-verify) so the FULL
     proxy (including congestion) gates the final acceptance.

Soft macros are allowed to overlap (per challenge rules), so no legalization
needed. Score uses partial-proxy because soft moves typically change WL+D
much more than cong (cong is dominated by hard-macro routes and Steiner
trees between hard pins).
"""
from __future__ import annotations

import time
import numpy as np
import torch

from macro_place.benchmark import Benchmark
from script4 import IncrementalProxy


def soft_centroid_pull(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    *,
    fractions: tuple = (0.25, 0.5),
    seed: int = 42,
    time_cap: float = 60.0,
    max_macros: int = 200,
    verbose: bool = False,
) -> torch.Tensor:
    """Move soft macros toward weighted net-partner centroids.

    Returns the post-pull placement (potentially with soft macros moved).
    Caller MUST wrap through _record / outer-verify to gate on the full
    TILOS proxy — this function only guarantees partial-proxy improvement.

    Args
    ----
    placement       : input [num_macros, 2] tensor (centers).
    fractions       : per-macro move trials. Default (0.25, 0.5) gives
                      one quarter-step and one half-step toward centroid.
    seed            : RNG seed (only used to randomize macro tie-breaking).
    time_cap        : hard wallclock cap.
    max_macros      : process at most this many soft macros (ranked by
                      net-bbox stress so the highest-leverage ones go first).
    verbose         : print summary at end.
    """
    n_hard = int(benchmark.num_hard_macros)
    n_macros = int(benchmark.num_macros)
    n_soft = n_macros - n_hard
    if n_soft == 0:
        return placement.clone()

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    movable = benchmark.get_movable_mask().cpu().numpy()

    # Set up IncrementalProxy with the current placement.
    saved_pos = benchmark.macro_positions.clone()
    benchmark.macro_positions = placement[:n_macros].clone()
    try:
        inc = IncrementalProxy(benchmark, plc=plc)
    except Exception:
        benchmark.macro_positions = saved_pos
        return placement.clone()
    benchmark.macro_positions = saved_pos

    nets = benchmark.net_pin_nodes
    if not nets:
        return placement.clone()

    # Build owner -> [(net_idx_in_inc, my_pin_idx)] for SOFT macros only.
    # Note: inc.num_nets_used skips degenerate nets (< 2 pins), so we have
    # to walk inc.net_pin_owner to find which (net_idx, pin_slot) a given
    # soft macro appears in.
    owner_to_nets: dict[int, list[tuple[int, int]]] = {}
    for ni in range(inc.num_nets_used):
        owners = inc.net_pin_owner[ni]
        for k in range(owners.shape[0]):
            o = int(owners[k])
            if n_hard <= o < n_macros:
                owner_to_nets.setdefault(o, []).append((ni, k))

    if not owner_to_nets:
        return placement.clone()

    # Rank soft macros by sum of their nets' current bbox stress.
    soft_score: list[tuple[float, int]] = []
    for m in range(n_hard, n_macros):
        if not movable[m] or m not in owner_to_nets:
            continue
        s = 0.0
        for (ni, _) in owner_to_nets[m]:
            if ni < inc.num_nets_used:
                s += float(inc.net_weights[ni]) * float(inc.net_bbox_total[ni])
        if s > 0:
            soft_score.append((s, m))
    soft_score.sort(reverse=True)
    targets = [m for _, m in soft_score[:max_macros]]
    if not targets:
        return placement.clone()

    init_partial = inc.partial_proxy_no_cong()
    best_partial = init_partial
    accepts = 0
    t0 = time.time()

    for m in targets:
        if time.time() - t0 > time_cap:
            break
        # Weighted centroid of partners.
        sum_x = 0.0
        sum_y = 0.0
        total_w = 0.0
        for (ni, my_k) in owner_to_nets[m]:
            owners = inc.net_pin_owner[ni]
            xy = inc.net_pin_xy[ni]
            k_size = int(owners.shape[0])
            if k_size < 2:
                continue
            w = 1.0 / (k_size - 1)
            for p in range(k_size):
                if p == my_k:
                    continue
                sum_x += w * float(xy[p, 0])
                sum_y += w * float(xy[p, 1])
                total_w += w
        if total_w <= 0:
            continue
        cx = sum_x / total_w
        cy = sum_y / total_w

        ox = float(inc.pos_aug[m, 0])
        oy = float(inc.pos_aug[m, 1])
        hw = float(sizes[m, 0]) * 0.5
        hh = float(sizes[m, 1]) * 0.5

        for f in fractions:
            nx = ox + f * (cx - ox)
            ny = oy + f * (cy - oy)
            nx = max(hw, min(cw - hw, nx))
            ny = max(hh, min(ch - hh, ny))
            if abs(nx - ox) < 1e-6 and abs(ny - oy) < 1e-6:
                continue
            try:
                inc.try_move(m, nx, ny)
                new_partial = inc.partial_proxy_no_cong()
            except Exception:
                # Be defensive: any try/revert anomaly = skip this macro.
                try:
                    inc.revert_move()
                except Exception:
                    pass
                break
            if new_partial < best_partial - 1e-9:
                inc.commit_move()
                best_partial = new_partial
                accepts += 1
                ox, oy = nx, ny  # commit succeeded — next f starts from here
            else:
                inc.revert_move()

    if verbose:
        print(
            f"  [soft-centroid] {accepts} accepts (partial {init_partial:.6f} "
            f"-> {best_partial:.6f}) t={time.time() - t0:.0f}s",
            flush=True,
        )

    out = placement.clone()
    out[:n_macros] = torch.from_numpy(inc.pos_aug[:n_macros].astype(np.float32))
    return out
