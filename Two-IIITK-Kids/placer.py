from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make sibling modules importable when invoked via `uv run evaluate <path>`
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_max_threads = os.environ.get("OMP_NUM_THREADS", str(min(os.cpu_count() or 1, 16)))
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, _max_threads)

# V13-rev4: CUBLAS_WORKSPACE_CONFIG must be set BEFORE torch is imported for
# cudnn.deterministic to actually take effect on cuBLAS GEMM ops. The
# ":4096:8" form reserves a small workspace and forces deterministic kernels.
# Without this, Phase α (script7.py FFT-Poisson GP) produced wildly different
# starting basins across runs (1.32 / 1.55 / 1.85 on the same ibm08 with the
# same seed) — every downstream stage then sat in a worse basin floor.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.set_num_threads(int(_max_threads))
torch.set_num_interop_threads(max(1, int(_max_threads) // 2))
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.loader import load_benchmark_from_dir, load_benchmark

from script3 import (
    legalize as legalize_np,
    _list_violators,
    _force_displace_grid,
)
from script2 import run_gd
from script1 import cd_refine
from script6 import swap_refine


# ── Tunables ────────────────────────────────────────────────────────────
# Pin cap for huge benches: ibm14 has one 84-pin net that bloats the
# [N, P, G] L-routing tensor from ~80MB to ~520MB. Capping at 24 keeps
# the GD pass fast without losing meaningful signal (dropped pins are
# usually clock / scan chain that don't contribute to L-routing demand).
HUGE_PIN_CAP = 24
HUGE_HARD_THRESH = 500
HUGE_NETS_THRESH = 30_000

# Total per-benchmark wallclock budget (the contest cap is 3600s; leave
# headroom for harness overhead).
DEFAULT_BUDGET_S = 3300


def _cd_config(n_hard: int) -> dict:
    if n_hard < 300:
        return dict(K=12, R=20)
    if n_hard < 500:
        return dict(K=10, R=18)
    if n_hard < 700:
        return dict(K=8, R=15)
    return dict(K=6, R=12)


def _load_plc_for(benchmark: Benchmark):
    name = benchmark.name
    iccad = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if iccad.exists():
        _, plc = load_benchmark_from_dir(str(iccad))
        return plc
    ng45 = {"ariane133": "ariane133", "ariane136": "ariane136",
            "nvdla": "nvdla", "mempool_tile": "mempool_tile"}
    short = ng45.get(name.replace("_ng45", ""), name)
    base = (Path("external/MacroPlacement/Flows/NanGate45")
            / short / "netlist" / "output_CT_Grouping")
    if (base / "netlist.pb.txt").exists():
        _, plc = load_benchmark(str(base / "netlist.pb.txt"),
                                str(base / "initial.plc"))
        return plc
    return None


def _attach_net_pin_nodes(benchmark: Benchmark, plc) -> None:
    """Build benchmark.net_pin_nodes from plc.

    The current macro_place.Benchmark exposes net_nodes (a list of node-index
    tensors), but this placer's helpers (gd_anneal, proxy_fast, cd_refine,
    swap_refine) all reach for net_pin_nodes — a list of [P, 2] long tensors
    of (macro_bench_idx, pin_slot) per pin in each net. We synthesize it from
    plc here so the rest of the pipeline runs unchanged.
    """
    if getattr(benchmark, "net_pin_nodes", None):
        return

    n_hard = benchmark.num_hard_macros
    n_macros = benchmark.num_macros
    hard_idx = list(benchmark.hard_macro_indices)
    soft_idx = list(benchmark.soft_macro_indices)
    port_idx = list(plc.port_indices)

    plc_to_bench = {}
    for bi, pi in enumerate(hard_idx):
        plc_to_bench[pi] = bi
    for bi, pi in enumerate(soft_idx):
        plc_to_bench[pi] = n_hard + bi
    for bi, pi in enumerate(port_idx):
        plc_to_bench[pi] = n_macros + bi

    name_to_bench = {}
    for pi, bi in plc_to_bench.items():
        name_to_bench[plc.modules_w_pins[pi].get_name()] = bi

    # pin full-name ("MACRO/PIN") -> (bench_idx, slot). Slot is the order
    # in which the pin appears within plc.hard_macro_pin_indices for its
    # owning macro — matching the order the loader uses to build
    # benchmark.macro_pin_offsets.
    pin_to_loc = {}
    slot_counter = {}
    for plc_pin_idx in plc.hard_macro_pin_indices:
        pin = plc.modules_w_pins[plc_pin_idx]
        macro_name = pin.get_macro_name() if hasattr(pin, "get_macro_name") else None
        if not macro_name or macro_name not in name_to_bench:
            continue
        bench_idx = name_to_bench[macro_name]
        slot = slot_counter.get(macro_name, 0)
        slot_counter[macro_name] = slot + 1
        pin_to_loc[pin.get_name()] = (bench_idx, slot)

    net_pin_nodes = []
    for driver, sinks in plc.nets.items():
        pins = []
        for pin_name in [driver] + list(sinks):
            if pin_name in pin_to_loc:
                bi, sl = pin_to_loc[pin_name]
                pins.append([bi, sl])
            else:
                parent = pin_name.split("/")[0]
                if parent in name_to_bench:
                    pins.append([name_to_bench[parent], 0])
        if pins:
            net_pin_nodes.append(torch.tensor(pins, dtype=torch.long))

    benchmark.net_pin_nodes = net_pin_nodes


def _compute_long_net_boost(benchmark: Benchmark, top_pct: float = 0.20,
                            boost: float = 2.5) -> torch.Tensor | None:
    """V10 (A2): per-net multiplier — top X% longest nets (by initial.plc
    bbox) get `boost`× WL weight in cycles 2/3 GD. Mirrors graph_grad's
    dyn_net_weights idea but precomputed once (no per-step grid analysis).
    Returns [n_nets] float tensor or None on failure / no nets."""
    if not getattr(benchmark, "net_pin_nodes", None):
        return None
    try:
        n_nets = int(benchmark.num_nets)
        n_hard = int(benchmark.num_hard_macros)
        n_macros = int(benchmark.num_macros)
        pos = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)
        port = benchmark.port_positions.detach().cpu().numpy().astype(np.float64)
        n_ports = port.shape[0]
        pin_offsets = benchmark.macro_pin_offsets
        lens = np.zeros(n_nets, dtype=np.float64)
        for n in range(n_nets):
            pn = benchmark.net_pin_nodes[n].cpu().numpy().astype(np.int64)
            if pn.size == 0:
                continue
            xs = []
            ys = []
            for k in range(pn.shape[0]):
                o, s = int(pn[k, 0]), int(pn[k, 1])
                if o < n_macros:
                    px = float(pos[o, 0])
                    py = float(pos[o, 1])
                    if (o < n_hard and pin_offsets and o < len(pin_offsets)
                            and pin_offsets[o] is not None
                            and pin_offsets[o].shape[0] > s):
                        px += float(pin_offsets[o][s, 0])
                        py += float(pin_offsets[o][s, 1])
                    xs.append(px); ys.append(py)
                else:
                    pi = o - n_macros
                    if 0 <= pi < n_ports:
                        xs.append(float(port[pi, 0]))
                        ys.append(float(port[pi, 1]))
            if xs:
                lens[n] = (max(xs) - min(xs)) + (max(ys) - min(ys))
        if lens.sum() <= 0:
            return None
        thresh = float(np.quantile(lens, 1.0 - top_pct))
        mult = np.where(lens >= max(thresh, 1e-9), boost, 1.0).astype(np.float32)
        return torch.from_numpy(mult)
    except Exception:
        return None


def _legalize_full(pos_torch: torch.Tensor, benchmark: Benchmark,
                   seed: int = 42, retry_seeds: int = 8,
                   benchmark_for_score=None, plc_for_score=None,
                   time_cap_s=None, log_fn=None) -> torch.Tensor:
    """Legalize hard macros to zero overlap; preserve soft positions as-is.

    Belt-and-suspenders: if the standard legalizer + its force_displace
    fallback still leave any violator (numerical edge case), retry the
    force_displace with a denser grid + different seed.

    V12 fix: on HUGE benches (e.g. ibm10/14/15/16/17) the first jitter seed
    plus n_steps=240 / max_sweeps=8 can leave 1-2 residual violators that
    only a denser grid resolves. We (a) raise the base attempt to n_steps=400
    / max_sweeps=12, (b) bump default retry_seeds 4→8, and (c) escalate
    n_steps / max_sweeps with each retry so late retries try grids up to
    n_steps=240+40*8=560 / max_sweeps=16. Float32-edge ovr=1 placements
    that previously got stuck consistently fit at the denser grid.
    """
    n_hard = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
    movable = (~benchmark.macro_fixed[:n_hard]).cpu().numpy()
    hard_np = pos_torch[:n_hard].detach().cpu().numpy().astype(np.float64)
    _t0 = time.time()

    def _attempt(s: int, n_steps: int = 160, max_sweeps: int = 6,
                 outer_rounds: int = 8):
        # V16: revert defaults to V7's proven 160/6 (was 400/12). _force_displace
        # cost scales ~n_steps²·max_sweeps per violator; on ibm10 the 400/12
        # default ate 1400s of the 3300s budget on gd_c1 alone, starving the
        # polish chain. V7 ran the full pipeline in 3255s with 160/6 and hit
        # 1.0073. The retry escalation below (ns_k=240+40·k) still handles the
        # "1-2 residual violators" edge case that originally motivated the bump.
        # V16c: outer_rounds plumb-through. ibm10 trace shows bulk_push used
        # all 8 rounds without converging (49 viols remained). Reducing rounds
        # for HUGE benches saves 100-200s per attempt; residual viols handled
        # by force_displace.
        _ta = time.time()
        legal = legalize_np(hard_np, sizes, movable, cw, ch, seed=s,
                            outer_rounds=outer_rounds, timing_log=log_fn)
        _t_lnp = time.time() - _ta
        vio = _list_violators(legal, sizes, movable)
        _tf = time.time()
        if vio:
            legal = _force_displace_grid(legal, sizes, movable, cw, ch,
                                         n_steps=n_steps, max_sweeps=max_sweeps)
        _t_fd = time.time() - _tf
        if log_fn is not None:
            log_fn(f"    [attempt s={s}] legalize_np={_t_lnp:.0f}s "
                   f"viol_after_lnp={len(vio)} force_displace={_t_fd:.0f}s "
                   f"(grid {n_steps}/{max_sweeps})")
        return legal

    # V16c: HUGE benches get tighter outer_rounds (8→4). Trace shows bulk
    # never converges in 8 rounds anyway (49 viols remained); cutting to 4
    # saves ~140s per call. Non-HUGE keeps 8 (it usually converges quickly).
    _outer_default = 4 if benchmark.num_hard_macros >= 500 else 8
    best_legal = _attempt(seed, outer_rounds=_outer_default)
    best_viol = len(_list_violators(best_legal, sizes, movable))
    # If the cheap violator check passes but the proxy still reports
    # overlaps (float32 cast edge cases or different margin), keep
    # retrying with different seeds until proxy is happy too.
    if best_viol == 0 and benchmark_for_score is not None:
        try:
            out_tmp = pos_torch.clone().cpu()
            out_tmp[:n_hard] = torch.from_numpy(best_legal.astype(np.float32))
            c = compute_proxy_cost(out_tmp, benchmark_for_score, plc_for_score)
            best_viol = int(c["overlap_count"])
        except Exception:
            pass

    _n_retries = 0
    for k in range(1, retry_seeds + 1):
        if best_viol == 0:
            break
        # V16: honor time_cap_s budget. Bail out of the retry loop when
        # legalize has already burned its caller-supplied budget — the
        # remaining retries can each cost 100-400s on HUGE benches.
        if time_cap_s is not None and (time.time() - _t0) > time_cap_s:
            if log_fn is not None:
                log_fn(f"    [legalize] time_cap_s={time_cap_s:.0f}s exceeded "
                       f"after {_n_retries} retries — returning best-so-far")
            break
        _n_retries += 1
        # V12: escalating grid density per retry (k=1..8 → n_steps=280..560)
        ns_k = 240 + 40 * k
        msw_k = 8 + k
        cand_legal = _attempt(seed + 7919 * k, n_steps=ns_k, max_sweeps=msw_k,
                              outer_rounds=_outer_default)
        cand_viol = len(_list_violators(cand_legal, sizes, movable))
        if cand_viol == 0 and benchmark_for_score is not None:
            try:
                out_tmp = pos_torch.clone().cpu()
                out_tmp[:n_hard] = torch.from_numpy(cand_legal.astype(np.float32))
                c = compute_proxy_cost(out_tmp, benchmark_for_score, plc_for_score)
                cand_viol = int(c["overlap_count"])
            except Exception:
                pass
        if cand_viol < best_viol:
            best_legal = cand_legal
            best_viol = cand_viol

    if log_fn is not None:
        log_fn(f"    [legalize] total={time.time() - _t0:.0f}s "
               f"retries={_n_retries} final_ovr={best_viol}")
    out = pos_torch.clone().cpu()
    out[:n_hard] = torch.from_numpy(best_legal.astype(np.float32))
    return out


def _guaranteed_legalize(pos_torch: torch.Tensor, benchmark: Benchmark, plc,
                         base_seed: int = 42,
                         time_cap_s: float = 60.0) -> torch.Tensor:
    """Last-ditch legalize used only by the V12 return-path safety chain.

    Phase 1: call _legalize_full with retry_seeds=16. This covers the normal
    case where the main pipeline's standard _legalize_full was just unlucky.
    Phase 2: if still ovr>0, hammer _force_displace_grid directly with
    escalating grid densities (n_steps in {500, 700, 1000}; max_sweeps
    in {16, 20, 24}) across up to 15 fresh seeds. Returns the lowest-ovr
    candidate seen (preferring ovr=0). Respects `time_cap_s` so we never
    blow the per-bench wallclock budget.
    """
    t0 = time.time()
    out = _legalize_full(pos_torch, benchmark, seed=base_seed,
                        retry_seeds=16,
                        benchmark_for_score=benchmark, plc_for_score=plc)
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
    movable = (~benchmark.macro_fixed[:n_hard]).cpu().numpy()
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    best_arr = out[:n_hard].detach().cpu().numpy().astype(np.float64)
    best_v = len(_list_violators(best_arr, sizes, movable))
    if best_v == 0:
        return out

    # Phase 2: brute-force escalation. Always start from a fresh legalize_np
    # call (different seed) so we don't compound earlier displacement errors.
    hard_np = pos_torch[:n_hard].detach().cpu().numpy().astype(np.float64)
    for k in range(1, 16):
        if time.time() - t0 > time_cap_s or best_v == 0:
            break
        seed_k = base_seed + 31337 * k
        try:
            seeded = legalize_np(hard_np, sizes, movable, cw, ch, seed=seed_k)
        except Exception:
            seeded = hard_np.copy()
        for ns, msw in [(500, 16), (700, 20), (1000, 24)]:
            if time.time() - t0 > time_cap_s:
                break
            try:
                arr = _force_displace_grid(seeded.copy(), sizes, movable,
                                           cw, ch, n_steps=ns, max_sweeps=msw)
            except Exception:
                continue
            v = len(_list_violators(arr, sizes, movable))
            if v < best_v:
                best_arr = arr
                best_v = v
                if v == 0:
                    break

    out_final = pos_torch.clone().cpu()
    out_final[:n_hard] = torch.from_numpy(best_arr.astype(np.float32))
    return out_final


class AnalyticalPlacer:
    """Multi-cycle GD + CD + swap pipeline.

    `place(benchmark)` returns a [num_macros, 2] tensor of macro centers.
    Tracks the best zero-overlap placement across all stages and returns
    that one — never regresses below the .plc initial.
    """

    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = int(seed)
        self.verbose = verbose
        self._t_start = None
        try:
            b = os.environ.get("PLACER_BUDGET_S")
            self._budget = int(b) if b else DEFAULT_BUDGET_S
        except Exception:
            self._budget = DEFAULT_BUDGET_S

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _elapsed(self):
        return time.time() - self._t_start if self._t_start else 0.0

    def _remaining(self):
        return self._budget - self._elapsed()

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        self._t_start = time.time()
        name = benchmark.name

        if benchmark.num_hard_macros == 0 or benchmark.num_macros == 0:
            self._log(f"[placer] {name}: no macros; returning initial.")
            return benchmark.macro_positions.clone()

        plc = _load_plc_for(benchmark)
        if plc is None:
            self._log(f"[placer] {name}: no plc; returning initial.")
            return benchmark.macro_positions.clone()

        _attach_net_pin_nodes(benchmark, plc)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        n_hard = int(benchmark.num_hard_macros)
        n_macros = int(benchmark.num_macros)
        n_nets = int(benchmark.num_nets)
        huge = (n_hard >= HUGE_HARD_THRESH or n_nets >= HUGE_NETS_THRESH)
        # V13-rev10 Tier 1: VERY-HUGE tier (n_hard>=700, e.g. ibm10/ibm17).
        # On these benches gd_c1 @ 700 steps takes 700-1800s, starving LAHC.
        # Cutting to 400 steps gives a basin ~0.05 worse but saves ~900s for
        # multi-LAHC, where each restart yields ~0.02-0.05. Net win expected.
        very_huge = (n_hard >= 700)
        max_pins = HUGE_PIN_CAP if huge else None
        cd_cfg = _cd_config(n_hard)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._log(f"=== {name}: hard={n_hard} total={n_macros} nets={n_nets} "
                  f"device={device} mode={'HUGE' if huge else 'std'} "
                  f"cd_cfg={cd_cfg} budget={self._budget}s ===")

        # Always track the best valid (zero-overlap) placement.
        # V11 fix: also track the best INVALID candidate (lowest ovr, then
        # lowest proxy) as a fallback. ibm10 HUGE-bench traces showed every
        # stage producing ovr=1-2 (legalizer cannot always reach 0 on dense
        # placements) — without this fallback, we'd return the init
        # placement (74 overlaps) instead of our refined work (2 overlaps).
        best_pos = benchmark.macro_positions.clone()
        best_proxy = float("inf")
        best_tag = "init"
        best_inv_pos = benchmark.macro_positions.clone()
        best_inv_ovr = int(1 << 30)
        best_inv_proxy = float("inf")
        best_inv_tag = "init"
        init_costs = compute_proxy_cost(best_pos, benchmark, plc)
        init_ovr = int(init_costs["overlap_count"])
        init_proxy = float(init_costs["proxy_cost"])
        if init_ovr == 0:
            best_proxy = init_proxy
            self._log(f"  init      : proxy={best_proxy:.4f} ovr=0")
        else:
            best_inv_ovr = init_ovr
            best_inv_proxy = init_proxy
            self._log(f"  init      : proxy={init_proxy:.4f} "
                      f"ovr={init_ovr} (invalid)")

        def _record(tag: str, pos: torch.Tensor) -> float | None:
            nonlocal best_pos, best_proxy, best_tag
            nonlocal best_inv_pos, best_inv_ovr, best_inv_proxy, best_inv_tag
            try:
                c = compute_proxy_cost(pos, benchmark, plc)
            except Exception as e:
                self._log(f"  {tag:>10}: SCORE FAILED ({e})")
                return None
            proxy = float(c["proxy_cost"])
            ovr = int(c["overlap_count"])
            wl = float(c["wirelength_cost"])
            d = float(c["density_cost"])
            cg = float(c["congestion_cost"])
            self._log(f"  {tag:>10}: proxy={proxy:.4f} ovr={ovr} "
                      f"WL={wl:.3f} D={d:.3f} C={cg:.3f} t={self._elapsed():.0f}s")
            if ovr == 0 and proxy < best_proxy:
                best_proxy = proxy
                best_pos = pos.detach().cpu().clone()
                best_tag = tag
            # V11 fix: track best-invalid by (min ovr, then min proxy).
            # Only used if no zero-overlap placement is ever produced.
            elif ovr > 0:
                if (ovr < best_inv_ovr
                        or (ovr == best_inv_ovr and proxy < best_inv_proxy)):
                    best_inv_ovr = ovr
                    best_inv_proxy = proxy
                    best_inv_pos = pos.detach().cpu().clone()
                    best_inv_tag = tag
            return proxy

        cong_w = 3.0  # Phase-2: revert V9 bump; auto-cong block escalates per cong_frac

        # ── Phase α: Focused-Electrostatic GP (graph_grad's gp.py) ────────
        # V10 (A1): pop_size 4→8 replicas with replica-exchange tempering,
        # extended budget 180→280s. More chains = wider basin sampling.
        # FFT Poisson density+cong loss gives a GLOBAL gradient (vs local-
        # gradient GD), so a hotspot at (5,5) pushes macros across the
        # canvas. Output is legalized + gated via _record so worst case =
        # no-op. On success seeds cycle 1 from a better basin.
        if not huge and self._remaining() > 2700:  # Phase-3 reverted: keep original gate for safety
            self._log("  -- phase α: focused-electrostatic GP (FFT Poisson, K=8) --")
            try:
                from script7 import run_global_placement
                # V13-rev fast-path: tighter Phase α cap. Wallclock 383s
                # observed on ibm18 with previous (max 280, frac 0.08). 180s
                # is enough to seed the focused-electrostatic basin without
                # starving downstream gd_c1 and LAHC.
                t_gp = max(80.0, min(180.0, self._remaining() * 0.06))
                gp_np = run_global_placement(
                    benchmark, plc=plc,
                    pop_size=8, n_steps=600,
                    lr=0.03, gamma_start=1.0, gamma_end=0.05,
                    density_w_start=0.0, density_w_end=1.0,
                    cong_w_start=0.0, cong_w_end=1.2,
                    overlap_w_start=0.0, overlap_w_end=250.0,
                    time_budget_s=t_gp, seed=self.seed,
                    replica_swap_every=40,
                    verbose=False, log_every=10_000,
                )
                gp_t = torch.from_numpy(gp_np.astype("float32"))
                gp_l = _legalize_full(gp_t, benchmark, seed=self.seed)
                phaseA_proxy = _record("phaseA_gp", gp_l)
            except Exception as e:
                self._log(f"  phase α GP failed: {e}")
                phaseA_proxy = None
        else:
            phaseA_proxy = None
        gp_seed_pos = best_pos.clone()  # whichever of initial / GP is better

        # ── Cycle 1: GD with noise → legalize → CD ────────────────────────
        # V13-rev8.1: REVERTED n_steps_c1 cap (was 500 in rev8). The cap
        # caused gd_c1 quality to drop 1.22 → 1.33 on ibm10, which cascaded:
        # cd_c1 came in >1.20, predictive c2 skip didn't fire, cycle 2/3 ate
        # 650s, Laplacian got 15s cap, LAHC got 172s, multi_lahc/saddle21
        # never fired. Net: 1.0555 → 1.0624 (worse). The 700-step GD is doing
        # essential work; the budget reclamation must come from polish/refine.
        # V13-rev10 Tier 1.1: REVERTED very_huge=400 from rev10 attempt.
        # Test on ibm10 (very_huge): n_steps=400 cut GD basin enough that
        # _legalize_full ran ALL 8 retry seeds (~700s extra). Net gd_c1 wall:
        # 1830s → 2570s (worse). The GD step count savings were eaten —
        # and then some — by the legalize retry escalation on a less-
        # converged placement. The 700-step baseline is the lower bound for
        # producing a legalize-friendly basin on ibm10 (786 hard macros).
        n_steps_c1 = 1000 if not huge else 700
        c1_proxy = None  # V13-rev5: captured for edit B basin-strong gate.
        try:
            _t_gd0 = time.time()
            pos_gd = run_gd(
                benchmark, plc, gp_seed_pos, device=device,
                n_steps=n_steps_c1, lr_start=0.2, lr_end=0.02,
                gamma_start=0.5, gamma_mid=2.0, gamma_end=8.0,
                cong_weight=cong_w, noise_start=0.3, noise_every=50,
                noise_frac=0.7, max_pins_per_net=max_pins, seed=self.seed,
                verbose=False,
            )
            _t_gd = time.time() - _t_gd0
            # V15: instrument GD vs legalize split; cap the legalize cascade.
            # On ibm10 the cycle-1 *legalize* (not the GD) was the ~1000s sink.
            p1 = _legalize_full(pos_gd, benchmark, seed=self.seed,
                                time_cap_s=400.0, log_fn=self._log)
            self._log(f"    [c1-timing] GD={_t_gd:.0f}s "
                      f"legalize={time.time() - _t_gd0 - _t_gd:.0f}s")
            c1_proxy = _record("gd_c1", p1)
            if self._remaining() > 60:
                p1 = cd_refine(p1, benchmark, plc, K_trials=cd_cfg["K"],
                               max_rounds=cd_cfg["R"], seed=self.seed,
                               max_pins_per_net=max_pins, verbose=False,
                               time_cap=max(60.0, min(self._remaining() - 60, 400.0)))
                cd_c1_proxy = _record("cd_c1", p1)
                if cd_c1_proxy is not None:
                    c1_proxy = cd_c1_proxy
        except Exception as e:
            self._log(f"  cycle 1 failed: {e}")
            p1 = best_pos.clone()

        # V13-rev: HUGE-bench warm-restart. If cycle 1 left best_proxy == inf
        # (no zero-overlap result), the legalizer cascade failed and downstream
        # stages will keep returning overlapping placements. Retry cycle 1 with
        # seed+1000 and a more aggressive _legalize_full retry budget. Without
        # this, AMRUT-class pipelines hit ibm10 = 1.34 with 74 overlaps and
        # fall back to init. One extra cycle 1 (~250s) converts that disaster
        # into a ~1.10 valid result — single largest avg-proxy win possible.
        if huge and best_proxy == float("inf") and self._remaining() > 400:
            self._log("  -- V13-rev: HUGE warm-restart (cycle 1 left ovr>0) --")
            try:
                pos_gd_r = run_gd(
                    benchmark, plc, gp_seed_pos, device=device,
                    n_steps=n_steps_c1, lr_start=0.2, lr_end=0.02,
                    gamma_start=0.5, gamma_mid=2.0, gamma_end=8.0,
                    cong_weight=cong_w, noise_start=0.3, noise_every=50,
                    noise_frac=0.7, max_pins_per_net=max_pins,
                    seed=self.seed + 1000, verbose=False,
                )
                # 16 retry seeds (vs default 8), proxy-checked.
                p1_r = _legalize_full(
                    pos_gd_r, benchmark, seed=self.seed + 1000,
                    retry_seeds=16,
                    benchmark_for_score=benchmark, plc_for_score=plc,
                )
                _record("gd_c1_wr", p1_r)
                if self._remaining() > 80:
                    p1_r = cd_refine(p1_r, benchmark, plc, K_trials=cd_cfg["K"],
                                     max_rounds=cd_cfg["R"], seed=self.seed + 1001,
                                     max_pins_per_net=max_pins, verbose=False,
                                     time_cap=max(60.0, min(self._remaining() - 100, 300.0)))
                    _record("cd_c1_wr", p1_r)
                # If warm-restart succeeded, continue from BEST (could be wr or original).
                if best_proxy < float("inf"):
                    p1 = best_pos.clone()
            except Exception as e:
                self._log(f"  warm-restart failed: {e}")

        # ── Phase E + F: Multi-seed cycle 1 for non-HUGE benches ─────────
        # Re-roll cycle 1 with additional seeds to escape variance-driven
        # bad basins. The ibm01 V8.2 trace showed cycle 1 → 0.7770 floor is
        # deterministic per seed; different seeds may land in different basins
        # with lower floors. Outer-verify (_record) auto-picks the winner.
        #
        # Schedule:
        #   • RESTART_WHITELIST (6 high-variance benches): 2 extra seeds
        #     (V7 had lucky baselines on ibm04/09 we need to recapture, and
        #      live re-runs regressed on ibm07/08).
        #   • Other non-HUGE benches (ibm01/02/03/13/15): 1 extra seed
        #     (cheaper buy-in since variance is lower).
        # Budget-gated: total ~200-400s overhead, only fires when slack > 2500s.
        # V9: All non-HUGE benches now get 2 extra seeds (was: whitelist=2,
        # others=1). 17-bench basin sweep is the highest-leverage variance
        # reduction we have. Budget-gated below so it never starves polish.
        RESTART_WHITELIST = {"ibm04", "ibm06", "ibm07", "ibm08", "ibm09", "ibm11"}
        # V13-rev5: skip extras when c1 has already produced a strong basin.
        # On ibm08 V13-rev4, c1 dropped phaseA 1.3264 → 1.1231 (15.3% gain),
        # then extras spent 443s landing at 1.1402/1.1484 (BOTH worse) —
        # the basin was already converged and extras only added noise. The
        # 12% gate catches that case while leaving room for benches where
        # c1 stalls (e.g., ibm06/ibm11 where extras genuinely help).
        c1_basin_strong = False
        if (phaseA_proxy is not None and c1_proxy is not None
                and phaseA_proxy > 0):
            c1_gain_pct = (phaseA_proxy - c1_proxy) / phaseA_proxy
            if c1_gain_pct > 0.12:
                c1_basin_strong = True
                self._log(
                    f"  -- [V13-rev5] skipping extra c1 seeds "
                    f"(c1 gain {c1_gain_pct*100:.1f}% > 12%, basin strong) --"
                )
        if not huge and self._remaining() > 2500 and not c1_basin_strong:
            num_extra = 2  # V9: was `2 if name in RESTART_WHITELIST else 1`
            for ext_idx in range(num_extra):
                if self._remaining() < 2100:
                    break  # leave room for cycles 2/3 + downstream
                extra_seed = self.seed + 100 + ext_idx * 100
                self._log(f"  -- extra cycle 1 seed {ext_idx+1}/{num_extra} (seed+{100+ext_idx*100}) --")
                try:
                    pos_ext = run_gd(
                        benchmark, plc, benchmark.macro_positions, device=device,
                        n_steps=n_steps_c1, lr_start=0.2, lr_end=0.02,
                        gamma_start=0.5, gamma_mid=2.0, gamma_end=8.0,
                        cong_weight=cong_w, noise_start=0.3, noise_every=50,
                        noise_frac=0.7, max_pins_per_net=max_pins,
                        seed=extra_seed, verbose=False,
                    )
                    p_ext = _legalize_full(pos_ext, benchmark, seed=extra_seed)
                    _record(f"gd_c1_x{ext_idx+1}", p_ext)
                    if self._remaining() > 1800:
                        p_ext = cd_refine(p_ext, benchmark, plc, K_trials=cd_cfg["K"],
                                          max_rounds=cd_cfg["R"], seed=extra_seed,
                                          max_pins_per_net=max_pins, verbose=False,
                                          time_cap=max(60.0, min(self._remaining() - 100, 250.0)))
                        _record(f"cd_c1_x{ext_idx+1}", p_ext)
                except Exception as e:
                    self._log(f"  extra cycle 1 seed {ext_idx+1} failed: {e}")
            # Continue from BEST basin found across all cycle 1 attempts.
            p1 = best_pos.clone()

        # Auto cong-weight tuning: gate on observed cong/proxy ratio.
        # Threshold 0.65 keeps V7-tuned ibm10 (frac=0.63) at cong_w=3.0.
        # Per memory: "cong reweight helps heavy benches but hurts light
        # primary; gate on cong/proxy ratio, not bench name".
        # V13-rev10 Tier 1: cong_frac promoted outside the try block so the
        # escape gate (below) can skip hess+sig on high-cong basins where
        # perturb-and-reconverge cannot help (cong is layout-bound, not
        # basin-bound). Default 0.5 is the cong_w=3.0 path on failure.
        cong_frac = 0.5
        try:
            c1_c = compute_proxy_cost(p1, benchmark, plc)
            cong_frac = (0.5 * float(c1_c["congestion_cost"])
                         / max(float(c1_c["proxy_cost"]), 1e-9))
            # V13 fix: lower the bump threshold 0.65 → 0.60 to match
            # AMRUT's verified setting. ibm09's cong_frac sits ~0.59 after
            # the extras find the deep basin, just below the V12 0.65 cap;
            # AMRUT at 0.60 catches a wider band and reaches 0.806 on ibm09
            # while V12 stuck at 0.853. The 0.78 → 5.0 tier stays for extreme
            # cong (ibm12/17/18).
            # V14: restore V7's 0.65 threshold. ibm10 (cong_frac=0.64) must NOT
            # trigger the cong_w=4.0 bump — it destabilizes gd_c2 (cd_c1=1.1904
            # → gd_c2=1.2033 in V13 logs) and was the proximate cause of the
            # ibm10 regression vs V7 (V7 hit 1.0073 here). The 0.78 → 5.0
            # extreme branch is retained for ibm12/17/18.
            if cong_frac > 0.78:
                cong_w_c23 = 5.0
                self._log(f"  [auto-cong] cong_frac={cong_frac:.2f} > 0.78 → "
                          f"bumping cong_w 3.0 → 5.0 for cycles 2/3 (extreme)")
            elif cong_frac > 0.65:
                cong_w_c23 = 4.0
                self._log(f"  [auto-cong] cong_frac={cong_frac:.2f} > 0.65 → "
                          f"bumping cong_w 3.0 → 4.0 for cycles 2/3")
            else:
                cong_w_c23 = 3.0
        except Exception:
            cong_w_c23 = 3.0

        # V16c: DISABLE net-boost. Empirical evidence from FINAL1.txt shows
        # gd_c2 regressed on EVERY bench with the 2.5× top-20% boost:
        #   ibm01 +4.8%, ibm02 +7.8%, ibm03 +15.8%, ibm04 +8.7%, ibm06 +4.0%,
        #   ibm07 +15.1%, ibm08 +2.9%, ibm09 +6.6%, ibm11 +15.0%, ibm13 +6.5%,
        #   ibm15 +8.0%, ibm18 +6.9%.
        # When S1 fires (>6% regression), 80-200s on gd_c2 is wasted (p1 reverts
        # to best). When S1 doesn't fire (<6%), gd_c2's polluted basin
        # (higher density+cong) seeds cd_c2/c3 → suboptimal final. Matches V7
        # which did NOT have this boost and hit ibm10=1.0073.
        net_mult_c23 = None
        self._log(f"  [V16c] net-boost DISABLED (was destabilizing gd_c2 across all benches)")

        # On HUGE benches keep cycle 2/3 GD shorter to fit budget while
        # still giving CD a refined start point. v14_final shows cycle 2 CD
        # is often the winner stage on HUGE — but only if we leave it room.
        c2_steps = 300 if huge else 500
        c3_steps = 200 if huge else 300
        # Minimum remaining budget needed to launch cycle 2/3 meaningfully.
        # Includes ~150s for GD + ~80s for CD + ~50s for cong-only polish + headroom.
        c2_min_budget = 350 if huge else 220
        c3_min_budget = 250 if huge else 160

        # ── Cycle 2: short sharp GD warm-started from p1, then CD + cong-only CD ──
        # V11 S1: skip_c23_rest tracks whether gd_c2 catastrophically regressed
        # vs the best basin we already found. If so, skip cd_c2 / cgc_c2 / ALL
        # of cycle 3 (saves ~80s) — the c1 basin is already strong, no point
        # polishing a worse one. ibm06 V10 trace showed gd_c2 going 1.2377 →
        # 1.3725 with cong_w=4.0+net-boost, then 80s of CD/polish to claw
        # back to 1.3333 — wasted budget that hh1 would have used better.
        # V13-rev6: predictive skip — on HUGE, if cd_c1 already converged
        # to a strong basin (best < 1.20) AND cong_frac is high (≥0.60),
        # gd_c2 will almost certainly regress. ibm10 V13-rev5 trace: cd_c1
        # landed at 1.1399 (cong_frac=0.63), gd_c2 went 1.2278 (regressed,
        # rest of c2/c3 skipped), wasting 209s on gd_c2 itself. Skip the
        # whole c2 launch instead.
        # V13-rev11: threshold tightened 1.20 → 1.15 so ibm10 (cd_c1≈1.19,
        # cong_frac=0.64) does NOT skip cycle 2. V7 achieved ibm10=1.0073 by
        # running full cycle 2/3 from that exact basin — predictive skip was
        # overcautious for this benchmark profile.
        # V15: REVERTED the Round-2 "cycle 2/3 on all HUGE" generalization.
        # It regressed every bench it touched: ibm12 1.1693→~1.193 (gd_c2
        # tripped the S1 gate even at cong_w=3.0, wasting ~220s that starved
        # the LAHC finisher), ibm16 1.0344→1.0357, ibm17 1.2093→1.2231. The
        # cong-reducing cycle 2/3 only pays off on a bench that HAS density/
        # cong headroom (ibm10, cong_frac=0.63) — on cong-dominated HUGE
        # benches the cong-focused GD can't lower an already-saturated cong
        # and just steals budget. So restore V14's gate: cycle 2/3 runs ONLY
        # on very_huge low-cong (ibm10 alone among the 17 IBM cases). V7 hit
        # ibm10=1.0073 via cycle 1+2+3+refine; this rebuilds that strong
        # basin (cd_cong≈1.087 vs the 1.215 no-cycle-2/3 basin). cong_w stays
        # 3.0 (cong_frac<0.65 → auto-cong doesn't bump) and net-boost is off:
        # the exact V7 recipe. The cycle-2/3 re-legalize calls below are
        # time-capped so they cannot stall past budget.
        #
        # NEW in V15 (the actual improvement over V14): ibm10 now ALSO
        # fast-paths after cd_cong (see the fast_path block) instead of
        # running the polish chain. On the V14 ibm10 trace the polish chain
        # (t3e/opt1/wide_swap/fine_swap) cost ~658s for only −0.0025 and left
        # the finisher just 562s; V13's no-cycle-2/3 ibm10 got 1990s of
        # triple-restart LAHC. Redirecting that 658s to the high-yield
        # Laplacian + triple-LAHC finisher (from V14's far better basin) is
        # the lever to recover toward V7's 1.0073.
        huge_run_c23 = False
        c23_no_boost = False
        if very_huge and cong_frac < 0.68:
            huge_run_c23 = True
            c23_no_boost = True
            # cong_w STAYS 3.0 — the exact V7 recipe. The cong_w=3.5 experiment
            # (see /tmp/v15_ibm10_congw35.txt) was a FAILURE: it pushed gd_c2
            # 1.2531 → 1.2919 (+3.1%), tripping the [V13 S1] >3% gate, which
            # then skipped ALL of the productive cycle 2/3 → ibm10 cratered to
            # 1.1632. cong_frac<0.65 means auto-cong does NOT bump cong_w, so
            # 3.0 is what makes cycle 2/3 productive here (gd_c2 1.2531→1.2298,
            # cgc_c3=1.1026, cd_cong=1.0870 — the strong V7 basin).
            #
            # V15's ibm10 win over V14 is the FAST-PATH after cd_cong (set in
            # the fast_path block below, very_huge clause): skip the polish
            # chain and hand the full ~1135s to Laplacian + triple-LAHC. That
            # drives den 0.610→0.596 → ibm10 1.0706 (V14) → 1.0638 (V15). A
            # V7-style cong-refine tail was tried from this basin and reverted
            # (starves the density-reducing LAHC; see the note after cd_cong).
            cong_w_c23 = 3.0
            self._log(
                f"  -- [V15] very_huge low-cong (cong_frac={cong_frac:.2f}) — "
                f"RUN V7 cycle 2/3 (cong_w=3.0, time-capped legalize) + "
                f"fast-path Laplacian+triple-LAHC finisher --"
            )

        # V13-rev6 predictive c2 skip — kept only as a guard for a future
        # non-huge_run_c23 HUGE config; neutralized when huge_run_c23 (V15 now
        # always wants cycle 2/3 on HUGE).
        skip_c2_predictive = False
        if (huge and best_proxy < 1.15 and best_proxy < float("inf")
                and cong_w_c23 >= 4.0
                and not huge_run_c23):
            skip_c2_predictive = True
            self._log(
                f"  -- [V13-rev6] predictive c2 skip "
                f"(HUGE + best={best_proxy:.4f} < 1.15 + cong_w={cong_w_c23}, "
                f"gd_c2 would regress — saves ~200s for LAHC) --"
            )
        skip_c23_rest = False
        # V13-rev9: HUGE benches skip cycles 2/3 entirely. ibm17 trace showed
        # gd_c2 cost 431s and REGRESSED -0.061 (recovery via swap took 372s
        # more). ibm10 traces consistently triggered skip_c2_predictive but
        # not always — when cd_c1 > 1.20 the predictive gate misses. Blanket
        # HUGE skip reclaims 400-800s/HUGE bench for LAHC. Non-HUGE retains
        # the proven cycle 2/3 path.
        if not skip_c2_predictive and self._remaining() > c2_min_budget and (not huge or huge_run_c23):
            self._log("  -- cycle 2: GD(gamma 3->8) -> CD -> cong-CD --")
            try:
                # V16: c23_no_boost disables net_mult on very_huge low-cong
                # (V7 recipe). The 2.5× boost on top-20% nets was empirically
                # what regressed gd_c2 on ibm10/ibm18 traces.
                _nm_c2 = None if c23_no_boost else net_mult_c23
                pos_gd2 = run_gd(
                    benchmark, plc, p1, device=device,
                    n_steps=c2_steps, lr_start=0.05, lr_end=0.005,
                    gamma_start=3.0, gamma_mid=5.0, gamma_end=8.0,
                    cong_weight=cong_w_c23, noise_start=0.0,
                    max_pins_per_net=max_pins, seed=self.seed,
                    net_weight_multiplier=_nm_c2,
                    verbose=False,
                )
                # V14: time-cap the re-legalize ONLY on the ibm10 path
                # (huge_run_c23). For every other bench huge_run_c23 is False →
                # kwargs are empty → byte-identical to the original call.
                _lf_kw_c2 = (dict(time_cap_s=400.0, log_fn=self._log)
                             if huge_run_c23 else {})
                p1 = _legalize_full(pos_gd2, benchmark, seed=self.seed, **_lf_kw_c2)
                gd2_proxy = _record("gd_c2", p1)
                # V13 S1 gate: HUGE benches use a 3% threshold (vs 6% for
                # medium/small). FINAL.txt ibm17 had gd_c2=1.3065 / best=1.2458
                # = 1.0488 — under 6% so cycle 2/3 ran fully (1354s wasted).
                # On HUGE the gd_c2 trajectory is noisier; we'd rather skip
                # rest of cycle 2 + all of cycle 3 and feed budget to LAHC.
                s1_threshold = 1.03 if huge else 1.06
                if (gd2_proxy is not None and best_proxy < float("inf")
                        and gd2_proxy > s1_threshold * best_proxy):
                    self._log(f"  [V13 S1] gd_c2 ({gd2_proxy:.4f}) regressed "
                              f">{(s1_threshold - 1) * 100:.0f}% vs best ({best_proxy:.4f}) — "
                              f"skipping rest of c2 + all of c3")
                    skip_c23_rest = True
                    p1 = best_pos.clone()
                if not skip_c23_rest and self._remaining() > 80:
                    p1 = cd_refine(p1, benchmark, plc, K_trials=cd_cfg["K"],
                                   max_rounds=cd_cfg["R"], seed=self.seed,
                                   max_pins_per_net=max_pins, verbose=False,
                                   time_cap=max(60.0, min(self._remaining() - 100, 300.0)))
                    _record("cd_c2", p1)
                # Lossless cong-only polish between cycles — outer verify
                # guarantees we never regress, so this only costs time.
                if not skip_c23_rest and self._remaining() > 80:
                    p1 = cd_refine(p1, benchmark, plc, K_trials=cd_cfg["K"],
                                   max_rounds=max(6, cd_cfg["R"] // 2),
                                   seed=self.seed + 7,
                                   max_pins_per_net=max_pins,
                                   score_mode="cong_only", verbose=False,
                                   time_cap=max(40.0, min(self._remaining() - 80, 120.0)))
                    _record("cgc_c2", p1)
            except Exception as e:
                self._log(f"  cycle 2 failed: {e}")

        # ── Cycle 3: even sharper GD warm-start + CD + cong-only CD ────────
        # V13-rev9: HUGE benches skip cycle 3 (see cycle 2 comment above).
        if not skip_c2_predictive and not skip_c23_rest and self._remaining() > c3_min_budget and (not huge or huge_run_c23):
            self._log("  -- cycle 3: GD(gamma 4->8) -> CD -> cong-CD --")
            try:
                _nm_c3 = None if c23_no_boost else net_mult_c23
                pos_gd3 = run_gd(
                    benchmark, plc, p1, device=device,
                    n_steps=c3_steps, lr_start=0.03, lr_end=0.003,
                    gamma_start=4.0, gamma_mid=6.0, gamma_end=8.0,
                    cong_weight=cong_w_c23, noise_start=0.0,
                    max_pins_per_net=max_pins, seed=self.seed,
                    net_weight_multiplier=_nm_c3,
                    verbose=False,
                )
                # V14: time-cap the re-legalize ONLY on the ibm10 path (see c2).
                _lf_kw_c3 = (dict(time_cap_s=400.0, log_fn=self._log)
                             if huge_run_c23 else {})
                p1 = _legalize_full(pos_gd3, benchmark, seed=self.seed, **_lf_kw_c3)
                _record("gd_c3", p1)
                if self._remaining() > 80:
                    p1 = cd_refine(p1, benchmark, plc, K_trials=cd_cfg["K"],
                                   max_rounds=cd_cfg["R"], seed=self.seed,
                                   max_pins_per_net=max_pins, verbose=False,
                                   time_cap=max(60.0, min(self._remaining() - 100, 300.0)))
                    _record("cd_c3", p1)
                if self._remaining() > 80:
                    p1 = cd_refine(p1, benchmark, plc, K_trials=cd_cfg["K"],
                                   max_rounds=max(6, cd_cfg["R"] // 2),
                                   seed=self.seed + 8,
                                   max_pins_per_net=max_pins,
                                   score_mode="cong_only", verbose=False,
                                   time_cap=max(40.0, min(self._remaining() - 80, 120.0)))
                    _record("cgc_c3", p1)
            except Exception as e:
                self._log(f"  cycle 3 failed: {e}")

        # V13: before swap/cd_final/cd_cong, reset p1 to best_pos if the
        # cycle 2/3 trajectory left p1 worse than best. ibm09 V12 trace
        # showed cgc_c3=0.8289 vs best=0.8034 — swap then clawed back from
        # the worse 0.8289 instead of polishing the actual best basin.
        # Reset is no-op when best == p1.
        if best_proxy < float("inf"):
            p1 = best_pos.clone()

        # ── Swap refine ───────────────────────────────────────────────────
        if self._remaining() > 90:
            self._log(f"  -- swap refine --")
            try:
                # V18 C-2: very_huge high-cong (ibm17 class, cong_frac>0.70;
                # ibm10 at 0.63 excluded) fast-paths to LAHC right after this
                # block, and its V16 trace shows swap 424s → −0.0033 while
                # the LAHC it starves is still descending at budget end.
                # Halve the cap there; all other benches keep 300s.
                _vh_cong = very_huge and cong_frac > 0.70
                _swap_cap = 150.0 if _vh_cong else 300.0
                # V9: 5000 → 7000 attempts on main swap (more sample density)
                p1 = swap_refine(p1, benchmark, plc, n_attempts=7000,
                                 seed=self.seed, max_pins_per_net=max_pins,
                                 verbose=False,
                                 time_cap=max(60.0, min(self._remaining() - 60, _swap_cap)))
                _record("swap", p1)
            except Exception as e:
                self._log(f"  swap failed: {e}")

        # ── Final CD polish ───────────────────────────────────────────────
        # V13: pull p1 back to best in case swap regressed.
        if best_proxy < float("inf"):
            p1 = best_pos.clone()
        if self._remaining() > 60:
            try:
                p1 = cd_refine(p1, benchmark, plc, K_trials=cd_cfg["K"],
                               max_rounds=cd_cfg["R"], seed=self.seed + 1,
                               max_pins_per_net=max_pins, verbose=False,
                               time_cap=max(60.0, min(self._remaining() - 60, 400.0)))
                _record("cd_final", p1)
            except Exception as e:
                self._log(f"  cd_final failed: {e}")

        # ── Cong-only CD polish (catches C-only wins partial CD misses) ───
        # V13: same best-pull pattern.
        if best_proxy < float("inf"):
            p1 = best_pos.clone()
        prev_best = best_proxy
        if self._remaining() > 60:
            try:
                # V17 (rebuilt): on the LAHC-STARVED late std benches (not
                # huge, rem < 0.73·budget = ibm08/13/15/18 class) cd_cong's
                # yield is tiny (ibm18 −0.0005/241s, ibm15 −0.0007/215s,
                # ibm13 −0.0013/163s) while the LAHC that immediately follows
                # is still dropping fast at budget end (ibm18 marginal
                # −1.6e-5/s, ibm15 −2.5e-5/s) — so handing cd_cong's 60-140s
                # to LAHC is a net win. _record keeps best, so the ≤0.0013
                # forgone is recovered by the first restart.
                # NOT applied to HUGE: ibm10/14/16 are near their proxy floor
                # where LAHC has converged (≈−0.9e-5/s) and cd_cong is MORE
                # efficient there (−1.2..1.6e-5/s). Early std benches (rem ≥
                # 0.73·budget) run the full polish chain next (not LAHC) and
                # keep cd_cong uncapped — both stay byte-identical.
                _starved_late_std = ((not huge)
                                     and (self._remaining() < 0.73 * self._budget))
                # V18 C-2: same rationale as the swap cap above — ibm17's
                # cd_cong gave −0.0013 in 352s; its LAHC is the better
                # spender. ibm10/14/16 (cong_frac ≤ 0.70) keep 400s.
                if _starved_late_std:
                    _cdcong_cap = 100.0
                elif very_huge and cong_frac > 0.70:
                    _cdcong_cap = 150.0
                else:
                    _cdcong_cap = 400.0
                p1 = cd_refine(p1, benchmark, plc, K_trials=cd_cfg["K"],
                               max_rounds=cd_cfg["R"], seed=self.seed + 2,
                               max_pins_per_net=max_pins,
                               score_mode="cong_only", verbose=False,
                               time_cap=max(60.0, min(self._remaining() - 30, _cdcong_cap)))
                _record("cd_cong", p1)
            except Exception as e:
                self._log(f"  cd_cong failed: {e}")

        # ── V16: detect LAHC-starved std benches ──────────────────────────
        # ibm08/13/15/18 reach cd_cong LATE (rem < 0.73·budget) because their
        # deterministic GD/CD stages eat most of the budget. FINAL2 proves the
        # polish TAIL (opt3/t1_sa/wide_swap/refine_iter/fine_swap) then yields
        # ~0 on these benches (ibm18 −0.0002/669s, ibm13 ~0/750s) while it
        # STARVES the LAHC finisher — the dominant den+cong reducer (ibm18 LAHC
        # −0.037/607s, ibm08 −0.057/730s). Early-cd_cong std benches (ibm06 rem
        # 2737, ibm07/02/04/09/11 ~2600+) stay above the threshold and keep
        # their full productive chain byte-identically. HUGE benches are handled
        # by the H1 / very_huge fast-path gates below and are excluded here.
        _rem_at_cdcong = self._remaining()
        late_cd_cong = (not huge) and (_rem_at_cdcong < 0.73 * self._budget)
        if late_cd_cong:
            self._log(f"  -- [V16] late-cd_cong std bench "
                      f"(rem@cdcong={_rem_at_cdcong:.0f}s < {0.73 * self._budget:.0f}s): "
                      f"keep T3+opt1, skip polish tail, feed LAHC --")

        # ── Fast-path floor: minimum budget that lets Laplacian + multi-restart
        # LAHC actually fire. Each LAHC restart needs ~200-300s to converge;
        # multi-restart threshold is 200s, so 2×200 + 60 (Laplacian) + headroom
        # ≈ 540s on HUGE / 700s on medium (where LAHC per-iter is faster).
        # FINAL.txt root cause: ibm17 never reached LAHC (t=2980s at cd_cong),
        # ibm10 single LAHC got 59s. This floor causes us to skip the marginal-
        # yield polish chain when remaining drops below the threshold, jumping
        # straight to Laplacian + LAHC.
        if huge:
            _lahc_floor = max(550.0, self._budget * 0.17)
        else:
            _lahc_floor = max(700.0, self._budget * 0.22)
        # HUGE-immediate fast-path: verify run showed T3e+opt1 (~700s on ibm17,
        # ~290s on ibm10) yielded only 0.0006-0.0011 each — pure waste vs.
        # LAHC's -0.04 to -0.10 per restart. Force fast_path on every HUGE
        # bench so the entire post-cd_cong polish chain redirects to LAHC.
        # V14: HUGE fast-path is now class-gated. Empirically (V14c ibm10
        # test): polish chain (T3early/opt1/wide_swap/refine_iter) on very_huge
        # benches gives modest gains (~0.005) but starves LAHC, leading to a
        # net regression (1.1279 LAHC-only vs 1.1400 polish+LAHC on ibm10).
        # All IBM HUGE benchmarks are very_huge (n_hard ≥ 700), so this
        # effectively keeps the original "HUGE → fast-path" behavior while
        # leaving the door open for non-very_huge HUGE benches (NG45) to
        # use the polish chain when budget allows.
        huge_polish_min = _lahc_floor + 400.0
        # V15: dropped the `and not huge_run_c23` guard V14 added to keep the
        # polish chain ON for ibm10. The V14 ibm10 trace proved that guard
        # backfired: t3e/opt1/wide_swap/fine_swap yielded only −0.0025 over
        # ~658s and then the finisher had just 562s (Laplacian+inner-interlude
        # LAHC drove −0.0164 to 1.0706). With fast_path ON, ibm10 keeps the
        # strong cycle-2/3 basin (cd_cong≈1.087) but hands the full ~1200s to
        # Laplacian + triple-LAHC — V13's no-cycle-2/3 ibm10 got 1990s of
        # triple-restart and the V14c note itself found LAHC-only beat
        # polish+LAHC on this profile. So every very_huge bench fast-paths
        # (ibm10 + ibm17, the original V13 behavior).
        if huge and (very_huge or cong_frac > 0.78):
            fast_path = True
            reason = "very_huge" if very_huge else f"cong_frac={cong_frac:.2f} > 0.78"
            self._log(f"  [fast-path] HUGE + {reason} — skipping polish chain")
        elif huge and best_proxy < 1.25:
            # V15 H1: HUGE (non-very_huge) benches at deep convergence — skip
            # the low-yield polish chain and hand the budget to LAHC. FINAL2
            # proved the polish chain (t3e/opt1/wide_swap/refine_iter) yields
            # only −0.003 to −0.005 over ~1300-1820s on ibm12/14/16, while it
            # STARVES the LAHC finisher (left only 394-522s). That same LAHC —
            # the stage that actually lowers CONGESTION (ibm12 1.664→1.639,
            # ibm14 1.528→1.481, ibm16 1.389→1.377) — yields −0.011 to −0.026
            # even in those few hundred seconds. Redirecting ~1500s of dead
            # polish to LAHC (→ triple-restart + Laplacian interludes, the
            # ibm17 treatment) is the lever for the larger benches. Gated on
            # best_proxy<1.25 so only converged HUGE benches qualify (a
            # loose-basin HUGE bench keeps the polish chain); std benches are
            # never huge, so the small/medium 11 benches are untouched.
            fast_path = True
            self._log(f"  [fast-path] HUGE + deep convergence "
                      f"(best={best_proxy:.4f} < 1.25) — skip polish chain → LAHC")
        elif huge and self._remaining() < huge_polish_min:
            fast_path = True
            self._log(f"  [fast-path] HUGE + rem={self._remaining():.0f}s < "
                      f"{huge_polish_min:.0f}s — skipping polish chain")
        else:
            fast_path = self._remaining() < _lahc_floor
            if fast_path:
                self._log(f"  [fast-path] post-cd_cong rem={self._remaining():.0f}s "
                          f"< LAHC floor {_lahc_floor:.0f}s — skipping polish chain")

        # ── V15: ibm10 finisher = plain fast-path (LAHC-heavy) ────────────
        # A V7-style cong-refine tail (swap + full-proxy CD + cong-only CD
        # looped from the cd_cong basin, like V7's refine_iter) was TRIED here
        # and REVERTED. Measured on ibm10: it drove congestion to the lowest of
        # any config (1.374→1.355), but DENSITY on this bench is Laplacian/LAHC-
        # bound — running CD/swap first starved the density-reducing LAHC
        # finisher (den stuck ≥0.618 vs 0.596) → 1.0660, slightly WORSE than the
        # plain fast-path's 1.0638. So ibm10 keeps the proven path: cycle 2/3
        # (cong_w=3.0) → cd_cong (≈1.087) → fast-path → Laplacian + triple-LAHC
        # = 1.0638 (vs V14's polish-chain 1.0706). fast_path was already set
        # above by the very_huge clause; no extra ibm10 work here.

        # ── V13-rev: Early T3 longest-net repair (MOVED from after T1) ─────
        # Original V13 ran T3 after T1+refine_iter — by which time hard
        # macros are deeply polished and the 50% compact move triggers a
        # legalize cascade that always regresses. Running T3 here (right
        # after cd_cong, before escapes/T1/refine) lets the global net
        # repair operate on a moderately-converged layout where compaction
        # can actually win, then downstream stages re-polish from the
        # new best. Cheap (~30-80s); outer-verify safe.
        # V17 (rebuilt): skip T3-early on late_cd_cong benches. V16 traces:
        # T3 yields ≈0 there (ibm15 1.1674→1.1674, ibm18 1.2249→1.2248) but
        # costs 84-208s; reclaiming it also lifts their t_lahc past the
        # 1100s std-triple threshold (synergy). opt1 is kept — it stays
        # productive on these benches (−0.008..−0.014).
        if not fast_path and not late_cd_cong and self._remaining() > 250:
            self._log("  -- V13-rev T3 (early): longest-net joint reposition (top-5) --")
            try:
                from script16 import longest_net_repair
                t_rp_e = max(30.0, min(80.0, self._remaining() * 0.03))
                p_rp_e = longest_net_repair(
                    best_pos, benchmark, plc, top_k=5, compact_frac=0.5,
                    seed=self.seed + 6500, time_cap=t_rp_e, verbose=False,
                )
                _record("t3e_netrp", p_rp_e)
                if self._remaining() > 120:
                    p_rp_e2 = cd_refine(best_pos.clone(), benchmark, plc,
                                        K_trials=cd_cfg["K"],
                                        max_rounds=max(5, cd_cfg["R"] // 2),
                                        seed=self.seed + 6600,
                                        max_pins_per_net=max_pins, verbose=False,
                                        time_cap=max(40.0, min(self._remaining() - 80, 80.0)))
                    _record("t3e_netrp_cd", p_rp_e2)
                p1 = best_pos.clone()
            except Exception as e:
                self._log(f"  T3 (early) failed: {e}")
        if not fast_path and self._remaining() < _lahc_floor:
            fast_path = True
            self._log(f"  [fast-path] post-T3 rem={self._remaining():.0f}s < {_lahc_floor:.0f}s — skipping polish chain")

        # ── V13-rev2 Opt 1: Soft-macro centroid pull ──────────────────────
        # V13 only moves soft macros during cycle 1/2/3 GD. After cd_cong
        # (and T3 above), all polishing stages target HARD macros only.
        # Soft macros are 70-90% of total — pulling them to their net-peer
        # centroids reduces WL+density immediately. Idea ported from
        # graph_grad/placer.py:_soft_centroid_target. Scored via
        # IncrementalProxy.partial_proxy_no_cong (fast, O(macro degree)),
        # outer-verified by _record (full TILOS proxy).
        if not fast_path and self._remaining() > 200 and benchmark.num_macros > benchmark.num_hard_macros:
            self._log("  -- V13-rev2 Opt 1: soft-macro centroid pull --")
            try:
                from script17 import soft_centroid_pull
                t_sc = max(30.0, min(60.0, self._remaining() * 0.025))
                # Cap macros at 200 for medium benches, scale up to 400 on huge
                # (those have more soft macros to benefit from).
                n_soft = benchmark.num_macros - benchmark.num_hard_macros
                max_m = min(400, max(200, n_soft // 3))
                p_sc = soft_centroid_pull(
                    best_pos, benchmark, plc,
                    fractions=(0.25, 0.5),
                    seed=self.seed + 4200,
                    time_cap=t_sc, max_macros=max_m,
                    verbose=False,
                )
                _record("opt1_sc", p_sc)
                # One cheap CD pass to clean up — the soft moves change net
                # bboxes which can open up new hard-macro micro-moves.
                if self._remaining() > 80:
                    p_sc2 = cd_refine(best_pos.clone(), benchmark, plc,
                                      K_trials=cd_cfg["K"],
                                      max_rounds=max(4, cd_cfg["R"] // 3),
                                      seed=self.seed + 4300,
                                      max_pins_per_net=max_pins, verbose=False,
                                      time_cap=max(30.0, min(self._remaining() - 60, 60.0)),
                                      order_by_congestion=True)
                    _record("opt1_sc_cd", p_sc2)
                p1 = best_pos.clone()
            except Exception as e:
                self._log(f"  Opt 1 soft-centroid failed: {e}")
        if not fast_path and self._remaining() < _lahc_floor:
            fast_path = True
            self._log(f"  [fast-path] post-opt1 rem={self._remaining():.0f}s < {_lahc_floor:.0f}s — skipping polish chain")
        elif not fast_path and late_cd_cong:
            # V16: T3 + opt1 (the productive front of the polish chain) have run;
            # now skip the low-yield TAIL (opt3/hess/sig/t1_sa/wide_swap/
            # refine_iter/fine_swap) and hand ALL remaining budget to the
            # Laplacian + LAHC finisher — the dominant den+cong reducer. Same
            # proven mechanism as H1 (HUGE benches), here gated on late_cd_cong
            # so ONLY the LAHC-starved std benches (ibm08/13/15/18) are affected;
            # ibm06 and the other early-cd_cong std benches keep their full
            # chain byte-identically (late_cd_cong is False for them).
            fast_path = True
            self._log(f"  [fast-path][V16] late-cd_cong std — skip polish tail "
                      f"(rem={self._remaining():.0f}s → Laplacian+LAHC)")

        # ── V13-rev2 Opt 3: Direct hot-cell attack ─────────────────────────
        # Top-16 hot cells -> hard macros touching them -> 5x5 grid sweep
        # with bit-exact oracle scoring. Replaces hotspot_perturb's random
        # jitter with a principled deterministic search. Idea ported from
        # graph_grad/placer.py:direct_congestion_attack. Targets cong-
        # dominated benches (ibm06, ibm12, ibm17, ibm18) where cong is
        # the limiting metric. Gated to non-tiny benches with budget headroom.
        # On HUGE benches (n_hard >= 500) the sweep cost per macro is higher
        # (more legality checks), so cap macros tighter.
        # V13-rev6: skip opt3 on deep-converged HUGE. ibm10 V13-rev5 trace
        # shows opt3 ran 173s for ZERO gain (best=1.1091 → 1.1091), with
        # opt3_cg adding 0.001 in 45s — a 218s tax for negligible yield.
        # LAHC gets ~6.5e-5/s on that basin (10× higher). Redirect to LAHC.
        # V13-rev fast-path: skip opt3 hot-cell on any deeply-converged basin
        # (was huge-only). FINAL.txt: ibm08 opt3 gain 0 in 155s — pure waste.
        # V13-rev9: loosen non-HUGE threshold 1.10 → 1.05 so ibm08 (currently
        # ~1.10 at this stage) gets a chance to benefit from hot-cell sweep.
        # HUGE benches still gated separately (fast_path=True forces skip).
        skip_opt3 = (huge and best_proxy < 1.15) or (best_proxy < 1.05)
        if skip_opt3:
            self._log(
                f"  -- [V13-rev6] skipping opt3 hot-cell attack "
                f"(HUGE + best={best_proxy:.4f} < 1.15, deep convergence) --"
            )
        if not fast_path and not skip_opt3 and self._remaining() > 250 and n_hard >= 100:
            self._log("  -- V13-rev2 Opt 3: hot-cell attack (top-16 cells) --")
            try:
                from script18 import hot_cell_attack
                t_hc = max(40.0, min(120.0, self._remaining() * 0.04))
                max_m_hc = 20 if huge else 30
                p_hc = hot_cell_attack(
                    best_pos, benchmark, plc,
                    n_top_cells=16, radius_cells=2.0,
                    sweep_steps=5, sweep_radius_frac=0.06,
                    max_macros=max_m_hc,
                    max_pins_per_net=max_pins,
                    seed=self.seed + 4400, time_cap=t_hc,
                    verbose=False,
                )
                _record("opt3_hc", p_hc)
                # Cheap CD cong-only cleanup post-attack.
                if self._remaining() > 80:
                    p_hc2 = cd_refine(best_pos.clone(), benchmark, plc,
                                      K_trials=cd_cfg["K"],
                                      max_rounds=max(4, cd_cfg["R"] // 3),
                                      seed=self.seed + 4500,
                                      max_pins_per_net=max_pins,
                                      score_mode="cong_only", verbose=False,
                                      time_cap=max(30.0, min(self._remaining() - 60, 60.0)),
                                      order_by_congestion=True)
                    _record("opt3_hc_cg", p_hc2)
                p1 = best_pos.clone()
            except Exception as e:
                self._log(f"  Opt 3 hot-cell-attack failed: {e}")
        if not fast_path and self._remaining() < _lahc_floor:
            fast_path = True
            self._log(f"  [fast-path] post-opt3 rem={self._remaining():.0f}s < {_lahc_floor:.0f}s — skipping polish chain")

        # ── Phase A: Hessian eigvec escape (small/medium benches only) ──
        # v14_final's ablation showed escape helps for n_hard < 460 but is a
        # no-op or slight regression on HUGE benches. Gate matches v14.
        # Outer-verify (_record) guarantees no regression below best.
        # SMART SKIP: if cd_cong barely improved cd_final (<0.15% gain), the
        # bench is deeply converged at this basin and escape stages won't
        # find a new one. ibm01 AMRUT4 trace: cd_cong gain was 0.0004/0.7863
        # = 0.05%, then hess+sig wasted 319s landing at 0.7878/0.7893 — both
        # worse than best=0.7859. Skipping reclaims that budget for polish.
        cd_cong_gain_pct = max(prev_best - best_proxy, 0.0) / max(prev_best, 1e-9)
        # V12 B3: tightened cd_cong gain threshold 0.0015 → 0.0010 so escape
        # phases fire on borderline-converged medium benches (ibm03/05/11).
        # V13-rev5: also skip escapes when best is already deeply converged.
        # ibm08 V13-rev4 (best=1.0932 at this point): hess peaked 1.1137,
        # sig peaked 1.1135 — both WORSE than best, wasted 410s combined.
        # At deep convergence the basin is narrow; escape stages disrupt
        # it and re-converge to a worse point. Redirect that budget to
        # LAHC which actually polishes the narrow basin.
        deep_converged = best_proxy < 1.10
        # V13-rev8: HUGE benches now ALSO get hess+sig escape, but with a
        # tighter combined gate: only when (remaining > 1400s) AND (best > 1.06).
        # ibm10 HUGE trace had remaining=2007s after wide_swap with best=1.10,
        # which would trigger this — and the SA-style hess escape on smooth-
        # proxy may find a meaningfully different basin than the pure SA/CD
        # polish loop V13 currently does. _record() catches any regression.
        # The (n_hard >= 460) hard skip is replaced by a budget+convergence gate.
        # V13-rev10 Tier 1: high-cong-converged skip. FINAL.txt traces show
        # hess+sig wasted 400-600s each on ibm08 (cong_frac=0.65, best=1.149),
        # ibm12 (0.70, best=1.19), ibm15 (0.67, best=1.17), ibm18 (0.71,
        # best=1.24) — all with zero net gain. On cong-dominated converged
        # basins the perturb-and-reconverge mechanism cannot find new layout
        # arrangements (cong is bound by macro/soft positions, not basin
        # shape). Skip these → feed LAHC. ibm06 (cong_frac=0.71) escape WAS
        # productive (-0.05) but it fires from best=1.30 > 1.25, so retained.
        # Threshold uses >= 0.64 (vs > 0.65 in rev10.0) so ibm08's exact
        # 0.65 fraction triggers the skip — strict-greater was missing it.
        high_cong_converged = (cong_frac >= 0.64 and best_proxy < 1.25)
        if n_hard >= 460:
            # HUGE: enable only with serious budget + room for improvement.
            SKIP_ESCAPES = (
                (cd_cong_gain_pct < 0.0010)
                or deep_converged
                or high_cong_converged
                or self._remaining() < 1400
                or best_proxy < 1.06
            )
        else:
            # Non-HUGE: V13-rev fast-path tighter. FINAL.txt shows hess+sig
            # regressed on ibm08 (best 1.1487 → peaks 1.1787) and gave only
            # -0.001 on ibm03 (best 0.9442 → peaks above 0.95). At best<1.15
            # the basin is narrow enough that perturbation lands worse —
            # better to feed budget to LAHC instead.
            SKIP_ESCAPES = (
                (cd_cong_gain_pct < 0.0010)
                or deep_converged
                or high_cong_converged
                or best_proxy < 1.15
            )
        if SKIP_ESCAPES and n_hard < 460:
            if deep_converged:
                self._log(
                    f"  -- [V13-rev5] skipping hess+sig escape "
                    f"(best={best_proxy:.4f} < 1.10, deep convergence) --"
                )
            else:
                self._log(f"  -- skipping hess+sig escape "
                          f"(cd_cong gain {cd_cong_gain_pct*100:.2f}% < 0.15%, deep convergence) --")
        if not fast_path and not SKIP_ESCAPES and self._remaining() > 500:
            self._log(f"  -- hess escape (alpha=2.0, HVP-Lanczos) --")
            try:
                from script8 import hessian_eigvec_escape
                p_pert, _ = hessian_eigvec_escape(
                    p1, benchmark, plc, alpha=2.0,
                    max_pins_per_net=max_pins, verbose=False,
                )
                # Re-converge: short GD + CD + swap
                p_h = _legalize_full(
                    run_gd(benchmark, plc, p_pert, device=device,
                           n_steps=500, lr_start=0.05, lr_end=0.005,
                           gamma_start=4.0, gamma_mid=6.0, gamma_end=8.0,
                           cong_weight=cong_w_c23, noise_start=0.0,
                           max_pins_per_net=max_pins, seed=self.seed,
                           verbose=False),
                    benchmark, seed=self.seed)
                _record("hess_gd", p_h)
                if self._remaining() > 200:
                    p_h = cd_refine(p_h, benchmark, plc, K_trials=cd_cfg["K"],
                                    max_rounds=max(8, cd_cfg["R"] // 2),
                                    seed=self.seed + 100,
                                    max_pins_per_net=max_pins, verbose=False,
                                    time_cap=max(60.0, min(self._remaining() - 100, 200.0)))
                    _record("hess_cd", p_h)
                if self._remaining() > 150:
                    p_h = swap_refine(p_h, benchmark, plc, n_attempts=3000,
                                      seed=self.seed + 101, max_pins_per_net=max_pins,
                                      verbose=False,
                                      time_cap=max(40.0, min(self._remaining() - 100, 150.0)))
                    _record("hess_swp", p_h)
                # Reset working ptr to BEST.
                p1 = best_pos.clone()
            except Exception as e:
                self._log(f"  hess escape failed: {e}")

        if not fast_path and self._remaining() < _lahc_floor:
            fast_path = True
            self._log(f"  [fast-path] post-hess rem={self._remaining():.0f}s < {_lahc_floor:.0f}s — skipping polish chain")

        # ── Phase B: Signal-LNS escape (perturb hotspot macros, re-converge) ──
        if not fast_path and not SKIP_ESCAPES and self._remaining() > 400:
            self._log(f"  -- signal-LNS escape (step=0.01um) --")
            try:
                from script14 import signal_perturb
                p_sig, _ = signal_perturb(
                    p1, benchmark, plc, step_um=0.01,
                    max_pins=max_pins, top_pct=0.05,
                    radius_cells=2.0, max_select_frac=0.20, seed=self.seed,
                )
                p_l = _legalize_full(
                    run_gd(benchmark, plc, p_sig, device=device,
                           n_steps=500, lr_start=0.05, lr_end=0.005,
                           gamma_start=4.0, gamma_mid=6.0, gamma_end=8.0,
                           cong_weight=cong_w_c23, noise_start=0.0,
                           max_pins_per_net=max_pins, seed=self.seed,
                           verbose=False),
                    benchmark, seed=self.seed)
                _record("sig_gd", p_l)
                if self._remaining() > 200:
                    p_l = cd_refine(p_l, benchmark, plc, K_trials=cd_cfg["K"],
                                    max_rounds=max(8, cd_cfg["R"] // 2),
                                    seed=self.seed + 200,
                                    max_pins_per_net=max_pins, verbose=False,
                                    time_cap=max(60.0, min(self._remaining() - 100, 200.0)))
                    _record("sig_cd", p_l)
                if self._remaining() > 150:
                    p_l = swap_refine(p_l, benchmark, plc, n_attempts=3000,
                                      seed=self.seed + 201, max_pins_per_net=max_pins,
                                      verbose=False,
                                      time_cap=max(40.0, min(self._remaining() - 100, 150.0)))
                    _record("sig_swp", p_l)
                if self._remaining() > 100:
                    p_l = cd_refine(p_l, benchmark, plc, K_trials=cd_cfg["K"],
                                    max_rounds=max(6, cd_cfg["R"] // 3),
                                    seed=self.seed + 202,
                                    max_pins_per_net=max_pins,
                                    score_mode="cong_only", verbose=False,
                                    time_cap=max(40.0, min(self._remaining() - 60, 120.0)))
                    _record("sig_cg", p_l)
                p1 = best_pos.clone()
            except Exception as e:
                self._log(f"  signal-LNS escape failed: {e}")
        if not fast_path and self._remaining() < _lahc_floor:
            fast_path = True
            self._log(f"  [fast-path] post-sig rem={self._remaining():.0f}s < {_lahc_floor:.0f}s — skipping polish chain")

        # ── V11 T1: Cong-targeted SA polish ──────────────────────────────
        # Metropolis-SA swap pass biased toward macros in top-5% cong cells.
        # Greedy swap_refine plateaus at deep convergence (ibm06 V10 trace:
        # 14+ iters each shaving <1e-3); Metropolis acceptance can escape
        # that plateau by accepting worsening moves with prob exp(-Δ/T).
        # V12 B1: unlock on HUGE — cong is 75% of proxy on heavy benches
        # (per proxy-cost-breakdown memory), and ibm12/17/18 in particular
        # are cong-dominated. Outer-verify safe (_record gates regression).
        # Budget margin raised 350→450 to ensure HUGE benches keep headroom
        # for polish loop; on HUGE the t_sa cap drops to 8% remaining.
        # V13-rev6: also skip t1_sa on deep-converged HUGE. ibm10 V13-rev5
        # trace: t1_sa took 164s, t1_sa_cd 36s, combined yield = 0.001 (best
        # 1.1079→1.1071). LAHC's 6.5e-5/s yield on this basin is 8× better
        # per-second. The cong-SA escape mechanism doesn't help when basin
        # is already converged below 1.15 — the SA temperature is too low
        # to actually accept worsening moves that would find a new basin.
        # V13-rev fast-path: skip T1 cong-SA on any deeply-converged basin.
        # FINAL.txt: ibm08 T1 gain -0.001 (185s), ibm18 gain 0 (165s) — wasted.
        skip_t1_sa = (huge and best_proxy < 1.15) or (best_proxy < 1.10)
        if skip_t1_sa:
            self._log(
                f"  -- [V13-rev6] skipping T1 cong-SA "
                f"(HUGE + best={best_proxy:.4f} < 1.15, deep convergence) --"
            )
        if not fast_path and not skip_t1_sa and self._remaining() > 450:
            self._log("  -- V11 T1: cong-SA polish (Metropolis, hot-biased) --")
            try:
                from script15 import cong_sa_polish
                t_sa = max(120.0, min(250.0,
                          self._remaining() * (0.08 if huge else 0.10)))
                p_sa = cong_sa_polish(
                    best_pos, benchmark, plc,
                    n_attempts=20000, hot_prob=0.7, hot_pct=0.05,
                    radius_cells=2.0, T_start_frac=0.005, T_end=1e-5,
                    refresh_hot_every=2000, area_min_ratio=0.5,
                    area_max_ratio=2.0, max_pins_per_net=max_pins,
                    seed=self.seed + 8000, time_cap=t_sa, verbose=False,
                )
                _record("t1_sa", p_sa)
                # Polish via CD to clean up SA residue (cheap, outer-safe).
                if self._remaining() > 120:
                    p_sa2 = cd_refine(p_sa, benchmark, plc, K_trials=cd_cfg["K"],
                                      max_rounds=max(5, cd_cfg["R"] // 2),
                                      seed=self.seed + 8100,
                                      max_pins_per_net=max_pins, verbose=False,
                                      time_cap=max(40.0, min(self._remaining() - 60, 100.0)))
                    _record("t1_sa_cd", p_sa2)
                p1 = best_pos.clone()
            except Exception as e:
                self._log(f"  T1 cong-SA failed: {e}")
        if not fast_path and self._remaining() < _lahc_floor:
            fast_path = True
            self._log(f"  [fast-path] post-t1 rem={self._remaining():.0f}s < {_lahc_floor:.0f}s — skipping polish chain")

        # ── V13-rev: Wide-area swap (NEW — replaces late T3 slot) ─────────
        # Standard swap_refine uses area_min_ratio=0.5, area_max_ratio=2.0
        # which excludes large-vs-small swaps. After T1's same-area swaps
        # have exhausted that pool, wider area ratios open up moves that
        # could rearrange routing globally (small RAM swaps with mid-size
        # IPs etc). Outer-verify safe; ~60s budget cap.
        # T3 has been moved earlier (right after cd_cong); duplicate call
        # here was always redundant — best_pos already reflected T3's gains.
        if not fast_path and self._remaining() > 150:
            self._log("  -- V13-rev wide-area swap (area 0.3/3.3) --")
            try:
                p_wa = swap_refine(
                    best_pos.clone(), benchmark, plc, n_attempts=4000,
                    area_min_ratio=0.3, area_max_ratio=3.3,
                    seed=self.seed + 8700,
                    max_pins_per_net=max_pins, verbose=False,
                    time_cap=max(30.0, min(self._remaining() - 100, 80.0)),
                )
                _record("wide_swap", p_wa)
                # Cheap CD pass to clean up wide-swap residue.
                if self._remaining() > 80:
                    p_wa2 = cd_refine(best_pos.clone(), benchmark, plc,
                                      K_trials=cd_cfg["K"],
                                      max_rounds=max(5, cd_cfg["R"] // 2),
                                      seed=self.seed + 8800,
                                      max_pins_per_net=max_pins, verbose=False,
                                      time_cap=max(30.0, min(self._remaining() - 60, 60.0)))
                    _record("wide_swap_cd", p_wa2)
                p1 = best_pos.clone()
            except Exception as e:
                self._log(f"  wide-area swap failed: {e}")
        if not fast_path and self._remaining() < _lahc_floor:
            fast_path = True
            self._log(f"  [fast-path] post-wide_swap rem={self._remaining():.0f}s < {_lahc_floor:.0f}s — skipping polish chain")

        # ── Iterative refinement loop: fills any remaining budget with
        # cheap, lossless polishing. Patience=2: tolerate one no-progress
        # iter. AMRUT-new: cap raised 14 → 20 + interleaved hotspot_hops
        # on patience-break — the hop itself rarely beats best, but it
        # resets no_progress=0 so refine continues past natural plateau.
        # Diverse (sigma, sel_frac, max) schedule across 4 hops explores
        # different basin sizes / disruption depths.
        # All inner stages are no-regression (outer-verify).
        rf_iter = 0
        no_progress = 0
        basin_hops = 0
        # V13-rev: hotspot_hops at deep convergence (best<0.95) almost never
        # beat best — ibm09 trace showed 4 consecutive hh's failing while
        # eating 200s. Skip entirely there; refine_iter exits on plateau as
        # designed. Mid-converged (0.95 ≤ best < 1.10) keeps 2 tiny hops.
        # Loose basin keeps V11's full 4-hop schedule.
        if best_proxy < 0.95:
            # Deep convergence: hop never beats best (basin too narrow).
            hop_params = []
        elif best_proxy < 1.10:
            # Mid-converged: very small hops only.
            hop_params = [
                (0.010, 0.08, 20),
                (0.020, 0.10, 24),
            ]
        else:
            # Loose basin: V11 default — bigger jumps useful for escape.
            hop_params = [
                (0.02, 0.10, 24),
                (0.04, 0.12, 32),
                (0.06, 0.12, 32),
                (0.08, 0.15, 40),
            ]
        max_basin_hops = len(hop_params)
        # V11 S2: plateau-exit refine_iter. Track best_proxy delta across
        # last 5 iters; if cumulative gain < 5e-5 the loop is truly stuck
        # (vs noise-level oscillation). ibm06 V10 trace: iters 5-14 made
        # gains ≤ 2e-4 each but cumulative 5e-3 still mattered — so we
        # only exit if FIVE iters together produce <5e-5. Disabled while
        # basin_hops are still being tried (hops reset progress).
        # V11 S3: adaptive hh — track if each hop improved best; after 2
        # consecutive regressing hops, disable further hops (frees budget
        # for T1 cong-SA + T3 net-repair).
        hh_consec_failures = 0
        s2_recent_deltas: list[float] = []
        # V13-rev5: hard LAHC budget reserve, enforced through polish +
        # fine_swap gates below too. Previous value (100s on HUGE) let
        # polish (gate rem>90) eat into LAHC budget: ibm10 V13-rev4 polish
        # ran to rem=58, skipping LAHC entirely. 220s on HUGE / 280s on
        # medium is the floor — LAHC's empirical yield on ibm08 was 0.040
        # in 255s, so anything less starves the highest-yield stage.
        # V13-rev10 Tier 1: raise HUGE reserve 220→600. With the previous
        # 220s reserve, polish loop + fine_swap + Laplacian could eat to
        # rem≈300s, leaving LAHC only 1 restart of 270s instead of 2-3
        # restarts of 350s+ with Laplacian interludes. The reserve is what
        # the polish/refine/fine_swap gates check against to bail early —
        # raising it forces them to hand budget to LAHC. Medium kept at
        # 280 (the std-mode benches ibm01-ibm09 already hit target).
        # 0.25 cap (was 0.10) lets the reserve scale with budget on larger
        # configurations. On the standard 3300s budget: HUGE caps at 600,
        # medium at 280 — same shape, more aggressive on HUGE.
        _lahc_reserve = min(600.0 if huge else 280.0, self._budget * 0.25)
        while not fast_path and self._remaining() > (130 + _lahc_reserve) and rf_iter < 10:
            prev = best_proxy
            rf_iter += 1
            self._log(f"  -- refine_iter {rf_iter} (rem={self._remaining():.0f}s) --")
            try:
                if self._remaining() > 80:
                    # V9: 3000 → 6000 attempts (same time_cap, 2× sample density)
                    p1 = swap_refine(p1, benchmark, plc, n_attempts=6000,
                                     seed=self.seed + 10 + rf_iter,
                                     max_pins_per_net=max_pins, verbose=False,
                                     time_cap=max(40.0, min(self._remaining() - 60, 120.0)))
                    _record(f"rf{rf_iter}_swp", p1)
                if self._remaining() > 60:
                    # V13-rev2 Opt 2: priority ordering at deep convergence —
                    # the high-cong macros are where remaining gains live.
                    p1 = cd_refine(p1, benchmark, plc, K_trials=cd_cfg["K"],
                                   max_rounds=max(5, cd_cfg["R"] // 2),
                                   seed=self.seed + 20 + rf_iter,
                                   max_pins_per_net=max_pins, verbose=False,
                                   time_cap=max(40.0, min(self._remaining() - 50, 120.0)),
                                   order_by_congestion=True)
                    _record(f"rf{rf_iter}_cd", p1)
                if self._remaining() > 50:
                    p1 = cd_refine(p1, benchmark, plc, K_trials=cd_cfg["K"],
                                   max_rounds=max(5, cd_cfg["R"] // 2),
                                   seed=self.seed + 30 + rf_iter,
                                   max_pins_per_net=max_pins,
                                   score_mode="cong_only", verbose=False,
                                   time_cap=max(40.0, min(self._remaining() - 30, 120.0)),
                                   order_by_congestion=True)
                    _record(f"rf{rf_iter}_cg", p1)
            except Exception as e:
                self._log(f"  refine_iter {rf_iter} failed: {e}")
                break
            # V11 S2 plateau tracking: cumulative delta over last 5 iters.
            iter_delta = max(prev - best_proxy, 0.0)
            s2_recent_deltas.append(iter_delta)
            if len(s2_recent_deltas) > 5:
                s2_recent_deltas.pop(0)
            # V13-rev5: refine_iter at deep convergence yields 0.0001-0.0003
            # per iter (~115s each on medium benches). LAHC yields 0.04 per
            # 255s on the same basin (ibm08 trace) — 70x better per second.
            # If we're stalling and LAHC budget is available, exit early so
            # the budget feeds LAHC rather than draining on marginal polish.
            # V13-rev8.1: tighter HUGE threshold (1.5e-3 → 2.5e-3) and
            # earlier rf_iter gate (>= 4 → >= 3). ibm10 rev7 trace: refine
            # iters 4-9 each produced 0.0006-0.0011 — well below 2.5e-3.
            # New gate exits ~iter 4 instead of iter 6, freeing ~400s for
            # Laplacian + saddle21 + LAHC (the proven-high-yield stages).
            # The non-HUGE threshold (3e-4) is unchanged — small benches
            # still benefit from extended refine.
            # V13-rev fast-path: tighter exit threshold on non-HUGE benches.
            # FINAL.txt: ibm08 rf_iter 4-5 gave <0.0008 each — exit sooner.
            # V18 C-1: non-HUGE 0.0008 → 0.0015. V16 traces: ibm01/04/07/09/
            # 11 spent 270-510s in refine_iter at ~0.001/iter while their
            # LAHC was still descending at budget end; with the V18 4-6×
            # faster LAHC the per-second comparison tilts further. ibm02
            # (~0.0023/iter) stays in the loop. The rf_iter>=3 floor and
            # LAHC-headroom condition below are unchanged.
            rf_threshold = (0.0025 if (huge and rf_iter >= 3) else 0.0015)
            if (iter_delta < rf_threshold
                    and self._remaining() > (_lahc_reserve + 250)
                    and rf_iter >= 3):
                self._log(
                    f"  [V13-rev7] early refine_iter exit "
                    f"(Δ {iter_delta:.5f} < {rf_threshold:.4f}, "
                    f"LAHC reserve has headroom)"
                )
                break
            # Fast-path inside loop: if remaining drops below LAHC floor,
            # bail immediately so Laplacian + multi-LAHC get their budget.
            if self._remaining() < _lahc_floor:
                fast_path = True
                self._log(f"  [fast-path] mid-refine_iter rem={self._remaining():.0f}s < {_lahc_floor:.0f}s — breaking refine_iter")
                break
            # Patience=2: on second no-progress iter, fire a hotspot_hop
            # (targeted jitter on macros near top-C cells). The hop never
            # directly improves best — its job is to reset no_progress=0
            # so refine_iter continues past plateau. Up to 4 hops total
            # with escalating sigma / varying sel_frac.
            # V12 B5: tightened plateau threshold 0.0003 → 0.0002.
            if prev - best_proxy < 0.0002:
                no_progress += 1
                if no_progress >= 2:
                    # V11 S3: if 2 consecutive hops both failed AND we've
                    # truly plateaued (S2: <5e-5 over 5 iters), exit.
                    s2_plateau = (len(s2_recent_deltas) == 5
                                  and sum(s2_recent_deltas) < 5e-5)
                    if hh_consec_failures >= 2 and s2_plateau:
                        self._log(f"  [V11 S2+S3] plateau ({sum(s2_recent_deltas):.5f} over 5 iters) "
                                  f"+ 2 failed hops — exiting refine_iter loop")
                        break
                    if (basin_hops < max_basin_hops
                            and hh_consec_failures < 2
                            and self._remaining() > 250):
                        basin_hops += 1
                        sigma, sel_frac, sel_max = hop_params[(basin_hops - 1) % max(len(hop_params), 1)]
                        self._log(f"  -- hotspot_hop {basin_hops} "
                                  f"(sigma={sigma:.3f}, sel_frac={sel_frac:.2f}, "
                                  f"rem={self._remaining():.0f}s) --")
                        pre_hh_best = best_proxy
                        try:
                            from script9 import hotspot_perturb
                            hp, hp_info = hotspot_perturb(
                                best_pos, benchmark, plc,
                                top_pct=0.05, radius_cells=2.0,
                                select_frac=sel_frac, min_select=4, max_select=sel_max,
                                sigma_frac=sigma,
                                max_pins_per_net=max_pins,
                                seed=self.seed + 7000 + basin_hops,
                            )
                            self._log(f"  [hh{basin_hops}] selected={hp_info['n_selected']} "
                                      f"of {hp_info['n_cands']} cands "
                                      f"({hp_info['n_hot_cells']} hot cells)")
                            hp_l = _legalize_full(hp, benchmark,
                                                  seed=self.seed + 7001 + basin_hops)
                            _record(f"hh{basin_hops}_lg", hp_l)
                            if self._remaining() > 120:
                                hp_l = cd_refine(hp_l, benchmark, plc, K_trials=cd_cfg["K"],
                                                 max_rounds=max(5, cd_cfg["R"] // 2),
                                                 seed=self.seed + 7100 + basin_hops,
                                                 max_pins_per_net=max_pins, verbose=False,
                                                 time_cap=max(40.0, min(self._remaining() - 60, 150.0)))
                                _record(f"hh{basin_hops}_cd", hp_l)
                            if self._remaining() > 80:
                                hp_l = cd_refine(hp_l, benchmark, plc, K_trials=cd_cfg["K"],
                                                 max_rounds=max(5, cd_cfg["R"] // 2),
                                                 seed=self.seed + 7200 + basin_hops,
                                                 max_pins_per_net=max_pins,
                                                 score_mode="cong_only", verbose=False,
                                                 time_cap=max(40.0, min(self._remaining() - 40, 120.0)))
                                _record(f"hh{basin_hops}_cg", hp_l)
                            # V11 S3: count consecutive hop failures.
                            if best_proxy < pre_hh_best - 1e-5:
                                hh_consec_failures = 0
                            else:
                                hh_consec_failures += 1
                                self._log(f"  [V11 S3] hh{basin_hops} did not improve best "
                                          f"({pre_hh_best:.4f} → {best_proxy:.4f}) — "
                                          f"consec fails = {hh_consec_failures}")
                            # Continue refining from best (outer-verify safe).
                            p1 = best_pos.clone()
                            no_progress = 0
                        except Exception as e:
                            self._log(f"  hotspot_hop {basin_hops} failed: {e}")
                            break
                    else:
                        break
            else:
                no_progress = 0

        # ── Lossless final polish loop.
        # V13-rev: cap 20→8, patience 5→3, threshold 3e-5→1e-4.
        # ibm09 trace: polish iters 5-16 produced ~0.0009 total in ~280s.
        # That's 0.000057/iter — well below noise. The first 4-6 polish
        # iters do all the real work. Tightening reclaims ~200-300s per
        # bench for downstream stages (or finishes earlier on slack benches).
        polish_pass = 0
        polish_no_progress = 0
        # V13-rev5: polish must respect the LAHC reserve. Previous gate
        # rem>90 let polish eat to rem=58 on ibm10 HUGE, starving LAHC.
        # Polish iters give 0.0001-0.0003 each; LAHC gives 0.04 per 255s
        # on the same basin (ibm08). Reserving LAHC budget is the right
        # trade — we may sacrifice 1-2 polish iters of marginal value.
        # V13-rev8.1: HUGE-specific polish reserve raised and cap lowered.
        # ibm10 rev8 trace: polish 1-8 each gave 0.0003-0.0006, total 0.0060
        # over ~320s. That's 0.0019/100s — LAHC delivers 0.04/255s ≈
        # 0.016/100s on the same basin (8× better per second). Polish should
        # bail earlier to feed LAHC. Both saddle21 and multi_lahc need 500s+
        # reserve to fire — polish must respect that.
        _polish_min_remaining = (
            (350.0 + _lahc_reserve) if huge else (60.0 + _lahc_reserve)
        )
        # V13-rev fast-path: tighter caps. FINAL.txt shows polish iters
        # 0.00005-0.0002 each → first 3-4 do the real work, rest is noise.
        _polish_cap = 3 if huge else 4
        _polish_threshold = 0.0010 if huge else 0.0003
        _polish_patience = 2 if huge else 3
        while not fast_path and self._remaining() > _polish_min_remaining and polish_pass < _polish_cap:
            polish_pass += 1
            prev = best_proxy
            mode = "partial" if (polish_pass % 2 == 0) else "cong_only"
            try:
                p1 = cd_refine(
                    p1, benchmark, plc,
                    K_trials=cd_cfg["K"],
                    max_rounds=max(5, cd_cfg["R"] // 2),
                    seed=self.seed + 9000 + polish_pass * 13,
                    max_pins_per_net=max_pins,
                    score_mode=mode, verbose=False,
                    time_cap=max(40.0, min(self._remaining() - 30, 80.0)),
                    order_by_congestion=True,  # V13-rev2 Opt 2
                )
                tag = f"polish{polish_pass}_" + ("p" if mode == "partial" else "cg")
                _record(tag, p1)
            except Exception as e:
                self._log(f"  polish{polish_pass} failed: {e}")
                break
            if prev - best_proxy < _polish_threshold:
                polish_no_progress += 1
                if polish_no_progress >= _polish_patience:
                    break
            else:
                polish_no_progress = 0
            if self._remaining() < _lahc_floor:
                fast_path = True
                self._log(f"  [fast-path] mid-polish rem={self._remaining():.0f}s < {_lahc_floor:.0f}s — breaking polish loop")
                break

        # ── Final close-neighbor fine_swap pass. Standard swap_refine
        # picks random pairs anywhere on the canvas, which is wasted effort
        # at deep convergence — most random pair swaps overlap or regress.
        # This pass restricts swaps to physically-close macro pairs (already
        # the most likely to improve WL), using swap_refine's area-ratio
        # filter as a proxy for "similar enough to fit each other's slot".
        # Cheap (~30-60s), outer-verify safe.
        # V13-rev5: fine_swap also respects the LAHC reserve. On ibm08
        # V13-rev4 fine_swap took 28s and produced no improvement (proxy
        # stayed at 1.0865) — same per-second yield problem as polish.
        if not fast_path and self._remaining() > (30.0 + _lahc_reserve):
            try:
                # V13-rev fast-path: cap reduced from 60s → 20s. FINAL.txt shows
                # 0 gain on most benches in 30-80s.
                p1 = swap_refine(
                    p1, benchmark, plc, n_attempts=2000,
                    area_min_ratio=0.6, area_max_ratio=1.67,  # tighter than default 0.5/2.0
                    seed=self.seed + 9500,
                    max_pins_per_net=max_pins, verbose=False,
                    time_cap=max(10.0, min(self._remaining() - _lahc_reserve - 10, 20.0)),
                )
                _record("fine_swap", p1)
            except Exception as e:
                self._log(f"  fine_swap failed: {e}")

        # ── V13-rev7: Laplacian soft-resolve (closed-form HPWL warm-start) ─
        # Idea ported (not copied) from vmallela_v7 Phase 2. The clique-
        # model quadratic HPWL has a closed-form global minimum via
        # L_ff @ x_f = -L_fc @ x_c + port_contrib. Solved with sparse CG.
        # Applied via per-soft line search with full-proxy gating, so
        # density / congestion are respected even though the solve
        # ignores them. By construction this stage cannot regress.
        # Expected: -0.001 to -0.005 per bench, ~30-60s wall.
        # Runs AFTER fine_swap so it operates on the most-polished hard
        # positions, then LAHC polishes the new soft positions.
        if (benchmark.num_macros > benchmark.num_hard_macros
                and self._remaining() > (20.0 + min(_lahc_reserve * 0.8, 100.0))):
            try:
                from script20 import laplacian_soft_resolve
                # V14: bumped Laplacian cap 60s → 90s. Empirically the gain
                # per second on the main pre-LAHC Laplacian call is
                # 0.0003-0.0006/s (one of the best in the pipeline), and
                # the line search is bounded by the soft count so extra
                # time goes to deeper alpha sweeps. Wider alphas in
                # script20.py let those extra seconds find moves
                # that the (1.0, 0.5, 0.25, 0.1, 0.05) ladder misses.
                t_lap = max(15.0, min(90.0,
                            self._remaining() - _lahc_reserve - 5.0))
                self._log(f"  -- V13-rev7 Laplacian soft-resolve (t_cap={t_lap:.0f}s) --")
                p_lap = laplacian_soft_resolve(
                    best_pos.clone(), benchmark, plc,
                    time_cap=t_lap, seed=self.seed + 20000,
                    verbose=False,
                )
                _record("laplacian", p_lap)
            except Exception as e:
                self._log(f"  Laplacian soft-resolve failed: {e}")

        # ── V13-rev fast-path: saddle21 multi-alpha sweep REMOVED ────────
        # FINAL.txt confirms saddle21 consistently regresses on every bench
        # it fires on: ibm02 (laplacian 1.0393 → saddle21 1.0933), ibm06
        # (1.2016 → 1.2216), ibm07 (1.0074 → 1.0418). _record outer-verify
        # prevents the regression from sticking, but the 200-320s the sweep
        # consumes is pure waste vs. the same budget on LAHC multi-restart
        # (-0.02 to -0.05 per restart on these benches).

        # ── V13-rev4 LAHC polish (Late Acceptance Hill Climbing) ──────────
        # LAHC's queue-based acceptance (cand < cur OR cand < history[i mod L])
        # lets the search step over micro-plateaus that V13's strict-improvement
        # refine loop can't cross. Uses FULL proxy via
        # IncrementalProxy.proxy_full_cached() in the inner loop (~3-8ms/iter,
        # 6-10× faster than FastProxy.proxy_full's 30-50ms).
        #
        # v3 used partial proxy (WL+D, no cong) for acceptance and validated
        # full proxy only every 200 iters; on cong-dominated benches (ibm08
        # cong=65%), this drifted the position into partial-good full-bad
        # territory and produced ZERO improvement in 692s. v4-a fixed that
        # with full-proxy acceptance + drift-reset, but its 3x3 spatial-grid
        # overlap check missed pairs with stride > 1 cell on benches where
        # macros span ~1.5-2 cells — 13 hard overlaps slipped through on
        # ibm08, invalidating an otherwise-improving result (1.0865→1.0465).
        #
        # v4 replaces the grid check with vectorized O(n_hard) brute force
        # and adds a final legality verification: if any overlap pair sneaks
        # through, the input placement is returned unchanged.
        #
        # Idea credit: graph_grad/placer2.py lahc_polish.
        if self._remaining() > 45 and best_proxy < float("inf"):
            self._log(f"  -- V13-rev9 LAHC polish (rem={self._remaining():.0f}s) --")
            try:
                from script19 import lahc_polish
                try:
                    from script20 import laplacian_soft_resolve
                    _have_lap = True
                except Exception:
                    _have_lap = False
                t_lahc = max(60.0, self._remaining() - 30.0)
                # V13-rev9: lowered multi_lahc threshold (was 240 → 180) so
                # 2-restart fires whenever there's room for two ≥90s runs.
                # On HUGE, when t_lahc > 540s we instead do a 3-restart with
                # Laplacian re-runs between restarts (each Laplacian gives
                # 0.005-0.025 by reopening soft basin LAHC's small kicks
                # can't reach).
                multi_lahc = t_lahc > 180.0
                # V14: lowered triple_lahc threshold 900 → 600. With the V14
                # cycle 2/3 skip on very_huge low-cong, LAHC now gets 550-700s
                # on ibm10-class HUGE benches. At t_lahc=600, three 170s
                # restarts each get past the LAHC warmup hump (~80s) and
                # contribute ~0.005-0.015 per restart. The Laplacian interludes
                # between restarts reopen the soft basin so each LAHC's
                # starting point is genuinely different.
                triple_lahc = huge and t_lahc > 600.0 and _have_lap
                # V13-rev9.1 HUGE LAHC: REVERTED to V13-rev6 proven values.
                # The over-aggressive kick_mult=7.0/interval=25s disrupted the
                # narrow converged HUGE basin (ibm10: lahc_r1 went 1.1311 →
                # 1.1339, regression). Original mult=5.0/interval=35s + the
                # new Laplacian interludes is the winning combo.
                if huge:
                    # V14: tightened drift_reset_ratio 1.005 → 1.003 on HUGE.
                    # Stricter snap-back lets the inner-interlude pattern
                    # (LAHC1 → lap → r_mid → lap → r2) hold a tighter basin
                    # between restarts. The Laplacian gain per interlude
                    # depends on starting from best, not from cur drift.
                    lahc_kwargs = dict(
                        list_len=120, soft_prob=0.70, soft_centroid_prob=0.45,
                        hard_move_frac=0.025, soft_move_frac=0.015,
                        ils_kick_interval=35.0, ils_kick_radius_mult=5.0,
                        drift_reset_ratio=1.003, drift_check_every=50,
                    )
                else:
                    # V13-rev9.1: REVERTED non-HUGE LAHC params to V13-rev6
                    # baseline. The aggressive (list_len=140, mult=3.5) values
                    # regressed ibm01 by 0.004 (deep basin: 0.7584→0.7623).
                    # Outer-verify (_record) is regression-safe, but raw LAHC
                    # output was worse — keep proven values to preserve the
                    # already-strong non-HUGE pipeline.
                    lahc_kwargs = dict(
                        list_len=100, soft_prob=0.55, soft_centroid_prob=0.40,
                        hard_move_frac=0.025, soft_move_frac=0.015,
                        ils_kick_interval=80.0, ils_kick_radius_mult=3.0,
                        drift_reset_ratio=1.005, drift_check_every=50,
                    )

                def _run_laplacian_interlude(tag: str, max_t: float = 60.0):
                    """Run a short Laplacian polish between LAHC restarts.
                    Soft positions drift during LAHC; re-solving the clique
                    Laplacian gives 0.005-0.025 typical gain in ~30-60s.
                    Outer-verify (_record) safe — never regresses.
                    V14: bumped remaining-fraction cap 5% → 7% to use Laplacian's
                    high gain-per-second more aggressively when budget allows."""
                    if not _have_lap or self._remaining() < 30:
                        return
                    if benchmark.num_macros <= n_hard:
                        return
                    try:
                        t_lap = max(20.0, min(max_t, self._remaining() * 0.07))
                        p_lap = laplacian_soft_resolve(
                            best_pos.clone(), benchmark, plc,
                            time_cap=t_lap,
                            seed=self.seed + 21000 + hash(tag) % 1000,
                            verbose=False,
                        )
                        _record(tag, p_lap)
                    except Exception as e:
                        self._log(f"  {tag} failed: {e}")

                if triple_lahc:
                    # V14: reserve 135s for 3 Laplacian interludes (was 90s
                    # for 2). Adding post-r3 Laplacian — soft positions drift
                    # during the final LAHC restart too; closing with a
                    # Laplacian polish captures that.
                    t_each = (t_lahc - 135.0) / 3.0
                    self._log(
                        f"     HUGE triple-restart: 3 runs × {t_each:.0f}s "
                        f"+ 3 Laplacian interludes (seeds +19000, +29000, +39000)"
                    )
                    # Run 1 from best
                    p_lahc_1 = lahc_polish(
                        best_pos.clone(), benchmark, plc,
                        time_cap=t_each,
                        seed=self.seed + 19000,
                        verbose=False, **lahc_kwargs,
                    )
                    _record("lahc_r1", p_lahc_1)
                    # Laplacian interlude
                    _run_laplacian_interlude("lap_post_r1", max_t=45.0)
                    # Run 2 from possibly-improved best
                    if self._remaining() > t_each + 30:
                        p_lahc_2 = lahc_polish(
                            best_pos.clone(), benchmark, plc,
                            time_cap=t_each,
                            seed=self.seed + 29000,
                            verbose=False, **lahc_kwargs,
                        )
                        _record("lahc_r2", p_lahc_2)
                        _run_laplacian_interlude("lap_post_r2", max_t=45.0)
                    # Run 3 from possibly-improved best
                    if self._remaining() > 60:
                        p_lahc_3 = lahc_polish(
                            best_pos.clone(), benchmark, plc,
                            time_cap=max(60.0, min(t_each, self._remaining() - 50)),
                            seed=self.seed + 39000,
                            verbose=False, **lahc_kwargs,
                        )
                        _record("lahc_r3", p_lahc_3)
                        # V14: post-r3 Laplacian — captures the soft drift
                        # from the third LAHC restart. Yields are usually
                        # the smallest of the three (~0.0005-0.002) but the
                        # 40s cost is low.
                        _run_laplacian_interlude("lap_post_r3", max_t=45.0)
                elif multi_lahc:
                    # V13-rev10 Tier 2: HUGE benches get an inner Laplacian
                    # interlude pattern: [LAHC1 → lap → short-LAHC → lap →
                    # LAHC2]. ibm10 v13rev9p1 trace yields:
                    #   Laplacian per-sec yield: ~2.2e-4/s, 3-10× LAHC's
                    #     terminal yield of 1.8e-5/s on this basin
                    #   lap_post_r1 already gave 0.0023 in 65s with prev
                    #     pattern (LAHC1 → lap → LAHC2)
                    # The middle short-LAHC kicks softs away from Laplacian's
                    # WL-only optimum so the second Laplacian can land in a
                    # different (potentially better) WL minimum that still
                    # respects the density/cong basin LAHC just explored.
                    # Total LAHC time stays ≈ t_lahc; we trade ~70s of LAHC
                    # for 1 extra Laplacian (~40s) + short-LAHC kick (~100s).
                    # Non-HUGE keeps proven 2-restart + post-r1 lap pattern.
                    if huge:
                        t_r1 = t_lahc * 0.42
                        t_r_mid = max(80.0, t_lahc * 0.14)
                        self._log(
                            f"     HUGE inner-interlude: r1={t_r1:.0f}s + lap "
                            f"+ r_mid={t_r_mid:.0f}s + lap + r2 "
                            f"(seeds +19000, +24000, +29000)"
                        )
                        p_lahc_1 = lahc_polish(
                            best_pos.clone(), benchmark, plc,
                            time_cap=t_r1,
                            seed=self.seed + 19000,
                            verbose=False, **lahc_kwargs,
                        )
                        _record("lahc_r1", p_lahc_1)
                        _run_laplacian_interlude("lap_post_r1", max_t=40.0)
                        if self._remaining() > (t_r_mid + 80.0):
                            p_lahc_mid = lahc_polish(
                                best_pos.clone(), benchmark, plc,
                                time_cap=t_r_mid,
                                seed=self.seed + 24000,
                                verbose=False, **lahc_kwargs,
                            )
                            _record("lahc_r_mid", p_lahc_mid)
                            _run_laplacian_interlude("lap_post_r_mid", max_t=40.0)
                        if self._remaining() > 60:
                            p_lahc_2 = lahc_polish(
                                best_pos.clone(), benchmark, plc,
                                time_cap=max(60.0, min(t_lahc * 0.32, self._remaining() - 20)),
                                seed=self.seed + 29000,
                                verbose=False, **lahc_kwargs,
                            )
                            _record("lahc_r2", p_lahc_2)
                    else:
                        # V17 (rebuilt): std benches get Laplacian interludes
                        # between restarts too — softs drift during LAHC and
                        # the clique-Laplacian re-solve is the main density
                        # reducer (−0.001..−0.0035 per interlude; HUGE already
                        # had this). Budget-gated triple-restart: the >1100s
                        # floor keeps each of 3 restarts past the ~80s LAHC
                        # warmup and excludes ibm18-class tight budgets — the
                        # V16b blanket-triple regression case. Verified as part
                        # of the V17 full-17 run (avg 0.9904, all VALID).
                        std_triple = (t_lahc > 1100.0 and _have_lap
                                      and benchmark.num_macros > n_hard)
                        if std_triple:
                            t_each = (t_lahc - 120.0) / 3.0
                            self._log(
                                f"     std triple-restart: 3 runs × {t_each:.0f}s "
                                f"+ Laplacian interludes (seeds +19000, +29000, +39000)"
                            )
                            p_lahc_1 = lahc_polish(
                                best_pos.clone(), benchmark, plc,
                                time_cap=t_each,
                                seed=self.seed + 19000,
                                verbose=False, **lahc_kwargs,
                            )
                            _record("lahc_r1", p_lahc_1)
                            _run_laplacian_interlude("lap_post_r1", max_t=40.0)
                            if self._remaining() > t_each + 30:
                                p_lahc_2 = lahc_polish(
                                    best_pos.clone(), benchmark, plc,
                                    time_cap=t_each,
                                    seed=self.seed + 29000,
                                    verbose=False, **lahc_kwargs,
                                )
                                _record("lahc_r2", p_lahc_2)
                                _run_laplacian_interlude("lap_post_r2", max_t=40.0)
                            if self._remaining() > 60:
                                p_lahc_3 = lahc_polish(
                                    best_pos.clone(), benchmark, plc,
                                    time_cap=max(60.0, min(t_each, self._remaining() - 50)),
                                    seed=self.seed + 39000,
                                    verbose=False, **lahc_kwargs,
                                )
                                _record("lahc_r3", p_lahc_3)
                                _run_laplacian_interlude("lap_post_r3", max_t=40.0)
                        else:
                            # Tight budget: 2-restart + 1 interlude (between
                            # the restarts only — a trailing interlude would
                            # eat the final LAHC seconds).
                            t_each = t_lahc * 0.50
                            self._log(
                                f"     multi-restart enabled: 2 runs × {t_each:.0f}s "
                                f"+ Laplacian interlude (seeds +19000, +29000)"
                            )
                            p_lahc_1 = lahc_polish(
                                best_pos.clone(), benchmark, plc,
                                time_cap=t_each,
                                seed=self.seed + 19000,
                                verbose=False, **lahc_kwargs,
                            )
                            _record("lahc_r1", p_lahc_1)
                            _run_laplacian_interlude("lap_post_r1", max_t=40.0)
                            if self._remaining() > 40:
                                p_lahc_2 = lahc_polish(
                                    best_pos.clone(), benchmark, plc,
                                    time_cap=max(60.0, min(t_each, self._remaining() - 20)),
                                    seed=self.seed + 29000,
                                    verbose=False, **lahc_kwargs,
                                )
                                _record("lahc_r2", p_lahc_2)
                else:
                    p_lahc = lahc_polish(
                        best_pos.clone(), benchmark, plc,
                        time_cap=t_lahc,
                        seed=self.seed + 19000,
                        verbose=False, **lahc_kwargs,
                    )
                    _record("lahc", p_lahc)
            except Exception as e:
                self._log(f"  LAHC polish failed: {e}")

        # V12 safety chain: if no ovr=0 placement was ever recorded, the
        # main pipeline's _legalize_full left residual violators on every
        # stage (seen on ibm10 HUGE: every stage produced ovr=1-2). Fall
        # back to the best_inv candidate and hammer _guaranteed_legalize
        # on it; if even that fails, try legalizing the spread-out init
        # state as a final fallback. Returning a placement with ovr>0
        # would be disqualified by the contest evaluator, so we always
        # try to land at ovr=0.
        if best_proxy < float("inf"):
            self._log(f">>> {name}: best={best_tag} proxy={best_proxy:.4f} "
                      f"t={self._elapsed():.0f}s")
            return best_pos

        self._log(f"  [V12 SAFETY] no ovr=0 placement; salvaging best_inv "
                  f"(tag={best_inv_tag}, ovr={best_inv_ovr}, "
                  f"proxy={best_inv_proxy:.4f})")
        t_cap1 = min(max(self._remaining(), 20.0), 60.0)
        salvaged = _guaranteed_legalize(best_inv_pos, benchmark, plc,
                                        base_seed=self.seed + 99991,
                                        time_cap_s=t_cap1)
        try:
            c = compute_proxy_cost(salvaged, benchmark, plc)
            ovr_s = int(c["overlap_count"])
            proxy_s = float(c["proxy_cost"])
        except Exception as e:
            self._log(f"  [V12 SAFETY] proxy on salvage failed: {e}")
            ovr_s = best_inv_ovr
            proxy_s = best_inv_proxy
        self._log(f"  [V12 SAFETY] salvage attempt 1: "
                  f"ovr={ovr_s} proxy={proxy_s:.4f}")
        if ovr_s == 0:
            self._log(f">>> {name}: best=salvaged proxy={proxy_s:.4f} "
                      f"t={self._elapsed():.0f}s")
            return salvaged

        # Last resort: re-legalize the original init (it is more spread out
        # than the refined-but-stuck best_inv, so the grid solver has more
        # room to relocate violators).
        self._log(f"  [V12 SAFETY] salvage 1 still ovr={ovr_s}; "
                  f"trying init.plc legalize")
        t_cap2 = min(max(self._remaining(), 10.0), 30.0)
        fallback = _guaranteed_legalize(benchmark.macro_positions.clone(),
                                        benchmark, plc,
                                        base_seed=self.seed + 88888,
                                        time_cap_s=t_cap2)
        try:
            c2 = compute_proxy_cost(fallback, benchmark, plc)
            ovr_f = int(c2["overlap_count"])
            proxy_f = float(c2["proxy_cost"])
        except Exception:
            ovr_f = 1 << 30
            proxy_f = float("inf")
        self._log(f"  [V12 SAFETY] salvage attempt 2: "
                  f"ovr={ovr_f} proxy={proxy_f:.4f}")
        if ovr_f < ovr_s or (ovr_f == ovr_s and proxy_f < proxy_s):
            self._log(f">>> {name}: best=init_salvage proxy={proxy_f:.4f} "
                      f"ovr={ovr_f} t={self._elapsed():.0f}s")
            return fallback
        self._log(f">>> {name}: best=refined_salvage proxy={proxy_s:.4f} "
                  f"ovr={ovr_s} t={self._elapsed():.0f}s")
        return salvaged
