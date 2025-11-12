"""
Geometry tuning utility focused on finding waveguide combinations whose Δw solver
converges via a true, physically valid bracket (i.e., the sampled G(Δw) curve crosses
zero with feasible L₅₀ values, and the root falls within tolerance).

Usage:
    python tune_geometry.py

The script sweeps configurable ranges of (w, g, t), samples G(Δw) on a dense grid to
verify sign changes, runs the full delta width solver, and ranks the geometries that
deliver a usable bracketed zero. Results are printed in two sections:
    1. Geometries with bracketed zeros that meet the target tolerance
    2. The closest misses (smallest residual mismatch) together with the dominant
       failure reason (no sign change, solver fallback, etc.)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from types import SimpleNamespace
from tqdm import tqdm

from delta_width import solve_delta_w_star, _G_of_dw
import csv
import pathlib


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SignScan:
    """Summary of the sampled G(Δw) curve for a single geometry."""

    has_sign_change: bool
    bracket_hint: Optional[Tuple[float, float]]
    sample_points: int
    skipped_points: int
    best_abs_err: float
    best_dw: Optional[float]


@dataclass
class GeometryResult:
    """Aggregate information for a (w, g, t) combination."""

    w: float
    g: float
    t: float
    sign_scan: SignScan
    delta_w_star: Optional[float]
    diagnostics: dict

    @property
    def abs_err(self) -> float:
        return float(self.diagnostics.get("abs_err", np.inf))

    @property
    def rel_err(self) -> float:
        return float(self.diagnostics.get("rel_err", np.inf))

    @property
    def has_bracket(self) -> bool:
        return bool(self.diagnostics.get("n_brackets", 0))

    @property
    def tolerance_met(self) -> bool:
        return bool(self.diagnostics.get("tolerance_met", False))

    @property
    def usable_zero(self) -> bool:
        """True only if a sign change exists AND the solver hit tolerance via a bracket."""
        return self.sign_scan.has_sign_change and self.has_bracket and self.tolerance_met

    def failure_reason(self) -> str:
        if self.sign_scan.sample_points == 0:
            return "mode solver infeasible for all Δw samples"
        if not self.sign_scan.has_sign_change:
            return "no sign change in sampled G(Δw)"
        if not self.has_bracket:
            return "solver could not retain bracket"
        if not self.tolerance_met:
            return f"L50 mismatch {self.abs_err:.3f} µm (rel {self.rel_err*100:.2f}%)"
        return "success"


# ---------------------------------------------------------------------------
# Core analysis helpers
# ---------------------------------------------------------------------------


def _make_param_instance(param_template, w: float, g: float, t: float) -> SimpleNamespace:
    """Clone param_template and override geometry values."""
    param = SimpleNamespace(**param_template.__dict__)
    param.wg_width = float(w)
    param.coupling_gap = float(g)
    param.wg_thick = float(t)
    return param


def sample_G_curve(
    param_template,
    w: float,
    g: float,
    t: float,
    lambda0: float,
    cache_dir: str,
    search_min: float,
    search_max: float,
    dw_step: float,
) -> SignScan:
    """Sample G(Δw) uniformly and report sign-change and best-mismatch stats."""
    param = _make_param_instance(param_template, w, g, t)

    dw_values = np.arange(search_min, search_max + dw_step * 0.5, dw_step)
    collected: List[Tuple[float, float, float, float]] = []
    last_sample: Optional[Tuple[float, float]] = None
    first_bracket: Optional[Tuple[float, float]] = None

    for dw in dw_values:
        G_val, L50_te, L50_tm = _G_of_dw(dw, param, lambda0, cache_dir)
        if G_val is None:
            continue
        collected.append((dw, G_val, L50_te, L50_tm))

        if last_sample is not None and (G_val * last_sample[1]) < 0.0:
            first_bracket = (last_sample[0], dw)
        last_sample = (dw, G_val)

    if not collected:
        return SignScan(False, None, 0, len(dw_values), np.inf, None)

    has_pos = any(item[1] > 0.0 for item in collected)
    has_neg = any(item[1] < 0.0 for item in collected)
    has_sign_change = has_pos and has_neg

    best_point = min(collected, key=lambda item: abs(item[2] - item[3]))
    best_abs_err = abs(best_point[2] - best_point[3])
    best_dw = best_point[0]

    bracket_hint = None
    if has_sign_change:
        if first_bracket is not None:
            bracket_hint = first_bracket
        else:
            # Fall back to the first positive/negative samples if no consecutive pair exists
            neg_dw = next((dw for dw, G_val, *_ in collected if G_val < 0), None)
            pos_dw = next((dw for dw, G_val, *_ in collected if G_val > 0), None)
            if neg_dw is not None and pos_dw is not None:
                bracket_hint = (neg_dw, pos_dw)

    skipped = len(dw_values) - len(collected)
    return SignScan(has_sign_change, bracket_hint, len(collected), skipped, best_abs_err, best_dw)


def evaluate_geometry(
    w: float,
    g: float,
    t: float,
    param_template,
    lambda0: float,
    cache_dir: str,
    abs_tol: float,
    rel_tol: float,
    search_min: float,
    search_max: float,
    hard_min: float,
    hard_max: float,
    dw_step: float,
    seed_step: float,
    max_iter: int,
) -> GeometryResult:
    """Full evaluation pipeline for a single geometry."""
    sign_info = sample_G_curve(
        param_template=param_template,
        w=w,
        g=g,
        t=t,
        lambda0=lambda0,
        cache_dir=cache_dir,
        search_min=search_min,
        search_max=search_max,
        dw_step=dw_step,
    )

    delta_w_star, diagnostics = solve_delta_w_star(
        w=w,
        g=g,
        t=t,
        lambda0=lambda0,
        param_template=param_template,
        cache_dir=cache_dir,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        search_min=search_min,
        search_max=search_max,
        hard_min=hard_min,
        hard_max=hard_max,
        seed_step=seed_step,
        max_iter=max_iter,
    )

    return GeometryResult(
        w=float(w),
        g=float(g),
        t=float(t),
        sign_scan=sign_info,
        delta_w_star=None if delta_w_star is None else float(delta_w_star),
        diagnostics=diagnostics or {},
    )


# ---------------------------------------------------------------------------
# Sweeping and reporting
# ---------------------------------------------------------------------------


def _build_axis(range_tuple: Tuple[float, float, float]) -> np.ndarray:
    start, stop, step = range_tuple
    return np.round(np.arange(start, stop + step * 0.5, step), 4)


def _log_geometry_attempt(w: float, g: float, t: float) -> None:
    print(f"[Geometry] Trying w={w:.3f} µm, g={g:.3f} µm, t={t:.3f} µm", flush=True)


def _log_geometry_result(result: GeometryResult) -> None:
    delta_w = result.delta_w_star
    abs_err = result.abs_err
    rel_err = result.rel_err
    diag = result.diagnostics
    L50_te = diag.get("L50_te", np.nan)
    L50_tm = diag.get("L50_tm", np.nan)
    bracket_count = diag.get("n_brackets", 0)
    tol = diag.get("tolerance_met", False)
    method = "bracket" if bracket_count else "newton"
    usable = "usable" if result.usable_zero else "fallback"
    delta_w_str = f"{delta_w:+.4f}" if (delta_w is not None and np.isfinite(delta_w)) else "n/a"
    abs_err_str = f"{abs_err:.4f}" if np.isfinite(abs_err) else "nan"
    rel_err_pct = rel_err * 100 if np.isfinite(rel_err) else np.nan
    L50_te_str = f"{L50_te:.3f}" if np.isfinite(L50_te) else "nan"
    L50_tm_str = f"{L50_tm:.3f}" if np.isfinite(L50_tm) else "nan"
    print(
        "[Geometry] Result: "
        f"Δw*={delta_w_str} µm, |Δ|={abs_err_str} µm ({rel_err_pct:.2f}%), "
        f"L50_TE={L50_te_str} µm, L50_TM={L50_tm_str} µm, "
        f"brackets={bracket_count}, tol={'yes' if tol else 'no'}, path={method}, {usable}",
        flush=True,
    )


def _load_processed_keys(csv_path: str) -> set:
    path = pathlib.Path(csv_path)
    if not path.exists():
        return set()
    processed = set()
    try:
        with path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    w = round(float(row["w_um"]), 6)
                    g = round(float(row["g_um"]), 6)
                    t = round(float(row["t_um"]), 6)
                    processed.add((w, g, t))
                except (KeyError, ValueError):
                    continue
    except FileNotFoundError:
        return set()
    return processed


def scan_geometry_grid(
    w_range: Tuple[float, float, float],
    g_range: Tuple[float, float, float],
    t_range: Tuple[float, float, float],
    param_template,
    lambda0: float,
    cache_dir: str = "data",
    abs_tol: float = 0.05,
    rel_tol: float = 0.01,
    search_min: float = -0.30,
    search_max: float = 0.30,
    hard_min: float = -0.35,
    hard_max: float = 0.35,
    dw_step: float = 0.01,
    seed_step: float = 0.005,
    max_iter: int = 80,
    csv_path: Optional[str] = None,
    resume: bool = False,
) -> List[GeometryResult]:
    """Sweep the provided ranges and collect GeometryResult objects."""
    w_vals = _build_axis(w_range)
    g_vals = _build_axis(g_range)
    t_vals = _build_axis(t_range)

    combos = len(w_vals) * len(g_vals) * len(t_vals)
    results: List[GeometryResult] = []

    iterator = tqdm(
        total=combos,
        desc="Scanning geometries",
        unit="geom",
        ncols=100,
        mininterval=0.5,
    )

    processed_keys = set()
    if csv_path and resume:
        processed_keys = _load_processed_keys(csv_path)

    for w in w_vals:
        for g in g_vals:
            for t in t_vals:
                key = (round(float(w), 6), round(float(g), 6), round(float(t), 6))
                if key in processed_keys:
                    iterator.update(1)
                    continue
                _log_geometry_attempt(w, g, t)
                result = evaluate_geometry(
                    w=w,
                    g=g,
                    t=t,
                    param_template=param_template,
                    lambda0=lambda0,
                    cache_dir=cache_dir,
                    abs_tol=abs_tol,
                    rel_tol=rel_tol,
                    search_min=search_min,
                    search_max=search_max,
                    hard_min=hard_min,
                    hard_max=hard_max,
                    dw_step=dw_step,
                    seed_step=seed_step,
                    max_iter=max_iter,
                )
                results.append(result)
                _log_geometry_result(result)
                iterator.update(1)
                if csv_path:
                    write_csv(results, csv_path)

    iterator.close()
    return results


def print_summary(results: Sequence[GeometryResult], top_k: int = 10) -> None:
    """Pretty-print usable geometries and closest failures."""
    usable = [res for res in results if res.usable_zero]
    usable.sort(key=ranking_score)

    print("\n=== Geometries with bracketed, tolerance-meeting zeros ===")
    if not usable:
        print("  (none)")
    else:
        for res in usable[:top_k]:
            bracket = res.sign_scan.bracket_hint or ("?", "?")
            print(
                f"  w={res.w:.3f} µm, g={res.g:.3f} µm, t={res.t:.3f} µm  |  "
                f"Δw*={res.delta_w_star:+.4f} µm  |  "
                f"|L50_TE-L50_TM|={res.abs_err:.4f} µm ({res.rel_err*100:.2f}%)  |  "
                f"bracket≈{bracket}"
            )

    failing = [res for res in results if not res.usable_zero and res.sign_scan.sample_points > 0]
    failing.sort(key=ranking_score)

    print("\n=== Closest mismatches (no usable zero) ===")
    if not failing:
        print("  (none)")
        return

    for res in failing[:top_k]:
        reason = res.failure_reason()
        bracket = res.sign_scan.bracket_hint or ("?", "?")
        best_hint = (
            f"best sampled |Δ|={res.sign_scan.best_abs_err:.3f} µm at Δw={res.sign_scan.best_dw:+.3f} µm"
            if np.isfinite(res.sign_scan.best_abs_err)
            else "no feasible samples"
        )
        print(
            f"  w={res.w:.3f}, g={res.g:.3f}, t={res.t:.3f}  |  "
            f"solver Δw*={res.delta_w_star if res.delta_w_star is not None else 'n/a'}  |  "
            f"{best_hint}  |  bracket≈{bracket}  |  reason: {reason}"
        )


def ranking_score(res: GeometryResult) -> float:
    abs_err = res.abs_err if np.isfinite(res.abs_err) else 1e3
    penalty = 0.0
    if not res.sign_scan.has_sign_change:
        penalty += 100.0
    if res.sign_scan.sample_points == 0:
        penalty += 500.0
    if not res.has_bracket:
        penalty += 10.0
    if not res.tolerance_met:
        penalty += 1.0
    return abs_err + penalty


def write_csv(results: Sequence[GeometryResult], path: str) -> None:
    fieldnames = [
        "w_um",
        "g_um",
        "t_um",
        "sign_change",
        "sign_samples",
        "sign_skipped",
        "bracket_start_um",
        "bracket_end_um",
        "delta_w_star_um",
        "abs_err_um",
        "rel_err",
        "n_brackets",
        "tolerance_met",
        "usable_zero",
        "fallback_reason",
        "best_sample_abs_err_um",
        "best_sample_dw_um",
        "ranking_score",
        "L50_te_um",
        "L50_tm_um",
    ]
    csv_path = pathlib.Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read existing data to preserve it
    existing_results = {}
    if csv_path.exists():
        try:
            with csv_path.open("r", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        key = (round(float(row["w_um"]), 6), 
                               round(float(row["g_um"]), 6), 
                               round(float(row["t_um"]), 6))
                        existing_results[key] = row
                    except (KeyError, ValueError):
                        continue
        except Exception:
            # If reading fails, start fresh
            existing_results = {}
    
    # Merge new results with existing (new results overwrite old ones for same geometry)
    for res in results:
        key = (round(float(res.w), 6), round(float(res.g), 6), round(float(res.t), 6))
        bracket = res.sign_scan.bracket_hint or (None, None)
        existing_results[key] = {
            "w_um": res.w,
            "g_um": res.g,
            "t_um": res.t,
            "sign_change": res.sign_scan.has_sign_change,
            "sign_samples": res.sign_scan.sample_points,
            "sign_skipped": res.sign_scan.skipped_points,
            "bracket_start_um": bracket[0],
            "bracket_end_um": bracket[1],
            "delta_w_star_um": res.delta_w_star,
            "abs_err_um": res.abs_err,
            "rel_err": res.rel_err,
            "n_brackets": res.diagnostics.get("n_brackets", 0),
            "tolerance_met": res.diagnostics.get("tolerance_met", False),
            "usable_zero": res.usable_zero,
            "fallback_reason": res.failure_reason(),
            "best_sample_abs_err_um": res.sign_scan.best_abs_err,
            "best_sample_dw_um": res.sign_scan.best_dw,
            "ranking_score": ranking_score(res),
            "L50_te_um": res.diagnostics.get("L50_te", np.nan),
            "L50_tm_um": res.diagnostics.get("L50_tm", np.nan),
        }
    
    # Convert back to GeometryResult-like objects for sorting, then write
    # We'll create a simple dict-based sorting
    all_rows = list(existing_results.values())
    # Sort by ranking_score (convert to float, handle missing values)
    all_rows.sort(key=lambda r: float(r.get("ranking_score", 1e9) if r.get("ranking_score") not in (None, "", "nan") else 1e9))
    
    # Write all results back to file
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan geometries for usable Δw brackets.")
    parser.add_argument("--w-range", type=float, nargs=3, default=(1.05, 1.25, 0.05), metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--g-range", type=float, nargs=3, default=(0.26, 0.34, 0.02), metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--t-range", type=float, nargs=3, default=(0.28, 0.34, 0.02), metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--abs-tol", type=float, default=0.05)
    parser.add_argument("--rel-tol", type=float, default=0.01)
    parser.add_argument("--search-min", type=float, default=-0.30)
    parser.add_argument("--search-max", type=float, default=0.30)
    parser.add_argument("--hard-min", type=float, default=-0.35)
    parser.add_argument("--hard-max", type=float, default=0.35)
    parser.add_argument("--dw-step", type=float, default=0.01, help="Δw sampling step for the sign scan")
    parser.add_argument("--seed-step", type=float, default=0.005, help="Seed scan step for solve_delta_w_star")
    parser.add_argument("--max-iter", type=int, default=80, help="Maximum Newton iterations")
    parser.add_argument("--csv", type=str, default="results/sweeps/tune_geometry.csv", help="Path to save detailed CSV results")
    parser.add_argument("--resume", type=bool, default=True, help="Skip geometries already present in the CSV")
    parser.add_argument("--rewrite", type=bool, default=False, help="Ignore existing CSV contents and start over")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from dc_study import param as DEFAULT_PARAM_TEMPLATE
        lambda0 = float(getattr(DEFAULT_PARAM_TEMPLATE, "wl_0", 1.55))
    except ImportError as exc:
        raise RuntimeError("Unable to import dc_study.param as the default template") from exc

    results = scan_geometry_grid(
        w_range=tuple(args.w_range),
        g_range=tuple(args.g_range),
        t_range=tuple(args.t_range),
        param_template=DEFAULT_PARAM_TEMPLATE,
        lambda0=lambda0,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        search_min=args.search_min,
        search_max=args.search_max,
        hard_min=args.hard_min,
        hard_max=args.hard_max,
        dw_step=args.dw_step,
        seed_step=args.seed_step,
        max_iter=args.max_iter,
        csv_path=args.csv,
        resume=args.resume and not args.rewrite,
    )
    print_summary(results)


if __name__ == "__main__":
    main()
