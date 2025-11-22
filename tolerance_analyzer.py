

#!/usr/bin/env python3

"""Tolerance analysis helper for DC Monte Carlo CSVs.

This script parses the tolerance_*.csv files produced by the Monte Carlo
runs (single-parameter and joint) and reports the numerical metrics used in
the report:

- For each single-parameter file (Variable: w, g, or t):
  * Per-polarization statistics for η and V (mean, std, 95% interval).
  * 95% half-range Δη_95^{(p,pol)} and ΔV_95^{(p,pol)}.
  * Worst-case half-range over polarizations Δη_95^{(p)}, ΔV_95^{(p)}.

- For the joint file (Variable: joint):
  * 95% intervals and extrema for η and V for TE and TM.
  * Yield for a given spec: |η−0.5| ≤ δη_spec and V ≥ V_min for both pols.

- Overall tolerance penalty σ_tol used in the FOM:

    σ_tol = max_p (Δη_95^{(p)} + ΔV_95^{(p)})

Run from the directory containing the CSVs, e.g.:

    python tolerance_analyzer.py

or specify a directory:

    python tolerance_analyzer.py --data-dir path/to/csvs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Utility: parse header comments
# -----------------------------------------------------------------------------


def parse_header(path: Path) -> Dict[str, str]:
    """Parse the leading commented header lines into a dict.

    Expected format (lines starting with '#'):
        # Key: value

    Returns a dict mapping "Key" -> "value" (both stripped).
    """

    header: Dict[str, str] = {}
    with path.open("r") as f:
        for line in f:
            if not line.startswith("#"):
                break
            # Drop leading '#', strip spaces
            line = line[1:].strip()
            if ":" in line:
                key, val = line.split(":", 1)
                header[key.strip()] = val.strip()
    return header


# -----------------------------------------------------------------------------
# Metrics for single-parameter tolerance sweeps
# -----------------------------------------------------------------------------


def compute_single_param_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute per-polarization statistics and 95% half-ranges for a
    single-parameter tolerance sweep.

    Assumes columns: eta_TE, eta_TM, V_TE, V_TM.

    Returns a nested dict with keys:
        metrics[pol]['eta_mean'], 'eta_std', 'eta_p2.5', 'eta_p97.5',
        'eta_half_range_95', 'eta_dev95_from_ideal',
        and similarly for V, plus
        metrics['delta_eta95_param'], metrics['delta_V95_param'].
    """

    metrics: Dict[str, Dict[str, float]] = {}

    for pol in ("TE", "TM"):
        eta = df[f"eta_{pol}"].to_numpy()
        V = df[f"V_{pol}"].to_numpy()

        p2_eta, p97_eta = np.percentile(eta, [2.5, 97.5])
        p2_V, p97_V = np.percentile(V, [2.5, 97.5])

        metrics[pol] = {
            "eta_mean": float(np.mean(eta)),
            "eta_std": float(np.std(eta, ddof=1)),
            "eta_p2.5": float(p2_eta),
            "eta_p97.5": float(p97_eta),
            # central 95% half-range
            "eta_half_range_95": float(0.5 * (p97_eta - p2_eta)),
            # deviation from ideal 0.5 at 95% level
            "eta_dev95_from_ideal": float(
                max(abs(p2_eta - 0.5), abs(p97_eta - 0.5))
            ),
            "V_mean": float(np.mean(V)),
            "V_std": float(np.std(V, ddof=1)),
            "V_p2.5": float(p2_V),
            "V_p97.5": float(p97_V),
            "V_half_range_95": float(0.5 * (p97_V - p2_V)),
            # deviation from ideal 1.0 at 95% level
            "V_dev95_from_ideal": float(
                max(abs(p2_V - 1.0), abs(p97_V - 1.0))
            ),
        }

    # Worst-case 95% half-ranges over polarizations
    delta_eta95 = max(metrics[pol]["eta_half_range_95"] for pol in ("TE", "TM"))
    delta_V95 = max(metrics[pol]["V_half_range_95"] for pol in ("TE", "TM"))

    metrics["delta_eta95_param"] = float(delta_eta95)
    metrics["delta_V95_param"] = float(delta_V95)

    return metrics


# -----------------------------------------------------------------------------
# Metrics for joint tolerance (w,g,t perturbed together)
# -----------------------------------------------------------------------------


def compute_joint_metrics(
    df: pd.DataFrame,
    spec_eta_tol: float = 0.02,
    spec_V_min: float = 0.98,
) -> Dict[str, Dict[str, float]]:
    """Compute statistics and yield for a joint tolerance run.

    Assumes columns: eta_TE, eta_TM, V_TE, V_TM.

    Returns a dict with keys 'TE', 'TM', and yields:
        metrics['TE']['eta_min'], 'eta_max', 'eta_p2.5', 'eta_p97.5', ...
        metrics['TM'][...]
        metrics['yield_TE'], metrics['yield_TM'], metrics['yield_both'].
    """

    metrics: Dict[str, Dict[str, float]] = {}

    for pol in ("TE", "TM"):
        eta = df[f"eta_{pol}"].to_numpy()
        V = df[f"V_{pol}"].to_numpy()

        p2_eta, p97_eta = np.percentile(eta, [2.5, 97.5])
        p2_V, p97_V = np.percentile(V, [2.5, 97.5])

        metrics[pol] = {
            "eta_min": float(np.min(eta)),
            "eta_max": float(np.max(eta)),
            "eta_p2.5": float(p2_eta),
            "eta_p97.5": float(p97_eta),
            "V_min": float(np.min(V)),
            "V_max": float(np.max(V)),
            "V_p2.5": float(p2_V),
            "V_p97.5": float(p97_V),
        }

    # Spec-based yield: |η-0.5| ≤ spec_eta_tol and V ≥ spec_V_min
    eta_TE = df["eta_TE"].to_numpy()
    eta_TM = df["eta_TM"].to_numpy()
    V_TE = df["V_TE"].to_numpy()
    V_TM = df["V_TM"].to_numpy()

    cond_TE = (np.abs(eta_TE - 0.5) <= spec_eta_tol) & (V_TE >= spec_V_min)
    cond_TM = (np.abs(eta_TM - 0.5) <= spec_eta_tol) & (V_TM >= spec_V_min)
    cond_both = cond_TE & cond_TM

    metrics["yield_TE"] = float(cond_TE.mean())
    metrics["yield_TM"] = float(cond_TM.mean())
    metrics["yield_both"] = float(cond_both.mean())

    return metrics


# -----------------------------------------------------------------------------
# File-level analyzers
# -----------------------------------------------------------------------------


def analyze_single_param_file(path: Path) -> Tuple[str, Dict[str, float]]:
    """Analyze a single-parameter tolerance CSV (Variable: w, g, or t)."""

    header = parse_header(path)
    var = header.get("Variable", "?")
    sigma = header.get("Sigma_nm", "?")

    df = pd.read_csv(path, comment="#")
    metrics = compute_single_param_metrics(df)

    print(f"\n=== Single-parameter tolerance: {path.name} ===")
    print(f"Parameter: {var}, σ = {sigma} nm, N = {len(df)}")

    for pol in ("TE", "TM"):
        m = metrics[pol]
        print(f"  {pol}:")
        print(
            "    η:   mean={eta_mean:.6f}, std={eta_std:.6f}, "
            "95%=[{eta_p2:.6f}, {eta_p97:.6f}], "
            "half-range={eta_hr:.6f}, dev_from_0.5_95={eta_dev:.6f}".format(
                eta_mean=m["eta_mean"],
                eta_std=m["eta_std"],
                eta_p2=m["eta_p2.5"],
                eta_p97=m["eta_p97.5"],
                eta_hr=m["eta_half_range_95"],
                eta_dev=m["eta_dev95_from_ideal"],
            )
        )
        print(
            "    V:   mean={V_mean:.6f}, std={V_std:.6f}, "
            "95%=[{V_p2:.6f}, {V_p97:.6f}], "
            "half-range={V_hr:.6f}, dev_from_1.0_95={V_dev:.6f}".format(
                V_mean=m["V_mean"],
                V_std=m["V_std"],
                V_p2=m["V_p2.5"],
                V_p97=m["V_p97.5"],
                V_hr=m["V_half_range_95"],
                V_dev=m["V_dev95_from_ideal"],
            )
        )

    print(
        "  Worst-case 95% half-ranges over pols: "
        f"Δη_95^({var}) = {metrics['delta_eta95_param']:.6f}, "
        f"ΔV_95^({var}) = {metrics['delta_V95_param']:.6f}"
    )

    return var, metrics


def analyze_joint_file(
    path: Path, spec_eta_tol: float = 0.02, spec_V_min: float = 0.98
) -> Dict[str, Dict[str, float]]:
    """Analyze a joint tolerance CSV (Variable: joint)."""

    header = parse_header(path)
    df = pd.read_csv(path, comment="#")
    metrics = compute_joint_metrics(df, spec_eta_tol, spec_V_min)

    print(f"\n=== Joint tolerance: {path.name} ===")

    sig_w = header.get("Sigma_w_nm", "?")
    sig_g = header.get("Sigma_g_nm", "?")
    sig_t = header.get("Sigma_t_nm", "?")
    print(f"Sigmas: σ_w={sig_w} nm, σ_g={sig_g} nm, σ_t={sig_t} nm, N={len(df)}")

    for pol in ("TE", "TM"):
        m = metrics[pol]
        print(f"  {pol}:")
        print(
            "    η:   min={eta_min:.6f}, max={eta_max:.6f}, "
            "95%=[{eta_p2:.6f}, {eta_p97:.6f}]".format(
                eta_min=m["eta_min"],
                eta_max=m["eta_max"],
                eta_p2=m["eta_p2.5"],
                eta_p97=m["eta_p97.5"],
            )
        )
        print(
            "    V:   min={V_min:.6f}, max={V_max:.6f}, "
            "95%=[{V_p2:.6f}, {V_p97:.6f}]".format(
                V_min=m["V_min"],
                V_max=m["V_max"],
                V_p2=m["V_p2.5"],
                V_p97=m["V_p97.5"],
            )
        )

    print(
        f"  Spec: |η-0.5| ≤ {spec_eta_tol}, V ≥ {spec_V_min}\n"
        f"  Yield: TE={metrics['yield_TE']*100:.1f}%, "
        f"TM={metrics['yield_TM']*100:.1f}%, "
        f"both={metrics['yield_both']*100:.1f}%"
    )

    return metrics


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze tolerance_*.csv files and print Δη_95, ΔV_95, and yield "
            "metrics used in the design study report."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results/tolerance",
        help="Directory containing tolerance_*.csv files (default: current directory)",
    )
    parser.add_argument(
        "--eta-spec",
        type=float,
        default=0.02,
        help="Spec on |η-0.5| for yield calculation (default: 0.02)",
    )
    parser.add_argument(
        "--V-spec",
        type=float,
        default=0.98,
        help="Minimum HOM visibility for yield calculation (default: 0.98)",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    csv_paths = sorted(data_dir.glob("tolerance_*.csv"))
    if not csv_paths:
        print(f"No tolerance_*.csv files found in {data_dir.resolve()}")
        return

    # Collect single-parameter metrics to compute σ_tol
    single_param_metrics: Dict[str, Dict[str, float]] = {}
    joint_metrics: Dict[str, Dict[str, float]] | None = None

    for path in csv_paths:
        header = parse_header(path)
        var = header.get("Variable", "?")

        if var in {"w", "g", "t"}:
            p, metrics = analyze_single_param_file(path)
            single_param_metrics[p] = metrics
        elif var == "joint":
            joint_metrics = analyze_joint_file(
                path, spec_eta_tol=args.eta_spec, spec_V_min=args.V_spec
            )
        else:
            print(f"\n[Warning] Skipping {path.name}: unknown Variable='{var}'")

    # Compute global σ_tol if we have single-parameter sweeps
    if single_param_metrics:
        print("\n=== Aggregated tolerance penalty σ_tol ===")
        sigma_tol = 0.0
        for p, m in single_param_metrics.items():
            sigma_p = m["delta_eta95_param"] + m["delta_V95_param"]
            print(
                f"  p={p}: Δη_95^{p}={m['delta_eta95_param']:.6f}, "
                f"ΔV_95^{p}={m['delta_V95_param']:.6f}, "
                f"σ_p={sigma_p:.6f}"
            )
            sigma_tol = max(sigma_tol, sigma_p)

        print(f"\n  σ_tol = max_p σ_p = {sigma_tol:.6f}\n")
    else:
        print("\n[Info] No single-parameter tolerance files found; σ_tol not computed.\n")


if __name__ == "__main__":
    main()