"""
Tolerance analysis script for directional coupler geometry parameters.

This script performs Monte Carlo tolerance analysis by perturbing geometry
parameters (t, w, g) around manually specified nominal values. It can run in two modes:
1. Single variable mode: perturb one parameter at a time (t, w, or g)
2. Joint mode: perturb all three parameters simultaneously

It freezes delta_w_star and L_c at their nominal designed values to model realistic
fabrication tolerance/yield analysis.

Results are saved to CSV files in results/tolerance/ and plots are generated.
You can regenerate plots from existing CSV files without re-running simulations.

Usage:
    # Run simulations and generate plots for a single variable
    python test_tolerance.py --variable w --w 1.1 --g 0.28 --t 0.28 --n-samples 20 --sigma-w-nm 15.0
    
    # Run simulations for all variables (t, w, g) sequentially
    python test_tolerance.py --variable all --w 1.1 --g 0.28 --t 0.28 --n-samples 20
    
    # Run joint Monte Carlo (all parameters perturbed simultaneously)
    python test_tolerance.py --variable joint --w 1.1 --g 0.28 --t 0.28 --n-samples 20
    
    # Run with fixed delta_w (skip solving)
    python test_tolerance.py --variable w --w 1.1 --g 0.28 --t 0.28 --delta-w -0.32 --n-samples 20 --sigma-w-nm 15.0
    
    # Regenerate plots from existing CSV (no simulations)
    python test_tolerance.py --variable w --plot-only
    
    # Regenerate plots for all variables from existing CSV files
    python test_tolerance.py --variable all --plot-only
    
    # Regenerate plots for joint analysis
    python test_tolerance.py --variable joint --plot-only
"""

from __future__ import annotations

import os, sys

print(">>> CWD:", os.getcwd())
print(">>> PYTHON:", sys.executable)
print(">>> sys.path:")
for p in sys.path:
    print("   ", p)

import argparse
import csv
import copy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dc_study import param as DEFAULT_PARAM_TEMPLATE, run_single
from building_utils import update_param_derived
from delta_width import solve_delta_w_star


def compute_nominal_delta_w_star(w_nom: float, g_nom: float, t_nom: float,
                                 param_template, lambda0: float) -> Tuple[float, dict]:
    """
    Compute nominal delta_w_star for the given geometry.
    
    Args:
        w_nom: Nominal waveguide width
        g_nom: Nominal coupling gap
        t_nom: Nominal thickness
        param_template: Parameter template from dc_study
        lambda0: Design wavelength
    
    Returns:
        Tuple of (delta_w_star_nom, diagnostics_dict)
    """
    # Get tolerance and search bound hyperparameters from param_template
    abs_tol = getattr(param_template, 'delta_w_abs_tol', 0.05)
    rel_tol = getattr(param_template, 'delta_w_rel_tol', 0.01)
    search_min = getattr(param_template, 'delta_w_search_min', -0.30)
    search_max = getattr(param_template, 'delta_w_search_max', 0.30)
    hard_min = getattr(param_template, 'delta_w_hard_min', -0.35)
    hard_max = getattr(param_template, 'delta_w_hard_max', 0.35)
    seed_step = getattr(param_template, 'delta_w_h_init', 0.01)  # Use h_init as seed_step
    max_iter = getattr(param_template, 'delta_w_max_iter', 80)
    
    print(f"[Setup] Solving for Δw*: w={w_nom:.3f} µm, g={g_nom:.3f} µm, t={t_nom:.3f} µm")
    
    delta_w_star, diagnostics = solve_delta_w_star(
        w=w_nom,
        g=g_nom,
        t=t_nom,
        lambda0=lambda0,
        param_template=param_template,
        cache_dir="data",
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        search_min=search_min,
        search_max=search_max,
        hard_min=hard_min,
        hard_max=hard_max,
        seed_step=seed_step,
        max_iter=max_iter,
    )
    
    if delta_w_star is None:
        raise RuntimeError(
            f"Failed to solve for delta_w_star for geometry w={w_nom:.3f}, g={g_nom:.3f}, t={t_nom:.3f}. "
            f"Check diagnostics or try different geometry."
        )
    
    print(f"[Setup] Computed Δw* = {delta_w_star:+.4f} µm")
    
    return float(delta_w_star), diagnostics


def compute_nominal_Lc(w_nom: float, g_nom: float, t_nom: float, delta_w_star_nom: float,
                       param_template, lambda0: float) -> float:
    """
    Compute nominal L_c for the given geometry.
    
    Args:
        w_nom: Nominal waveguide width
        g_nom: Nominal coupling gap
        t_nom: Nominal thickness
        delta_w_star_nom: Nominal delta_w_star
        param_template: Parameter template from dc_study
        lambda0: Design wavelength
    
    Returns:
        Nominal coupling length L_c
    """
    # Create temporary param to compute L_c
    temp_param = copy.deepcopy(param_template)
    temp_param.wg_width = float(w_nom)
    temp_param.coupling_gap = float(g_nom)
    temp_param.wg_thick = float(t_nom)
    temp_param.delta_w = float(delta_w_star_nom)
    temp_param.freeze_l_c = True  # This won't prevent initial computation
    
    # Compute L_c (this will compute it once)
    update_param_derived(temp_param, solve_delta_w=False)
    
    L_c_nom = float(temp_param.coupling_length)
    print(f"[Setup] Computed L_c_nom = {L_c_nom:.3f} µm")
    
    return L_c_nom


def perturb_parameter(base_value: float, sigma_nm: float, n_samples: int,
                     seed: Optional[int] = None) -> np.ndarray:
    """
    Generate Monte Carlo samples using normal distribution.
    
    Args:
        base_value: Nominal value to perturb around (in µm)
        sigma_nm: Standard deviation in nanometers
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        Array of perturbed values (in µm)
    """
    rng = np.random.default_rng(seed)
    std_dev = sigma_nm * 1e-3  # nm → µm
    samples = rng.normal(loc=base_value, scale=std_dev, size=n_samples)
    return samples


def perturb_joint_triplets(w_nom: float, g_nom: float, t_nom: float,
                           sigma_w_nm: float, sigma_g_nm: float, sigma_t_nm: float,
                           n_samples: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Monte Carlo samples for all three parameters simultaneously.
    
    Args:
        w_nom: Nominal width (µm)
        g_nom: Nominal gap (µm)
        t_nom: Nominal thickness (µm)
        sigma_w_nm: Standard deviation for width in nm
        sigma_g_nm: Standard deviation for gap in nm
        sigma_t_nm: Standard deviation for thickness in nm
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (w_samples, g_samples, t_samples) arrays (in µm)
    """
    rng = np.random.default_rng(seed)
    std_w = sigma_w_nm * 1e-3  # nm → µm
    std_g = sigma_g_nm * 1e-3
    std_t = sigma_t_nm * 1e-3
    
    w_samples = rng.normal(loc=w_nom, scale=std_w, size=n_samples)
    g_samples = rng.normal(loc=g_nom, scale=std_g, size=n_samples)
    t_samples = rng.normal(loc=t_nom, scale=std_t, size=n_samples)
    
    return w_samples, g_samples, t_samples


def create_perturbed_param(base_param, variable: str, perturbed_value: float,
                           w_nom: float, g_nom: float, t_nom: float,
                           delta_w_star_nom: float, L_c_nom: float) -> SimpleNamespace:
    """
    Create a parameter instance with perturbed geometry.
    
    Args:
        base_param: Base parameter namespace
        variable: Which variable to perturb ('t', 'w', or 'g')
        perturbed_value: The perturbed value
        w_nom: Nominal width
        g_nom: Nominal gap
        t_nom: Nominal thickness
        delta_w_star_nom: Nominal delta_w_star (frozen)
        L_c_nom: Nominal L_c (frozen)
    
    Returns:
        New parameter namespace with perturbed geometry
    """
    param = copy.deepcopy(base_param)
    
    if variable == "w":
        # Apply common width bias to both arms
        # w1 = w_nom + delta_w_star_nom/2 + δw
        # w2 = w_nom - delta_w_star_nom/2 + δw
        # This keeps delta_w = delta_w_star_nom constant
        delta_w_bias = perturbed_value - w_nom  # This is δw, the bias applied to both arms
        param.wg_width = float(perturbed_value)  # Average width = w_nom + δw
        param.delta_w = float(delta_w_star_nom)  # Keep delta_w frozen
        param.wg_width_left = float(w_nom + delta_w_star_nom / 2 + delta_w_bias)
        param.wg_width_right = float(w_nom - delta_w_star_nom / 2 + delta_w_bias)
        param.coupling_gap = float(g_nom)
        param.wg_thick = float(t_nom)
    elif variable == "g":
        param.wg_width = float(w_nom)
        param.delta_w = float(delta_w_star_nom)
        param.wg_width_left = float(w_nom + delta_w_star_nom / 2)
        param.wg_width_right = float(w_nom - delta_w_star_nom / 2)
        param.coupling_gap = float(perturbed_value)
        param.wg_thick = float(t_nom)
    elif variable == "t":
        param.wg_width = float(w_nom)
        param.delta_w = float(delta_w_star_nom)
        param.wg_width_left = float(w_nom + delta_w_star_nom / 2)
        param.wg_width_right = float(w_nom - delta_w_star_nom / 2)
        param.coupling_gap = float(g_nom)
        param.wg_thick = float(perturbed_value)
    else:
        raise ValueError(f"Unknown variable: {variable}. Must be 't', 'w', or 'g'")
    
    # Freeze L_c and delta_w - do NOT recompute
    param.coupling_length = float(L_c_nom)
    param.delta_w = float(delta_w_star_nom)
    
    # Update domain sizes manually (freeze size_x since it depends on L_c)
    # Only update size_y if gap or width changed significantly
    w_max = max(param.wg_width_left, param.wg_width_right)
    pad_extra = getattr(param, "pad_extra", 0.05)
    param.size_y = (
        2 * (param.sbend_height + param.wl_0)
        + w_max
        + param.coupling_gap
        + max(0.0, float(w_max) - float(param.wl_0))
        + float(pad_extra)
    )
    # size_x stays frozen (depends on L_c which is frozen)
    # size_z stays frozen (depends on wl_0 and t, but t changes are small)
    
    # Prevent update_param_derived from recomputing L_c by setting geometry signature
    # This tricks it into thinking geometry hasn't changed
    param._last_geometry_for_lc = (
        param.wg_width_left,
        param.wg_width_right,
        float(param.coupling_gap),
        float(param.wg_thick),
        float(getattr(param.medium.SiN, 'permittivity', 1.0)),
        float(getattr(param.medium.SiO2, 'permittivity', 1.0)),
    )
    
    return param


def create_perturbed_param_joint(base_param, w_perturbed: float, g_perturbed: float, t_perturbed: float,
                                 w_nom: float, g_nom: float, t_nom: float,
                                 delta_w_star_nom: float, L_c_nom: float) -> SimpleNamespace:
    """
    Create a parameter instance with all three geometry parameters perturbed simultaneously.
    
    Args:
        base_param: Base parameter namespace
        w_perturbed: Perturbed width
        g_perturbed: Perturbed gap
        t_perturbed: Perturbed thickness
        w_nom: Nominal width
        g_nom: Nominal gap
        t_nom: Nominal thickness
        delta_w_star_nom: Nominal delta_w_star (frozen)
        L_c_nom: Nominal L_c (frozen)
    
    Returns:
        New parameter namespace with perturbed geometry
    """
    param = copy.deepcopy(base_param)
    
    # Apply common width bias to both arms
    # w1 = w_nom + delta_w_star_nom/2 + δw
    # w2 = w_nom - delta_w_star_nom/2 + δw
    # This keeps delta_w = delta_w_star_nom constant
    delta_w_bias = w_perturbed - w_nom  # This is δw, the bias applied to both arms
    param.wg_width = float(w_perturbed)  # Average width = w_nom + δw
    param.delta_w = float(delta_w_star_nom)  # Keep delta_w frozen
    param.wg_width_left = float(w_nom + delta_w_star_nom / 2 + delta_w_bias)
    param.wg_width_right = float(w_nom - delta_w_star_nom / 2 + delta_w_bias)
    param.coupling_gap = float(g_perturbed)
    param.wg_thick = float(t_perturbed)
    
    # Freeze L_c and delta_w - do NOT recompute
    param.coupling_length = float(L_c_nom)
    param.delta_w = float(delta_w_star_nom)
    
    # Update domain sizes manually (freeze size_x since it depends on L_c)
    # Only update size_y if gap or width changed significantly
    w_max = max(param.wg_width_left, param.wg_width_right)
    pad_extra = getattr(param, "pad_extra", 0.05)
    param.size_y = (
        2 * (param.sbend_height + param.wl_0)
        + w_max
        + param.coupling_gap
        + max(0.0, float(w_max) - float(param.wl_0))
        + float(pad_extra)
    )
    # size_x stays frozen (depends on L_c which is frozen)
    # size_z stays frozen (depends on wl_0 and t, but t changes are small)
    
    # Prevent update_param_derived from recomputing L_c by setting geometry signature
    # This tricks it into thinking geometry hasn't changed
    param._last_geometry_for_lc = (
        param.wg_width_left,
        param.wg_width_right,
        float(param.coupling_gap),
        float(param.wg_thick),
        float(getattr(param.medium.SiN, 'permittivity', 1.0)),
        float(getattr(param.medium.SiO2, 'permittivity', 1.0)),
    )
    
    return param


def run_tolerance_simulation(param, lambda0: float, task_tag: str) -> Optional[dict]:
    """
    Run simulation for perturbed geometry and extract eta/V at lambda0.
    
    Args:
        param: Parameter namespace with perturbed geometry
        lambda0: Wavelength to simulate at (µm)
        task_tag: Tag for simulation task name
    
    Returns:
        Dictionary with eta_TE, eta_TM, V_TE, V_TM at lambda0, or None if failed
    """
    results = {}
    
    for pol in ["te", "tm"]:
        try:
            summary = run_single(param, pol=pol, task_tag=task_tag, dry_run=False, lambda_single=lambda0)
            if summary is None:
                return None
            
            lam = summary["lam"]
            idx = int(np.argmin(np.abs(lam - lambda0)))
            
            results[f"eta_{pol}"] = float(summary["eta"][idx])
            results[f"V_{pol}"] = float(summary["V"][idx])
        except Exception as e:
            print(f"[Simulation] Failed for {pol}: {e}")
            return None
    
    return results


def load_tolerance_csv(csv_path: Path) -> Tuple[np.ndarray, dict, str]:
    """
    Load tolerance data from CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Tuple of (perturbed_values, results_dict, variable)
    """
    metadata = {}
    variable = None
    
    with csv_path.open("r", newline="") as fh:
        # Read metadata from comments and find where data starts
        lines = fh.readlines()
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            if line.startswith("#"):
                if "Nominal geometry:" in line:
                    # Parse: # Nominal geometry: w=1.100000, g=0.280000, t=0.280000
                    parts = line.split(":")[1].strip().split(",")
                    for part in parts:
                        key, value = part.strip().split("=")
                        metadata[key] = float(value)
                elif "Delta_w_star:" in line:
                    metadata["delta_w_star"] = float(line.split(":")[1].strip())
                elif "L_c:" in line:
                    metadata["L_c"] = float(line.split(":")[1].strip())
                elif "Lambda0:" in line:
                    metadata["lambda0"] = float(line.split(":")[1].strip())
                elif "Variable:" in line:
                    variable = line.split(":")[1].strip()
            else:
                # Found first non-comment line (should be header)
                data_start_idx = i
                break
        
        # Read CSV data starting from data_start_idx
        data_lines = "".join(lines[data_start_idx:])
        reader = csv.DictReader(StringIO(data_lines))
        rows = list(reader)
    
    if not rows:
        raise ValueError(f"No data found in {csv_path}")
    
    # Extract data
    perturbed_values = np.array([float(row["variable_value"]) for row in rows])
    results_dict = {
        "eta_te": np.array([float(row["eta_TE"]) for row in rows]),
        "eta_tm": np.array([float(row["eta_TM"]) for row in rows]),
        "V_te": np.array([float(row["V_TE"]) for row in rows]),
        "V_tm": np.array([float(row["V_TM"]) for row in rows]),
    }
    
    return perturbed_values, results_dict, variable


def load_existing_samples_single(csv_path: Path) -> Optional[np.ndarray]:
    """
    Load existing sample values from a single-variable CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Array of existing variable values, or None if file doesn't exist or is empty
    """
    if not csv_path.exists():
        return None
    
    try:
        with csv_path.open("r", newline="") as fh:
            lines = fh.readlines()
            data_start_idx = 0
            
            # Find where data starts (skip comment lines)
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    data_start_idx = i
                    break
            
            if data_start_idx == 0:
                return None
            
            # Read CSV data
            data_lines = "".join(lines[data_start_idx:])
            reader = csv.DictReader(StringIO(data_lines))
            rows = []
            for row in reader:
                # Filter out empty rows
                if row.get("variable_value", "").strip():
                    rows.append(row)
            
            if not rows:
                return None
            
            existing_values = np.array([float(row["variable_value"]) for row in rows])
            return existing_values
    except Exception as e:
        print(f"[Warning] Could not load existing samples from {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_existing_samples_joint(csv_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load existing sample triplets from a joint CSV file.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Tuple of (w_samples, g_samples, t_samples), or None if file doesn't exist or is empty
    """
    if not csv_path.exists():
        return None
    
    try:
        with csv_path.open("r", newline="") as fh:
            lines = fh.readlines()
            data_start_idx = 0
            
            # Find where data starts (skip comment lines)
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    data_start_idx = i
                    break
            
            if data_start_idx == 0:
                return None
            
            # Read CSV data
            data_lines = "".join(lines[data_start_idx:])
            reader = csv.DictReader(StringIO(data_lines))
            rows = []
            for row in reader:
                # Filter out empty rows (rows where all values are empty strings)
                if any(row.get(key, "").strip() for key in ["w_value", "g_value", "t_value"]):
                    rows.append(row)
            
            if not rows:
                return None
            
            w_samples = np.array([float(row["w_value"]) for row in rows])
            g_samples = np.array([float(row["g_value"]) for row in rows])
            t_samples = np.array([float(row["t_value"]) for row in rows])
            
            return (w_samples, g_samples, t_samples)
    except Exception as e:
        print(f"[Warning] Could not load existing samples from {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def filter_duplicates_single(new_samples: np.ndarray, existing_samples: np.ndarray, 
                             tolerance: float = 1e-6) -> np.ndarray:
    """
    Filter out samples that are too close to existing samples.
    
    Args:
        new_samples: Array of new sample values
        existing_samples: Array of existing sample values
        tolerance: Tolerance for considering samples as duplicates (in µm)
    
    Returns:
        Array of new samples that are not duplicates
    """
    if len(existing_samples) == 0:
        return new_samples
    
    # For each new sample, check if it's close to any existing sample
    keep_mask = np.ones(len(new_samples), dtype=bool)
    
    for i, new_val in enumerate(new_samples):
        if np.any(np.abs(existing_samples - new_val) < tolerance):
            keep_mask[i] = False
    
    filtered = new_samples[keep_mask]
    n_filtered = len(new_samples) - len(filtered)
    if n_filtered > 0:
        print(f"[Filter] Removed {n_filtered} duplicate samples (tolerance: {tolerance:.2e} µm)")
    
    return filtered


def filter_duplicates_joint(new_w: np.ndarray, new_g: np.ndarray, new_t: np.ndarray,
                            existing_w: np.ndarray, existing_g: np.ndarray, existing_t: np.ndarray,
                            tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out triplets that are too close to existing triplets.
    
    Args:
        new_w, new_g, new_t: Arrays of new sample values
        existing_w, existing_g, existing_t: Arrays of existing sample values
        tolerance: Tolerance for considering samples as duplicates (in µm)
    
    Returns:
        Tuple of filtered (w, g, t) arrays
    """
    if len(existing_w) == 0:
        return (new_w, new_g, new_t)
    
    # For each new triplet, check if it's close to any existing triplet
    keep_mask = np.ones(len(new_w), dtype=bool)
    
    for i in range(len(new_w)):
        # Check if this triplet is close to any existing triplet
        distances = np.sqrt(
            (existing_w - new_w[i])**2 + 
            (existing_g - new_g[i])**2 + 
            (existing_t - new_t[i])**2
        )
        if np.any(distances < tolerance):
            keep_mask[i] = False
    
    filtered_w = new_w[keep_mask]
    filtered_g = new_g[keep_mask]
    filtered_t = new_t[keep_mask]
    
    n_filtered = len(new_w) - len(filtered_w)
    if n_filtered > 0:
        print(f"[Filter] Removed {n_filtered} duplicate triplets (tolerance: {tolerance:.2e} µm)")
    
    return (filtered_w, filtered_g, filtered_t)


def generate_tolerance_filename(variable: str, w_nom: float, g_nom: float, t_nom: float,
                                sigma_nm: float = None, sigma_w_nm: float = None,
                                sigma_g_nm: float = None, sigma_t_nm: float = None,
                                extension: str = "csv") -> str:
    """
    Generate a unique filename for tolerance results based on run parameters.
    
    Args:
        variable: Which variable was perturbed ('t', 'w', 'g', or 'joint')
        w_nom: Nominal width
        g_nom: Nominal gap
        t_nom: Nominal thickness
        sigma_nm: Standard deviation in nm (for single variable mode)
        sigma_w_nm: Standard deviation for width in nm (for joint mode)
        sigma_g_nm: Standard deviation for gap in nm (for joint mode)
        sigma_t_nm: Standard deviation for thickness in nm (for joint mode)
        extension: File extension ('csv' or 'png')
    
    Returns:
        Filename string
    """
    # Format geometry values: replace decimal point with 'p' and format to 3 decimal places
    w_str = f"{w_nom:.3f}".replace('.', 'p')
    g_str = f"{g_nom:.3f}".replace('.', 'p')
    t_str = f"{t_nom:.3f}".replace('.', 'p')
    
    if variable == "joint":
        sigma_w_str = f"{sigma_w_nm:.1f}".replace('.', 'p')
        sigma_g_str = f"{sigma_g_nm:.1f}".replace('.', 'p')
        sigma_t_str = f"{sigma_t_nm:.1f}".replace('.', 'p')
        filename = f"tolerance_joint_w{w_str}_g{g_str}_t{t_str}_sw{sigma_w_str}_sg{sigma_g_str}_st{sigma_t_str}.{extension}"
    else:
        sigma_str = f"{sigma_nm:.1f}".replace('.', 'p')
        filename = f"tolerance_{variable}_w{w_str}_g{g_str}_t{t_str}_sigma{sigma_str}.{extension}"
    
    return filename


def plot_tolerance_results(perturbed_values: np.ndarray, results_list: list, variable: str,
                          w_nom: float, g_nom: float, t_nom: float, output_dir: Path,
                          filename_base: Optional[str] = None):
    """
    Generate plots showing tolerance behavior.

    This condensed version uses two subplots:
      - Left:  η vs parameter, TE and TM overlaid with different colors/markers
      - Right: V vs parameter, TE and TM overlaid with different colors/markers
    """
    # Handle both formats: list of dicts (from simulations) or dict of arrays (from CSV)
    if isinstance(results_list, dict):
        # Data from CSV
        valid_values = perturbed_values
        eta_te = results_list["eta_te"]
        eta_tm = results_list["eta_tm"]
        V_te = results_list["V_te"]
        V_tm = results_list["V_tm"]
    else:
        # Data from simulations (list of dicts)
        valid_indices = [i for i, r in enumerate(results_list) if r is not None]
        valid_values = perturbed_values[valid_indices]

        eta_te = np.array([results_list[i]["eta_te"] for i in valid_indices])
        eta_tm = np.array([results_list[i]["eta_tm"] for i in valid_indices])
        V_te = np.array([results_list[i]["V_te"] for i in valid_indices])
        V_tm = np.array([results_list[i]["V_tm"] for i in valid_indices])

    # Variable labels
    var_labels = {"t": "Thickness t (µm)", "w": "Width w (µm)", "g": "Gap g (µm)"}
    var_label = var_labels.get(variable, variable)

    # Nominal (unperturbed) value for this variable
    if variable == 'w':
        nominal_value = w_nom
    elif variable == 'g':
        nominal_value = g_nom
    elif variable == 't':
        nominal_value = t_nom
    else:
        nominal_value = None

    # Create 2-panel plot (η on the left, V on the right)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Tolerance Analysis: {variable.upper()} perturbation\n"
        f"Nominal: w={w_nom:.3f} µm, g={g_nom:.3f} µm, t={t_nom:.3f} µm",
        fontsize=14,
    )

    # Ensure axes is a 1D array for consistent indexing
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # --- Left subplot: η vs parameter (TE and TM) ---
    ax_eta = axes[0]
    ax_eta.scatter(valid_values, eta_te, alpha=0.6, s=30, label="TE")
    ax_eta.scatter(valid_values, eta_tm, alpha=0.6, s=30, label="TM")
    ax_eta.axhline(0.5, linestyle="--", label="Target (0.5)")
    # Add vertical line for nominal value if available
    if nominal_value is not None:
        ax_eta.axvline(nominal_value, linestyle=':', alpha=0.7 , color='r')
    ax_eta.set_xlabel(var_label)
    ax_eta.set_ylabel("η (Coupling Ratio)")
    ax_eta.set_title("η (TE and TM)")
    ax_eta.grid(True, alpha=0.3)
    ax_eta.legend()

    # --- Right subplot: V vs parameter (TE and TM) ---
    ax_V = axes[1]
    ax_V.scatter(valid_values, V_te, alpha=0.6, s=30, label="TE")
    ax_V.scatter(valid_values, V_tm, alpha=0.6, s=30, label="TM")
    ax_V.axhline(1.0, linestyle="--", label="Target (1.0)")
    # Add vertical line for nominal value if available
    if nominal_value is not None:
        ax_V.axvline(nominal_value, linestyle=':', alpha=0.7, color='r')
    ax_V.set_xlabel(var_label)
    ax_V.set_ylabel("V (Visibility)")
    ax_V.set_title("V (TE and TM)")
    ax_V.grid(True, alpha=0.3)
    ax_V.legend()

    plt.tight_layout()

    # Save plot
    if filename_base is None:
        # This shouldn't happen in normal flow, but provide fallback
        plot_path = output_dir / f"tolerance_{variable}.png"
    else:
        plot_path = output_dir / f"{filename_base}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Plot] Saved tolerance plot: {plot_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo tolerance analysis for directional coupler geometry parameters"
    )
    parser.add_argument(
        "--variable",
        type=str,
        choices=["t", "w", "g", "all", "joint"],
        default="joint",
        help="Which parameter to perturb: 't' (thickness), 'w' (width), 'g' (gap), 'all' (run all three separately), or 'joint' (perturb all three simultaneously)"
    )
    parser.add_argument(
        "--w",
        type=float,
        default=1.08,
        help="Nominal waveguide width in µm"
    )
    parser.add_argument(
        "--g",
        type=float,
        default=0.255,
        help="Nominal coupling gap in µm"
    )
    parser.add_argument(
        "--t",
        type=float,
        default=0.275,
        help="Nominal waveguide thickness in µm"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Number of Monte Carlo samples (default: 20)"
    )
    parser.add_argument(
        "--sigma-w-nm",
        type=float,
        default=7,
        help="Standard deviation for width perturbation in nm (default: 15.0)"
    )
    parser.add_argument(
        "--sigma-g-nm",
        type=float,
        default=7,
        help="Standard deviation for gap perturbation in nm (default: 15.0)"
    )
    parser.add_argument(
        "--sigma-t-nm",
        type=float,
        default=3.5,
        help="Standard deviation for thickness perturbation in nm (default: 7.5)"
    )
    parser.add_argument(
        "--lambda0",
        type=float,
        default=1.55,
        help="Design wavelength in µm (default: 1.55)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tolerance",
        help="Output directory for plots and CSV (default: results/tolerance)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)"
    )
    parser.add_argument(
        "--plot-only",
        type=bool,
        default=True,
        help="Only plot from existing CSV file, skip simulations"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="results/tolerance/tolerance_joint_w1p080_g0p255_t0p275_sw7p0_sg7p0_st3p5.csv",
        help="Path to CSV file for plot-only mode (optional)"
    )
    
    parser.add_argument(
        "--delta-w",
        type=float,
        default=-0.323581,
        help="Fixed delta_w value in µm (if provided, skips solving for delta_w* and uses this value directly)"
    )
    
    return parser.parse_args()


def run_tolerance_for_variable(variable: str, args: argparse.Namespace, 
                                base_param, w_nom: float, g_nom: float, t_nom: float,
                                delta_w_star_nom: float, L_c_nom: float,
                                size_x_nom: float, size_z_nom: float,
                                output_dir: Path) -> None:
    """
    Run tolerance analysis for a single variable.
    
    Args:
        variable: Which variable to perturb ('t', 'w', or 'g')
        args: Parsed command-line arguments
        base_param: Base parameter namespace
        w_nom: Nominal width
        g_nom: Nominal gap
        t_nom: Nominal thickness
        delta_w_star_nom: Nominal delta_w_star
        L_c_nom: Nominal L_c
        size_x_nom: Nominal size_x
        size_z_nom: Nominal size_z
        output_dir: Output directory for results
    """
    print(f"\n{'='*60}")
    print(f"Running tolerance analysis for variable: {variable.upper()}")
    print(f"{'='*60}\n")
    
    # Determine which parameter to perturb and select appropriate sigma
    if variable == "w":
        base_value = w_nom
        sigma_nm = args.sigma_w_nm
    elif variable == "g":
        base_value = g_nom
        sigma_nm = args.sigma_g_nm
    elif variable == "t":
        base_value = t_nom
        sigma_nm = args.sigma_t_nm
    else:
        raise ValueError(f"Unknown variable: {variable}")
    
    # Generate unique filename based on run parameters
    filename_base = generate_tolerance_filename(
        variable, w_nom, g_nom, t_nom, sigma_nm, extension=""
    ).rstrip(".")
    
    csv_path = output_dir / f"{filename_base}.csv"
    
    # Check for existing samples
    existing_samples = load_existing_samples_single(csv_path)
    file_exists = csv_path.exists() and existing_samples is not None
    
    if file_exists:
        print(f"[Resume] Found existing CSV with {len(existing_samples)} samples")
        print(f"  Existing range: [{existing_samples.min():.4f}, {existing_samples.max():.4f}] µm")
    
    # Generate Monte Carlo samples
    perturbed_values_all = perturb_parameter(
        base_value, sigma_nm, args.n_samples, args.seed
    )
    
    # Filter out duplicates if file exists
    if file_exists:
        perturbed_values = filter_duplicates_single(perturbed_values_all, existing_samples)
        if len(perturbed_values) == 0:
            print(f"[Info] All {len(perturbed_values_all)} new samples are duplicates. Nothing to simulate.")
            return
        print(f"[Filter] After filtering: {len(perturbed_values)} new samples to simulate")
    else:
        perturbed_values = perturbed_values_all
    
    std_dev_um = sigma_nm * 1e-3
    print(f"[Monte Carlo] Generated {len(perturbed_values)} samples")
    print(f"  Base value: {base_value:.4f} µm")
    print(f"  Std dev: {sigma_nm:.1f} nm ({std_dev_um:.4f} µm)")
    print(f"  Range: [{perturbed_values.min():.4f}, {perturbed_values.max():.4f}] µm")
    
    # Open CSV file for incremental writing (append if exists, create new if not)
    if file_exists:
        # Ensure file ends with newline before appending
        with csv_path.open("r", newline="") as check_file:
            content = check_file.read()
            if content and not content.endswith("\n"):
                # File doesn't end with newline, add one
                with csv_path.open("a", newline="") as fix_file:
                    fix_file.write("\n")
        
        csv_file = csv_path.open("a", newline="", buffering=1)  # Append mode with line buffering
        print(f"[CSV] Appending to existing file: {csv_path}")
    else:
        csv_file = csv_path.open("w", newline="", buffering=1)  # Write mode with line buffering
        fieldnames = ["variable_value", "eta_TE", "eta_TM", "V_TE", "V_TM"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write metadata as comments first
        csv_file.write(f"# Nominal geometry: w={w_nom:.6f}, g={g_nom:.6f}, t={t_nom:.6f}\n")
        csv_file.write(f"# Delta_w_star: {delta_w_star_nom:.6f}\n")
        if args.delta_w is not None:
            csv_file.write(f"# Delta_w_manual: True\n")
        csv_file.write(f"# L_c: {L_c_nom:.6f}\n")
        csv_file.write(f"# Lambda0: {args.lambda0:.6f}\n")
        csv_file.write(f"# Variable: {variable}\n")
        csv_file.write(f"# Sigma_nm: {sigma_nm:.1f}\n")
        csv_file.flush()
        
        # Write CSV header
        writer.writeheader()
        csv_file.flush()
        print(f"[CSV] Created new file for incremental writing: {csv_path}")
    
    fieldnames = ["variable_value", "eta_TE", "eta_TM", "V_TE", "V_TM"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Run simulations
    results_list = []
    
    n_total = len(perturbed_values)
    progress_bar = tqdm(enumerate(perturbed_values), total=n_total, 
                        desc=f"Monte Carlo samples ({variable})", unit="sample")
    
    for i, perturbed_val in progress_bar:
        # Update progress bar description with current sample number
        progress_bar.set_description(f"Sample {i+1}/{n_total} ({variable})")
        
        # Create perturbed parameter
        param = create_perturbed_param(
            base_param, variable, perturbed_val,
            w_nom, g_nom, t_nom, delta_w_star_nom, L_c_nom
        )
        
        # Ensure size_x is frozen (depends on L_c which is frozen)
        # size_y is updated in create_perturbed_param based on perturbed geometry
        param.size_x = size_x_nom
        param.size_z = size_z_nom
        
        # Run simulation
        task_tag = f"tolerance_{variable}_{i:03d}"
        result = run_tolerance_simulation(param, args.lambda0, task_tag)
        
        if result is not None:
            results_list.append(result)
            # Write row immediately to CSV
            row = {
                "variable_value": perturbed_val,
                "eta_TE": result["eta_te"],
                "eta_TM": result["eta_tm"],
                "V_TE": result["V_te"],
                "V_TM": result["V_tm"],
            }
            writer.writerow(row)
            csv_file.flush()  # Ensure data is written to buffer
            os.fsync(csv_file.fileno())  # Force OS to write to disk immediately
            print(f"[CSV] Wrote sample {i+1}/{n_total} to CSV ({variable}={perturbed_val:.6f})")
        else:
            results_list.append(None)
            progress_bar.write(f"[Warning] Simulation {i+1}/{n_total} failed for {variable}={perturbed_val:.4f} µm")
    
    progress_bar.close()
    
    # Close CSV file
    csv_file.close()
    print(f"[CSV] Saved results to {csv_path}")
    
    # Compute tolerance statistics
    if results_list and any(r is not None for r in results_list):
        valid_results = [r for r in results_list if r is not None]
        eta_te_arr = np.array([r["eta_te"] for r in valid_results])
        eta_tm_arr = np.array([r["eta_tm"] for r in valid_results])
        V_te_arr = np.array([r["V_te"] for r in valid_results])
        V_tm_arr = np.array([r["V_tm"] for r in valid_results])
        delta_eta_arr = np.abs(eta_te_arr - eta_tm_arr)
        
        print(f"\n[Tolerance Statistics - {variable.upper()}]")
        print(f"  η_TE: mean={eta_te_arr.mean():.4f}, std={eta_te_arr.std():.4f}, "
              f"95%% range=[{np.percentile(eta_te_arr, 2.5):.4f}, {np.percentile(eta_te_arr, 97.5):.4f}]")
        print(f"  η_TM: mean={eta_tm_arr.mean():.4f}, std={eta_tm_arr.std():.4f}, "
              f"95%% range=[{np.percentile(eta_tm_arr, 2.5):.4f}, {np.percentile(eta_tm_arr, 97.5):.4f}]")
        print(f"  V_TE: mean={V_te_arr.mean():.4f}, std={V_te_arr.std():.4f}, "
              f"95%% range=[{np.percentile(V_te_arr, 2.5):.4f}, {np.percentile(V_te_arr, 97.5):.4f}]")
        print(f"  V_TM: mean={V_tm_arr.mean():.4f}, std={V_tm_arr.std():.4f}, "
              f"95%% range=[{np.percentile(V_tm_arr, 2.5):.4f}, {np.percentile(V_tm_arr, 97.5):.4f}]")
        print(f"  Δη: mean={delta_eta_arr.mean():.4f}, std={delta_eta_arr.std():.4f}, "
              f"95%% range=[{np.percentile(delta_eta_arr, 2.5):.4f}, {np.percentile(delta_eta_arr, 97.5):.4f}]")
    
    # Generate plots
    if results_list and any(r is not None for r in results_list):
        plot_tolerance_results(
            perturbed_values, results_list, variable,
            w_nom, g_nom, t_nom, output_dir, filename_base=filename_base
        )
    else:
        print(f"[Warning] No successful simulations to plot for {variable}")


def run_joint_tolerance(args: argparse.Namespace, 
                        base_param, w_nom: float, g_nom: float, t_nom: float,
                        delta_w_star_nom: float, L_c_nom: float,
                        size_x_nom: float, size_z_nom: float,
                        output_dir: Path) -> None:
    """
    Run joint tolerance analysis where all three parameters (t, w, g) are perturbed simultaneously.
    
    Args:
        args: Parsed command-line arguments
        base_param: Base parameter namespace
        w_nom: Nominal width
        g_nom: Nominal gap
        t_nom: Nominal thickness
        delta_w_star_nom: Nominal delta_w_star
        L_c_nom: Nominal L_c
        size_x_nom: Nominal size_x
        size_z_nom: Nominal size_z
        output_dir: Output directory for results
    """
    print(f"\n{'='*60}")
    print(f"Running JOINT tolerance analysis (all parameters perturbed simultaneously)")
    print(f"{'='*60}\n")
    
    # Generate unique filename based on run parameters
    filename_base = generate_tolerance_filename(
        "joint", w_nom, g_nom, t_nom,
        sigma_w_nm=args.sigma_w_nm, sigma_g_nm=args.sigma_g_nm, sigma_t_nm=args.sigma_t_nm,
        extension=""
    ).rstrip(".")
    
    csv_path = output_dir / f"{filename_base}.csv"
    
    # Check for existing samples
    print(f"[Debug] Checking for existing file: {csv_path}")
    print(f"[Debug] File exists: {csv_path.exists()}")
    existing_data = load_existing_samples_joint(csv_path)
    file_exists = csv_path.exists() and existing_data is not None
    print(f"[Debug] Existing data loaded: {existing_data is not None}")
    
    if file_exists:
        existing_w, existing_g, existing_t = existing_data
        print(f"[Resume] Found existing CSV with {len(existing_w)} triplets")
        print(f"  Existing ranges:")
        print(f"    Width: [{existing_w.min():.4f}, {existing_w.max():.4f}] µm")
        print(f"    Gap: [{existing_g.min():.4f}, {existing_g.max():.4f}] µm")
        print(f"    Thickness: [{existing_t.min():.4f}, {existing_t.max():.4f}] µm")
        print(f"  Existing samples:")
        for i, (w, g, t) in enumerate(zip(existing_w, existing_g, existing_t)):
            print(f"    Sample {i+1}: w={w:.6f}, g={g:.6f}, t={t:.6f}")
    
    # Generate Monte Carlo triplets
    print(f"[Monte Carlo] Generating {args.n_samples} new triplets...")
    w_samples_all, g_samples_all, t_samples_all = perturb_joint_triplets(
        w_nom, g_nom, t_nom,
        args.sigma_w_nm, args.sigma_g_nm, args.sigma_t_nm,
        args.n_samples, args.seed
    )
    print(f"[Monte Carlo] Generated {len(w_samples_all)} triplets")
    
    # Filter out duplicates if file exists
    if file_exists:
        print(f"[Filter] Checking for duplicates against {len(existing_w)} existing samples...")
        w_samples, g_samples, t_samples = filter_duplicates_joint(
            w_samples_all, g_samples_all, t_samples_all,
            existing_w, existing_g, existing_t
        )
        if len(w_samples) == 0:
            print(f"[Info] All {len(w_samples_all)} new triplets are duplicates. Nothing to simulate.")
            return
        print(f"[Filter] After filtering: {len(w_samples)} unique triplets to simulate (removed {len(w_samples_all) - len(w_samples)} duplicates)")
    else:
        w_samples = w_samples_all
        g_samples = g_samples_all
        t_samples = t_samples_all
        print(f"[Filter] No existing file found, all {len(w_samples)} triplets are new")
    
    std_w_um = args.sigma_w_nm * 1e-3
    std_g_um = args.sigma_g_nm * 1e-3
    std_t_um = args.sigma_t_nm * 1e-3
    
    print(f"[Monte Carlo] Generated {len(w_samples)} joint triplets")
    print(f"  Width: base={w_nom:.4f} µm, std={args.sigma_w_nm:.1f} nm ({std_w_um:.4f} µm), "
          f"range=[{w_samples.min():.4f}, {w_samples.max():.4f}] µm")
    print(f"  Gap: base={g_nom:.4f} µm, std={args.sigma_g_nm:.1f} nm ({std_g_um:.4f} µm), "
          f"range=[{g_samples.min():.4f}, {g_samples.max():.4f}] µm")
    print(f"  Thickness: base={t_nom:.4f} µm, std={args.sigma_t_nm:.1f} nm ({std_t_um:.4f} µm), "
          f"range=[{t_samples.min():.4f}, {t_samples.max():.4f}] µm")
    
    # Open CSV file for incremental writing (append if exists, create new if not)
    if file_exists:
        # Ensure file ends with newline before appending
        try:
            with csv_path.open("rb") as check_file:
                check_file.seek(-1, 2)  # Go to last byte
                last_char = check_file.read(1)
                if last_char != b"\n":
                    # File doesn't end with newline, add one
                    with csv_path.open("a", newline="") as fix_file:
                        fix_file.write("\n")
        except (OSError, IOError):
            # File might be empty or have issues, try text mode
            with csv_path.open("r", newline="") as check_file:
                content = check_file.read()
                if content and not content.endswith("\n"):
                    with csv_path.open("a", newline="") as fix_file:
                        fix_file.write("\n")
        
        csv_file = csv_path.open("a", newline="", buffering=1)  # Append mode with line buffering
        print(f"[CSV] Appending to existing file: {csv_path}")
        print(f"[Debug] File position after opening: {csv_file.tell()}")
    else:
        csv_file = csv_path.open("w", newline="", buffering=1)  # Write mode with line buffering
        fieldnames = ["w_value", "g_value", "t_value", "eta_TE", "eta_TM", "V_TE", "V_TM"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write metadata as comments first
        csv_file.write(f"# Nominal geometry: w={w_nom:.6f}, g={g_nom:.6f}, t={t_nom:.6f}\n")
        csv_file.write(f"# Delta_w_star: {delta_w_star_nom:.6f}\n")
        if args.delta_w is not None:
            csv_file.write(f"# Delta_w_manual: True\n")
        csv_file.write(f"# L_c: {L_c_nom:.6f}\n")
        csv_file.write(f"# Lambda0: {args.lambda0:.6f}\n")
        csv_file.write(f"# Variable: joint\n")
        csv_file.write(f"# Sigma_w_nm: {args.sigma_w_nm:.1f}\n")
        csv_file.write(f"# Sigma_g_nm: {args.sigma_g_nm:.1f}\n")
        csv_file.write(f"# Sigma_t_nm: {args.sigma_t_nm:.1f}\n")
        csv_file.flush()
        
        # Write CSV header
        writer.writeheader()
        csv_file.flush()
        print(f"[CSV] Created new file for incremental writing: {csv_path}")
    
    fieldnames = ["w_value", "g_value", "t_value", "eta_TE", "eta_TM", "V_TE", "V_TM"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Run simulations
    results_list = []
    
    n_total = len(w_samples)
    progress_bar = tqdm(zip(w_samples, g_samples, t_samples), total=n_total,
                        desc="Monte Carlo samples (joint)", unit="sample")
    
    for i, (w_pert, g_pert, t_pert) in enumerate(progress_bar):
        # Update progress bar description
        progress_bar.set_description(f"Sample {i+1}/{n_total} (joint)")
        
        # Create perturbed parameter with all three values
        param = create_perturbed_param_joint(
            base_param, w_pert, g_pert, t_pert,
            w_nom, g_nom, t_nom, delta_w_star_nom, L_c_nom
        )
        
        # Ensure size_x is frozen (depends on L_c which is frozen)
        param.size_x = size_x_nom
        param.size_z = size_z_nom
        
        # Run simulation
        task_tag = f"tolerance_joint_{i:03d}"
        result = run_tolerance_simulation(param, args.lambda0, task_tag)
        
        if result is not None:
            results_list.append(result)
            # Write row immediately to CSV
            row = {
                "w_value": w_pert,
                "g_value": g_pert,
                "t_value": t_pert,
                "eta_TE": result["eta_te"],
                "eta_TM": result["eta_tm"],
                "V_TE": result["V_te"],
                "V_TM": result["V_tm"],
            }
            # Get file size before write
            file_size_before = csv_path.stat().st_size if csv_path.exists() else 0
            
            # Write row
            writer.writerow(row)
            csv_file.flush()  # Ensure data is written to buffer
            os.fsync(csv_file.fileno())  # Force OS to write to disk immediately
            
            # Verify write succeeded
            file_size_after = csv_path.stat().st_size if csv_path.exists() else 0
            if file_size_after > file_size_before:
                print(f"[CSV] ✓ Wrote sample {i+1}/{n_total} to CSV (w={w_pert:.6f}, g={g_pert:.6f}, t={t_pert:.6f}) - file size: {file_size_before} → {file_size_after} bytes")
            else:
                print(f"[CSV] ✗ WARNING: Write may have failed! File size unchanged: {file_size_before} bytes")
                # Try writing directly as fallback
                csv_line = f"{w_pert},{g_pert},{t_pert},{result['eta_te']},{result['eta_tm']},{result['V_te']},{result['V_tm']}\n"
                csv_file.write(csv_line)
                csv_file.flush()
                os.fsync(csv_file.fileno())
                print(f"[CSV] ✓ Fallback write completed")
        else:
            results_list.append(None)
            progress_bar.write(f"[Warning] Simulation {i+1}/{n_total} failed for w={w_pert:.4f}, g={g_pert:.4f}, t={t_pert:.4f} µm")
    
    progress_bar.close()
    
    # Close CSV file
    csv_file.close()
    print(f"[CSV] Saved results to {csv_path}")
    
    # Compute tolerance statistics
    if results_list and any(r is not None for r in results_list):
        valid_results = [r for r in results_list if r is not None]
        eta_te_arr = np.array([r["eta_te"] for r in valid_results])
        eta_tm_arr = np.array([r["eta_tm"] for r in valid_results])
        V_te_arr = np.array([r["V_te"] for r in valid_results])
        V_tm_arr = np.array([r["V_tm"] for r in valid_results])
        delta_eta_arr = np.abs(eta_te_arr - eta_tm_arr)
        
        print(f"\n[Tolerance Statistics - JOINT]")
        print(f"  η_TE: mean={eta_te_arr.mean():.4f}, std={eta_te_arr.std():.4f}, "
              f"95%% range=[{np.percentile(eta_te_arr, 2.5):.4f}, {np.percentile(eta_te_arr, 97.5):.4f}]")
        print(f"  η_TM: mean={eta_tm_arr.mean():.4f}, std={eta_tm_arr.std():.4f}, "
              f"95%% range=[{np.percentile(eta_tm_arr, 2.5):.4f}, {np.percentile(eta_tm_arr, 97.5):.4f}]")
        print(f"  V_TE: mean={V_te_arr.mean():.4f}, std={V_te_arr.std():.4f}, "
              f"95%% range=[{np.percentile(V_te_arr, 2.5):.4f}, {np.percentile(V_te_arr, 97.5):.4f}]")
        print(f"  V_TM: mean={V_tm_arr.mean():.4f}, std={V_tm_arr.std():.4f}, "
              f"95%% range=[{np.percentile(V_tm_arr, 2.5):.4f}, {np.percentile(V_tm_arr, 97.5):.4f}]")
        print(f"  Δη: mean={delta_eta_arr.mean():.4f}, std={delta_eta_arr.std():.4f}, "
              f"95%% range=[{np.percentile(delta_eta_arr, 2.5):.4f}, {np.percentile(delta_eta_arr, 97.5):.4f}]")
    
    # Generate plots (for joint mode, we'll create scatter plots showing results vs each parameter)
    if results_list and any(r is not None for r in results_list):
        plot_joint_tolerance_results(
            w_samples, g_samples, t_samples, results_list,
            w_nom, g_nom, t_nom, output_dir, filename_base=filename_base
        )
    else:
        print(f"[Warning] No successful simulations to plot for joint analysis")


def plot_joint_tolerance_results(w_samples: np.ndarray, g_samples: np.ndarray, t_samples: np.ndarray,
                                 results_list: list, w_nom: float, g_nom: float, t_nom: float,
                                 output_dir: Path, filename_base: str) -> None:
    """
    Generate plots for joint tolerance analysis showing results vs each parameter.
    
    Args:
        w_samples: Array of perturbed width values
        g_samples: Array of perturbed gap values
        t_samples: Array of perturbed thickness values
        results_list: List of result dictionaries (may contain None for failed runs)
        w_nom: Nominal width (for title)
        g_nom: Nominal gap (for title)
        t_nom: Nominal thickness (for title)
        output_dir: Output directory for plots
        filename_base: Base filename (without extension)
    """
    # Filter valid results
    valid_indices = [i for i, r in enumerate(results_list) if r is not None]
    w_valid = w_samples[valid_indices]
    g_valid = g_samples[valid_indices]
    t_valid = t_samples[valid_indices]
    
    eta_te = np.array([results_list[i]["eta_te"] for i in valid_indices])
    eta_tm = np.array([results_list[i]["eta_tm"] for i in valid_indices])
    V_te = np.array([results_list[i]["V_te"] for i in valid_indices])
    V_tm = np.array([results_list[i]["V_tm"] for i in valid_indices])
    
    # Create 4-panel plot (one for each output metric)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Joint Tolerance Analysis: All Parameters Perturbed Simultaneously\n"
                 f"Nominal: w={w_nom:.3f} µm, g={g_nom:.3f} µm, t={t_nom:.3f} µm",
                 fontsize=14)
    
    # Plot eta_TE vs each parameter
    ax = axes[0, 0]
    ax.scatter(w_valid, eta_te, alpha=0.5, s=20, label='w', c='blue')
    ax.scatter(g_valid, eta_te, alpha=0.5, s=20, label='g', c='green')
    ax.scatter(t_valid, eta_te, alpha=0.5, s=20, label='t', c='red')
    ax.axhline(0.5, color='k', linestyle='--', label='Target (0.5)')
    ax.set_xlabel("Parameter Value (µm)")
    ax.set_ylabel("η (Coupling Ratio)")
    ax.set_title("η_TE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot eta_TM vs each parameter
    ax = axes[0, 1]
    ax.scatter(w_valid, eta_tm, alpha=0.5, s=20, label='w', c='blue')
    ax.scatter(g_valid, eta_tm, alpha=0.5, s=20, label='g', c='green')
    ax.scatter(t_valid, eta_tm, alpha=0.5, s=20, label='t', c='red')
    ax.axhline(0.5, color='k', linestyle='--', label='Target (0.5)')
    ax.set_xlabel("Parameter Value (µm)")
    ax.set_ylabel("η (Coupling Ratio)")
    ax.set_title("η_TM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot V_TE vs each parameter
    ax = axes[1, 0]
    ax.scatter(w_valid, V_te, alpha=0.5, s=20, label='w', c='blue')
    ax.scatter(g_valid, V_te, alpha=0.5, s=20, label='g', c='green')
    ax.scatter(t_valid, V_te, alpha=0.5, s=20, label='t', c='red')
    ax.axhline(1.0, color='k', linestyle='--', label='Target (1.0)')
    ax.set_xlabel("Parameter Value (µm)")
    ax.set_ylabel("V (Visibility)")
    ax.set_title("V_TE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot V_TM vs each parameter
    ax = axes[1, 1]
    ax.scatter(w_valid, V_tm, alpha=0.5, s=20, label='w', c='blue')
    ax.scatter(g_valid, V_tm, alpha=0.5, s=20, label='g', c='green')
    ax.scatter(t_valid, V_tm, alpha=0.5, s=20, label='t', c='red')
    ax.axhline(1.0, color='k', linestyle='--', label='Target (1.0)')
    ax.set_xlabel("Parameter Value (µm)")
    ax.set_ylabel("V (Visibility)")
    ax.set_title("V_TM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"{filename_base}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Plot] Saved joint tolerance plot: {plot_path}")


def main() -> None:
    """Main execution function."""
    args = parse_args()
    
    # Create output directory relative to this script's directory
    script_dir = Path(__file__).resolve().parent
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle plot-only mode
    if args.plot_only:
        # Handle joint mode separately
        if args.variable == "joint":
            # Search for joint CSV file
            if args.csv_file:
                csv_path = Path(args.csv_file)
                # Resolve relative paths relative to script directory
                if not csv_path.is_absolute():
                    csv_path = (script_dir / csv_path).resolve()
            else:
                pattern = "tolerance_joint_*.csv"
                matching_files = list(output_dir.glob(pattern))
                if not matching_files:
                    raise FileNotFoundError(
                        f"No joint CSV file found matching pattern '{pattern}' in {output_dir}. "
                        f"Run simulations first or specify --csv-file."
                    )
                if len(matching_files) > 1:
                    print(f"[Warning] Multiple joint CSV files found. Using: {matching_files[0]}")
                csv_path = matching_files[0]
            
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            print(f"[Plot Only] Loading joint data from {csv_path}")
            # For joint mode, we need to load all three parameter columns
            w_samples = []
            g_samples = []
            t_samples = []
            results_dict = {"eta_te": [], "eta_tm": [], "V_te": [], "V_tm": []}
            
            with csv_path.open("r", newline="") as fh:
                # Filter out comment lines and empty lines
                filtered_lines = []
                for line in fh:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        filtered_lines.append(line)
                # Create a StringIO object from filtered lines for DictReader
                csv_content = StringIO("".join(filtered_lines))
                reader = csv.DictReader(csv_content)
                
                for row in reader:
                    w_val = row.get("w_value", "").strip()
                    g_val = row.get("g_value", "").strip()
                    t_val = row.get("t_value", "").strip()
                    if w_val and g_val and t_val:
                        try:
                            w_samples.append(float(w_val))
                            g_samples.append(float(g_val))
                            t_samples.append(float(t_val))
                            results_dict["eta_te"].append(float(row.get("eta_TE", "").strip()))
                            results_dict["eta_tm"].append(float(row.get("eta_TM", "").strip()))
                            results_dict["V_te"].append(float(row.get("V_TE", "").strip()))
                            results_dict["V_tm"].append(float(row.get("V_TM", "").strip()))
                        except (ValueError, KeyError) as e:
                            print(f"[Warning] Skipping invalid row: {row}, error: {e}")
                            continue
            
            if len(w_samples) == 0:
                raise ValueError(f"No valid data rows found in CSV file: {csv_path}")
            
            w_samples = np.array(w_samples)
            g_samples = np.array(g_samples)
            t_samples = np.array(t_samples)
            results_dict = {k: np.array(v) for k, v in results_dict.items()}
            
            # Extract metadata
            metadata = {}
            with csv_path.open("r", newline="") as fh:
                for line in fh:
                    if line.startswith("# Nominal geometry:"):
                        parts = line.split(":")[1].strip().split(",")
                        for part in parts:
                            key, value = part.strip().split("=")
                            metadata[key] = float(value)
            
            w_nom = metadata.get("w", args.w if hasattr(args, 'w') else 0.0)
            g_nom = metadata.get("g", args.g if hasattr(args, 'g') else 0.0)
            t_nom = metadata.get("t", args.t if hasattr(args, 't') else 0.0)
            
            filename_base = csv_path.stem
            plot_joint_tolerance_results(
                w_samples, g_samples, t_samples, 
                [{"eta_te": eta, "eta_tm": etm, "V_te": vte, "V_tm": vtm} 
                 for eta, etm, vte, vtm in zip(results_dict["eta_te"], results_dict["eta_tm"], 
                                               results_dict["V_te"], results_dict["V_tm"])],
                w_nom, g_nom, t_nom, output_dir, filename_base=filename_base
            )
            
            # Print statistics
            eta_te_arr = results_dict["eta_te"]
            eta_tm_arr = results_dict["eta_tm"]
            V_te_arr = results_dict["V_te"]
            V_tm_arr = results_dict["V_tm"]
            
            if len(eta_te_arr) == 0:
                raise ValueError("No data loaded from CSV file. Cannot compute statistics.")
            
            delta_eta_arr = np.abs(eta_te_arr - eta_tm_arr)
            
            print(f"\n[Tolerance Statistics - JOINT]")
            print(f"  Samples: {len(eta_te_arr)}")
            print(f"  η_TE: mean={eta_te_arr.mean():.4f}, std={eta_te_arr.std():.4f}, "
                  f"95%% range=[{np.percentile(eta_te_arr, 2.5):.4f}, {np.percentile(eta_te_arr, 97.5):.4f}]")
            print(f"  η_TM: mean={eta_tm_arr.mean():.4f}, std={eta_tm_arr.std():.4f}, "
                  f"95%% range=[{np.percentile(eta_tm_arr, 2.5):.4f}, {np.percentile(eta_tm_arr, 97.5):.4f}]")
            print(f"  V_TE: mean={V_te_arr.mean():.4f}, std={V_te_arr.std():.4f}, "
                  f"95%% range=[{np.percentile(V_te_arr, 2.5):.4f}, {np.percentile(V_te_arr, 97.5):.4f}]")
            print(f"  V_TM: mean={V_tm_arr.mean():.4f}, std={V_tm_arr.std():.4f}, "
                  f"95%% range=[{np.percentile(V_tm_arr, 2.5):.4f}, {np.percentile(V_tm_arr, 97.5):.4f}]")
            print(f"  Δη: mean={delta_eta_arr.mean():.4f}, std={delta_eta_arr.std():.4f}, "
                  f"95%% range=[{np.percentile(delta_eta_arr, 2.5):.4f}, {np.percentile(delta_eta_arr, 97.5):.4f}]")
            
            return
        
        # Determine which variables to process
        if args.variable == "all":
            variables_to_process = ["t", "w", "g"]
        else:
            variables_to_process = [args.variable]
        
        for variable in variables_to_process:
            # Determine CSV file path
            if args.csv_file and len(variables_to_process) == 1:
                # Only use --csv-file if processing a single variable
                csv_path = Path(args.csv_file)
                # Resolve relative paths relative to script directory
                if not csv_path.is_absolute():
                    csv_path = (script_dir / csv_path).resolve()
            else:
                # Search for CSV files matching the variable pattern
                pattern = f"tolerance_{variable}_*.csv"
                matching_files = list(output_dir.glob(pattern))
                if not matching_files:
                    print(f"[Warning] No CSV file found matching pattern '{pattern}' in {output_dir}. Skipping {variable}.")
                    continue
                if len(matching_files) > 1:
                    print(f"[Warning] Multiple CSV files found matching pattern for {variable}. Using: {matching_files[0]}")
                csv_path = matching_files[0]
            
            if not csv_path.exists():
                print(f"[Warning] CSV file not found: {csv_path}. Skipping {variable}.")
                continue
            
            print(f"\n[Plot Only] Loading data from {csv_path} (variable: {variable})")
            perturbed_values, results_dict, variable_from_csv = load_tolerance_csv(csv_path)
            
            # Use variable from CSV if available, otherwise use the one we're processing
            actual_variable = variable_from_csv or variable
        
        # Extract metadata (w, g, t, sigma_nm from CSV comments)
        metadata = {}
        with csv_path.open("r", newline="") as fh:
            for line in fh:
                if line.startswith("# Nominal geometry:"):
                    parts = line.split(":")[1].strip().split(",")
                    for part in parts:
                        key, value = part.strip().split("=")
                        metadata[key] = float(value)
                elif line.startswith("# Sigma_nm:"):
                    metadata["sigma_nm"] = float(line.split(":")[1].strip())
        
        w_nom = metadata.get("w", args.w if hasattr(args, 'w') else 0.0)
        g_nom = metadata.get("g", args.g if hasattr(args, 'g') else 0.0)
        t_nom = metadata.get("t", args.t if hasattr(args, 't') else 0.0)
        sigma_nm = metadata.get("sigma_nm", 0.0)
        
        # Generate filename base from CSV metadata
        filename_base = csv_path.stem  # Use CSV filename without extension
        
        # Generate plots from CSV data
        plot_tolerance_results(
            perturbed_values, results_dict, actual_variable,
            w_nom, g_nom, t_nom, output_dir, filename_base=filename_base
        )
        
        # Compute and print statistics
        eta_te_arr = results_dict["eta_te"]
        eta_tm_arr = results_dict["eta_tm"]
        V_te_arr = results_dict["V_te"]
        V_tm_arr = results_dict["V_tm"]
        delta_eta_arr = np.abs(eta_te_arr - eta_tm_arr)
        
        print(f"\n[Tolerance Statistics - {actual_variable.upper()}]")
        print(f"  η_TE: mean={eta_te_arr.mean():.4f}, std={eta_te_arr.std():.4f}, "
              f"95%% range=[{np.percentile(eta_te_arr, 2.5):.4f}, {np.percentile(eta_te_arr, 97.5):.4f}]")
        print(f"  η_TM: mean={eta_tm_arr.mean():.4f}, std={eta_tm_arr.std():.4f}, "
              f"95%% range=[{np.percentile(eta_tm_arr, 2.5):.4f}, {np.percentile(eta_tm_arr, 97.5):.4f}]")
        print(f"  V_TE: mean={V_te_arr.mean():.4f}, std={V_te_arr.std():.4f}, "
              f"95%% range=[{np.percentile(V_te_arr, 2.5):.4f}, {np.percentile(V_te_arr, 97.5):.4f}]")
        print(f"  V_TM: mean={V_tm_arr.mean():.4f}, std={V_tm_arr.std():.4f}, "
              f"95%% range=[{np.percentile(V_tm_arr, 2.5):.4f}, {np.percentile(V_tm_arr, 97.5):.4f}]")
        print(f"  Δη: mean={delta_eta_arr.mean():.4f}, std={delta_eta_arr.std():.4f}, "
              f"95%% range=[{np.percentile(delta_eta_arr, 2.5):.4f}, {np.percentile(delta_eta_arr, 97.5):.4f}]")
        
        return
    
    # Get nominal geometry from command-line arguments
    w_nom = float(args.w)
    g_nom = float(args.g)
    t_nom = float(args.t)
    
    print(f"[Setup] Nominal geometry: w={w_nom:.3f} µm, g={g_nom:.3f} µm, t={t_nom:.3f} µm")
    
    # Create base parameter from template
    base_param = copy.deepcopy(DEFAULT_PARAM_TEMPLATE)
    base_param.wg_width = float(w_nom)
    base_param.coupling_gap = float(g_nom)
    base_param.wg_thick = float(t_nom)
    
    # Compute nominal delta_w_star from geometry (or use provided fixed value)
    if args.delta_w is not None:
        # Use manually specified delta_w
        delta_w_star_nom = float(args.delta_w)
        delta_w_diagnostics = None
        print(f"[Setup] Using manually specified Δw* = {delta_w_star_nom:+.4f} µm (skipping optimization)")
    else:
        # Solve for delta_w_star
        delta_w_star_nom, delta_w_diagnostics = compute_nominal_delta_w_star(
            w_nom, g_nom, t_nom,
            base_param, args.lambda0
        )
    
    # Set delta_w in base_param
    base_param.delta_w = float(delta_w_star_nom)
    
    # Compute nominal L_c
    L_c_nom = compute_nominal_Lc(
        w_nom, g_nom, t_nom, delta_w_star_nom,
        base_param, args.lambda0
    )
    
    # Set L_c in base_param and call update_param_derived ONCE
    base_param.coupling_length = float(L_c_nom)
    update_param_derived(base_param, solve_delta_w=False)
    
    # Ensure L_c and delta_w are frozen (update_param_derived may have recomputed)
    base_param.coupling_length = float(L_c_nom)
    base_param.delta_w = float(delta_w_star_nom)
    base_param.wg_width_left = float(w_nom + delta_w_star_nom / 2)
    base_param.wg_width_right = float(w_nom - delta_w_star_nom / 2)
    
    # Store frozen domain sizes (size_x depends on L_c which is frozen)
    size_x_nom = base_param.size_x
    size_z_nom = base_param.size_z
    
    print(f"[Setup] Frozen values: L_c={L_c_nom:.3f} µm, Δw*={delta_w_star_nom:+.4f} µm")
    
    # Handle joint mode separately
    if args.variable == "joint":
        print(f"\n[Setup] Running joint tolerance analysis (all parameters perturbed simultaneously)")
        run_joint_tolerance(
            args, base_param,
            w_nom, g_nom, t_nom,
            delta_w_star_nom, L_c_nom,
            size_x_nom, size_z_nom,
            output_dir
        )
        print(f"\n{'='*60}")
        print("Completed joint tolerance analysis")
        print(f"{'='*60}")
        return
    
    # Determine which variables to process
    if args.variable == "all":
        variables_to_process = ["t", "w", "g"]
        print(f"\n[Setup] Running tolerance analysis for all variables: {variables_to_process}")
    else:
        variables_to_process = [args.variable]
    
    # Run tolerance analysis for each variable
    for variable in variables_to_process:
        run_tolerance_for_variable(
            variable, args, base_param,
            w_nom, g_nom, t_nom,
            delta_w_star_nom, L_c_nom,
            size_x_nom, size_z_nom,
            output_dir
        )
    
    if args.variable == "all":
        print(f"\n{'='*60}")
        print("Completed tolerance analysis for all variables")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

