import tidy3d as td
import numpy as np
import tidy3d.web as web
from types import SimpleNamespace
from pathlib import Path
import shutil
import copy
import csv
from datetime import datetime
from simulation_utils import (
    build_sim,
    preflight,
    summarize_and_save,
    plot_eta_overlay,
    plot_visibility_overlay,
    plot_delta_phi_overlay,
    save_combined_te_tm_csv,
    compute_and_save_band_averaged_visibility,
    plot_delta_eta,
    plot_delta_visibility,
    pick_mode_index_at_source,
    compute_mode_solver_diagnostics,
)
from building_utils import generate_object, update_param_derived
import matplotlib.pyplot as plt
from material_dispersion import get_permittivity_SiO2, get_permittivity_SiN
# import need be changed in some cases


def get_geometry_folder_name(wg_width, wg_thick, coupling_gap, blend_policy):
    """
    Generate a folder name based on geometry parameters.
    
    Format: w{width}_t{thick}_g{gap}_b{blend_policy}
    Example: w1p150_t0p320_g0p250_bmedian
    
    Args:
        wg_width: Waveguide width in µm
        wg_thick: Waveguide thickness in µm
        coupling_gap: Coupling gap in µm
        blend_policy: Blend policy string ("median", "te", "tm", "balance")
    
    Returns:
        Folder name string
    """
    # Format numbers: replace decimal point with 'p' and format to 3 decimal places
    w_str = f"{wg_width:.3f}".replace('.', 'p')
    t_str = f"{wg_thick:.3f}".replace('.', 'p')
    g_str = f"{coupling_gap:.3f}".replace('.', 'p')
    
    folder_name = f"w{w_str}_t{t_str}_g{g_str}_b{blend_policy}"
    return folder_name


DRY_RUN = False# set False to actually simulate
# --- central hyperparameters (edit once, propagate everywhere) ---
GRID_STEPS_PER_WVL_X = 6
GRID_STEPS_PER_WVL_Y = 6
GRID_STEPS_PER_WVL_Z = 10
RUN_TIME = 1e-11 #s
SHUTOFF = 1e-6#s
SOURCE_DLAMBDA = 0.06  # µm span driving the Gaussian pulse
SOURCE_NUM_FREQS = 20  # tidy3d validation prefers <= 20
MONITOR_LAMBDA_START = 1.530 #nm
MONITOR_LAMBDA_STOP = 1.565 #nm
MONITOR_LAMBDA_POINTS = 36  # Adjust as needed to trade spectral resolution for speed
Y_PAD_EXTRA = 0.2  # µm additional y-padding to ensure ≥ λ0/2 clearance to PML
ENABLE_FIELD_MONITOR = True # set False to disable costly field monitor

# ---- multi-objective weights (Stage 1 defaults) ----
# Score = Vbar_avg - alpha*DeltaEta_avg - beta*sigma_tol - gamma*(L_c/L_ref)
# For Stage 1 we set beta=0 (no full tolerance yet). Adjust as needed.
WEIGHTS = {
    "alpha": 2.0,   # penalize polarization mismatch strongly
    "beta": 0.0,    # no tolerance penalty in Stage 1 (set >0 in Stage 2)
    "gamma": 0.05,  # light compactness penalty per L_ref
    "L_ref": 10.0,  # µm reference length
}

# --- Coupling length derivation parameters ---
COUPLING_TRIM_FACTOR = 0.0  # 10% trim for bend/transition effects (δ ∈ [0.05, 0.10] per derivation)
                              # The balance optimization now accounts for trim factor, so this can be 
                              # set to the typical value. Adjust based on FDTD calibration if needed.
COUPLING_LENGTH_BOUNDS = (3.0, 50.0)  # µm min/max bounds for derived L_c
FREEZE_L_C = True  # If True (default), compute L_c once at design wavelength and reuse for all λ.
                   # If False, recompute L_c(λ) per wavelength (for CMT checks, not device spectra).
                   # Default=True keeps geometry fixed during wavelength sweeps.
COUPLING_BLEND_POLICY = "balance"  # How to blend TE/TM L50: "median" (default), "te", "tm", or "balance"
                                    # "balance" optimizes L_c to minimize |η_TE-0.5|+|η_TM-0.5|
SOLVE_DELTA_W = True  # If True, solve for delta_w* to equalize L_50_TE and L_50_TM (asymmetry-first strategy)
                      # If False, use symmetric mode (delta_w=0) or manually set delta_w
DELTA_W_ABS_TOL = 0.05  # Absolute tolerance for L50_TE - L50_TM matching (µm)
DELTA_W_REL_TOL = 0.01  # Relative tolerance for L50_TE - L50_TM matching (1% = 0.01)
DELTA_W_SEARCH_MIN = -0.6  # Primary search box minimum (µm)
DELTA_W_SEARCH_MAX = +0.6  # Primary search box maximum (µm)
DELTA_W_HARD_MIN = -1  # Hard clip minimum (µm) - fabrication constraint
DELTA_W_HARD_MAX = +1  # Hard clip maximum (µm) - fabrication constraint
# Newton solver hyperparameters
DELTA_W_H_INIT = 0.01  # Initial step size for finite differences (µm)
DELTA_W_H_MIN = 0.001  # Minimum step size for finite differences (µm)
DELTA_W_MAX_ITER = 200  # Maximum Newton iterations
DELTA_W_EPS_MIN = 1e-6  # Minimum |F'| threshold for derivative computation

# --- Material dispersion configuration ---
USE_DISPERSIVE_MATERIALS = True  # If True, use wavelength-dependent refractive indices (Sellmeier)
                                  # If False, use constant values (backward compatibility)

#script parameters, input parameters for the simulation
# Note: coupling_length will be derived from supermode analysis in update_param_derived()
# Note: delta_w will be solved (not set) when SOLVE_DELTA_W=True (asymmetry-first strategy)
sbend_length = 8  #µm
sbend_height = 0.5  #µm
wg_length = 5  #µm
wg_width = 1.1 #µm
wg_thick = 0.28  #µm
wl_0 = 1.55  #µm
coupling_gap = 0.28  #µm


# Optional fine-tuning parameters for L_c (only if needed)
ENABLE_L_FINE_TUNE = True  # Set True to add trim-factor sweep around derived L_c
L_TUNE_TRIM_RANGE = (-0.15, 0.15)  # Fractional trim offsets (e.g., ±15%)
L_TUNE_POINTS = 4  # Number of trim samples within the range

#calculate the size of the simulation domain (coupling_length will be computed later)
size_x = 2*(wg_length+sbend_length) + 0.0  # Will be computed from derived coupling_length
size_z = 3*wl_0+wg_thick
freq_0 = td.C_0/wl_0
size_y = 2*(sbend_height + wl_0) + wg_width + coupling_gap + max(0.0, wg_width - wl_0) + Y_PAD_EXTRA

# Material definitions with wavelength-dependent support
def create_SiO2_medium(lambda_um=None):
    """
    Create SiO2 medium with wavelength-dependent permittivity if enabled.
    
    Args:
        lambda_um: Wavelength in micrometers. If None and USE_DISPERSIVE_MATERIALS=True,
                   uses wl_0 (1.55 µm). If USE_DISPERSIVE_MATERIALS=False, uses constant value.
    
    Returns:
        td.Medium object
    """
    if USE_DISPERSIVE_MATERIALS:
        if lambda_um is None:
            lambda_um = wl_0
        permittivity = get_permittivity_SiO2(lambda_um)
    else:
        # Constant value at 1550nm: n ≈ 1.444, ε ≈ 2.085136
        permittivity = 2.085136
    return td.Medium(name='SiO2', permittivity=permittivity)


def create_SiN_medium(lambda_um=None):
    """
    Create SiN medium with wavelength-dependent permittivity if enabled.
    
    Args:
        lambda_um: Wavelength in micrometers. If None and USE_DISPERSIVE_MATERIALS=True,
                   uses wl_0 (1.55 µm). If USE_DISPERSIVE_MATERIALS=False, uses constant value.
    
    Returns:
        td.Medium object
    """
    if USE_DISPERSIVE_MATERIALS:
        if lambda_um is None:
            lambda_um = wl_0
        permittivity = get_permittivity_SiN(lambda_um)
    else:
        # Constant value: n = 2.0, ε = 4.0
        permittivity = 2.0**2
    return td.Medium(name='SiN', permittivity=permittivity)


# Create initial materials at design wavelength
SiO2 = create_SiO2_medium(wl_0)
SiN = create_SiN_medium(wl_0)

# Mode specs (realistic neff guesses for SiN platform)
MODE_SPEC_TE = td.ModeSpec(num_modes=1, filter_pol='te', target_neff=1.7)
MODE_SPEC_TM = td.ModeSpec(num_modes=1, filter_pol='tm', target_neff=1.5)


param = SimpleNamespace(
    # coupling_length will be derived from supermode analysis in update_param_derived()
    sbend_length=sbend_length,
    wg_length=wg_length,
    wg_width=wg_width,
    wg_thick=wg_thick,
    sbend_height=sbend_height,
    wl_0=wl_0,
    size_x=size_x,
    size_y=size_y,
    size_z=size_z,
    freq_0=freq_0,
    coupling_gap=coupling_gap,
    delta_w=0.0  # Will be solved if SOLVE_DELTA_W=True, otherwise symmetric (0.0)
)
param.pad_extra = Y_PAD_EXTRA
param.medium = SimpleNamespace(
    Vacuum=td.Medium(permittivity=1.0),
    SiO2=SiO2,  # Initial medium at design wavelength
    SiN=SiN,    # Initial medium at design wavelength
    create_SiO2=create_SiO2_medium,  # Factory function for wavelength-dependent SiO2
    create_SiN=create_SiN_medium,    # Factory function for wavelength-dependent SiN
    use_dispersive=USE_DISPERSIVE_MATERIALS  # Flag for dispersion mode
)
param.mode_specs = {
    "te": MODE_SPEC_TE,
    "tm": MODE_SPEC_TM,
}
param.mode_indices = {}
RUN_POLS = ("te", "tm")

# propagate hyperparameters so helper modules can read them
param.grid_steps_per_wvl = (
    GRID_STEPS_PER_WVL_X,
    GRID_STEPS_PER_WVL_Y,
    GRID_STEPS_PER_WVL_Z,
)
param.run_time = RUN_TIME
param.shutoff = SHUTOFF
param.source_dlambda = SOURCE_DLAMBDA
param.source_num_freqs = SOURCE_NUM_FREQS
param.monitor_lambda_start = MONITOR_LAMBDA_START
param.monitor_lambda_stop = MONITOR_LAMBDA_STOP
param.monitor_lambda_points = MONITOR_LAMBDA_POINTS
param.enable_field_monitor = ENABLE_FIELD_MONITOR
param.coupling_trim_factor = COUPLING_TRIM_FACTOR
param.coupling_length_bounds = COUPLING_LENGTH_BOUNDS
param.freeze_l_c = FREEZE_L_C  # Controls whether L_c is wavelength-dependent
param.coupling_blend_policy = COUPLING_BLEND_POLICY  # Controls how TE/TM L50 values are blended
param.delta_w_abs_tol = DELTA_W_ABS_TOL  # Absolute tolerance for delta_w* solver (µm)
param.delta_w_rel_tol = DELTA_W_REL_TOL  # Relative tolerance for delta_w* solver (fraction, e.g., 0.01 = 1%)
param.delta_w_search_min = DELTA_W_SEARCH_MIN  # Primary search box minimum (µm)
param.delta_w_search_max = DELTA_W_SEARCH_MAX  # Primary search box maximum (µm)
param.delta_w_hard_min = DELTA_W_HARD_MIN  # Hard clip minimum (µm)
param.delta_w_hard_max = DELTA_W_HARD_MAX  # Hard clip maximum (µm)
param.delta_w_h_init = DELTA_W_H_INIT  # Initial step size for finite differences (µm)
param.delta_w_h_min = DELTA_W_H_MIN  # Minimum step size for finite differences (µm)
param.delta_w_max_iter = DELTA_W_MAX_ITER  # Maximum Newton iterations
param.delta_w_eps_min = DELTA_W_EPS_MIN  # Minimum |F'| threshold for derivative computation

# Compute geometry-specific results folder name
RESULTS_BASE_DIR = Path("results")
GEOMETRY_FOLDER_NAME = get_geometry_folder_name(
    wg_width=wg_width,
    wg_thick=wg_thick,
    coupling_gap=coupling_gap,
    blend_policy=COUPLING_BLEND_POLICY
)
RESULTS_DIR = RESULTS_BASE_DIR / GEOMETRY_FOLDER_NAME
TRIM_LOG_DIR = Path("results/sweeps")
TRIM_LOG_PATH = TRIM_LOG_DIR / "trim_fine_tune.csv"
TRIM_LOG_FIELDS = [
    "timestamp",
    "geometry",
    "wg_width_um",
    "wg_thick_um",
    "gap_um",
    "delta_w_um",
    "w1_um",
    "w2_um",
    "trim_offset",
    "total_trim_factor",
    "coupling_length_um",
    "eta_te_1550",
    "eta_tm_1550",
    "V_te_1550",
    "V_tm_1550",
    "score",
    "DeltaEta_avg",
    "Vbar_avg",
]

def _coupling_metric(results_cache):
    te = results_cache.get("te")
    tm = results_cache.get("tm")
    if te is None or tm is None:
        return float("inf")
    lam = te["lam"]
    idx = int(np.argmin(np.abs(lam - 1.55)))
    eta_te = float(te["eta"][idx])
    eta_tm = float(tm["eta"][idx])
    return abs(eta_te - 0.5) + abs(eta_tm - 0.5)


def _trim_already_computed(geometry, trim_offset):
    """Check if a trim configuration was already computed and logged."""
    if not TRIM_LOG_PATH.exists():
        return None
    try:
        with TRIM_LOG_PATH.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row_geometry = row.get("geometry", "")
                try:
                    row_trim = round(float(row.get("trim_offset", 0.0)), 6)
                except (TypeError, ValueError):
                    row_trim = 0.0
                if row_geometry == geometry and row_trim == round(float(trim_offset), 6):
                    # Return the existing row data
                    return row
    except Exception:
        pass
    return None


def _append_trim_log(row_dict):
    TRIM_LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    if TRIM_LOG_PATH.exists():
        with TRIM_LOG_PATH.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                existing.append(row)

    def _key(row):
        try:
            trim = float(row.get("trim_offset", 0.0))
        except (TypeError, ValueError):
            trim = 0.0
        return (row.get("geometry", ""), round(trim, 6))

    new_key = (row_dict.get("geometry", ""), round(float(row_dict.get("trim_offset", 0.0)), 6))
    filtered = [row for row in existing if _key(row) != new_key]
    filtered.append({k: row_dict.get(k) for k in TRIM_LOG_FIELDS})
    filtered.sort(key=lambda r: float(r.get("score", -1e9)), reverse=True)

    with TRIM_LOG_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRIM_LOG_FIELDS)
        writer.writeheader()
        for row in filtered:
            writer.writerow(row)


def compute_objective(results_te, results_tm, param, weights):
    lam = results_te["lam"]
    mask = (lam >= 1.530) & (lam <= 1.565)
    def _avg(arr):
        arr = np.asarray(arr)
        sel = arr[mask] if mask.any() else arr
        return float(np.mean(sel))
    Vbar_te = _avg(results_te["V"])
    Vbar_tm = _avg(results_tm["V"])
    Vbar_avg = 0.5 * (Vbar_te + Vbar_tm)
    DeltaEta = np.abs(np.asarray(results_te["eta"]) - np.asarray(results_tm["eta"]))
    DeltaEta_avg = _avg(DeltaEta)
    sigma_tol = 0.0
    w = weights or WEIGHTS
    alpha = float(w.get("alpha", WEIGHTS["alpha"]))
    beta = float(w.get("beta", WEIGHTS["beta"]))
    gamma = float(w.get("gamma", WEIGHTS["gamma"]))
    L_ref = float(w.get("L_ref", WEIGHTS["L_ref"]))
    Lc = float(getattr(param, "coupling_length"))
    Score = Vbar_avg - alpha * DeltaEta_avg - beta * sigma_tol - gamma * (Lc / L_ref)
    return {
        "Vbar_te": Vbar_te,
        "Vbar_tm": Vbar_tm,
        "Vbar_avg": Vbar_avg,
        "DeltaEta_avg": DeltaEta_avg,
        "sigma_tol": sigma_tol,
        "Lc": Lc,
        "Score": float(Score),
    }

# Create geometry directory only for real runs (not dry runs)
# Move existing results to geometry-specific folder if they exist in base results folder
if not DRY_RUN:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Only move files if geometry folder is empty and base folder has files
    if RESULTS_BASE_DIR.exists() and not any(RESULTS_DIR.iterdir()):
        existing_files = list(RESULTS_BASE_DIR.glob("*"))
        # Filter out directories and only move files
        existing_files = [f for f in existing_files if f.is_file()]
        if existing_files:
            for file_path in existing_files:
                dest_path = RESULTS_DIR / file_path.name
                if not dest_path.exists():  # Only move if destination doesn't exist
                    shutil.move(str(file_path), str(dest_path))
                    print(f"[RESULTS] Moved {file_path.name} to {RESULTS_DIR}")

# Compute derived coupling_length before building geometry
# If SOLVE_DELTA_W=True, delta_w* will be solved to equalize L_50_TE and L_50_TM
# Skip if being imported by tune_geometry.py to avoid unnecessary computation
_is_tune_geometry_import = any('tune_geometry' in str(frame.filename) for frame in __import__('inspect').stack())
if not _is_tune_geometry_import:
    update_param_derived(param, solve_delta_w=SOLVE_DELTA_W)
    generate_object_result = generate_object(param)
    generate_object_result = generate_object_result if isinstance(generate_object_result, list) else [generate_object_result]
else:
    # When imported by tune_geometry, skip generate_object (not needed)
    generate_object_result = []

# --- runner ---
def run_single(param, pol='te', task_tag='single', dry_run=False, lambda_single=None):
    """Build, preflight, and optionally run one polarization. Returns summary dict or None if dry_run."""
    if not hasattr(param, "mode_indices") or param.mode_indices is None:
        param.mode_indices = {}

     # --- SNAPSHOT monitor settings (treat as local to this call) ---
    _orig_mon = (
        param.monitor_lambda_start,
        param.monitor_lambda_stop,
        param.monitor_lambda_points,
        param.source_num_freqs,
        param.source_dlambda,
    )

    try:
        if lambda_single is not None:
                param.monitor_lambda_start = lambda_single
                param.monitor_lambda_stop = lambda_single
                param.monitor_lambda_points = 1
                param.source_num_freqs = 1
                param.source_dlambda = 0.0005

        if not dry_run:
            # remove any stale index before probing
            param.mode_indices.pop(pol, None)
            probe_sim = build_sim(param, pol=pol)
            lambda_probe = lambda_single or getattr(param, "wl_0", 1.55)
            mode_index = pick_mode_index_at_source(
                probe_sim,
                param,
                pol,
                lambda_um=lambda_probe,
                n_modes=6,
            )
            param.mode_indices[pol] = mode_index

        sim = build_sim(param, pol=pol)
        mode_solver_diag = None if dry_run else compute_mode_solver_diagnostics(sim, param, pol)
        cost_estimate = preflight(sim, do_server_estimate=True)  # Returns estimate; only used when dry_run=True
        if dry_run:
            return {"cost_estimate": cost_estimate}
        # Use blocking run compatible with installed tidy3d.web version (no 'folder' kwarg)
        res = web.run(sim, task_name=f"DC_{task_tag}_{pol}", path="./data/sim_data.hdf5")
        # Support both return types: prefer .results() when available; otherwise use object directly
        try:
            data = res.results()
        except AttributeError:
            data = res
        return summarize_and_save(data, pol, outdir=str(RESULTS_DIR), mode_solver_diag=mode_solver_diag, param=param)
    finally:
         # --- RESTORE monitor settings no matter what ---
        (
            param.monitor_lambda_start,
            param.monitor_lambda_stop,
            param.monitor_lambda_points,
            param.source_num_freqs,
            param.source_dlambda,
        ) = _orig_mon


if __name__ == "__main__":

    results_cache = {}

    # single TE/TM run (with optional L_c fine-tune via trim offsets)
    trim_values = [0.0]
    if ENABLE_L_FINE_TUNE and not DRY_RUN:
        trim_values = np.linspace(L_TUNE_TRIM_RANGE[0], L_TUNE_TRIM_RANGE[1], L_TUNE_POINTS)
    base_trim = getattr(param, "coupling_trim_factor", COUPLING_TRIM_FACTOR)
    base_param_snapshot = copy.deepcopy(param)
    best_entry = None  # (metric, trim, param_state, results_cache)

    trial_records = []
    for trim in trim_values:
        # Check if this trim was already computed
        existing_entry = _trim_already_computed(GEOMETRY_FOLDER_NAME, trim) if not DRY_RUN else None
        if existing_entry is not None:
            print(f"[L_c Fine Tune] Skipping trim_offset={trim:+.3f} (already computed, score={existing_entry.get('score', 'N/A')})")
            # Still update the log to ensure it's sorted (refresh timestamp)
            log_payload_skip = {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "geometry": existing_entry.get("geometry", GEOMETRY_FOLDER_NAME),
                "wg_width_um": existing_entry.get("wg_width_um", ""),
                "wg_thick_um": existing_entry.get("wg_thick_um", ""),
                "gap_um": existing_entry.get("gap_um", ""),
                "delta_w_um": existing_entry.get("delta_w_um", ""),
                "w1_um": existing_entry.get("w1_um", ""),
                "w2_um": existing_entry.get("w2_um", ""),
                "trim_offset": existing_entry.get("trim_offset", trim),
                "total_trim_factor": existing_entry.get("total_trim_factor", ""),
                "coupling_length_um": existing_entry.get("coupling_length_um", ""),
                "eta_te_1550": existing_entry.get("eta_te_1550", ""),
                "eta_tm_1550": existing_entry.get("eta_tm_1550", ""),
                "V_te_1550": existing_entry.get("V_te_1550", ""),
                "V_tm_1550": existing_entry.get("V_tm_1550", ""),
                "score": existing_entry.get("score", ""),
                "DeltaEta_avg": existing_entry.get("DeltaEta_avg", ""),
                "Vbar_avg": existing_entry.get("Vbar_avg", ""),
            }
            _append_trim_log(log_payload_skip)
            continue
        
        trial_param = copy.deepcopy(base_param_snapshot)
        trial_param.coupling_trim_factor = base_trim + trim
        update_param_derived(trial_param, solve_delta_w=SOLVE_DELTA_W)
        trial_results = {}
        for pol in RUN_POLS:
            summary = run_single(trial_param, pol=pol, task_tag=f"full_trim_{trim:+.3f}", dry_run=DRY_RUN)
            if summary is not None:
                trial_results[pol] = summary
        if DRY_RUN:
            best_entry = (0.0, trim, trial_param, trial_results)
            break
        if len(trial_results) == len(RUN_POLS):
            metric = _coupling_metric(trial_results)
            log_payload = None
            if not DRY_RUN:
                metrics_tmp = compute_objective(trial_results["te"], trial_results["tm"], trial_param, WEIGHTS)
                lam_tmp = trial_results["te"]["lam"]
                idx_tmp = int(np.argmin(np.abs(lam_tmp - 1.55)))
                log_payload = {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                    "geometry": GEOMETRY_FOLDER_NAME,
                    "wg_width_um": trial_param.wg_width,
                    "wg_thick_um": trial_param.wg_thick,
                    "gap_um": trial_param.coupling_gap,
                    "delta_w_um": getattr(trial_param, "delta_w", float("nan")),
                    "w1_um": getattr(trial_param, "wg_width_left", float("nan")),
                    "w2_um": getattr(trial_param, "wg_width_right", float("nan")),
                    "trim_offset": trim,
                    "total_trim_factor": trial_param.coupling_trim_factor,
                    "coupling_length_um": metrics_tmp["Lc"],
                    "eta_te_1550": float(trial_results["te"]["eta"][idx_tmp]),
                    "eta_tm_1550": float(trial_results["tm"]["eta"][idx_tmp]),
                    "V_te_1550": float(trial_results["te"]["V"][idx_tmp]),
                    "V_tm_1550": float(trial_results["tm"]["V"][idx_tmp]),
                    "score": metrics_tmp["Score"],
                    "DeltaEta_avg": metrics_tmp["DeltaEta_avg"],
                    "Vbar_avg": metrics_tmp["Vbar_avg"],
                }
                trial_records.append(log_payload)
                _append_trim_log(log_payload)
            if best_entry is None or metric < best_entry[0]:
                best_entry = (metric, trim, copy.deepcopy(trial_param), trial_results)

    if best_entry is None:
        raise RuntimeError("No successful simulations were recorded for the requested single run.")

    _, best_trim, best_param_state, best_results_cache = best_entry
    if ENABLE_L_FINE_TUNE and not DRY_RUN:
        print(f"[L_c Fine Tune] Selected trim offset {best_trim:+.3%} (coupling_trim_factor={best_param_state.coupling_trim_factor:+.4f})")

    param = best_param_state
    results_cache = best_results_cache
    if not DRY_RUN and len(results_cache) == len(RUN_POLS):
        metrics = compute_objective(results_cache["te"], results_cache["tm"], param, WEIGHTS)
        lam = results_cache["te"]["lam"]
        idx = int(np.argmin(np.abs(lam - 1.55)))
        eta_te_1550 = float(results_cache["te"]["eta"][idx])
        eta_tm_1550 = float(results_cache["tm"]["eta"][idx])
        V_te_1550 = float(results_cache["te"]["V"][idx])
        V_tm_1550 = float(results_cache["tm"]["V"][idx])
        print("\n[L_c Fine Tune Report]")
        print(f"  Geometry        : w={param.wg_width:.3f} µm, g={param.coupling_gap:.3f} µm, t={param.wg_thick:.3f} µm")
        print(f"  Δw* / widths    : Δw={param.delta_w:+.4f} µm (w1={getattr(param,'wg_width_left',param.wg_width):.3f} µm, w2={getattr(param,'wg_width_right',param.wg_width):.3f} µm)")
        print(f"  Trim selection  : offset={best_trim:+.3%}, total factor={param.coupling_trim_factor:+.4f}")
        print(f"  Coupling length : {metrics['Lc']:.3f} µm")
        print(f"  η(1550 nm)      : TE={eta_te_1550:.3f}, TM={eta_tm_1550:.3f}")
        print(f"  V(1550 nm)      : TE={V_te_1550:.3f}, TM={V_tm_1550:.3f}")
        print(f"  Score components: V̄_avg={metrics['Vbar_avg']:.4f}, Δη̄={metrics['DeltaEta_avg']:.4f}")
        print(f"  Final Score     : {metrics['Score']:.4f}")
        final_log = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "geometry": GEOMETRY_FOLDER_NAME,
            "wg_width_um": param.wg_width,
            "wg_thick_um": param.wg_thick,
            "gap_um": param.coupling_gap,
            "delta_w_um": getattr(param, "delta_w", float("nan")),
            "w1_um": getattr(param, "wg_width_left", float("nan")),
            "w2_um": getattr(param, "wg_width_right", float("nan")),
            "trim_offset": best_trim,
            "total_trim_factor": param.coupling_trim_factor,
            "coupling_length_um": metrics["Lc"],
            "eta_te_1550": eta_te_1550,
            "eta_tm_1550": eta_tm_1550,
            "V_te_1550": V_te_1550,
            "V_tm_1550": V_tm_1550,
            "score": metrics["Score"],
            "DeltaEta_avg": metrics["DeltaEta_avg"],
            "Vbar_avg": metrics["Vbar_avg"],
        }
        _append_trim_log(final_log)

        # If both runs were executed, save an overlay comparison plot AND a combined CSV with Δη(λ) and band-averaged visibility
        if not DRY_RUN and all(p in results_cache for p in ("te", "tm")):
            # Debug: verify data before plotting
            idx_1550_te = int(np.argmin(np.abs(results_cache["te"]["lam"] - 1.55)))
            idx_1550_tm = int(np.argmin(np.abs(results_cache["tm"]["lam"] - 1.55)))
            print(f"\n[PLOT DEBUG] TE at 1550nm: eta={results_cache['te']['eta'][idx_1550_te]:.3f}, V={results_cache['te']['V'][idx_1550_te]:.3f}")
            print(f"[PLOT DEBUG] TM at 1550nm: eta={results_cache['tm']['eta'][idx_1550_tm]:.3f}, V={results_cache['tm']['V'][idx_1550_tm]:.3f}")

            # Overlay plots for coupling ratio and visibility
            plot_eta_overlay(results_cache["te"], results_cache["tm"], outdir=str(RESULTS_DIR))
            plot_visibility_overlay(results_cache["te"], results_cache["tm"], outdir=str(RESULTS_DIR))
            plot_delta_phi_overlay(results_cache["te"], results_cache["tm"], outdir=str(RESULTS_DIR))

            # Combine TE/TM spectra into a single CSV
            delta_eta = save_combined_te_tm_csv(results_cache["te"], results_cache["tm"], outdir=str(RESULTS_DIR))

            # Compute and save band-averaged visibilities
            compute_and_save_band_averaged_visibility(results_cache["te"], results_cache["tm"], param, outdir=str(RESULTS_DIR))

            # Plot polarization imbalance Δη(λ) and ΔV(λ)
            plot_delta_eta(results_cache["te"]["lam"], delta_eta, outdir=str(RESULTS_DIR))
            plot_delta_visibility(results_cache["te"], results_cache["tm"], outdir=str(RESULTS_DIR))
