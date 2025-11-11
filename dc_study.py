import tidy3d as td
import numpy as np
import tidy3d.web as web
from types import SimpleNamespace
from pathlib import Path
import shutil
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
from sweep_utils import sweep
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
DO_SWEEP = False # set True to run a coarse one-parameter sweep
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
# delta_w is now solved (not a free parameter) when SOLVE_DELTA_W=True

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

# --- runners & sweep utilities ---
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
    # To run the 3D multi-objective sweep, set:
    #   DO_SWEEP = True
    #   SWEEP_PARAM = "3d"
    # Adjust WEIGHTS at the top of the file to trade off polarization balance (alpha),
    # tolerance (beta — Stage 1 keep 0.0), and compactness (gamma; L_ref is the scale).

    # Optional fine-tuning parameters for L_c (only if needed)
    ENABLE_L_FINE_TUNE = False  # Set True to add ±15% fine-tuning sweep around derived L_c
    L_TUNE_RANGE = (0.85, 1.15)  # Multiplier range around derived L_c for fine-tuning
    L_TUNE_POINTS = 5  # Number of points in fine-tuning grid

    SWEEP_PARAM = "coupling_length"  # Legacy: coupling_length is now derived, not swept
    # define coarse grids for common parameters (units in µm)
    GRID = {
        "coupling_gap":    np.round(np.linspace(0.24, 0.34, 6), 3),
    }

    results_cache = {}

    if DO_SWEEP:
            GRID3D = {
                "coupling_gap":    np.round(np.linspace(0.24, 0.34, 2), 3),   # µm
                "wg_width":        np.round(np.linspace(1.20, 1.60, 2), 3),   # µm
                # delta_w is now solved (not swept) using asymmetry-first strategy
                # coupling_length is now derived, not swept in main grid
            }
            
            # Optional fine-tuning grid for L_c
            grid_length = None  # Default: use derived value only
            if ENABLE_L_FINE_TUNE:
                # Derive base L_c first to create fine-tuning grid
                # This will be recomputed per (gap, width) in the sweep, but we need a preview
                temp_param = SimpleNamespace(**param.__dict__)
                update_param_derived(temp_param, solve_delta_w=SOLVE_DELTA_W)
                base_Lc = temp_param.coupling_length
                grid_length = np.linspace(L_TUNE_RANGE[0]*base_Lc, L_TUNE_RANGE[1]*base_Lc, L_TUNE_POINTS)
                print(f"[SWEEP] L_c fine-tuning enabled: {L_TUNE_RANGE[0]:.1%} to {L_TUNE_RANGE[1]:.1%} × {base_Lc:.3f}µm = [{grid_length[0]:.3f}, {grid_length[-1]:.3f}] µm")
            
            sweep(
                GRID3D["coupling_gap"], 
                GRID3D["wg_width"], 
                grid_length,  # None for derived-only, or array for fine-tuning
                param=param,
                weights=WEIGHTS,
                run_single_fn=run_single,
                dry_run=DRY_RUN, 
                save_top_k=10, 
                outdir=str(RESULTS_DIR),
                lambda_single=1.55
            )
      
    else:
        # single TE/TM run (no sweep)
        for pol in RUN_POLS:
            summary = run_single(param, pol=pol, task_tag="full", dry_run=DRY_RUN)
            if summary is not None:
                results_cache[pol] = summary

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
