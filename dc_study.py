import tidy3d as td
import numpy as np
import tidy3d.web as web
from types import SimpleNamespace
from simulation_utils import (
    build_sim,
    preflight,
    summarize_and_save,
    plot_eta_overlay,
    plot_visibility_overlay,
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
# import need be changed in some cases


DRY_RUN = False # set False to actually simulate
DO_SWEEP = False # set True to run a coarse one-parameter sweep
# --- central hyperparameters (edit once, propagate everywhere) ---
GRID_STEPS_PER_WVL_X = 6
GRID_STEPS_PER_WVL_Y = 6
GRID_STEPS_PER_WVL_Z = 10
RUN_TIME = 1e-11
SHUTOFF = 5e-4
SOURCE_DLAMBDA = 0.06  # µm span driving the Gaussian pulse
SOURCE_NUM_FREQS = 20  # tidy3d validation prefers <= 20
MONITOR_LAMBDA_START = 1.530
MONITOR_LAMBDA_STOP = 1.565
MONITOR_LAMBDA_POINTS = 36  # Adjust as needed to trade spectral resolution for speed
Y_PAD_EXTRA = 0.05  # µm additional y-padding to ensure ≥ λ0/2 clearance to PML
ENABLE_FIELD_MONITOR = False  # set False to disable costly field monitor

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
COUPLING_TRIM_FACTOR = 0.05  # 7.5% empirical trim for bend/transition effects (configurable)
FREEZE_L_C = True  # If True (default), compute L_c once at design wavelength and reuse for all λ.
                   # If False, recompute L_c(λ) per wavelength (for CMT checks, not device spectra).
                   # Default=True keeps geometry fixed during wavelength sweeps.

#script parameters, input parameters for the simulation
# Note: coupling_length will be derived from supermode analysis in update_param_derived()
sbend_length = 8
sbend_height = 0.5
wg_length = 5
wg_width = 1.15
wg_thick = 0.32
wl_0 = 1.55
coupling_gap = 0.275

#calculate the size of the simulation domain (coupling_length will be computed later)
size_x = 2*(wg_length+sbend_length) + 0.0  # Will be computed from derived coupling_length
size_z = 3*wl_0+wg_thick
freq_0 = td.C_0/wl_0
size_y = 2*(sbend_height + wl_0) + wg_width + coupling_gap + max(0.0, wg_width - wl_0) + Y_PAD_EXTRA

#Everything not occupied by a structure uses this medium
SiO2 = td.Medium(
    name = 'SiO2', 
    permittivity = 2.085136, 
)

# Waveguide material: SiN instead of Si (n≈2.0)
SiN = td.Medium(name='SiN', permittivity=2.0**2)

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
    coupling_gap=coupling_gap
)
param.pad_extra = Y_PAD_EXTRA
param.medium = SimpleNamespace(
    Vacuum=td.Medium(permittivity=1.0),
    SiO2=SiO2,
    SiN=SiN
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
param.freeze_l_c = FREEZE_L_C  # Controls whether L_c is wavelength-dependent

# Compute derived coupling_length before building geometry
update_param_derived(param)

generate_object_result = generate_object(param)
generate_object_result = generate_object_result if isinstance(generate_object_result, list) else [generate_object_result]

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
            mode_index = pick_mode_index_at_source(probe_sim, param, pol)
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
        return summarize_and_save(data, pol, outdir="results", mode_solver_diag=mode_solver_diag)
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
                # coupling_length is now derived, not swept in main grid
            }
            
            # Optional fine-tuning grid for L_c
            grid_length = None  # Default: use derived value only
            if ENABLE_L_FINE_TUNE:
                # Derive base L_c first to create fine-tuning grid
                # This will be recomputed per (gap, width) in the sweep, but we need a preview
                temp_param = SimpleNamespace(**param.__dict__)
                update_param_derived(temp_param)
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
            # Overlay plots for coupling ratio and visibility
            plot_eta_overlay(results_cache["te"], results_cache["tm"], outdir="results")
            plot_visibility_overlay(results_cache["te"], results_cache["tm"], outdir="results")

            # Combine TE/TM spectra into a single CSV
            delta_eta = save_combined_te_tm_csv(results_cache["te"], results_cache["tm"], outdir="results")

            # Compute and save band-averaged visibilities
            compute_and_save_band_averaged_visibility(results_cache["te"], results_cache["tm"], param, outdir="results")

            # Plot polarization imbalance Δη(λ) and ΔV(λ)
            plot_delta_eta(results_cache["te"]["lam"], delta_eta, outdir="results")
            plot_delta_visibility(results_cache["te"], results_cache["tm"], outdir="results")
