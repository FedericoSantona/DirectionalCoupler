import tidy3d as td
import numpy as np
import tidy3d.web as web
from types import SimpleNamespace
from simulation_utils import (
    build_sim,
    preflight,
    summarize_and_save,
    plot_eta_overlay,
    save_combined_te_tm_csv,
    compute_and_save_band_averaged_visibility,
    plot_delta_eta,
    pick_mode_index_at_source,
    compute_mode_solver_diagnostics,
)
from building_utils import generate_object, update_param_derived
import matplotlib.pyplot as plt
# import need be changed in some cases


DRY_RUN = False  # set False to actually simulate
DO_SWEEP = False # set True to run a coarse one-parameter sweep
# --- central hyperparameters (edit once, propagate everywhere) ---
GRID_STEPS_PER_WVL_X = 12
GRID_STEPS_PER_WVL_Y = 12
GRID_STEPS_PER_WVL_Z = 16
RUN_TIME = 3e-11
SHUTOFF = 1e-4
SOURCE_DLAMBDA = 0.06  # µm span driving the Gaussian pulse
SOURCE_NUM_FREQS = 20  # tidy3d validation prefers <= 20
MONITOR_LAMBDA_START = 1.530
MONITOR_LAMBDA_STOP = 1.565
MONITOR_LAMBDA_POINTS = 36  # Adjust as needed to trade spectral resolution for speed

#script parameters, input parameters for the simulation
coupling_length = 8.8
sbend_length = 6
wg_length = 5
wg_width = 1.35
wg_thick = 0.35
sbend_height = 0.75
wl_0 = 1.55
coupling_gap = 0.25

#calculate the size of the simulation domain
size_x = 2*(wg_length+sbend_length)+coupling_length
size_z = 3*wl_0+wg_thick
freq_0 = td.C_0/wl_0
size_y = 2*(sbend_height+wl_0)+wg_width+coupling_gap

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
    coupling_length=coupling_length,
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

generate_object_result = generate_object(param)
generate_object_result = generate_object_result if isinstance(generate_object_result, list) else [generate_object_result]

# --- runners & sweep utilities ---
def run_single(param, pol='te', task_tag='single', dry_run=False):
    """Build, preflight, and optionally run one polarization. Returns summary dict or None if dry_run."""
    if not hasattr(param, "mode_indices") or param.mode_indices is None:
        param.mode_indices = {}

    if not dry_run:
        # remove any stale index before probing
        param.mode_indices.pop(pol, None)
        probe_sim = build_sim(param, pol=pol)
        mode_index = pick_mode_index_at_source(probe_sim, param, pol)
        param.mode_indices[pol] = mode_index

    sim = build_sim(param, pol=pol)
    mode_solver_diag = None if dry_run else compute_mode_solver_diagnostics(sim, param, pol)
    preflight(sim, do_server_estimate=True)
    if dry_run:
        return None
    # Use blocking run compatible with installed tidy3d.web version (no 'folder' kwarg)
    res = web.run(sim, task_name=f"DC_{task_tag}_{pol}", path="./data/sim_data.hdf5")
    # Support both return types: prefer .results() when available; otherwise use object directly
    try:
        data = res.results()
    except AttributeError:
        data = res
    return summarize_and_save(data, pol, outdir="results", mode_solver_diag=mode_solver_diag)

def eta_at_1550_from_summary(summary):
    lam = summary["lam"]
    eta = summary["eta"]
    k = int(np.argmin(np.abs(lam - 1.55)))
    return float(eta[k])

def sweep_one_param(param_name, values, pols=('te','tm'), dry_run=False, folder="DC_Study"):
    """
    Coarse grid search over one parameter (e.g., 'coupling_length' or 'coupling_gap').
    For each value, updates param, rebuilds derived sizes, runs TE/TM, and saves a summary CSV and a plot of eta(1550nm) vs param.
    """
    from pathlib import Path
    import csv
    outdir = Path("results"); outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"summary_sweep_{param_name}.csv"
    print(f"\n[SWEEP] {param_name} over {list(values)}")

    # keep original to restore
    original = getattr(param, param_name)

    # accumulate for plotting
    xs, eta_te_list, eta_tm_list = [], [], []
    minV_te_list, minV_tm_list = [], []

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([param_name, "eta_te_1550", "eta_tm_1550", "delta_eta_1550", "minV_te", "minV_tm"])
        for v in values:
            setattr(param, param_name, float(v))
            update_param_derived(param)

            print(f"\n[SWEEP] {param_name}={v:.4f}")
            summaries = {}
            for pol in pols:
                s = run_single(param, pol=pol, task_tag=f"sweep_{param_name}{v:.4f}", dry_run=dry_run)
                if s is not None:
                    summaries[pol] = s

            # Only write row if we actually ran (i.e., not dry_run)
            if summaries:
                xs.append(float(v))
                eta_te = eta_at_1550_from_summary(summaries["te"]) if "te" in summaries else np.nan
                eta_tm = eta_at_1550_from_summary(summaries["tm"]) if "tm" in summaries else np.nan
                d_eta = abs(eta_te - eta_tm) if np.isfinite(eta_te) and np.isfinite(eta_tm) else np.nan
                minV_te = summaries["te"]["min_V"] if "te" in summaries else np.nan
                minV_tm = summaries["tm"]["min_V"] if "tm" in summaries else np.nan
                eta_te_list.append(eta_te); eta_tm_list.append(eta_tm)
                minV_te_list.append(minV_te); minV_tm_list.append(minV_tm)
                w.writerow([f"{v:.6f}", f"{eta_te:.6f}", f"{eta_tm:.6f}", f"{d_eta:.6f}", f"{minV_te:.6f}", f"{minV_tm:.6f}"])

    # restore original
    setattr(param, param_name, original)
    update_param_derived(param)

    # make quick plot if data exist
    if len(xs) > 0:
        xs = np.array(xs)
        plt.figure()
        plt.plot(xs, eta_te_list, marker='o', label='TE @ 1550 nm')
        if len(eta_tm_list) == len(xs):
            plt.plot(xs, eta_tm_list, marker='o', label='TM @ 1550 nm')
        plt.axhline(0.5, ls="--")
        plt.xlabel(param_name.replace("_"," ") + " (µm)")
        plt.ylabel("Coupling ratio η at 1550 nm")
        plt.title(f"η(1550 nm) vs {param_name}")
        out = outdir / f"eta1550_vs_{param_name}.png"
        plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
        print(f"[SWEEP] saved CSV: {csv_path}")
        print(f"[SWEEP] saved plot: {out}")

if __name__ == "__main__":
    SWEEP_PARAM = "coupling_length"  # e.g., "coupling_length" or "coupling_gap"
    # define coarse grids for common parameters (units in µm)
    GRID = {
        "coupling_length": np.linspace(8.0, 20.0, 7),
        "coupling_gap":    np.round(np.linspace(0.24, 0.34, 6), 3),
    }

    results_cache = {}

    if DO_SWEEP:
        vals = GRID.get(SWEEP_PARAM, None)
        if vals is None:
            raise ValueError(f"No default grid defined for '{SWEEP_PARAM}'. Add it to GRID dict.")
        # Note: dry_run=False will actually simulate; True will only preflight
        sweep_one_param(SWEEP_PARAM, vals, pols=('te','tm'), dry_run=DRY_RUN)
    else:
        # single TE/TM run (no sweep)
        for pol in RUN_POLS:
            summary = run_single(param, pol=pol, task_tag="full", dry_run=DRY_RUN)
            if summary is not None:
                results_cache[pol] = summary

        # If both runs were executed, save an overlay comparison plot AND a combined CSV with Δη(λ) and band-averaged visibility
        if not DRY_RUN and all(p in results_cache for p in ("te", "tm")):
            # Overlay plot (existing behavior)
            plot_eta_overlay(results_cache["te"], results_cache["tm"], outdir="results")

            # Combine TE/TM spectra into a single CSV
            delta_eta = save_combined_te_tm_csv(results_cache["te"], results_cache["tm"], outdir="results")

            # Compute and save band-averaged visibilities
            compute_and_save_band_averaged_visibility(results_cache["te"], results_cache["tm"], outdir="results")

            # Plot polarization imbalance Δη(λ)
            plot_delta_eta(results_cache["te"]["lam"], delta_eta, outdir="results")
