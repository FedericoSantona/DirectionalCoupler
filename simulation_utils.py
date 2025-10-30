import tidy3d as td
import tidy3d.web as web
from tidy3d.web.api import webapi as api

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from building_utils import generate_object, make_source, make_monitors, resolve_mode_spec

try:
    from tidy3d.plugins.mode import ModeSolver as _ModeSolver
except Exception:
    _ModeSolver = None


def _get_monitor(sim_data, name):
    """Compatibility helper for accessing monitor data across tidy3d versions."""
    if hasattr(sim_data, "monitor_data") and name in sim_data.monitor_data:
        return sim_data.monitor_data[name]
    return sim_data[name]


def _monitor_freqs(mon):
    """Return frequency grid (Hz) from a mode monitor dataset, handling API changes."""
    if hasattr(mon, "freqs"):
        freqs = mon.freqs
    elif hasattr(mon, "monitor") and hasattr(mon.monitor, "freqs"):
        freqs = mon.monitor.freqs
    elif hasattr(mon, "amps"):
        amps = mon.amps
        if hasattr(amps, "coords") and "f" in amps.coords:
            freqs = amps.coords["f"].values
        elif hasattr(amps, "f"):
            freqs = amps.f
        else:
            freqs = None
    else:
        freqs = None
    if freqs is None:
        raise AttributeError("Could not extract frequency grid from monitor data.")
    return np.array(freqs, dtype=float)


def getObjectInArray(className, items):
    return list(filter(lambda item: isinstance(item, className), items))


def _mode_solver_field_fraction(mode_field):
    """Return (fx, fy) = normalized |Ex|^2 and |Ey|^2 fractions for a mode solver field."""
    try:
        if isinstance(mode_field, (list, tuple)) and len(mode_field) > 0:
            mode_field = mode_field[0]
        ex = None
        ey = None
        for attr in ("Ex", "ex", "E_x", "e_x"):
            if hasattr(mode_field, attr):
                ex = getattr(mode_field, attr)
                break
        for attr in ("Ey", "ey", "E_y", "e_y"):
            if hasattr(mode_field, attr):
                ey = getattr(mode_field, attr)
                break
        if ex is None or ey is None:
            return None, None
        ex = np.array(ex)
        ey = np.array(ey)
        ex_energy = float(np.sum(np.abs(ex) ** 2))
        ey_energy = float(np.sum(np.abs(ey) ** 2))
        total = ex_energy + ey_energy
        if total <= 0:
            return None, None
        return ex_energy / total, ey_energy / total
    except Exception:
        return None, None


def compute_mode_solver_diagnostics(sim, param, pol, lambda_eval=1.55):
    """Run a Tidy3D mode solver at the through monitor plane and return diagnostics."""
    diag = {}
    try:
        if _ModeSolver is None:
            raise AttributeError("ModeSolver plugin is unavailable in this tidy3d build")
        mode_spec = resolve_mode_spec(param, pol)
        plane_box = td.Box(
            center=[param.size_x / 2 - param.wl_0, 0.0, 0.0],
            size=[0.0, 4 * param.wg_width, 5 * param.wg_thick],
        )
        ms = _ModeSolver(
            simulation=sim,
            plane=plane_box,
            mode_spec=mode_spec,
            freqs=[td.C_0 / lambda_eval],
            fields=("Ey", "Ez"),
        )
        try:
            task_name = f"ModeDiag_{pol}_{lambda_eval:.3f}"
            path = f"./data/mode_solver_{pol}.hdf5"
            res = web.run(ms, task_name=task_name, path=path)
            try:
                sol = res.results()
            except AttributeError:
                sol = res
        except Exception as remote_exc:
            # fall back to local solver if remote execution is unavailable
            diag["warning"] = f"remote mode solver failed: {remote_exc}"
            sol = ms.solve()
        neff = getattr(sol, "n_eff", None)
        if neff is not None:
            neff_arr = np.array(neff)
            diag["n_eff"] = float(np.real(neff_arr.ravel()[0]))
        diag["mode_pol"] = getattr(mode_spec, "filter_pol", pol)
        ey_e, ez_e = _mode_solver_component_energies(sol, ("Ey","Ez"))
        if ey_e is not None and ez_e is not None:
            total = ey_e + ez_e + 1e-18
            diag["solver_fy"] = float((ey_e / total)[0, 0])
            diag["solver_fz"] = float((ez_e / total)[0, 0])
    except Exception as exc:
        diag["error"] = str(exc)
    return diag


def _mode_solver_component_energies(sol, comps=("Ey","Ez")):
    """Return per-component energies integrated over space -> arrays shape (f, mode)."""
    outs = []
    for comp in comps:
        try:
            arr = np.asarray(getattr(sol, comp))
        except Exception:
            return (None,) * len(comps)
        if arr.size == 0:
            return (None,) * len(comps)
        # expected dims ~ (f, Ny, Nz, mode, ...)
        # integrate over spatial dims (Ny, Nz) = axes 1,2 when present
        arr2 = np.abs(arr) ** 2
        if arr2.ndim >= 3:
            # sum over axes 1 and 2 if they exist
            axes = []
            if arr2.shape[1] > 1:
                axes.append(1)
            if arr2.ndim > 2 and arr2.shape[2] > 1:
                axes.append(2)
            if axes:
                eng = np.sum(arr2, axis=tuple(axes))
            else:
                eng = arr2
        else:
            eng = arr2
        # now eng dims ~ (f, mode, ...). Squeeze trailing singleton dims.
        eng = np.array(eng)
        while eng.ndim > 2:
            eng = eng[..., 0]
        outs.append(eng)
    return tuple(outs)


def pick_mode_index_at_source(sim, param, pol, lambda_um=1.55, n_modes=6):
    """Determine the best mode index to inject for the requested polarization."""
    if _ModeSolver is None:
        raise AttributeError("ModeSolver plugin is unavailable in this tidy3d build")

    pol_key = pol.lower()
    sources = [s for s in getattr(sim, "sources", []) if isinstance(s, td.ModeSource)]
    if not sources:
        raise ValueError("No ModeSource found in simulation for mode selection")
    source = sources[0]

    plane_box = td.Box(center=list(source.center), size=list(source.size))
    base_spec = getattr(source, "mode_spec", None) or resolve_mode_spec(param, pol_key)
    target_neff = getattr(base_spec, "target_neff", None)
    mode_spec = td.ModeSpec(num_modes=int(n_modes), filter_pol=None, target_neff=target_neff)

    solver = _ModeSolver(
        simulation=sim,
        plane=plane_box,
        mode_spec=mode_spec,
        freqs=[td.C_0 / lambda_um],
        direction=getattr(source, "direction", "+"),
        fields=("Ey", "Ez"),
    )

    sol = solver.solve()

    neff_arr = np.array(sol.n_eff)
    if neff_arr.ndim >= 2:
        neff_arr = np.real(neff_arr.reshape(neff_arr.shape[0], -1))[0]
    else:
        neff_arr = np.real(neff_arr.reshape(-1))

    ey_e, ez_e = _mode_solver_component_energies(sol, ("Ey","Ez"))
    if ey_e is None or ez_e is None:
        raise ValueError("Mode solver did not return field components for mode selection")

    # Convert to fractions per mode at first frequency
    ey_e = np.asarray(ey_e)
    ez_e = np.asarray(ez_e)
    # ensure shape (f, mode)
    while ey_e.ndim > 2:
        ey_e = ey_e[..., 0]
    while ez_e.ndim > 2:
        ez_e = ez_e[..., 0]
    ey0 = ey_e[0]
    ez0 = ez_e[0]
    total = ey0 + ez0 + 1e-18
    fy = ey0 / total
    fz = ez0 / total

    if neff_arr.shape[0] < fy.shape[0]:
        neff_arr = np.pad(neff_arr, (0, fy.shape[0] - neff_arr.shape[0]), mode='edge')

    if pol_key == "te":
        # TE: Ey-dominant on y–z cross-section
        candidates = np.where(fy > fz)[0]
        best = candidates[np.argmax(fy[candidates] - fz[candidates])] if candidates.size else int(np.argmax(neff_arr))
    elif pol_key == "tm":
        # TM: Ez-dominant on y–z cross-section
        candidates = np.where(fz > fy)[0]
        best = candidates[np.argmax(fz[candidates] - fy[candidates])] if candidates.size else int(np.argmax(neff_arr))
    else:
        raise ValueError(f"Unsupported polarization '{pol}'")

    return int(best)


def _field_component_array(field_data, component):
    """Best-effort extraction of a field component array from a FieldMonitor result."""
    candidates = (
        component,
        component.lower(),
        f"E{component[-1]}",
        f"E_{component[-1]}",
        f"e{component[-1]}",
        f"e_{component[-1]}",
    )
    for attr in candidates:
        if hasattr(field_data, attr):
            return np.array(getattr(field_data, attr))
    if hasattr(field_data, "fields"):
        fields = getattr(field_data, "fields")
        if isinstance(fields, dict) and component in fields:
            return np.array(fields[component])
    return None


def _field_energy_fraction(field_data):
    """Compute (fx, fy) from |Ex|^2 and |Ey|^2 integrals over the field monitor plane."""
    try:
        ex = _field_component_array(field_data, "Ex")
        ey = _field_component_array(field_data, "Ey")
        if ex is None or ey is None:
            return None, None
        ex_energy = float(np.sum(np.abs(np.asarray(ex)) ** 2))
        ey_energy = float(np.sum(np.abs(np.asarray(ey)) ** 2))
        total = ex_energy + ey_energy
        if total <= 0:
            return None, None
        return ex_energy / total, ey_energy / total
    except Exception:
        return None, None


def compute_field_monitor_polarization(sim_data, pol):
    """Return polarization fractions (fx, fy) from the recorded field monitor."""
    try:
        field_mon = _get_monitor(sim_data, f"field_xy_{pol}")
    except Exception:
        field_mon = None
    if field_mon is None:
        return None, None
    return _field_energy_fraction(field_mon)

def _mode_spec_from_monitor_data(mon):
    """Best-effort: fetch the ModeSpec used by a ModeMonitor result."""
    try:
        if hasattr(mon, "monitor") and hasattr(mon.monitor, "mode_spec"):
            return mon.monitor.mode_spec
    except Exception:
        pass
    try:
        if hasattr(mon, "mode_spec"):
            return mon.mode_spec
    except Exception:
        pass
    return None

def _extract_neff_at_index(mon, idx):
    """Best-effort extraction of n_eff at frequency index idx from a mode monitor dataset.

    Tries several tidy3d data layouts. Returns float or None if unavailable.
    """
    # direct attributes found in some versions
    for attr in ("n_eff", "neff", "mode_neff"):
        try:
            if hasattr(mon, attr):
                arr = getattr(mon, attr)
                a = np.array(arr)
                if a.ndim == 0:
                    return float(a)
                if a.ndim == 1:
                    return float(a[min(idx, a.shape[0]-1)])
                if a.ndim >= 2:
                    # assume (freq, mode, ...); pick first mode
                    fdim = 0 if a.shape[0] >= a.shape[1] else 1
                    if fdim == 0:
                        return float(a[min(idx, a.shape[0]-1), 0])
                    else:
                        return float(a[0, min(idx, a.shape[1]-1)])
        except Exception:
            pass

    # nested containers that may hold modes with attributes
    for container_name in ("mode_data", "modes"):
        try:
            container = getattr(mon, container_name, None)
            if container is None:
                continue
            # list-like of modes with .n_eff
            if isinstance(container, (list, tuple)) and len(container) > 0:
                m0 = container[0]
                if hasattr(m0, "n_eff"):
                    val = getattr(m0, "n_eff")
                    return float(val) if np.isscalar(val) else float(np.array(val).squeeze())
            # object with array attribute
            for attr in ("n_eff", "neff"):
                if hasattr(container, attr):
                    arr = getattr(container, attr)
                    a = np.array(arr)
                    if a.ndim == 0:
                        return float(a)
                    if a.ndim == 1:
                        return float(a[min(idx, a.shape[0]-1)])
                    if a.ndim >= 2:
                        return float(a[min(idx, a.shape[0]-1), 0])
        except Exception:
            pass

    # xarray DataArray route via amplitudes metadata/coords
    try:
        if hasattr(mon, "amps"):
            amps = mon.amps
            # coords
            try:
                if hasattr(amps, "coords") and "n_eff" in amps.coords:
                    a = np.array(amps.coords["n_eff"])  # shape (freq, mode)
                    if a.ndim == 1:
                        return float(a[min(idx, a.shape[0]-1)])
                    if a.ndim >= 2:
                        return float(a[min(idx, a.shape[0]-1), 0])
            except Exception:
                pass
            # attrs
            try:
                if hasattr(amps, "attrs") and isinstance(amps.attrs, dict) and "n_eff" in amps.attrs:
                    a = np.array(amps.attrs["n_eff"])  # shape (freq, mode) in some versions
                    if a.ndim == 0:
                        return float(a)
                    if a.ndim == 1:
                        return float(a[min(idx, a.shape[0]-1)])
                    if a.ndim >= 2:
                        return float(a[min(idx, a.shape[0]-1), 0])
            except Exception:
                pass
    except Exception:
        pass

    return None

def server_estimate(sim, task_name="estimate_probe"):
    """Upload-only to server and request official cost estimate; then delete the temp task."""
    # Upload (does not start the simulation)
    task_id = api.upload(simulation=sim, task_name=task_name, verbose=False)
    # Query server-side estimate
    est = api.estimate_cost(task_id)

    # Handle both float and dict-style returns gracefully
    if isinstance(est, (float, int)):
        print(f"[estimate] Maximum FlexCredit cost: {est:.3f}")
    elif isinstance(est, dict):
        print(f"[estimate] credits={est.get('credits', 'N/A')}, runtime={est.get('runtime', 'N/A')}, mem={est.get('memory', 'N/A')}")
    else:
        try:
            credits = getattr(est, "credits", None)
            runtime = getattr(est, "runtime", None)
            memory = getattr(est, "memory", None)
            print(f"[estimate] credits={credits}, runtime={runtime}, mem={memory}")
        except Exception:
            print(f"[estimate] {est}")

    # Clean up server-side placeholder task
    try:
        api.delete(task_id)
    except Exception:
        pass

    return est

def build_sim(param, pol='te'):
    # build geometry
    structures = [*getObjectInArray(td.Structure, generate_object(param))]
    # sources and monitors
    src = make_source(param, pol=pol)
    mon_s31, mon_s41, mon_field = make_monitors(param, pol=pol)
    steps = getattr(param, "grid_steps_per_wvl", 18)
    if isinstance(steps, (tuple, list)) and len(steps) >= 3:
        step_x, step_y, step_z = steps[:3]
    else:
        step_x = step_y = step_z = steps
    runtime = getattr(param, "run_time", 5e-11)
    shutoff = getattr(param, "shutoff", 1e-5)
    sim = td.Simulation(
        size=[param.size_x, param.size_y, param.size_z],
        symmetry=[0, 0, 1],
        grid_spec=td.GridSpec(
            grid_x=td.AutoGrid(min_steps_per_wvl=step_x),
            grid_y=td.AutoGrid(min_steps_per_wvl=step_y),
            grid_z=td.AutoGrid(min_steps_per_wvl=step_z),
            wavelength=param.wl_0
        ),
        version='2.9.1',
        subpixel=td.SubpixelSpec(pec=td.PECConformal()),
        run_time=runtime,
        shutoff=shutoff,
        medium=param.medium.SiO2,
        sources=[src],
        monitors=[mon_s31, mon_s41, mon_field],
        structures=structures
    )
    return sim

def preflight(sim, do_server_estimate=True):
    """Version-agnostic validation/summary + official server-side cost estimate (optional)."""
    # --- validate ---
    try:
        sim.validate()
    except TypeError:
        # older pydantic-style classmethod
        td.Simulation.validate(sim)
    except AttributeError:
        print("[preflight] validate not available.")

    # --- summary (best-effort) ---
    try:
        sim.summary()
    except Exception:
        print("[preflight] summary not available.")
        #print(sim)

    # --- official server-side estimate ---
    if do_server_estimate:
        try:
            server_estimate(sim)
        except Exception as e:
            print(f"[estimate] failed: {e}")
    else:
        print("[estimate] skipped.")

def hom_visibility(eta):
    """Normalized HOM visibility in 0–1 range.

    V(eta) = 0.5 * 4*eta*(1-eta)/(eta^2 + (1-eta)^2)
    The 0.5 factor ensures V<=1 with peak 1 at eta=0.5.
    """
    return 0.5 * 4 * eta * (1 - eta) / (eta**2 + (1 - eta)**2 + 1e-18)

def extract_eta(sim_data, pol):
    # robust access to powers from mode monitors
    mon_t = _get_monitor(sim_data, f"monitor_s31_{pol}")
    mon_c = _get_monitor(sim_data, f"monitor_s41_{pol}")
    def _p(m):
        if hasattr(m, "powers"):
            data = np.array(m.powers)
        elif hasattr(m, "mode_powers"):
            data = np.array(m.mode_powers)
        elif hasattr(m, "amps"):
            amps = m.amps
            try:
                amps_dir = amps.sel(direction="+")
            except Exception:
                amps_dir = amps[0]
            data = np.sum(np.abs(np.array(amps_dir))**2, axis=-1)
        else:
            raise AttributeError("Unsupported monitor data structure.")
        return np.array(data).squeeze()
    P_t = _p(mon_t)
    P_c = _p(mon_c)
    eta = P_c / (P_c + P_t + 1e-18)
    return eta



# --- helpers for reporting & plots ---
def _lambda_from_monitor(sim_data, pol):
    """Return wavelength grid (µm) from the through monitor of a given polarization."""
    mon = _get_monitor(sim_data, f"monitor_s31_{pol}")
    freqs = _monitor_freqs(mon)
    return td.C_0 / freqs

def summarize_and_save(sim_data, pol, outdir="results", mode_solver_diag=None):
    """
    Extract eta(λ) and V(λ) for a polarization, print clear summary,
    save CSV and per-pol plots to disk. Returns a dict of key metrics.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    lam = _lambda_from_monitor(sim_data, pol)
    eta = extract_eta(sim_data, pol)
    V = hom_visibility(eta)

    # pick 1550 nm index
    k = int(np.argmin(np.abs(lam - 1.55)))
    eta_1550 = float(eta[k])
    V_1550 = float(V[k])

    # band stats over C-band grid used in the sim (assumed 1.50–1.60 µm)
    dev_from_50 = np.abs(eta - 0.5)
    max_dev = float(np.max(dev_from_50))
    min_V = float(np.min(V))

    # ---- diagnostics: field polarization fractions ----
    field_fx, field_fy = compute_field_monitor_polarization(sim_data, pol)

    # ---- print a concise summary ----
    print(f"\n[{pol.upper()}] SUMMARY")
    print(f"  η(1550 nm)     : {eta_1550:.3f}")
    print(f"  V(1550 nm)     : {V_1550:.3f}")
    print(f"  max |η-0.5|    : {max_dev:.3f}  (over {lam[0]:.3f}–{lam[-1]:.3f} µm)")
    print(f"  min V(λ)       : {min_V:.3f}")
    if mode_solver_diag:
        neff_val = mode_solver_diag.get("n_eff")
        pol_guess = mode_solver_diag.get("mode_pol")
        target_pol = pol_guess.upper() if isinstance(pol_guess, str) else "?"
        if isinstance(neff_val, (float, int)):
            print(f"  mode solver    : n_eff@1550≈{neff_val:.6f} (target pol={target_pol})")
        if "solver_fy" in mode_solver_diag and "solver_fz" in mode_solver_diag:
            fy = mode_solver_diag["solver_fy"]
            fz = mode_solver_diag["solver_fz"]
            print(f"                    solver field fractions Ey={fy:.3f}, Ez={fz:.3f}")
        if "warning" in mode_solver_diag:
            print(f"  mode solver    : note -> {mode_solver_diag['warning']}")
        if "error" in mode_solver_diag:
            print(f"  mode solver    : warning -> {mode_solver_diag['error']}")

    if field_fx is not None and field_fy is not None:
        print(f"  field monitor  : fx={field_fx:.3f}, fy={field_fy:.3f} (|E|^2 fractions)")

    # ---- save CSV ----
    csv_path = Path(outdir) / f"spectra_{pol}.csv"
    np.savetxt(csv_path, np.c_[lam, eta, V], delimiter=",",
               header="lambda_um,eta,V", comments="")
    print(f"  saved CSV      : {csv_path}")

    # ---- plots: η(λ) and V(λ) ----
    plt.figure()
    plt.plot(lam, eta)
    plt.axhline(0.5, ls="--")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Coupling ratio η")
    plt.title(f"Coupling ratio vs wavelength ({pol.upper()})")
    fig1 = Path(outdir) / f"eta_vs_lambda_{pol}.png"
    plt.tight_layout(); plt.savefig(fig1, dpi=200); plt.close()

    plt.figure()
    plt.plot(lam, V)
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("HOM visibility V (0–1)")
    plt.title(f"HOM visibility vs wavelength ({pol.upper()})")
    fig2 = Path(outdir) / f"visibility_vs_lambda_{pol}.png"
    plt.tight_layout(); plt.savefig(fig2, dpi=200); plt.close()

    print(f"  saved plots    : {fig1.name}, {fig2.name}")

    return {
        "lam": lam, "eta": eta, "V": V,
        "eta_1550": eta_1550, "V_1550": V_1550,
        "max_dev": max_dev, "min_V": min_V,
        "csv": str(csv_path), "fig_eta": str(fig1), "fig_V": str(fig2),
        "field_fx": field_fx, "field_fy": field_fy,
        "mode_solver": mode_solver_diag,
    }

def plot_eta_overlay(results_te, results_tm, outdir="results"):
    """Overlay η_TE(λ) and η_TM(λ) for quick polarization comparison."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    lam = results_te["lam"]
    plt.figure()
    plt.plot(lam, results_te["eta"], label="TE")
    plt.plot(lam, results_tm["eta"], label="TM")
    plt.axhline(0.5, ls="--")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Coupling ratio η")
    plt.title("Coupling ratio vs wavelength (TE vs TM)")
    plt.legend()
    out = Path(outdir) / "eta_vs_lambda_TE_TM.png"
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
    print(f"\n[OVERLAY] saved plot: {out}")

def save_combined_te_tm_csv(results_te, results_tm, outdir="results"):
    """
    Combine TE/TM spectra into a single CSV: lambda_um, eta_TE, eta_TM, delta_eta, V_TE, V_TM.
    Returns delta_eta array for use in other functions.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    lam = results_te["lam"]
    if lam.shape != results_tm["lam"].shape or np.max(np.abs(lam - results_tm["lam"])) > 1e-9:
        raise ValueError("TE and TM wavelength grids do not match; cannot create combined CSV.")

    delta_eta = np.abs(results_te["eta"] - results_tm["eta"])  # Δη(λ)
    combined = np.c_[lam, results_te["eta"], results_tm["eta"], delta_eta, results_te["V"], results_tm["V"]]
    header = "lambda_um,eta_TE,eta_TM,delta_eta,V_TE,V_TM"
    csv_path = Path(outdir) / "combined_TE_TM.csv"
    np.savetxt(csv_path, combined, delimiter=",", header=header, comments="")
    print("[COMBINE] saved combined TE/TM CSV -> " + str(csv_path))
    # quick sanity: are TE/TM essentially identical?
    try:
        max_d = float(np.max(delta_eta))
        if max_d < 1e-6:
            print("[WARN] η_TE and η_TM are nearly identical (Δη < 1e-6).\n"
                  "       Verify mode separation: check ModeSpec filter_pol and n_eff logs.")
    except Exception:
        pass
    return delta_eta

def compute_and_save_band_averaged_visibility(results_te, results_tm, outdir="results"):
    """
    Compute band-averaged visibilities over the telecom C-band (1530–1565 nm) if available;
    otherwise average over the simulated band. Save summary to text file.
    Returns dict with Vbar_TE, Vbar_TM, and Vbar_avg.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    lam = results_te["lam"]
    mask = (lam >= 1.530) & (lam <= 1.565)
    if not np.any(mask):
        mask = slice(None)
    Vbar_te = float(np.mean(results_te["V"][mask]))
    Vbar_tm = float(np.mean(results_tm["V"][mask]))
    Vbar_avg = 0.5 * (Vbar_te + Vbar_tm)

    # Save a small text summary with V̄_TE, V̄_TM, and V̄_avg
    summary_path = Path(outdir) / "combined_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Vbar_TE={Vbar_te:.6f}\n")
        f.write(f"Vbar_TM={Vbar_tm:.6f}\n")
        f.write(f"Vbar_avg={Vbar_avg:.6f}\n")
    print(f"[COMBINE] V̄_TE={Vbar_te:.4f}, V̄_TM={Vbar_tm:.4f}, V̄_avg={Vbar_avg:.4f}")
    return {"Vbar_TE": Vbar_te, "Vbar_TM": Vbar_tm, "Vbar_avg": Vbar_avg}

def plot_delta_eta(lam, delta_eta, outdir="results"):
    """Plot polarization imbalance Δη(λ) vs wavelength."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(lam, delta_eta)
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Δη = |η_TE - η_TM|")
    plt.title("Polarization imbalance Δη vs wavelength")
    out = Path(outdir) / "delta_eta_vs_lambda.png"
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
    print("[COMBINE] saved plot: " + str(out))
