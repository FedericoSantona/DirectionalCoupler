import tidy3d as td
import numpy as np
from simulation_utils import _ModeSolver
from building_utils import resolve_mode_spec
from pathlib import Path
import pickle

# Try to import scipy.optimize for balance policy optimization
try:
    from scipy.optimize import minimize_scalar
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_COUPLING_CACHE_REV = 2

# Cache for coupling length computations
_coupling_cache_per_pol = {}  # Level 1: per-polarization supermode data
_coupling_cache_blended = {}  # Level 2: blended L_c values


def _eta_from_cmt(kappa, Delta, Omega, L):
    """
    Compute coupling ratio η(L) from CMT parameters.
    
    Formula: η(L) = Cmax * sin²(Ω*L)
    where Cmax = κ²/(κ²+Δ²) and Ω = sqrt(κ²+Δ²)
    
    Args:
        kappa: Coupling coefficient κ
        Delta: Detuning parameter Δ
        Omega: Rabi frequency Ω = sqrt(κ²+Δ²)
        L: Coupling length in µm
    
    Returns:
        Coupling ratio η(L) in [0, Cmax]
    """
    kappa_sq = float(kappa) ** 2
    Delta_sq = float(Delta) ** 2
    den = kappa_sq + Delta_sq
    if den <= 0:
        return 0.0
    Cmax = kappa_sq / den
    Omega_L = float(Omega) * float(L)
    sin_sq = np.sin(Omega_L) ** 2
    eta = Cmax * sin_sq
    return float(np.clip(eta, 0.0, 1.0))


def _optimize_Lc_balance(kappa_dict, Delta_dict, Omega_dict, min_len, max_len):
    """
    Optimize L_c to minimize |η_TE(L) - 0.5| + |η_TM(L) - 0.5|.
    
    Uses CMT predictions from supermode (κ, Δ) parameters.
    Constrains search to first few lobes to prefer shortest workable L.
    
    Args:
        kappa_dict: Dict with 'te' and 'tm' keys containing κ values
        Delta_dict: Dict with 'te' and 'tm' keys containing Δ values
        Omega_dict: Dict with 'te' and 'tm' keys containing Ω values
        min_len: Minimum coupling length bound in µm (safety limit)
        max_len: Maximum coupling length bound in µm (safety limit)
    
    Returns:
        Optimal L_c in µm, or None if optimization fails
    """
    # Check that we have valid parameters for both polarizations
    for pol in ("te", "tm"):
        if pol not in kappa_dict or pol not in Delta_dict or pol not in Omega_dict:
            return None
        kappa = kappa_dict[pol]
        Delta = Delta_dict[pol]
        Omega = Omega_dict[pol]
        
        # Check for invalid parameters
        if not np.isfinite(kappa) or not np.isfinite(Delta) or not np.isfinite(Omega):
            return None
        if Omega <= 0:
            return None
    
    # Compute first-0.5 lengths for each polarization: L_50_first = π/(4*Ω)
    L_50_first_te = np.pi / (4.0 * Omega_dict["te"])
    L_50_first_tm = np.pi / (4.0 * Omega_dict["tm"])
    
    # Constrain search window to first few lobes
    # L_min = 0.5 * min(L_50_TE_first, L_50_TM_first)
    # L_max = 2.0 * max(L_50_TE_first, L_50_TM_first)
    search_min = 0.5 * min(L_50_first_te, L_50_first_tm)
    search_max = 2.0 * max(L_50_first_te, L_50_first_tm)
    
    # Clip to safety bounds
    search_min = max(search_min, min_len)
    search_max = min(search_max, max_len)
    
    if search_min >= search_max:
        # Fallback: use safety bounds if computed window is invalid
        search_min = min_len
        search_max = max_len
    
    # Objective function: |η_TE(L) - 0.5| + |η_TM(L) - 0.5| + tiny length regularizer
    def objective(L):
        eta_te = _eta_from_cmt(kappa_dict["te"], Delta_dict["te"], Omega_dict["te"], L)
        eta_tm = _eta_from_cmt(kappa_dict["tm"], Delta_dict["tm"], Omega_dict["tm"], L)
        cost = abs(eta_te - 0.5) + abs(eta_tm - 0.5) + 1e-3 * (L / max_len)
        return cost
    
    # Try scipy optimization if available
    if _HAS_SCIPY:
        try:
            result = minimize_scalar(
                objective,
                bounds=(search_min, search_max),
                method='bounded',
                options={'xatol': 1e-6}
            )
            if result.success:
                return float(result.x)
        except Exception:
            pass
    
    # Fallback: grid search
    n_points = 1000
    L_grid = np.linspace(search_min, search_max, n_points)
    costs = [objective(L) for L in L_grid]
    best_idx = np.argmin(costs)
    return float(L_grid[best_idx])






def compute_Lc(
    param,
    lambda0=None,
    trim_factor=0.075,
    cache_dir="data",
    blend_policy="median",
    length_bounds=None,
):
    """
    Compute coupling length L_c for directional coupler (symmetric or asymmetric).
    Gracefully handles symmetric case (delta_w=0) where Δ→0 and math reduces to symmetric formula.
    
    Args:
        param: Parameter namespace with wg_width, delta_w, coupling_gap, wg_thick, medium.SiN, medium.SiO2.
        lambda0: Wavelength in µm (default: param.wl_0)
        trim_factor: Empirical trim factor for bends/transitions (default: 0.075)
        cache_dir: Directory for cache file (default: "data")
        blend_policy: How to blend TE/TM results: "median" (default), "te", "tm", or "balance".
                       "balance" minimizes |η_TE-0.5|+|η_TM-0.5| using CMT predictions.
        length_bounds: Optional tuple (min_len, max_len) to clip L_c; defaults to (3, 50) µm
    
    Returns:
        L_c in µm (clipped to provided safety bounds)
    """
    if _ModeSolver is None:
        raise AttributeError("ModeSolver plugin is unavailable in this tidy3d build")
    
    if lambda0 is None:
        lambda0 = getattr(param, 'lambda_single', param.wl_0)
    lambda0 = float(lambda0)
    
    w1, w2 = _adc__get_w1_w2(param)
    
    # Symmetric detection: if widths are nearly equal, log but use same math (Δ→0 naturally)
    epsilon_w = 0.002  # 2 nm threshold for symmetric detection
    if abs(w1 - w2) < epsilon_w:
        print(f"[L_c (ADC)] symmetric detected (|w1-w2|={abs(w1-w2)*1000:.2f}nm < {epsilon_w*1000:.0f}nm)")
    coupling_gap = float(param.coupling_gap)
    wg_thick = float(param.wg_thick)
    eps_SiN = float(param.medium.SiN.permittivity)
    eps_SiO2 = float(param.medium.SiO2.permittivity)
    
    # Resolve safety bounds for L_c clipping
    if length_bounds is None:
        min_len, max_len = 3.0, 50.0
    else:
        if len(length_bounds) != 2:
            raise ValueError("length_bounds must be a tuple (min_len, max_len)")
        min_len, max_len = map(float, length_bounds)
        if min_len <= 0 or max_len <= 0 or min_len >= max_len:
            raise ValueError("length_bounds must satisfy 0 < min_len < max_len")
    
    pols = ("te", "tm")
    L_50_dict = {}
    kappa_dict = {}
    Delta_dict = {}
    Omega_dict = {}
    Cmax_dict = {}
    n1_dict = {}
    n2_dict = {}
    
    for pol in pols:
        per_pol_key = (_COUPLING_CACHE_REV, "adc", w1, w2, coupling_gap, wg_thick, lambda0, eps_SiN, eps_SiO2, pol)
        if per_pol_key in _coupling_cache_per_pol:
            cached = _coupling_cache_per_pol[per_pol_key]
            n1 = cached["n1"]
            n2 = cached["n2"]
            S = cached["beta_split"]
            Delta = cached["Delta"]
            kappa = cached["kappa"]
            L_50 = cached["L_50"]
            Omega = S / 2  # Recompute Omega from cached S
            # Recompute Cmax on cache hit (was not stored in older cache revs)
            kappa_sq_local = float(kappa) ** 2
            den_local = kappa_sq_local + float(Delta) ** 2
            Cmax = (kappa_sq_local / den_local) if den_local > 0 else 0.0
        else:
            # Isolated arms
            n1 = _adc__isolated_neff(w1, param, pol, lambda0)
            n2 = _adc__isolated_neff(w2, param, pol, lambda0)
            
            k0 = 2 * np.pi / lambda0
            beta1 = k0 * n1
            beta2 = k0 * n2
            Delta = (beta1 - beta2) / 2
            
            # Coupled pair simulation
            # Build two waveguides at y positions + and - (gap + widths)/2
            # Correct center-to-center spacing between waveguides
            d_center = coupling_gap + 0.5 * (w1 + w2)
            y_upper = +0.5 * d_center
            y_lower = -0.5 * d_center
            t = wg_thick
            
            domain_x = 2.0 * max(w1, w2)
            pad_y = max(1.0 * lambda0, 2.0 * max(w1, w2))
            pad_z = max(0.5 * lambda0, 2.0 * t)
            domain_y = (coupling_gap + w1 + w2) + 2.0 * pad_y
            domain_z = t + 2.0 * pad_z
            
            upper_wg = td.Structure(
                geometry=td.Box(size=(w1, w1, t), center=(0, y_upper, 0)),
                medium=param.medium.SiN,
                name="upper_wg"
            )
            lower_wg = td.Structure(
                geometry=td.Box(size=(w2, w2, t), center=(0, y_lower, 0)),
                medium=param.medium.SiN,
                name="lower_wg"
            )
            
            sim_cross = td.Simulation(
                size=(domain_x, domain_y, domain_z),
                medium=param.medium.SiO2,
                structures=[upper_wg, lower_wg],
                grid_spec=td.GridSpec(
                    grid_x=td.AutoGrid(min_steps_per_wvl=6),
                    grid_y=td.AutoGrid(min_steps_per_wvl=6),
                    grid_z=td.AutoGrid(min_steps_per_wvl=10),
                    wavelength=lambda0
                ),
                boundary_spec=td.BoundarySpec(
                    x=td.Boundary.pml(),
                    y=td.Boundary.pml(),
                    z=td.Boundary.pml(),
                ),
                run_time=1e-12
            )
            
            base_spec = resolve_mode_spec(param, pol)
            mode_spec = td.ModeSpec(
                num_modes=2,
                filter_pol=getattr(base_spec, "filter_pol", pol),
                target_neff=getattr(base_spec, "target_neff", None),
            )
            plane = td.Box(center=(0, 0, 0), size=(0, 0.95 * domain_y, 0.95 * domain_z))
            
            try:
                solver = _ModeSolver(
                    simulation=sim_cross,
                    plane=plane,
                    mode_spec=mode_spec,
                    freqs=[td.C_0 / lambda0],
                    direction="+",
                    fields=("Ex", "Ey", "Ez"),
                )
                sol = solver.solve()
                n_eff_arr = np.array(sol.n_eff)
                if n_eff_arr.ndim >= 2:
                    n_eff_arr = np.real(n_eff_arr.reshape(-1))[:2]
                else:
                    n_eff_arr = np.real(n_eff_arr[:2])
                if len(n_eff_arr) < 2:
                    raise ValueError(f"Mode solver returned <2 modes for {pol} in ADC")
                
                # Sort so beta_plus >= beta_minus (n_eff sorted descending)
                sorted_indices = np.argsort(n_eff_arr)[::-1]
                n_plus = float(n_eff_arr[sorted_indices[0]])
                n_minus = float(n_eff_arr[sorted_indices[1]])
                
                S = k0 * (n_plus - n_minus)  # supermode splitting in beta
            except Exception as e:
                print(f"[L_c (ADC) WARNING] Mode solver failed for {pol}: {e}")
                # fallback safe defaults
                S = 0.0
            
            Omega = S / 2
            kappa_sq = max(Omega**2 - Delta**2, 0.0)
            kappa = np.sqrt(kappa_sq)
            Cmax = kappa_sq / (kappa_sq + Delta**2) if (kappa_sq + Delta**2) > 0 else 0.0
            
            if kappa <= abs(Delta):
                # exact 50:50 not reachable
                L_50 = np.inf
                print(f"[L_c (ADC) WARNING] pol={pol.upper()}: κ={kappa:.4e} ≤ |Δ|={abs(Delta):.4e}, 50:50 coupling not reachable (Cmax={Cmax:.3f})")
            else:
                # Log reachability info
                print(f"[L_c (ADC)] pol={pol.upper()}: κ={kappa:.4e} > |Δ|={abs(Delta):.4e}, 50:50 reachable (Cmax={Cmax:.3f})")
                ratio = 0.5 * (1.0 + (Delta / kappa)**2)
                ratio = min(max(ratio, 0.0), 1.0)
                L_50 = (1.0 / Omega) * np.arcsin(np.sqrt(ratio))
            
            _coupling_cache_per_pol[per_pol_key] = {
                "n1": n1,
                "n2": n2,
                "beta_split": S,
                "Delta": Delta,
                "kappa": kappa,
                "Cmax": Cmax,
                "L_50": L_50,
            }
            _save_coupling_cache(cache_dir)
        
        # Ensure Cmax is defined (covers both cache and fresh-solve paths)
        if "Cmax" in locals():
            Cmax_local = Cmax
        else:
            kappa_sq_local = float(kappa) ** 2
            den_local = kappa_sq_local + float(Delta) ** 2
            Cmax_local = (kappa_sq_local / den_local) if den_local > 0 else 0.0

        n1_dict[pol] = n1
        n2_dict[pol] = n2
        Delta_dict[pol] = Delta
        kappa_dict[pol] = kappa
        Omega_dict[pol] = Omega
        Cmax_dict[pol] = Cmax_local
        L_50_dict[pol] = L_50
    
    # Blend according to policy
    L_c_raw = None
    if blend_policy == "median":
        finite_vals = [v for v in L_50_dict.values() if np.isfinite(v)]
        if len(finite_vals) == 0:
            # Both infinite => return upper bound clipped with warning
            L_c_raw = max_len
            print(f"[L_c (ADC) WARNING] Both TE and TM L_50 are infinite; returning upper bound {max_len:.1f} µm")
        elif len(finite_vals) == 1:
            L_c_raw = finite_vals[0]
        else:
            L_c_raw = np.median(finite_vals)
    elif blend_policy in ("te", "tm"):
        L_c_raw = L_50_dict.get(blend_policy, np.inf)
        if not np.isfinite(L_c_raw):
            print(f"[L_c (ADC) WARNING] Selected polarization {blend_policy.upper()} has infinite L_50; check coupling")
    elif blend_policy == "balance":
        # Optimize L_c to minimize |η_TE(L) - 0.5| + |η_TM(L) - 0.5|
        try:
            L_c_raw = _optimize_Lc_balance(kappa_dict, Delta_dict, Omega_dict, min_len, max_len)
            if L_c_raw is None:
                # Optimization failed, fall back to median
                print(f"[L_c (ADC) WARNING] Balance optimization failed, falling back to median policy")
                finite_vals = [v for v in L_50_dict.values() if np.isfinite(v)]
                if len(finite_vals) == 0:
                    L_c_raw = max_len
                elif len(finite_vals) == 1:
                    L_c_raw = finite_vals[0]
                else:
                    L_c_raw = np.median(finite_vals)
        except Exception as e:
            print(f"[L_c (ADC) WARNING] Balance optimization error: {e}, falling back to median policy")
            finite_vals = [v for v in L_50_dict.values() if np.isfinite(v)]
            if len(finite_vals) == 0:
                L_c_raw = max_len
            elif len(finite_vals) == 1:
                L_c_raw = finite_vals[0]
            else:
                L_c_raw = np.median(finite_vals)
    else:
        raise ValueError(f"Invalid blend_policy: {blend_policy}")
    
    # Apply trim factor and clip
    L_c_raw_trimmed = L_c_raw * (1 + trim_factor)
    L_c = float(np.clip(L_c_raw_trimmed, min_len, max_len))
    if L_c != L_c_raw_trimmed:
        print(f"[L_c (ADC) WARNING] L_c clipped from {L_c_raw_trimmed:.3f} to {L_c:.3f} µm (bounds [{min_len}, {max_len}])")
    
    # Store in blended cache
    blended_key = (
        _COUPLING_CACHE_REV,
        "adc",
        w1,
        w2,
        coupling_gap,
        wg_thick,
        lambda0,
        eps_SiN,
        eps_SiO2,
        trim_factor,
        blend_policy,
        min_len,
        max_len,
    )
    _coupling_cache_blended[blended_key] = L_c
    _save_coupling_cache(cache_dir)
    
    # Log summary line
    summary_parts = []
    for pol in pols:
        n1 = n1_dict[pol]
        n2 = n2_dict[pol]
        Delta = Delta_dict[pol]
        kappa = kappa_dict[pol]
        Cmax = Cmax_dict[pol]
        L50 = L_50_dict[pol]
        L50_str = f"{L50:.3f}" if np.isfinite(L50) else "INF"
        cmax_str = f"{Cmax:.3f}" if Cmax >= 0 else "N/A"
        summary_parts.append(f"pol={pol.upper()}: n1={n1:.4f}, n2={n2:.4f}, Δ={Delta:.4e}, κ={kappa:.4e}, Cmax={cmax_str}, L50={L50_str}")
    print(f"[L_c (ADC) derived] {'; '.join(summary_parts)}; policy={blend_policy}, trim={trim_factor:.1%}, L_c={L_c:.3f}µm")
    
    return L_c



def _adc__get_w1_w2(param):
    """
    Determine arm widths (w1, w2) for asymmetric directional coupler from param.
    Uses delta_w approach: w1 = wg_width + delta_w/2, w2 = wg_width - delta_w/2.
    
    Raises ValueError if wg_width or delta_w not available.
    """
    if not hasattr(param, 'wg_width'):
        raise ValueError("Asymmetric coupling length computation requires wg_width parameter")
    if not hasattr(param, 'delta_w'):
        raise ValueError("Asymmetric coupling length computation requires delta_w parameter")
    
    w_base = float(param.wg_width)
    delta_w = float(param.delta_w)
    w1 = w_base + delta_w / 2
    w2 = w_base - delta_w / 2
    return w1, w2


def _adc__isolated_neff(width, param, pol, lambda0):
    """
    Compute isolated waveguide effective index n_eff for given width, polarization, and wavelength.
    Single waveguide in SiO2 cladding.
    """
    wg_thick = float(param.wg_thick)
    eps_SiN = param.medium.SiN
    eps_SiO2 = param.medium.SiO2
    
    # Domain size chosen to ensure decay before PML
    pad_y = max(1.0 * lambda0, 2.0 * width)
    pad_z = max(0.5 * lambda0, 2.0 * wg_thick)
    domain_x = width * 2.0
    domain_y = width + 2.0 * pad_y
    domain_z = wg_thick + 2.0 * pad_z
    
    wg = td.Structure(
        geometry=td.Box(size=(width, width, wg_thick), center=(0, 0, 0)),
        medium=eps_SiN,
        name="isolated_wg"
    )
    
    sim_iso = td.Simulation(
        size=(domain_x, domain_y, domain_z),
        medium=eps_SiO2,
        structures=[wg],
        grid_spec=td.GridSpec(
            grid_x=td.AutoGrid(min_steps_per_wvl=6),
            grid_y=td.AutoGrid(min_steps_per_wvl=6),
            grid_z=td.AutoGrid(min_steps_per_wvl=10),
            wavelength=lambda0
        ),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary.pml(),
        ),
        run_time=1e-12
    )
    
    base_spec = resolve_mode_spec(param, pol)
    mode_spec = td.ModeSpec(
        num_modes=1,
        filter_pol=getattr(base_spec, "filter_pol", pol),
        target_neff=getattr(base_spec, "target_neff", None),
    )
    plane = td.Box(center=(0, 0, 0), size=(0, 0.95 * domain_y, 0.95 * domain_z))
    
    try:
        solver = _ModeSolver(
            simulation=sim_iso,
            plane=plane,
            mode_spec=mode_spec,
            freqs=[td.C_0 / lambda0],
            direction="+",
            fields=("Ex", "Ey", "Ez"),
        )
        sol = solver.solve()
        n_eff_arr = np.array(sol.n_eff)
        n_eff = float(np.real(n_eff_arr[0]))
    except Exception as e:
        print(f"[L_c (ADC) WARNING] Isolated mode solver failed for width={width}, pol={pol}: {e}")
        n_eff = 1.5  # fallback guess
    
    return n_eff


def _load_coupling_cache(cache_dir="data"):
    """Load coupling length cache from disk."""
    cache_path = Path(cache_dir) / "coupling_length_cache.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                return data.get("per_pol", {}), data.get("blended", {})
        except Exception as e:
            print(f"[cache] Failed to load coupling cache: {e}")
    return {}, {}


def _save_coupling_cache(cache_dir="data"):
    """Save coupling length cache to disk."""
    cache_path = Path(cache_dir) / "coupling_length_cache.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "per_pol": _coupling_cache_per_pol,
                "blended": _coupling_cache_blended
            }, f)
    except Exception as e:
        print(f"[cache] Failed to save coupling cache: {e}")


# Load cache on module import
_coupling_cache_per_pol, _coupling_cache_blended = _load_coupling_cache()
