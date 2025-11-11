import tidy3d as td
import numpy as np
import warnings
import logging
import sys
from simulation_utils import _ModeSolver
from building_utils import resolve_mode_spec
from pathlib import Path

# Suppress verbose tidy3d mode solver warnings about field decay at boundaries
# These warnings are common and don't affect accuracy for mode solving
warnings.filterwarnings('ignore', message='.*Mode field.*does not decay at the plane boundaries.*')
warnings.filterwarnings('ignore', message='.*Use the remote mode solver.*')
warnings.filterwarnings('ignore', category=UserWarning)

# Also suppress tidy3d logging if it uses logging module
logging.getLogger('tidy3d').setLevel(logging.ERROR)
logging.getLogger('tidy3d.plugins.mode').setLevel(logging.ERROR)

# Context manager to suppress warnings during mode solver calls
class _SuppressTidy3dWarnings:
    """Context manager to suppress tidy3d mode solver warnings."""
    def __init__(self):
        self._original_stderr = None
        self._original_stdout = None
        self._original_warnings = None
        self._devnull_stderr = None
        self._devnull_stdout = None
    
    def __enter__(self):
        import os
        self._original_warnings = warnings.filters[:]
        warnings.filterwarnings('ignore')
        # Redirect stderr to suppress tidy3d's direct print statements
        # Keep stdout separate so tqdm can still write progress bars
        self._devnull_stderr = open(os.devnull, 'w')
        self._devnull_stdout = open(os.devnull, 'w')
        self._original_stderr = sys.stderr
        self._original_stdout = sys.stdout
        sys.stderr = self._devnull_stderr
        sys.stdout = self._devnull_stdout
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stderr and stdout
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
        if self._devnull_stderr is not None:
            self._devnull_stderr.close()
        if self._devnull_stdout is not None:
            self._devnull_stdout.close()
        # Restore warnings
        if self._original_warnings is not None:
            warnings.filters[:] = self._original_warnings
        return False
import pickle

# Try to import scipy.optimize for balance policy optimization
try:
    from scipy.optimize import minimize_scalar, brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    brentq = None

_COUPLING_CACHE_REV = 7  # Bumped to invalidate cache after fixing balance optimization search bounds
                                 # to use actual L_50 values instead of π/(4*Ω) approximation

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


def compute_internal_phase(kappa, Delta, Omega, L_c):
    """
    Compute internal phase Δφ = arg(a₂/a₁) from CMT parameters.
    
    Formula: Δφ = arg(-jκ sin(ΩL_c) / (Ω cos(ΩL_c) - jΔ sin(ΩL_c)))
    
    For a canonical 3-dB coupler, Δφ ≈ π/2 at the design wavelength.
    
    Args:
        kappa: Coupling coefficient κ
        Delta: Detuning parameter Δ
        Omega: Rabi frequency Ω = sqrt(κ²+Δ²)
        L_c: Coupling length in µm
    
    Returns:
        Internal phase Δφ in radians
    """
    kappa = float(kappa)
    Delta = float(Delta)
    Omega = float(Omega)
    L_c = float(L_c)
    
    Omega_L = Omega * L_c
    sin_Omega_L = np.sin(Omega_L)
    cos_Omega_L = np.cos(Omega_L)
    
    # Numerator: -jκ sin(ΩL_c)
    numerator = -1j * kappa * sin_Omega_L
    
    # Denominator: Ω cos(ΩL_c) - jΔ sin(ΩL_c)
    denominator = Omega * cos_Omega_L - 1j * Delta * sin_Omega_L
    
    # Avoid division by zero
    if abs(denominator) < 1e-18:
        return np.nan
    
    # Compute ratio and extract phase
    ratio = numerator / denominator
    delta_phi = np.angle(ratio)
    
    return float(delta_phi)


def _optimize_Lc_balance(kappa_dict, Delta_dict, Omega_dict, min_len, max_len, trim_factor=0.0, Cmax_dict=None, L_50_dict=None):
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
        trim_factor: Trim factor to account in optimization (default: 0.0)
        Cmax_dict: Optional dict with Cmax values for normalization (default: None)
        L_50_dict: Optional dict with 'te' and 'tm' keys for actual L_50 values (used for search bounds)
    
    Returns:
        Optimal L_c_raw (before trim factor) in µm, or None if optimization fails
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
    
    # Use actual L_50 values for search bounds if available, otherwise fall back to π/(4*Ω)
    if L_50_dict is not None and all(pol in L_50_dict for pol in ("te", "tm")):
        L_50_te = L_50_dict["te"]
        L_50_tm = L_50_dict["tm"]
        # Use finite L_50 values only
        finite_L50 = [L for L in [L_50_te, L_50_tm] if np.isfinite(L) and L > 0]
        if len(finite_L50) >= 1:
            # Search around the actual L_50 values: 0.5x min to 1.5x max
            # This ensures we stay in the first lobe and don't jump to later lobes
            search_min = 0.5 * min(finite_L50)
            search_max = 1.5 * max(finite_L50)
        else:
            # Fallback to π/(4*Ω) if L_50 values are invalid
            L_50_first_te = np.pi / (4.0 * Omega_dict["te"])
            L_50_first_tm = np.pi / (4.0 * Omega_dict["tm"])
            search_min = 0.5 * min(L_50_first_te, L_50_first_tm)
            search_max = 2.0 * max(L_50_first_te, L_50_first_tm)
    else:
        # Fallback: compute first-0.5 lengths for each polarization: L_50_first = π/(4*Ω)
        L_50_first_te = np.pi / (4.0 * Omega_dict["te"])
        L_50_first_tm = np.pi / (4.0 * Omega_dict["tm"])
        search_min = 0.5 * min(L_50_first_te, L_50_first_tm)
        search_max = 2.0 * max(L_50_first_te, L_50_first_tm)
    
    # Clip to safety bounds
    search_min = max(search_min, min_len)
    search_max = min(search_max, max_len)
    
    if search_min >= search_max:
        # Fallback: use safety bounds if computed window is invalid
        search_min = min_len
        search_max = max_len
    
    # Objective function: balance both polarizations while penalizing undercoupling
    # Strategy: minimize max error (ensure neither is too far off) + imbalance penalty
    def objective(L_raw):
        L_final = L_raw * (1 + trim_factor)
        eta_te = _eta_from_cmt(kappa_dict["te"], Delta_dict["te"], Omega_dict["te"], L_final)
        eta_tm = _eta_from_cmt(kappa_dict["tm"], Delta_dict["tm"], Omega_dict["tm"], L_final)
        
        # Raw errors
        err_te_raw = abs(eta_te - 0.5)
        err_tm_raw = abs(eta_tm - 0.5)
        
        # Normalize by Cmax to account for different achievable coupling
        if Cmax_dict is not None:
            cmax_te = Cmax_dict.get("te", 1.0)
            cmax_tm = Cmax_dict.get("tm", 1.0)
            err_te = err_te_raw / max(cmax_te, 0.1)
            err_tm = err_tm_raw / max(cmax_tm, 0.1)
        else:
            err_te = err_te_raw
            err_tm = err_tm_raw
        
        # Mild penalty for undercoupling (1.2x instead of 1.5x)
        if eta_te < 0.5:
            err_te *= 1.2
        if eta_tm < 0.5:
            err_tm *= 1.2
        
        # Primary objective: minimize maximum error (ensures both are balanced)
        max_err = max(err_te, err_tm)
        
        # Secondary: penalize imbalance between errors (prevent favoring one mode)
        imbalance = abs(err_te - err_tm)
        
        # Combined cost: prioritize max error, but also penalize imbalance
        cost = max_err + 0.3 * imbalance + 1e-3 * (L_final / max_len)
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


def _compute_L50_per_pol(param, lambda0, cache_dir="data"):
    """
    Compute L_50, kappa, Delta, and Omega for both TE and TM polarizations.
    Core computation function used by both compute_Lc() and solve_delta_w_star().
    
    Args:
        param: Parameter namespace with wg_width, delta_w, coupling_gap, wg_thick, medium
        lambda0: Wavelength in µm
        cache_dir: Directory for cache file
    
    Returns:
        dict with keys: 'L_50_dict', 'kappa_dict', 'Delta_dict', 'Omega_dict', 'Cmax_dict', 'n1_dict', 'n2_dict'
        Each dict has 'te' and 'tm' keys.
    """
    if _ModeSolver is None:
        raise AttributeError("ModeSolver plugin is unavailable in this tidy3d build")
    
    lambda0 = float(lambda0)
    w1, w2 = _adc__get_w1_w2(param)
    coupling_gap = float(param.coupling_gap)
    wg_thick = float(param.wg_thick)
    
    # Get wavelength-dependent permittivity if dispersive materials are enabled
    use_dispersive = getattr(param.medium, 'use_dispersive', False)
    if use_dispersive and hasattr(param.medium, 'create_SiN'):
        medium_SiN = param.medium.create_SiN(lambda_um=lambda0)
        medium_SiO2 = param.medium.create_SiO2(lambda_um=lambda0)
        eps_SiN = float(medium_SiN.permittivity)
        eps_SiO2 = float(medium_SiO2.permittivity)
    else:
        eps_SiN = float(param.medium.SiN.permittivity)
        eps_SiO2 = float(param.medium.SiO2.permittivity)
    
    pols = ("te", "tm")
    L_50_dict = {}
    kappa_dict = {}
    Delta_dict = {}
    Omega_dict = {}
    Cmax_dict = {}
    n1_dict = {}
    n2_dict = {}
    
    # Include dispersion model identifier in cache key
    dispersion_id = "dispersive" if use_dispersive else "constant"
    
    # Numerical guard thresholds
    KAPPA_MIN = 5e-4
    DELTA_EPSILON = 1e-3
    
    for pol in pols:
        per_pol_key = (_COUPLING_CACHE_REV, "adc", w1, w2, coupling_gap, wg_thick, lambda0, eps_SiN, eps_SiO2, dispersion_id, pol)
        if per_pol_key in _coupling_cache_per_pol:
            cached = _coupling_cache_per_pol[per_pol_key]
            n1 = cached["n1"]
            n2 = cached["n2"]
            S = cached["beta_split"]
            Delta = cached["Delta"]
            kappa = cached["kappa"]
            L_50 = cached["L_50"]
            # Compute Omega correctly: Ω = √(κ² + Δ²) as per CMT derivation
            kappa_sq_local = float(kappa) ** 2
            Omega = np.sqrt(kappa_sq_local + Delta**2)
            # Recompute Cmax on cache hit (was not stored in older cache revs)
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
            d_center = coupling_gap + 0.5 * (w1 + w2)
            y_upper = +0.5 * d_center
            y_lower = -0.5 * d_center
            t = wg_thick
            
            domain_x = 2.0 * max(w1, w2)
            pad_y = max(1.0 * lambda0, 2.0 * max(w1, w2))
            pad_z = max(0.5 * lambda0, 2.0 * t)
            domain_y = (coupling_gap + w1 + w2) + 2.0 * pad_y
            domain_z = t + 2.0 * pad_z
            
            # Use wavelength-specific materials if dispersive mode is enabled
            if use_dispersive and hasattr(param.medium, 'create_SiN'):
                medium_SiN_lambda = param.medium.create_SiN(lambda_um=lambda0)
                medium_SiO2_lambda = param.medium.create_SiO2(lambda_um=lambda0)
            else:
                medium_SiN_lambda = param.medium.SiN
                medium_SiO2_lambda = param.medium.SiO2
            
            upper_wg = td.Structure(
                geometry=td.Box(size=(w1, w1, t), center=(0, y_upper, 0)),
                medium=medium_SiN_lambda,
                name="upper_wg"
            )
            lower_wg = td.Structure(
                geometry=td.Box(size=(w2, w2, t), center=(0, y_lower, 0)),
                medium=medium_SiN_lambda,
                name="lower_wg"
            )
            
            sim_cross = td.Simulation(
                size=(domain_x, domain_y, domain_z),
                medium=medium_SiO2_lambda,
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
                with _SuppressTidy3dWarnings():
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
                
                sorted_indices = np.argsort(n_eff_arr)[::-1]
                n_plus = float(n_eff_arr[sorted_indices[0]])
                n_minus = float(n_eff_arr[sorted_indices[1]])
                
                S = k0 * (n_plus - n_minus)
            except Exception as e:
                print(f"[L_c Computation] ⚠ WARNING: Mode solver failed for {pol.upper()}: {e}")
                S = 0.0
            
            # According to CMT: κ = (β_even - β_odd)/2 = k₀(n_even - n_odd)/2 = S/2
            kappa = S / 2
            # Ω = √(κ² + Δ²) as per CMT derivation
            kappa_sq = kappa ** 2
            Omega = np.sqrt(kappa_sq + Delta**2)
            Cmax = kappa_sq / (kappa_sq + Delta**2) if (kappa_sq + Delta**2) > 0 else 0.0
            
            # Numerical guards: check before computing L_50
            if kappa < KAPPA_MIN or abs(Delta) > kappa * (1 - DELTA_EPSILON):
                # Near-singular point: mark as invalid
                L_50 = np.inf
            elif kappa <= abs(Delta):
                # Exact 50:50 not reachable
                L_50 = np.inf
                # Warning message is now shown in the summary table, so we skip verbose logging here
            else:
                # Compute L_50 (reachability info shown in summary table)
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
        
        n1_dict[pol] = n1
        n2_dict[pol] = n2
        Delta_dict[pol] = Delta
        kappa_dict[pol] = kappa
        Omega_dict[pol] = Omega
        Cmax_dict[pol] = Cmax
        L_50_dict[pol] = L_50
    
    return {
        'L_50_dict': L_50_dict,
        'kappa_dict': kappa_dict,
        'Delta_dict': Delta_dict,
        'Omega_dict': Omega_dict,
        'Cmax_dict': Cmax_dict,
        'n1_dict': n1_dict,
        'n2_dict': n2_dict,
    }






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
        print(f"[L_c Computation] ℹ INFO: Symmetric coupler detected (|w₁-w₂|={abs(w1-w2)*1000:.2f} nm < {epsilon_w*1000:.0f} nm)")
    coupling_gap = float(param.coupling_gap)
    wg_thick = float(param.wg_thick)
    
    # Get wavelength-dependent permittivity if dispersive materials are enabled
    use_dispersive = getattr(param.medium, 'use_dispersive', False)
    if use_dispersive and hasattr(param.medium, 'create_SiN'):
        # Use wavelength-dependent materials
        medium_SiN = param.medium.create_SiN(lambda_um=lambda0)
        medium_SiO2 = param.medium.create_SiO2(lambda_um=lambda0)
        eps_SiN = float(medium_SiN.permittivity)
        eps_SiO2 = float(medium_SiO2.permittivity)
    else:
        # Fallback to constant permittivity (backward compatibility)
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
    
    # Use the shared computation function
    result = _compute_L50_per_pol(param, lambda0, cache_dir)
    L_50_dict = result['L_50_dict']
    kappa_dict = result['kappa_dict']
    Delta_dict = result['Delta_dict']
    Omega_dict = result['Omega_dict']
    Cmax_dict = result['Cmax_dict']
    n1_dict = result['n1_dict']
    n2_dict = result['n2_dict']
    
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
            print(f"[L_c Computation] ⚠ WARNING: Selected polarization {blend_policy.upper()} has infinite L_50; check coupling")
    elif blend_policy == "balance":
        # Optimize L_c to minimize |η_TE(L_final) - 0.5| + |η_TM(L_final) - 0.5|
        # where L_final = L_raw * (1 + trim_factor)
        try:
            L_c_raw = _optimize_Lc_balance(kappa_dict, Delta_dict, Omega_dict, min_len, max_len, trim_factor, Cmax_dict, L_50_dict)
            if L_c_raw is None:
                # Optimization failed, fall back to median
                print(f"[L_c Computation] ⚠ WARNING: Balance optimization failed, falling back to median policy")
                finite_vals = [v for v in L_50_dict.values() if np.isfinite(v)]
                if len(finite_vals) == 0:
                    L_c_raw = max_len
                elif len(finite_vals) == 1:
                    L_c_raw = finite_vals[0]
                else:
                    L_c_raw = np.median(finite_vals)
        except Exception as e:
            print(f"[L_c Computation] ⚠ WARNING: Balance optimization error: {e}, falling back to median policy")
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
        print(f"[L_c Computation] ⚠ WARNING: L_c clipped from {L_c_raw_trimmed:.3f} to {L_c:.3f} µm (bounds [{min_len:.1f}, {max_len:.1f}] µm)")
    
    # Store in blended cache (include dispersion model identifier)
    use_dispersive = getattr(param.medium, 'use_dispersive', False)
    dispersion_id = "dispersive" if use_dispersive else "constant"
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
        dispersion_id,
        trim_factor,
        blend_policy,
        min_len,
        max_len,
    )
    _coupling_cache_blended[blended_key] = L_c
    _save_coupling_cache(cache_dir)
    
    # Log summary line with improved formatting
    print(f"\n{'='*80}")
    print(f"[L_c Computation] Coupling Length Derived")
    print(f"{'─'*80}")
    
    pols = ("te", "tm")
    for pol in pols:
        # Explicitly get values for this polarization from dictionaries
        n1_pol = n1_dict[pol]
        n2_pol = n2_dict[pol]
        Delta_pol = Delta_dict[pol]
        kappa_pol = kappa_dict[pol]
        Cmax_pol = Cmax_dict[pol]
        L50_pol = L_50_dict[pol]
        L50_str = f"{L50_pol:.3f} µm" if np.isfinite(L50_pol) else "INF"
        cmax_str = f"{Cmax_pol:.3f}" if Cmax_pol >= 0 else "N/A"
        
        # Format reachability status
        reachable = "✓" if (kappa_pol > abs(Delta_pol) and np.isfinite(L50_pol)) else "✗"
        
        print(f"  {pol.upper():3s}: {reachable}  n₁={n1_pol:.4f}, n₂={n2_pol:.4f}  |  Δ={Delta_pol:+.4e}, κ={kappa_pol:.4e}  |  Cmax={cmax_str:>6s}, L₅₀={L50_str:>10s}")
    
    print(f"{'─'*80}")
    print(f"  Blend policy: {blend_policy:8s}  |  Trim factor: {trim_factor:.1%}  |  Final L_c: {L_c:.3f} µm")
    print(f"{'='*80}\n")
    
    # Store CMT parameters in param namespace for later use (e.g., Δφ computation)
    param.cmt_params = {
        "te": {
            "kappa": kappa_dict.get("te"),
            "Delta": Delta_dict.get("te"),
            "Omega": Omega_dict.get("te"),
            "L_c": L_c,
            "lambda0": lambda0,
        },
        "tm": {
            "kappa": kappa_dict.get("tm"),
            "Delta": Delta_dict.get("tm"),
            "Omega": Omega_dict.get("tm"),
            "L_c": L_c,
            "lambda0": lambda0,
        }
    }
    
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
    
    # Use wavelength-specific materials if dispersive mode is enabled
    use_dispersive = getattr(param.medium, 'use_dispersive', False)
    if use_dispersive and hasattr(param.medium, 'create_SiN'):
        eps_SiN = param.medium.create_SiN(lambda_um=lambda0)
        eps_SiO2 = param.medium.create_SiO2(lambda_um=lambda0)
    else:
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
        with _SuppressTidy3dWarnings():
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
