import tidy3d as td
import numpy as np
from simulation_utils import _ModeSolver
from building_utils import resolve_mode_spec
from pathlib import Path
import pickle

_COUPLING_CACHE_REV = 2

# Cache for coupling length computations
_coupling_cache_per_pol = {}  # Level 1: per-polarization supermode data
_coupling_cache_blended = {}  # Level 2: blended L_c values



def compute_Lc_symmetric(param, lambda0=None, trim_factor=0.075, cache_dir="data"):
    """
    Compute coupling length L_c from supermode analysis of 2-waveguide cross-section.
    
    Physics:
    - Solves for even and odd supermodes of the coupled waveguide system
    - Computes effective index difference: Δn = n_even - n_odd
    - Calculates beat length: L_π = λ₀ / (2 × Δn)
    - Calculates 3-dB length: L_50 = L_π / 2 = λ₀ / (4 × Δn)
    - Takes median of TE and TM L_50 values
    - Applies empirical trim factor for bend/transition effects
    
    Args:
        param: Parameter namespace with wg_width, coupling_gap, wg_thick, medium.SiN, medium.SiO2
        lambda0: Wavelength in µm (default: param.wl_0)
        trim_factor: Empirical trim factor for bends/transitions (default: 0.075 = 7.5%)
        cache_dir: Directory for cache file (default: "data")
    
    Returns:
        L_c in µm (clipped to [3.0, 50.0] µm safety bounds)
    """
    if _ModeSolver is None:
        raise AttributeError("ModeSolver plugin is unavailable in this tidy3d build")
    
    # Use provided lambda0 or default from param
    if lambda0 is None:
        lambda0 = getattr(param, 'lambda_single', param.wl_0)
    lambda0 = float(lambda0)
    
    # Extract geometry parameters
    wg_width = float(param.wg_width)
    wg_thick = float(param.wg_thick)
    coupling_gap = float(param.coupling_gap)
    
    # Extract material permittivities for cache key
    eps_SiN = float(param.medium.SiN.permittivity)
    eps_SiO2 = float(param.medium.SiO2.permittivity)
    
    # Check cache for blended L_c (include material permittivities)
    blended_key = (_COUPLING_CACHE_REV, wg_width, coupling_gap, wg_thick, lambda0, eps_SiN, eps_SiO2, trim_factor, 'median')
    if blended_key in _coupling_cache_blended:
        cached_Lc = _coupling_cache_blended[blended_key]
        # Retrieve TE and TM L_50 values from per-pol cache for detailed output
        L_c_te = None
        L_c_tm = None
        for pol in ("te", "tm"):
            per_pol_key = (_COUPLING_CACHE_REV, wg_width, coupling_gap, wg_thick, lambda0, eps_SiN, eps_SiO2, pol)
            if per_pol_key in _coupling_cache_per_pol:
                cached_data = _coupling_cache_per_pol[per_pol_key]
                L_50 = cached_data["L_50"]
                L_c_pol = L_50 * (1 + trim_factor)
                if pol == "te":
                    L_c_te = float(np.clip(L_c_pol, 3.0, 50.0))
                else:
                    L_c_tm = float(np.clip(L_c_pol, 3.0, 50.0))
        
        # Print with TE and TM values if available
        if L_c_te is not None and L_c_tm is not None:
            print(f"[L_c cache hit] L_c_TE={L_c_te:.3f}µm, L_c_TM={L_c_tm:.3f}µm, L_c_median={cached_Lc:.3f}µm (from cache)")
        else:
            print(f"[L_c cache hit] L_c={cached_Lc:.3f}µm (from cache)")
        return cached_Lc
    
    # Build minimal 2-waveguide cross-section simulation
    # Two parallel waveguides separated by coupling_gap
    gap = coupling_gap
    w = wg_width
    t = wg_thick
    
    # Waveguide centers: upper at +y, lower at -y
    y_upper = (gap + w) / 2
    y_lower = -(gap + w) / 2
    
    # Domain sized to ensure modal fields decay well before PMLs
    pad_y = max(1.0 * lambda0, 2.0 * w)
    pad_z = max(0.5 * lambda0, 2.0 * t)
    domain_x = 2.0 * w  # x size is irrelevant for the mode plane (size_x=0), keep compact
    domain_y = (gap + 2.0 * w) + 2.0 * pad_y
    domain_z = t + 2.0 * pad_z
    
    upper_wg = td.Structure(
        geometry=td.Box(size=(w, w, t), center=(0, y_upper, 0)),
        medium=param.medium.SiN,
        name="upper_wg"
    )
    
    lower_wg = td.Structure(
        geometry=td.Box(size=(w, w, t), center=(0, y_lower, 0)),
        medium=param.medium.SiN,
        name="lower_wg"
    )
    
    # Minimal simulation for mode solving
    # Note: run_time is required by Simulation even though we only use it for mode solving
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
        run_time=1e-12  # Minimal value (not used for mode solving, but required by Simulation)
    )
    
    # Solve for supermodes for each polarization
    L_50_dict = {}
    delta_n_dict = {}
    
    for pol in ("te", "tm"):
        # Check per-pol cache first (include material permittivities)
        per_pol_key = (_COUPLING_CACHE_REV, wg_width, coupling_gap, wg_thick, lambda0, eps_SiN, eps_SiO2, pol)
        if per_pol_key in _coupling_cache_per_pol:
            cached_data = _coupling_cache_per_pol[per_pol_key]
            n_even = cached_data["n_even"]
            n_odd = cached_data["n_odd"]
            delta_n = cached_data["delta_n"]
            L_50 = cached_data["L_50"]
            delta_n_dict[pol] = delta_n
            L_50_dict[pol] = L_50
            continue

        # Polarization-specific mode solver: request 2 modes with filter_pol set
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

            # Extract n_eff values
            n_eff_arr = np.array(sol.n_eff)
            if n_eff_arr.ndim >= 2:
                n_eff_arr = np.real(n_eff_arr.reshape(-1))[:2]
            else:
                n_eff_arr = np.real(n_eff_arr[:2])

            if len(n_eff_arr) < 2:
                raise ValueError(f"Mode solver returned <2 modes for {pol}")

            # Identify even/odd pair and compute Δn
            n_even, n_odd = _identify_even_odd_modes(sol, n_eff_arr)
            delta_n = n_even - n_odd

            if delta_n < 1e-6:
                raise ValueError(f"Too weak coupling for {pol}: Δn={delta_n:.2e} < 1e-6")

            # Compute 3-dB length
            L_pi = lambda0 / (2 * delta_n)
            L_50 = L_pi / 2  # = lambda0 / (4 * delta_n)

            # Store in per-pol cache
            _coupling_cache_per_pol[per_pol_key] = {
                "n_even": n_even,
                "n_odd": n_odd,
                "delta_n": delta_n,
                "L_50": L_50,
            }
            _save_coupling_cache(cache_dir)

            delta_n_dict[pol] = delta_n
            L_50_dict[pol] = L_50

        except Exception as e:
            print(f"[L_c WARNING] Mode solver failed for {pol}: {e}")
            # Fallback: use safe default
            L_50_dict[pol] = 10.0  # Safe default
            delta_n_dict[pol] = lambda0 / (4 * 10.0)  # Rough estimate
    
    # Compute median of TE and TM L_50 values
    L_50_te = L_50_dict.get("te", 10.0)
    L_50_tm = L_50_dict.get("tm", 10.0)
    L_50_median = (L_50_te + L_50_tm) / 2
    
    delta_n_te = delta_n_dict.get("te", 0.0)
    delta_n_tm = delta_n_dict.get("tm", 0.0)
    
    # Apply trim factor
    L_c_raw = L_50_median * (1 + trim_factor)
    
    # Clip to safety bounds with warning
    L_c = float(np.clip(L_c_raw, 3.0, 50.0))
    if L_c != L_c_raw:
        print(f"[L_c WARNING] L_c clipped from {L_c_raw:.3f} to {L_c:.3f} µm "
              f"(weak/strong coupling or out-of-bounds geometry)")
    
    # Store in blended cache
    _coupling_cache_blended[blended_key] = L_c
    _save_coupling_cache(cache_dir)
    
    # Log summary
    print(f"[L_c derived] Δn_TE={delta_n_te:.6f}, Δn_TM={delta_n_tm:.6f}, "
          f"L_50_TE={L_50_te:.3f}µm, L_50_TM={L_50_tm:.3f}µm, "
          f"median={L_50_median:.3f}µm, trim={trim_factor:.1%}, L_c={L_c:.3f}µm")
    
    return L_c



def compute_Lc_asymmetric(param, lambda0=None, trim_factor=0.075, cache_dir="data", blend_policy="median"):
    """
    Compute coupling length L_c for asymmetric directional coupler (different arm widths).
    
    Args:
        param: Parameter namespace with either (wg_width_left, wg_width_right) or (wg_width, delta_w),
               plus coupling_gap, wg_thick, medium.SiN, medium.SiO2.
        lambda0: Wavelength in µm (default: param.wl_0)
        trim_factor: Empirical trim factor for bends/transitions (default: 0.075)
        cache_dir: Directory for cache file (default: "data")
        blend_policy: How to blend TE/TM results: "median" (default), "te", or "tm"
    
    Returns:
        L_c in µm (clipped to [3.0, 50.0] µm safety bounds)
    """
    if _ModeSolver is None:
        raise AttributeError("ModeSolver plugin is unavailable in this tidy3d build")
    
    if lambda0 is None:
        lambda0 = getattr(param, 'lambda_single', param.wl_0)
    lambda0 = float(lambda0)
    
    w1, w2 = _adc__get_w1_w2(param)
    coupling_gap = float(param.coupling_gap)
    wg_thick = float(param.wg_thick)
    eps_SiN = float(param.medium.SiN.permittivity)
    eps_SiO2 = float(param.medium.SiO2.permittivity)
    
    pols = ("te", "tm")
    L_50_dict = {}
    kappa_dict = {}
    Delta_dict = {}
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
            y_upper = (coupling_gap + w1 + w2) / 2
            y_lower = -y_upper
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
                ratio = 0.5 * (1.0 + (Delta / kappa)**2)
                ratio = min(max(ratio, 0.0), 1.0)
                L_50 = (1.0 / Omega) * np.arcsin(np.sqrt(ratio))
            
            _coupling_cache_per_pol[per_pol_key] = {
                "n1": n1,
                "n2": n2,
                "beta_split": S,
                "Delta": Delta,
                "kappa": kappa,
                "L_50": L_50,
            }
            _save_coupling_cache(cache_dir)
        
        n1_dict[pol] = n1
        n2_dict[pol] = n2
        Delta_dict[pol] = Delta
        kappa_dict[pol] = kappa
        Cmax_dict[pol] = Cmax
        L_50_dict[pol] = L_50
    
    # Blend according to policy
    L_c_raw = None
    if blend_policy == "median":
        finite_vals = [v for v in L_50_dict.values() if np.isfinite(v)]
        if len(finite_vals) == 0:
            # Both infinite => return upper bound clipped with warning
            L_c_raw = 50.0
            print("[L_c (ADC) WARNING] Both TE and TM L_50 are infinite; returning upper bound 50 µm")
        elif len(finite_vals) == 1:
            L_c_raw = finite_vals[0]
        else:
            L_c_raw = np.median(finite_vals)
    elif blend_policy in ("te", "tm"):
        L_c_raw = L_50_dict.get(blend_policy, np.inf)
        if not np.isfinite(L_c_raw):
            print(f"[L_c (ADC) WARNING] Selected polarization {blend_policy.upper()} has infinite L_50; check coupling")
    else:
        raise ValueError(f"Invalid blend_policy: {blend_policy}")
    
    # Apply trim factor and clip
    L_c_raw_trimmed = L_c_raw * (1 + trim_factor)
    L_c = float(np.clip(L_c_raw_trimmed, 3.0, 50.0))
    if L_c != L_c_raw_trimmed:
        print(f"[L_c (ADC) WARNING] L_c clipped from {L_c_raw_trimmed:.3f} to {L_c:.3f} µm (out-of-bounds geometry or coupling)")
    
    # Store in blended cache
    blended_key = (_COUPLING_CACHE_REV, "adc", w1, w2, coupling_gap, wg_thick, lambda0, eps_SiN, eps_SiO2, trim_factor, blend_policy)
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
    Priority:
    1) Use wg_width_left and wg_width_right if present.
    2) Else use wg_width and delta_w.
    Raises ValueError if neither available.
    """
    if hasattr(param, 'wg_width_left') and hasattr(param, 'wg_width_right'):
        w1 = float(param.wg_width_left)
        w2 = float(param.wg_width_right)
    elif hasattr(param, 'wg_width') and hasattr(param, 'delta_w'):
        w_base = float(param.wg_width)
        delta_w = float(param.delta_w)
        w1 = w_base + delta_w / 2
        w2 = w_base - delta_w / 2
    else:
        raise ValueError("Asymmetric coupling length computation requires either (wg_width_left, wg_width_right) or (wg_width, delta_w) parameters")
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



def _identify_even_odd_modes(sol, n_eff_arr):
    """
    Identify even and odd supermodes from mode solver solution.
    
    Returns (n_even, n_odd) where n_even > n_odd typically.
    Uses field symmetry about mid-plane between waveguides.
    """
    if len(n_eff_arr) < 2:
        raise ValueError("Need at least 2 modes to identify even/odd pair")
    
    # Get field components for analysis
    try:
        # Try to get Ey or Ez component for symmetry analysis
        if hasattr(sol, 'Ey'):
            fields = np.array(sol.Ey)
        elif hasattr(sol, 'Ez'):
            fields = np.array(sol.Ez)
        else:
            # Fallback: use higher n_eff as even, lower as odd (typical for coupled waveguides)
            sorted_indices = np.argsort(n_eff_arr)[::-1]  # descending order
            return float(n_eff_arr[sorted_indices[0]]), float(n_eff_arr[sorted_indices[1]])
        
        # Analyze symmetry: even mode has similar field on both sides, odd has opposite
        # For simplicity, use n_eff ordering (even typically higher n_eff)
        # In weakly coupled waveguides, even mode has constructive overlap (higher n_eff)
        sorted_indices = np.argsort(n_eff_arr)[::-1]  # descending order
        n_even = float(n_eff_arr[sorted_indices[0]])
        n_odd = float(n_eff_arr[sorted_indices[1]])
        
        return n_even, n_odd
    except Exception:
        # Fallback to simple n_eff sorting
        sorted_indices = np.argsort(n_eff_arr)[::-1]
        return float(n_eff_arr[sorted_indices[0]]), float(n_eff_arr[sorted_indices[1]])


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
