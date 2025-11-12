"""
Delta width solver module for asymmetry-first directional coupler design.

This module provides utilities to solve for delta_w* that equalizes L_50_TE and L_50_TM
at the design wavelength, implementing the asymmetry-first strategy.
"""

import numpy as np
import sys
import os
from coupling_length import (
    _compute_L50_per_pol,
    _SuppressTidy3dWarnings,
)

# Try to import scipy.optimize for root finding and minimization
try:
    from scipy.optimize import minimize_scalar, brentq, root_scalar
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    brentq = None
    root_scalar = None

# Import tqdm for progress bars (required)
from tqdm import tqdm
from math import log

_LAST_DELTA_W_HINT = None


def _evaluate_L50(delta_w, param, lambda0, cache_dir, eval_cache=None, cache_precision=8, verbose=False):
    """
    Evaluate coupled-mode quantities for a given Δw, reusing cached results when possible.
    """
    key = round(float(delta_w), cache_precision)
    if eval_cache is not None and key in eval_cache:
        if verbose:
            print(f"[Mode Solver] Cache hit for Δw={delta_w:.6f} µm", flush=True)
        return eval_cache[key]
    
    if verbose:
        print(f"[Mode Solver] Computing for Δw={delta_w:.6f} µm (this may take a while)...", flush=True)
    
    from types import SimpleNamespace
    temp_param = SimpleNamespace(**param.__dict__)
    temp_param.delta_w = float(delta_w)
    _validate_param(temp_param)
    try:
        with _SuppressTidy3dWarnings():
            result = _compute_L50_per_pol(temp_param, lambda0, cache_dir)
        if verbose and result is not None:
            L50_te = result.get('L_50_dict', {}).get('te', np.nan)
            L50_tm = result.get('L_50_dict', {}).get('tm', np.nan)
            print(f"[Mode Solver] Complete: L50_TE={L50_te:.3f} µm, L50_TM={L50_tm:.3f} µm", flush=True)
    except Exception as e:
        if verbose:
            print(f"[Mode Solver] Failed: {e}", flush=True)
        result = None
    if eval_cache is not None:
        eval_cache[key] = result
    return result

def _validate_param(param):
    """Validate that param has required geometry attributes."""
    if not hasattr(param, 'wg_width') or param.wg_width is None:
        raise ValueError("param must have wg_width set")
    if not hasattr(param, 'coupling_gap') or param.coupling_gap is None:
        raise ValueError("param must have coupling_gap set")
    if not hasattr(param, 'wg_thick') or param.wg_thick is None:
        raise ValueError("param must have wg_thick set")

def _G_of_dw(delta_w, param, lambda0, cache_dir, eval_cache=None, verbose=False):
    """
    Evaluate G(Δw) = log(L50_TE / L50_TM) using natural log.
    Returns (G_val, L50_te, L50_tm) or (None, None, None) if infeasible.
    """
    result = _evaluate_L50(delta_w, param, lambda0, cache_dir, eval_cache, verbose=verbose)
    if result is None:
        return None, None, None
    L50_te = result['L_50_dict']['te']
    L50_tm = result['L_50_dict']['tm']
    kappa_te = result['kappa_dict']['te']
    kappa_tm = result['kappa_dict']['tm']
    Delta_te = result['Delta_dict']['te']
    Delta_tm = result['Delta_dict']['tm']
    # feasibility
    if abs(Delta_te) > kappa_te or abs(Delta_tm) > kappa_tm:
        return None, None, None
    if not (np.isfinite(L50_te) and np.isfinite(L50_tm)) or (L50_te <= 0) or (L50_tm <= 0):
        return None, None, None
    G_val = log(L50_te / L50_tm)
    return float(G_val), float(L50_te), float(L50_tm)

def _find_feasible_bracket(param, lambda0, cache_dir, search_min, search_max,
                           hard_min, hard_max, center_guess=0.0, step=0.01,
                           eval_cache=None):
    """
    Adaptively expand from center_guess with exponentially growing window to find a sign-change bracket.
    Returns (a, b, Ga, Gb) if found, else (None, None, None, None).
    """
    center = float(np.clip(center_guess, search_min, search_max))
    step = max(step, 0.002)

    # Build a uniform grid inside [search_min, search_max] and evaluate points
    grid = []
    x = search_min
    while x <= search_max + 1e-12:
        grid.append(round(float(np.clip(x, hard_min, hard_max)), 6))
        x += step

    # Make sure edges are included even if rounding skipped them
    grid.append(round(float(np.clip(search_min, hard_min, hard_max)), 6))
    grid.append(round(float(np.clip(search_max, hard_min, hard_max)), 6))
    grid.append(round(float(np.clip(center, hard_min, hard_max)), 6))

    # Deduplicate while sorting by distance to center so we still search outward
    unique = sorted(set(grid), key=lambda val: (abs(val - center), val))
    ordered = [float(val) for val in unique]

    last_x, last_G = None, None
    for x in ordered:
        Gx, _, _ = _G_of_dw(x, param, lambda0, cache_dir, eval_cache)
        if Gx is None:
            continue
        if last_G is not None and (Gx * last_G) < 0.0:
            return last_x, x, last_G, Gx
        last_x, last_G = x, Gx
    return None, None, None, None


def _heuristic_seed_search(param, lambda0, cache_dir, search_min, search_max,
                           center_guess, seed_step, abs_tol,
                           eval_cache=None, max_evals=40):
    """
    Lightweight seed scan that walks outward from center_guess with early stopping.
    Returns (best_seed_tuple, best_abs_err) where tuple = (dw, G, L50_te, L50_tm).
    """
    visited = set()
    best_seed = None
    best_abs_err = np.inf
    last_err = {+1: np.inf, -1: np.inf}
    unimproved = {+1: 0, -1: 0}
    evaluations = 0
    progress_interval = max(1, max_evals // 10)  # Print progress every ~10% of max_evals

    def _eval_dw(dw, direction=None):
        nonlocal best_seed, best_abs_err, evaluations
        if evaluations >= max_evals:
            return
        dw = float(np.clip(dw, search_min, search_max))
        key = round(dw, 6)
        if key in visited:
            return
        visited.add(key)
        evaluations += 1
        
        # Print progress periodically
        if evaluations % progress_interval == 0 or evaluations == 1:
            if np.isfinite(best_abs_err):
                print(f"[Δw* Seed Search] Evaluation {evaluations}/{max_evals}, "
                      f"best |L50_TE-L50_TM|={best_abs_err:.4f} µm", flush=True)
            else:
                print(f"[Δw* Seed Search] Evaluation {evaluations}/{max_evals}, "
                      f"searching for feasible points...", flush=True)
        
        # Always show when starting a new evaluation (helps identify slow ones)
        if evaluations > 1 and evaluations % progress_interval != 0:
            print(f"[Δw* Seed Search] Evaluation {evaluations}/{max_evals}: computing at Δw={dw:.6f} µm...", flush=True)
        
        # Show verbose mode solver messages for all evaluations (helps identify slow mode solves)
        verbose_mode = True
        result = _G_of_dw(dw, param, lambda0, cache_dir, eval_cache, verbose=verbose_mode)
        if result[0] is None:
            if evaluations > 1 and evaluations % progress_interval != 0:
                print(f"[Δw* Seed Search] Evaluation {evaluations}: infeasible point, skipping", flush=True)
            return
        G_val, L50_te, L50_tm = result
        abs_err = abs(L50_te - L50_tm)
        improved = abs_err < (best_abs_err - 1e-9)
        if improved:
            best_seed = (dw, G_val, L50_te, L50_tm)
            best_abs_err = abs_err
            if abs_err <= abs_tol:
                print(f"[Δw* Seed Search] ✓ Found seed meeting tolerance: "
                      f"Δw={dw:.6f} µm, |L50_TE-L50_TM|={abs_err:.4f} µm", flush=True)
            elif evaluations > 1 and evaluations % progress_interval != 0:
                print(f"[Δw* Seed Search] Evaluation {evaluations}: improved best to |L50_TE-L50_TM|={abs_err:.4f} µm", flush=True)
        if direction is not None:
            if improved:
                unimproved[direction] = 0
                last_err[direction] = abs_err
            else:
                if abs_err >= last_err[direction] - 1e-9:
                    unimproved[direction] += 1
                last_err[direction] = min(last_err[direction], abs_err)

    _eval_dw(center_guess)
    if best_abs_err <= abs_tol:
        return best_seed, best_abs_err

    radius = max(seed_step, 0.0025)
    max_radius = max(search_max - search_min, radius)
    print(f"[Δw* Seed Search] Starting radius search: initial radius={radius:.6f} µm, max_radius={max_radius:.6f} µm, "
          f"evaluations so far={evaluations}/{max_evals}", flush=True)
    iteration = 0
    stuck_iterations = 0
    last_evaluations = evaluations
    while evaluations < max_evals and radius <= max_radius + 1e-12:
        iteration += 1
        if iteration > 1:
            print(f"[Δw* Seed Search] Radius iteration {iteration}: radius={radius:.6f} µm, evaluations={evaluations}/{max_evals}", flush=True)
        
        evaluations_before_loop = evaluations
        for direction in (+1, -1):
            candidate = center_guess + direction * radius
            # Check if candidate is already visited before calling _eval_dw
            candidate_key = round(float(np.clip(candidate, search_min, search_max)), 6)
            if candidate_key in visited:
                continue  # Skip already visited points
            _eval_dw(candidate, direction)
            if best_abs_err <= abs_tol:
                return best_seed, best_abs_err
        
        # Check if we made any progress (new evaluations)
        if evaluations == evaluations_before_loop:
            stuck_iterations += 1
            if stuck_iterations >= 3:
                print(f"[Δw* Seed Search] No progress for {stuck_iterations} iterations (all candidates already visited), breaking", flush=True)
                break
        else:
            stuck_iterations = 0
        
        # Show progress after completing both directions for this radius
        if iteration > 1 or evaluations >= 25:
            print(f"[Δw* Seed Search] Completed radius {radius:.6f} µm: evaluations={evaluations}/{max_evals}, "
                  f"best |L50_TE-L50_TM|={best_abs_err:.4f} µm", flush=True)
        if unimproved[+1] >= 2 and unimproved[-1] >= 2:
            print(f"[Δw* Seed Search] Early stopping: no improvement in both directions "
                  f"(evaluations: {evaluations}/{max_evals})", flush=True)
            break
        old_radius = radius
        radius = min(radius * 1.5, max_radius)
        if radius == max_radius and old_radius < max_radius:
            print(f"[Δw* Seed Search] Reached max radius {max_radius:.6f} µm", flush=True)
            # If we've reached max radius and no new evaluations, break
            if evaluations == last_evaluations:
                print(f"[Δw* Seed Search] No new evaluations at max radius, breaking", flush=True)
                break
        last_evaluations = evaluations
    
    # Loop exited - show why
    if evaluations >= max_evals:
        print(f"[Δw* Seed Search] Loop exited: reached max evaluations ({max_evals})", flush=True)
    elif radius > max_radius + 1e-12:
        print(f"[Δw* Seed Search] Loop exited: radius {radius:.6f} µm exceeded max {max_radius:.6f} µm", flush=True)
    else:
        print(f"[Δw* Seed Search] Loop exited: unknown reason (evaluations={evaluations}, radius={radius:.6f})", flush=True)

    if best_seed is not None:
        print(f"[Δw* Seed Search] Completed: {evaluations} evaluations, "
              f"best |L50_TE-L50_TM|={best_abs_err:.4f} µm at Δw={best_seed[0]:.6f} µm", flush=True)
    else:
        print(f"[Δw* Seed Search] Completed: {evaluations} evaluations, no feasible seed found", flush=True)
    
    return best_seed, best_abs_err


def _F_of_dw(delta_w, param, lambda0, cache_dir, eval_cache=None):
    """
    Evaluate F(Δw) = L₅₀,TE(Δw) - L₅₀,TM(Δw).
    
    Args:
        delta_w: Width asymmetry parameter (µm)
        param: Parameter namespace template (must have wg_width, coupling_gap, wg_thick)
        lambda0: Design wavelength (µm)
        cache_dir: Directory for cache file
    
    Returns:
        F(Δw) (float) or None if infeasible
    """
    result = _evaluate_L50(delta_w, param, lambda0, cache_dir, eval_cache)
    if result is None:
        return None

    L50_te = result['L_50_dict']['te']
    L50_tm = result['L_50_dict']['tm']
    kappa_te = result['kappa_dict']['te']
    kappa_tm = result['kappa_dict']['tm']
    Delta_te = result['Delta_dict']['te']
    Delta_tm = result['Delta_dict']['tm']

    # Reject infeasible or non-finite points outright
    if (not (np.isfinite(L50_te) and np.isfinite(L50_tm)) or
            (L50_te <= 0) or (L50_tm <= 0)):
        return None
    if abs(Delta_te) > kappa_te or abs(Delta_tm) > kappa_tm:
        return None

    F_val = L50_te - L50_tm
    return float(F_val)


def _F_prime_of_dw(delta_w, h, param, lambda0, cache_dir,
                   eps_min=1e-6, allow_one_sided=False, eval_cache=None):
    """
    Compute F'(Δw) using finite differences.
    
    Args:
        delta_w: Width asymmetry parameter (µm)
        h: Step size for finite differences (µm)
        param: Parameter namespace template
        lambda0: Design wavelength (µm)
        cache_dir: Directory for cache file
        eps_min: Minimum |F'| threshold (default: 1e-6)
        allow_one_sided: If True, use one-sided derivative if one side is infeasible
    
    Returns:
        F'(Δw) (float) or None if infeasible/singular
    """
    # Evaluate F(Δw + h) and F(Δw - h)
    F_plus = _F_of_dw(delta_w + h, param, lambda0, cache_dir, eval_cache)
    F_minus = _F_of_dw(delta_w - h, param, lambda0, cache_dir, eval_cache)
    
    # Try central difference first
    if F_plus is not None and F_minus is not None:
        F_prime = (F_plus - F_minus) / (2.0 * h)
        if abs(F_prime) >= eps_min:
            return float(F_prime)
        return None
    
    # If one side is infeasible and one-sided is allowed, try forward/backward difference
    if allow_one_sided:
        if F_plus is not None and F_minus is None:
            # Forward difference: F' ≈ [F(Δw+h) - F(Δw)] / h
            F_center = _F_of_dw(delta_w, param, lambda0, cache_dir, eval_cache)
            if F_center is not None:
                F_prime = (F_plus - F_center) / h
                if abs(F_prime) >= eps_min:
                    return float(F_prime)
        elif F_minus is not None and F_plus is None:
            # Backward difference: F' ≈ [F(Δw) - F(Δw-h)] / h
            F_center = _F_of_dw(delta_w, param, lambda0, cache_dir, eval_cache)
            if F_center is not None:
                F_prime = (F_center - F_minus) / h
                if abs(F_prime) >= eps_min:
                    return float(F_prime)
    
    return None

def _F_prime_of_dw_norm(delta_w, h, param, lambda0, cache_dir,
                        eps_min=1e-6, allow_one_sided=False, eval_cache=None):
    """Compute normalized F'(Δw) using finite differences, reusing cached L50 values."""
    def eval_norm(dw):
        res = _evaluate_L50(dw, param, lambda0, cache_dir, eval_cache)
        if res is None:
            return None
        Lte = res['L_50_dict']['te']
        Ltm = res['L_50_dict']['tm']
        if not (np.isfinite(Lte) and np.isfinite(Ltm)) or (Lte <= 0) or (Ltm <= 0):
            return None
        m = 0.5 * (Lte + Ltm)
        return float((Lte - Ltm) / max(m, 1e-12))
    
    Fp = eval_norm(delta_w + h)
    Fm = eval_norm(delta_w - h)
    if Fp is not None and Fm is not None:
        d = (Fp - Fm) / (2.0 * h)
        return float(d) if abs(d) >= eps_min else None
    if allow_one_sided:
        Fc = eval_norm(delta_w)
        if Fc is not None and Fp is not None:
            d = (Fp - Fc) / h
            return float(d) if abs(d) >= eps_min else None
        if Fc is not None and Fm is not None:
            d = (Fc - Fm) / h
            return float(d) if abs(d) >= eps_min else None
    return None


def _solve_delta_w_star_newton(w, g, t, lambda0, param_template, delta_w_init=None, 
                                cache_dir="data", abs_tol=0.05, rel_tol=0.01,
                                search_min=-0.25, search_max=+0.25,
                                hard_min=-0.50, hard_max=+0.50,
                                h_init=0.01, h_min=0.001, max_iter=50, eps_min=1e-6,
                                eval_cache=None):
    """
    Newton-based solver for delta_w* that equalizes L₅₀,TE and L₅₀,TM.
    
    Args:
        w: Waveguide width (µm)
        g: Coupling gap (µm)
        t: Waveguide thickness (µm)
        lambda0: Design wavelength (µm)
        param_template: Parameter namespace template (for medium, etc.)
        delta_w_init: Optional initial guess (µm)
        cache_dir: Directory for cache file
        abs_tol: Absolute tolerance for |F| (µm)
        rel_tol: Relative tolerance for |F|/mean(L₅₀)
        search_min: Search box minimum (µm), default -0.25
        search_max: Search box maximum (µm), default +0.25
        hard_min: Hard clip minimum (µm), default -0.50
        hard_max: Hard clip maximum (µm), default +0.50
        h_init: Initial step size for finite differences (µm)
        h_min: Minimum step size (µm)
        max_iter: Maximum Newton iterations
        eps_min: Minimum |F'| threshold
    
    Returns:
        (delta_w_star, diagnostics_dict) or (None, diagnostics_dict) if infeasible
    """
    from types import SimpleNamespace
    
    # Create temporary param namespace for evaluation
    param = SimpleNamespace(**param_template.__dict__)
    param.wg_width = float(w)
    param.coupling_gap = float(g)
    param.wg_thick = float(t)
    
    # Initialize diagnostics
    diagnostics = {
        'converged': False,
        'n_iterations': 0,
        'abs_err': np.inf,
        'rel_err': np.inf,
        'L50_te': np.nan,
        'L50_tm': np.nan,
        'kappa_te': np.nan,
        'kappa_tm': np.nan,
        'Delta_te': np.nan,
        'Delta_tm': np.nan,
        'fallback_used': 'none',
        'final_F': np.nan,
        'final_F_prime': np.nan,
        'tolerance_met': False,
    }
    
    # Initialize Newton iteration
    delta_w_k = float(delta_w_init) if delta_w_init is not None else 0.0
    h = float(h_init)
    
    # Track best candidate
    best_candidate = None
    best_F_abs = np.inf
    
    # Track stuck detection
    stuck_counter = 0
    last_delta_w = None
    stuck_threshold = 5  # Number of iterations with no change before considering stuck
    
    # Progress bar for Newton iterations
    print(f"[Δw* Solver] Starting Newton iteration (max {max_iter} iterations)...", flush=True)
    
    if eval_cache is None:
        eval_cache = {}

    def _get_L50_result(dw, verbose=False):
        """Get L50 result, using shared cache if available."""
        if verbose:
            print(f"[Δw* Solver] Computing mode solver for Δw={dw:.6f} µm...", flush=True)
        result = _evaluate_L50(dw, param, lambda0, cache_dir, eval_cache)
        if verbose and result is not None:
            L50_te = result['L_50_dict']['te']
            L50_tm = result['L_50_dict']['tm']
            print(f"[Δw* Solver] Mode solver complete: L50_TE={L50_te:.3f} µm, L50_TM={L50_tm:.3f} µm", flush=True)
        return result
    
    def _Fnorm(dw):
        """Normalized objective: (L50_TE - L50_TM) / mean(L50)."""
        result = _get_L50_result(dw)
        if result is None:
            return None
        Lte = result['L_50_dict']['te']
        Ltm = result['L_50_dict']['tm']
        if not (np.isfinite(Lte) and np.isfinite(Ltm)) or (Lte <= 0) or (Ltm <= 0):
            return None
        m = 0.5 * (Lte + Ltm)
        return float((Lte - Ltm) / max(m, 1e-12))

    # Main Newton iteration loop
    print(f"[Δw* Solver] Iteration 1/{max_iter}: evaluating at Δw={delta_w_k:.6f} µm...", flush=True)
    for k in range(max_iter):
        diagnostics['n_iterations'] = k + 1
        
        if k > 0:
            print(f"[Δw* Solver] Iteration {k+1}/{max_iter}: evaluating at Δw={delta_w_k:.6f} µm...", flush=True)

        # Check if stuck (delta_w not changing)
        if last_delta_w is not None and abs(delta_w_k - last_delta_w) < 1e-8:
            stuck_counter += 1
            if stuck_counter >= stuck_threshold:
                print(f"[Δw* Solver] Stuck at Δw={delta_w_k:.6f} µm for {stuck_counter} iterations, breaking", flush=True)
                break
        else:
            stuck_counter = 0
        last_delta_w = delta_w_k

        # Evaluate F(Δw_k) and get L50 result (cached)
        # Only show verbose mode solver messages for first few iterations
        result_k = _get_L50_result(delta_w_k, verbose=(k < 3))
        if result_k is None:
            F_k = None
        else:
            L50_te_k = result_k['L_50_dict']['te']
            L50_tm_k = result_k['L_50_dict']['tm']
            kappa_te = result_k['kappa_dict']['te']
            kappa_tm = result_k['kappa_dict']['tm']
            Delta_te = result_k['Delta_dict']['te']
            Delta_tm = result_k['Delta_dict']['tm']
            
            # Check feasibility
            infeasible = False
            if abs(Delta_te) > kappa_te:
                infeasible = True
            if abs(Delta_tm) > kappa_tm:
                infeasible = True
            
            if not (np.isfinite(L50_te_k) and np.isfinite(L50_tm_k)):
                F_k = None
            elif infeasible:
                F_k = None
            else:
                F_k = L50_te_k - L50_tm_k

        if F_k is None:
            # Infeasible point - try to find nearby feasible point
            infeasible_dw = delta_w_k  # Store for diagnostics

            # Try to find a nearby feasible point by searching around the infeasible point
            search_offsets = [0.01, -0.01, 0.02, -0.02, 0.05, -0.05]
            feasible_found = False

            for offset in search_offsets:
                test_dw = delta_w_k + offset
                test_dw = np.clip(test_dw, hard_min, hard_max)
                result_test = _get_L50_result(test_dw)
                if result_test is not None:
                    L50_te_test = result_test['L_50_dict']['te']
                    L50_tm_test = result_test['L_50_dict']['tm']
                    kappa_te_test = result_test['kappa_dict']['te']
                    kappa_tm_test = result_test['kappa_dict']['tm']
                    Delta_te_test = result_test['Delta_dict']['te']
                    Delta_tm_test = result_test['Delta_dict']['tm']
                    
                    # Check feasibility
                    if (abs(Delta_te_test) <= kappa_te_test and 
                        abs(Delta_tm_test) <= kappa_tm_test and
                        np.isfinite(L50_te_test) and np.isfinite(L50_tm_test)):
                        # Found a feasible point nearby
                        delta_w_k = test_dw
                        result_k = result_test
                        F_k = L50_te_test - L50_tm_test
                        feasible_found = True
                        abs_err_test = abs(L50_te_test - L50_tm_test)
                        if k < 5:
                            print(f"[Δw* Solver] Found nearby feasible point at iteration {k+1}: Δw={infeasible_dw:.6f} → {test_dw:.6f} µm, |L50_TE - L50_TM|={abs_err_test:.6f} µm", flush=True)
                        break

            if not feasible_found:
                # No nearby feasible point found - use best candidate
                if best_candidate is not None:
                    delta_w_k = best_candidate[0]
                    F_k = best_candidate[1]
                    L50_te_final = best_candidate[2]
                    L50_tm_final = best_candidate[3]
                    abs_err_um = abs(L50_te_final - L50_tm_final)
                    diagnostics['fallback_used'] = 'best_candidate'
                    print(f"[Δw* Solver] Infeasible point at iteration {k+1} (Δw={infeasible_dw:.6f} µm), no nearby feasible point, using best candidate (Δw={delta_w_k:.6f} µm): |L50_TE - L50_TM| = {abs_err_um:.6f} µm", flush=True)
                    break
                else:
                    print(f"[Δw* Solver] No feasible points found after {k+1} iterations (Δw={delta_w_k:.6f} µm)", flush=True)
                    diagnostics['converged'] = False
                    return None, diagnostics

        # Get L50 values (already computed above, reuse from cache)
        if result_k is None:
            continue
        L50_te_k = result_k['L_50_dict']['te']
        L50_tm_k = result_k['L_50_dict']['tm']

        # compute normalized objective for Newton to improve conditioning
        mean_L50_k = 0.5 * (L50_te_k + L50_tm_k) if (np.isfinite(L50_te_k) and np.isfinite(L50_tm_k)) else np.inf
        F_norm_k = (L50_te_k - L50_tm_k) / max(mean_L50_k, 1e-12)

        # Update best candidate (store both raw and normalized)
        F_abs_k = abs(F_norm_k)
        if F_abs_k < best_F_abs:
            best_candidate = (delta_w_k, F_norm_k, L50_te_k, L50_tm_k)
            best_F_abs = F_abs_k

        # Check convergence (using abs error in microns for diagnostics)
        abs_err_microns = abs(L50_te_k - L50_tm_k)
        rel_err_k = abs_err_microns / mean_L50_k if mean_L50_k > 0 else np.inf
        
        # Print progress every few iterations
        if (k + 1) % 5 == 0 or k == 0:
            print(f"[Δw* Solver] Iteration {k+1}: Δw={delta_w_k:.6f} µm, "
                  f"|L50_TE-L50_TM|={abs_err_microns:.4f} µm (target: {abs_tol:.4f} µm)", flush=True)
        
        if abs_err_microns <= abs_tol or rel_err_k <= rel_tol:
            diagnostics['converged'] = True
            diagnostics['abs_err'] = abs_err_microns
            diagnostics['rel_err'] = rel_err_k
            diagnostics['L50_te'] = L50_te_k
            diagnostics['L50_tm'] = L50_tm_k
            diagnostics['final_F'] = F_k
            diagnostics['tolerance_met'] = True
            # Compute final derivative for diagnostics (raw, not normalized)
            F_prime_final = _F_prime_of_dw(delta_w_k, h, param, lambda0, cache_dir, eps_min, eval_cache=eval_cache)
            diagnostics['final_F_prime'] = F_prime_final if F_prime_final is not None else np.nan
            # Get final CMT parameters (already computed, reuse from cache)
            if result_k is not None:
                diagnostics['kappa_te'] = result_k['kappa_dict']['te']
                diagnostics['kappa_tm'] = result_k['kappa_dict']['tm']
                diagnostics['Delta_te'] = result_k['Delta_dict']['te']
                diagnostics['Delta_tm'] = result_k['Delta_dict']['tm']
            print(f"[Δw* Solver] ✓ Converged after {k+1} iterations: |L50_TE - L50_TM| = {abs_err_microns:.6f} µm", flush=True)
            return delta_w_k, diagnostics

        # Compute derivative on normalized objective
        F_prime_k = _F_prime_of_dw_norm(delta_w_k, h, param, lambda0, cache_dir, eps_min,
                                        allow_one_sided=False, eval_cache=eval_cache)
        if F_prime_k is None:
            F_prime_k = _F_prime_of_dw_norm(delta_w_k, h, param, lambda0, cache_dir, eps_min,
                                            allow_one_sided=True, eval_cache=eval_cache)
            if F_prime_k is not None and k < 5:
                print(f"[Δw* Solver] Using one-sided derivative (normalized) at iteration {k+1} (Δw={delta_w_k:.6f}, h={h:.6f})", flush=True)
        diagnostics['final_F_prime'] = F_prime_k if F_prime_k is not None else np.nan

        if F_prime_k is None:
            # Try reducing h and retrying with one-sided derivative
            if h > h_min * 2:
                h_reduced = max(h / 2.0, h_min)
                F_prime_k = _F_prime_of_dw_norm(delta_w_k, h_reduced, param, lambda0, cache_dir, eps_min,
                                                allow_one_sided=True, eval_cache=eval_cache)
                if F_prime_k is not None:
                    h = h_reduced
                    diagnostics['final_F_prime'] = F_prime_k
                    if k < 5:
                        print(f"[Δw* Solver] Reduced h to {h:.6f} and computed F' (normalized) using one-sided derivative", flush=True)
            # fallback to best candidate if still none
            if F_prime_k is None:
                if best_candidate is not None:
                    delta_w_k = best_candidate[0]
                    L50_te_final = best_candidate[2]
                    L50_tm_final = best_candidate[3]
                    abs_err_um = abs(L50_te_final - L50_tm_final)
                    diagnostics['fallback_used'] = 'best_candidate'
                    print(f"[Δw* Solver] Derivative (normalized) failed at iteration {k+1}, using best candidate: |L50_TE - L50_TM| = {abs_err_um:.6f} µm", flush=True)
                    break
                else:
                    diagnostics['converged'] = False
                    return None, diagnostics

        # Newton update: Δw_{k+1} = Δw_k - F_norm_k / F_prime_k
        raw_newton_step = -F_norm_k / F_prime_k
        # Limit step size to prevent huge jumps (max step = 20% of search range)
        max_step = 0.20 * (hard_max - hard_min)
        newton_step = np.clip(raw_newton_step, -max_step, max_step)
        # Apply damping if step was clipped OR if F' is small (indicates potential instability)
        damping_factor = 1.0
        if abs(raw_newton_step) > max_step:
            damping_factor = 0.5
            if k == 0:
                print(f"[Δw* Solver] Limiting Newton step size: raw={raw_newton_step:.6f} → clipped={newton_step:.6f} µm", flush=True)
        elif abs(F_prime_k) < eps_min * 10:
            damping_factor = max(0.3, abs(F_prime_k) / (eps_min * 10))
            if k < 3:
                print(f"[Δw* Solver] Applying damping (factor={damping_factor:.2f}) for small F'={F_prime_k:.6e} (normalized)", flush=True)
        newton_step = newton_step * damping_factor
        delta_w_new = delta_w_k + newton_step
        # Project to bounds
        delta_w_new = np.clip(delta_w_new, hard_min, hard_max)
        # Check for divergence (large step)
        step_size = abs(delta_w_new - delta_w_k)
        if step_size > 0.5 * (hard_max - hard_min):
            h = max(h / 2.0, h_min)
            if h <= h_min:
                if best_candidate is not None:
                    delta_w_k = best_candidate[0]
                    L50_te_final = best_candidate[2]
                    L50_tm_final = best_candidate[3]
                    abs_err_um = abs(L50_te_final - L50_tm_final)
                    diagnostics['fallback_used'] = 'best_candidate'
                    print(f"[Δw* Solver] Divergence detected at iteration {k+1}, using best candidate: |L50_TE - L50_TM| = {abs_err_um:.6f} µm", flush=True)
                    break
                else:
                    diagnostics['converged'] = False
                    return None, diagnostics
            continue
        # Update for next iteration
        delta_w_k = delta_w_new
    
    # Max iterations reached - return best candidate
    print(f"[Δw* Solver] Max iterations ({max_iter}) reached", flush=True)
    if best_candidate is not None:
        delta_w_k = best_candidate[0]
        F_k = best_candidate[1]  # normalized F, kept for reference but not used in diagnostics
        L50_te_final = best_candidate[2]
        L50_tm_final = best_candidate[3]
        
        # Compute physical mismatch
        abs_err_um = abs(L50_te_final - L50_tm_final)
        mean_L50 = 0.5 * (L50_te_final + L50_tm_final) if (np.isfinite(L50_te_final) and np.isfinite(L50_tm_final)) else np.inf
        
        diagnostics['converged'] = False  # Didn't converge within tolerance
        diagnostics['abs_err'] = abs_err_um
        diagnostics['rel_err'] = abs_err_um / mean_L50 if mean_L50 > 0 else np.inf
        diagnostics['L50_te'] = L50_te_final
        diagnostics['L50_tm'] = L50_tm_final
        diagnostics['final_F'] = abs_err_um  # keep final_F in physical units
        
        fallback_str = diagnostics.get('fallback_used', 'none')
        if fallback_str != 'none':
            print(f"[Δw* Solver] Using {fallback_str} fallback: |L50_TE - L50_TM| = {abs_err_um:.6f} µm", flush=True)
        else:
            print(f"[Δw* Solver] Final result: |L50_TE - L50_TM| = {abs_err_um:.6f} µm", flush=True)
        
        # Compute final derivative for diagnostics
        F_prime_final = _F_prime_of_dw(delta_w_k, h, param, lambda0, cache_dir, eps_min, eval_cache=eval_cache)
        diagnostics['final_F_prime'] = F_prime_final if F_prime_final is not None else np.nan
        
        # Check if best candidate meets tolerance
        if diagnostics['abs_err'] <= abs_tol or diagnostics['rel_err'] <= rel_tol:
            diagnostics['tolerance_met'] = True
        else:
            diagnostics['tolerance_met'] = False
        
        # Get final CMT parameters (reuse from cache if available)
        result_final = _get_L50_result(delta_w_k)
        if result_final is not None:
            diagnostics['kappa_te'] = result_final['kappa_dict']['te']
            diagnostics['kappa_tm'] = result_final['kappa_dict']['tm']
            diagnostics['Delta_te'] = result_final['Delta_dict']['te']
            diagnostics['Delta_tm'] = result_final['Delta_dict']['tm']
        
        return delta_w_k, diagnostics
    else:
        # No feasible points found
        diagnostics['converged'] = False
        return None, diagnostics


def solve_delta_w_star(w, g, t, lambda0, param_template, delta_w_init=None, cache_dir="data", 
                       abs_tol=0.05, rel_tol=0.01, search_min=-0.25, search_max=+0.25, 
                       hard_min=-0.50, hard_max=+0.50,
                       h_init=0.01, h_min=0.001, max_iter=50, eps_min=1e-6, seed_step=0.005):
    """
    Solve for delta_w* that equalizes L_50_TE and L_50_TM at lambda0.
    
    Uses Newton-based iterative solver with finite difference derivatives.
    No grid scanning or FDTD required - only local mode solves.
    
    Args:
        w: Waveguide width (µm)
        g: Coupling gap (µm)
        t: Waveguide thickness (µm)
        lambda0: Design wavelength (µm)
        param_template: Parameter namespace template (for medium, etc.)
        delta_w_init: Optional initial guess for warm-start (µm)
        cache_dir: Directory for cache file
        abs_tol: Absolute tolerance for L50_TE - L50_TM matching (µm), default 0.05
        rel_tol: Relative tolerance for L50_TE - L50_TM matching (fraction), default 0.01 (1%)
        search_min: Primary search box minimum (µm), default -0.25
        search_max: Primary search box maximum (µm), default +0.25
        hard_min: Hard clip minimum (µm), default -0.50
        hard_max: Hard clip maximum (µm), default +0.50
        seed_step: Step size for initial seed scanning (µm), default 0.005
        h_init: Initial step size for finite differences (µm), default 0.01
        h_min: Minimum step size for finite differences (µm), default 0.001
        max_iter: Maximum Newton iterations, default 50
        eps_min: Minimum |F'| threshold for derivative computation, default 1e-6
    
    Returns:
        (delta_w*, diagnostics_dict) or (None, diagnostics_dict) if infeasible
        diagnostics_dict contains: reached, n_brackets, final_J, abs_err, rel_err,
        L50_te, L50_tm, kappa_te, kappa_tm, Delta_te, Delta_tm, refine_used, tolerance_met
    """
    # Try a robust bracketed root first on G(Δw)=log(L50_TE/L50_TM)
    # Build a temporary param namespace
    from types import SimpleNamespace
    param = SimpleNamespace(**param_template.__dict__)
    param.wg_width = float(w); param.coupling_gap = float(g); param.wg_thick = float(t)
    eval_cache = {}
    global _LAST_DELTA_W_HINT
    center_guess = delta_w_init
    if center_guess is None and _LAST_DELTA_W_HINT is not None:
        center_guess = _LAST_DELTA_W_HINT
    if center_guess is None:
        center_guess = 0.0
    center_guess = float(np.clip(center_guess, search_min, search_max))

    a, b, Ga, Gb = _find_feasible_bracket(
        param,
        lambda0,
        cache_dir,
        search_min,
        search_max,
        hard_min,
        hard_max,
        center_guess=center_guess,
        step=max(seed_step, 0.01),
        eval_cache=eval_cache
    )
    if a is not None and b is not None:
        print(f"[Δw* Solver] Bracket found: Δw={a:+.4f} µm (G={Ga:+.3e}), Δw={b:+.4f} µm (G={Gb:+.3e})", flush=True)
    bracket_root = None
    bracket_diag = None

    if a is not None and b is not None and _HAS_SCIPY and (root_scalar is not None or brentq is not None):
        bracket_ok = True
        for frac in (0.25, 0.5, 0.75):
            mid = a + (b - a) * frac
            mid_val, _, _ = _G_of_dw(mid, param, lambda0, cache_dir, eval_cache)
            if mid_val is None:
                bracket_ok = False
                break
        if bracket_ok:
            def G(dw):
                val, _, _ = _G_of_dw(dw, param, lambda0, cache_dir, eval_cache)
                if val is None:
                    raise ValueError(f"Infeasible evaluation at Δw={dw:.6f} µm")
                return val
            try:
                if root_scalar is not None:
                    sol = root_scalar(G, bracket=[a, b], method='toms748', xtol=1e-6, rtol=1e-6, maxiter=80)
                    if not sol.converged:
                        raise RuntimeError("root_scalar did not converge")
                    root = float(sol.root)
                else:
                    root = brentq(G, a, b, xtol=1e-6)

                res = _evaluate_L50(root, param, lambda0, cache_dir, eval_cache)
                if res is not None:
                    L50_te = res['L_50_dict']['te']; L50_tm = res['L_50_dict']['tm']
                    kappa_te = res['kappa_dict']['te']; kappa_tm = res['kappa_dict']['tm']
                    Delta_te = res['Delta_dict']['te']; Delta_tm = res['Delta_dict']['tm']
                    mean_L50 = max(0.5 * (L50_te + L50_tm), 1e-12)
                    abs_err = abs(L50_te - L50_tm)
                    rel_err = abs_err / mean_L50
                    meets_tol = (abs_err <= abs_tol) or (rel_err <= rel_tol)
                    diagnostics = {
                        'reached': True,
                        'n_brackets': 1,
                        'final_J': abs_err,
                        'abs_err': abs_err,
                        'rel_err': rel_err,
                        'L50_te': L50_te, 'L50_tm': L50_tm,
                        'kappa_te': kappa_te, 'kappa_tm': kappa_tm,
                        'Delta_te': Delta_te, 'Delta_tm': Delta_tm,
                        'refine_used': False,
                        'tolerance_met': meets_tol,
                    }
                    if meets_tol:
                        _LAST_DELTA_W_HINT = float(root)
                        return float(root), diagnostics
                    else:
                        print(f"[Δw* Solver] Bracket residual |L50_TE - L50_TM| = {abs_err:.4f} µm (rel {rel_err*100:.2f}%), handing off to Newton for refinement", flush=True)
                        delta_w_init = float(root)
                        bracket_root = float(root)
                        bracket_diag = diagnostics
            except Exception:
                # fall through to Newton if bracket solve fails
                pass

    # If no bracket found and no initial guess provided, run lightweight seed search
    if a is None and delta_w_init is None:
        print(f"[Δw* Solver] No root bracket found, probing neighborhood of Δw={center_guess:.4f} µm (step {seed_step} µm)...", flush=True)
        best_seed, best_abs_err = _heuristic_seed_search(
            param,
            lambda0,
            cache_dir,
            search_min,
            search_max,
            center_guess=center_guess,
            seed_step=seed_step,
            abs_tol=abs_tol,
            eval_cache=eval_cache
        )
        if best_seed is not None:
            delta_w_init = float(best_seed[0])
            print(f"[Δw* Solver] Heuristic seed set to Δw={delta_w_init:.6f} µm (|L50_TE - L50_TM|={best_abs_err:.6f} µm)", flush=True)
        else:
            print(f"[Δw* Solver] Heuristic seed search failed, using Δw=0.0 µm", flush=True)

    # Call Newton-based solver as fallback
    delta_w_star, newton_diagnostics = _solve_delta_w_star_newton(
        w=w,
        g=g,
        t=t,
        lambda0=lambda0,
        param_template=param_template,
        delta_w_init=delta_w_init,
        cache_dir=cache_dir,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        search_min=search_min,
        search_max=search_max,
        hard_min=hard_min,
        hard_max=hard_max,
        h_init=h_init,
        h_min=h_min,
        max_iter=max_iter,
        eps_min=eps_min,
        eval_cache=eval_cache
    )

    # Map Newton diagnostics to expected format for backward compatibility
    diagnostics = {
        'reached': newton_diagnostics.get('converged', False) or (delta_w_star is not None),
        'n_brackets': 1 if bracket_diag is not None else 0,
        'final_J': newton_diagnostics.get('abs_err', np.inf),
        'abs_err': newton_diagnostics.get('abs_err', np.inf),
        'rel_err': newton_diagnostics.get('rel_err', np.inf),
        'L50_te': newton_diagnostics.get('L50_te', np.nan),
        'L50_tm': newton_diagnostics.get('L50_tm', np.nan),
        'kappa_te': newton_diagnostics.get('kappa_te', np.nan),
        'kappa_tm': newton_diagnostics.get('kappa_tm', np.nan),
        'Delta_te': newton_diagnostics.get('Delta_te', np.nan),
        'Delta_tm': newton_diagnostics.get('Delta_tm', np.nan),
        'refine_used': (bracket_diag is not None) or (newton_diagnostics.get('fallback_used', 'none') != 'none'),
        'tolerance_met': newton_diagnostics.get('tolerance_met', False),
    }

    if delta_w_star is not None:
        _LAST_DELTA_W_HINT = float(delta_w_star)

    return delta_w_star, diagnostics
