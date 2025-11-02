import numpy as np
from pathlib import Path
import csv
import bisect
from building_utils import update_param_derived


def _extract_cost_value(estimate):
    """Extract numeric cost value from estimate (handles float, dict, or object formats)."""
    if estimate is None:
        return 0.0
    if isinstance(estimate, (float, int)):
        return float(estimate)
    if isinstance(estimate, dict):
        return float(estimate.get('credits', estimate.get('cost', 0.0)))
    # Try to get credits attribute
    try:
        credits = getattr(estimate, 'credits', None)
        if credits is not None:
            return float(credits)
    except Exception:
        pass
    return 0.0


def eta_at_1550_from_summary(summary):
    """Extract coupling ratio at 1550nm from summary dict."""
    lam = summary["lam"]
    eta = summary["eta"]
    k = int(np.argmin(np.abs(lam - 1.55)))
    return float(eta[k])


def _cband_mask(lam, lam_lo=1.530, lam_hi=1.565):
    """Create boolean mask for C-band wavelengths."""
    lam = np.asarray(lam)
    return (lam >= lam_lo) & (lam <= lam_hi)


def _band_avg(arr, mask):
    """Compute average of array over mask."""
    arr = np.asarray(arr)
    if mask is None:
        return float(np.mean(arr))
    arr_m = arr[mask]
    return float(np.mean(arr_m)) if arr_m.size else float(np.mean(arr))


def compute_objective(results_te, results_tm, param, weights):
    """Return metrics and Score for a TE/TM pair.

    Metrics returned:
      - Vbar_te, Vbar_tm, Vbar_avg (band-averaged)
      - DeltaEta_avg (band-averaged |eta_te-eta_tm|)
      - Lc (current coupling length)
      - Score (to maximize)
    
    Args:
        results_te: TE mode results dict with 'lam', 'eta', 'V'
        results_tm: TM mode results dict with 'lam', 'eta', 'V'
        param: parameter namespace object
        weights: optimization weights dict with keys: alpha, beta, gamma, L_ref
    """
    lam = results_te["lam"]
    mask = _cband_mask(lam)
    # band-averaged visibility per pol
    Vbar_te = _band_avg(results_te["V"], mask)
    Vbar_tm = _band_avg(results_tm["V"], mask)
    Vbar_avg = 0.5 * (Vbar_te + Vbar_tm)
    # polarization imbalance (band-averaged)
    DeltaEta = np.abs(np.asarray(results_te["eta"]) - np.asarray(results_tm["eta"]))
    DeltaEta_avg = _band_avg(DeltaEta, mask)
    # tolerance placeholder (Stage 1)
    sigma_tol = 0.0
    alpha = float(weights.get("alpha", 2.0))
    beta = float(weights.get("beta", 0.0))
    gamma = float(weights.get("gamma", 0.05))
    L_ref = float(weights.get("L_ref", 10.0))
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


def sweep(grid_gap, grid_width, grid_length, param, weights, run_single_fn, dry_run=False, save_top_k=10, outdir="results",lambda_single=None):
    """Sweep over (coupling_gap, wg_width) with optional L_c fine-tuning.
    
    Note: coupling_length is now derived from supermode analysis by default.
    If grid_length is provided, it enables fine-tuning sweep around derived values.

    Args:
        grid_gap, grid_width: iterables of values in µm
        grid_length: Optional iterable for L_c fine-tuning (None for derived-only)
        param: parameter namespace object (will be modified during sweep)
        weights: optimization weights dict
        run_single_fn: function to run single simulation (from dc_study.run_single)
        dry_run: if True, only preflight without running
        save_top_k: number of top results to track
        outdir: output directory for CSV file
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "summary_sweep.csv"
    
    # Handle optional grid_length
    if grid_length is None:
        length_values = [None]  # Single derived value
    else:
        length_values = grid_length
    
    total_configs = len(grid_gap) * len(grid_width) * len(length_values)
    print(f"\n[SWEEP] |g|={len(grid_gap)} |w|={len(grid_width)} |L|={len(length_values)} total={total_configs} configurations")
    if grid_length is None:
        print("[SWEEP] L_c will be derived from supermode analysis (not swept)")
    else:
        print(f"[SWEEP] L_c fine-tuning enabled: {len(length_values)} values")

    # store best few results
    best = []  # list of (Score, row_dict)
    # accumulate cost estimates for dry_run
    total_cost = 0.0
    cost_count = 0
    
    # Store all rows in sorted order (by Score descending)
    all_rows = []  # list of (Score, row_dict) - sorted descending by Score
    
    # Define CSV header and column order (removed coupling_length_um, added coupling_trim_factor)
    csv_header = [
        "coupling_gap_um", "wg_width_um",
        "eta_te_1550", "eta_tm_1550", "DeltaEta_1550",
        "Vbar_te", "Vbar_tm", "Vbar_avg", "DeltaEta_avg", "sigma_tol", "Lc", "coupling_trim_factor", "Score"
    ]
    
    # Write initial CSV header
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header)

    # loop
    for g in grid_gap:
        for wwidth in grid_width:
            # Set geometry parameters
            setattr(param, "coupling_gap", float(g))
            setattr(param, "wg_width", float(wwidth))
            
            # coupling_length will be computed in update_param_derived()
            update_param_derived(param)  # Computes derived coupling_length
            base_Lc = param.coupling_length  # Store for reference
            
            # Fine-tuning loop (if enabled)
            for Lc in length_values:
                if Lc is not None:
                    # Override with fine-tuned value
                    setattr(param, "coupling_length", float(Lc))
                    update_param_derived(param)  # Recompute size_x with new Lc
                    Lc_display = Lc
                else:
                    # Use derived value (already computed above)
                    Lc_display = param.coupling_length
                
                print(f"\n[SWEEP] g={g:.3f}µm, w={wwidth:.3f}µm, Lc={Lc_display:.3f}µm")
                
                # Build task tag
                if Lc is not None:
                    task_tag = f"3d_g{g:.3f}_w{wwidth:.3f}_L{Lc:.3f}"
                else:
                    task_tag = f"3d_g{g:.3f}_w{wwidth:.3f}"
                
                summaries = {}
                config_cost = 0.0
                for pol in ("te", "tm"):
                    s = run_single_fn(param, pol=pol, task_tag=task_tag, dry_run=dry_run, lambda_single=lambda_single)
                    if s is not None:
                        summaries[pol] = s
                        # Accumulate cost if dry_run
                        if dry_run and "cost_estimate" in s:
                            cost_val = _extract_cost_value(s["cost_estimate"])
                            config_cost += cost_val
                            total_cost += cost_val
                            cost_count += 1

                if dry_run:
                    if config_cost > 0:
                        print(f"[SWEEP-3D] Config cost: {config_cost:.3f} FlexCredits (TE + TM)")
                    continue  # Skip objective computation in dry_run

                if not summaries or ("te" not in summaries) or ("tm" not in summaries):
                    continue  # skip if failed

                # point metrics (only when not dry_run)
                obj = compute_objective(summaries["te"], summaries["tm"], param, weights)
                eta_te_1550 = eta_at_1550_from_summary(summaries["te"]) 
                eta_tm_1550 = eta_at_1550_from_summary(summaries["tm"]) 
                d_eta_1550 = abs(eta_te_1550 - eta_tm_1550)

                row = {
                    "coupling_gap_um": f"{float(g):.6f}",
                    "wg_width_um": f"{float(wwidth):.6f}",
                    "eta_te_1550": f"{eta_te_1550:.6f}",
                    "eta_tm_1550": f"{eta_tm_1550:.6f}",
                    "DeltaEta_1550": f"{d_eta_1550:.6f}",
                    "Vbar_te": f"{obj['Vbar_te']:.6f}",
                    "Vbar_tm": f"{obj['Vbar_tm']:.6f}",
                    "Vbar_avg": f"{obj['Vbar_avg']:.6f}",
                    "DeltaEta_avg": f"{obj['DeltaEta_avg']:.6f}",
                    "sigma_tol": f"{obj['sigma_tol']:.6f}",
                    "Lc": f"{obj['Lc']:.6f}",
                    "coupling_trim_factor": f"{getattr(param, 'coupling_trim_factor', 0.075):.6f}",
                    "Score": f"{obj['Score']:.6f}",
                }

                # Insert row into sorted list (descending by Score)
                score_val = obj["Score"]
                # Use bisect to find insertion point for descending order
                # Since we want descending, we insert at position based on negative score
                insert_idx = bisect.bisect_left([-s for s, _ in all_rows], -score_val)
                all_rows.insert(insert_idx, (score_val, row))
                
                # Rewrite entire CSV file with sorted rows
                with csv_path.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(csv_header)
                    for _, r in all_rows:
                        # Write row in column order matching header
                        w.writerow([r[col] for col in csv_header])

                # maintain top-k
                best.append((obj["Score"], row))
                best.sort(key=lambda t: t[0], reverse=True)
                if len(best) > save_top_k:
                    best.pop()

    # print cost summary for dry_run
    if dry_run and cost_count > 0:
        avg_cost_per_simulation = total_cost / cost_count
        estimated_total = avg_cost_per_simulation * total_configs * 2  # *2 for TE + TM
        configs_sampled = cost_count // 2  # Each config has 2 simulations (TE + TM)
        print(f"\n[SWEEP-3D] Cost Estimate Summary (dry_run):")
        print(f"  Total configurations: {total_configs}")
        print(f"  Total simulations: {total_configs * 2} (TE + TM per config)")
        print(f"  Cost estimates collected: {cost_count} simulations from {configs_sampled} configurations")
        print(f"  Average cost per simulation: {avg_cost_per_simulation:.3f} FlexCredits")
        print(f"  Estimated total cost: {estimated_total:.3f} FlexCredits")

    # print top results
    if best:
        print("\n[SWEEP-3D] Top candidates (by Score):")
        for rank, (score, row) in enumerate(best, 1):
            print(f"  #{rank:02d} Score={float(score):.4f}  g={row['coupling_gap_um']} µm  w={row['wg_width_um']} µm  Lc={row['coupling_length_um']} µm  Vbar_avg={row['Vbar_avg']}  Δη_avg={row['DeltaEta_avg']}  Δη@1550={row['DeltaEta_1550']}")
        print(f"[SWEEP-3D] saved CSV: {csv_path}")

