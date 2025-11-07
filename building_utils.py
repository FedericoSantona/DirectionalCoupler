import tidy3d as td
import numpy as np
from shapely.geometry import Point, LineString
from tidy3d_lambda import entrypoint

# Default modal specifications used when callers do not supply custom ones.
DEFAULT_MODE_SPECS = {
    "te": td.ModeSpec(num_modes=1, filter_pol="te", target_neff=1.7),
    "tm": td.ModeSpec(num_modes=1, filter_pol="tm", target_neff=1.5),
}

#--helpers functions--#

def _resolve_mode_spec(param, pol):
    """Return a tidy3d.ModeSpec for the requested polarization."""
    pol_key = pol.lower()
    if pol_key not in DEFAULT_MODE_SPECS:
        raise ValueError(f"Unsupported polarization '{pol}'. Expected one of {tuple(DEFAULT_MODE_SPECS)}.")
    if hasattr(param, "mode_specs"):
        specs = param.mode_specs
        if isinstance(specs, dict) and pol_key in specs:
            return specs[pol_key]
        candidate = getattr(specs, pol_key, None)
        if candidate is not None:
            return candidate
    attr_name = f"mode_spec_{pol_key}"
    if hasattr(param, attr_name):
        return getattr(param, attr_name)
    return DEFAULT_MODE_SPECS[pol_key]


def resolve_mode_spec(param, pol):
    """Public wrapper used by other modules to obtain a ModeSpec."""
    return _resolve_mode_spec(param, pol)

# --- recompute derived domain sizes when a parameter changes ---
def update_param_derived(p, lambda_eval=None):
    """
    Recompute derived parameters including coupling_length from supermode analysis.
    
    Args:
        p: Parameter namespace
        lambda_eval: Optional wavelength for per-lambda L_c recomputation (if freeze_l_c=False)
    """
    # Import here to avoid circular dependency
    from coupling_length import compute_Lc_symmetric
    
    # Determine wavelength for coupling_length computation
    freeze = getattr(p, 'freeze_l_c', True)  # Default: freeze at design wavelength
    
    if freeze:
        # Use design wavelength (compute once, reuse for all λ)
        lambda0 = getattr(p, 'lambda_single', p.wl_0)
    else:
        # Per-lambda recomputation (for CMT validation)
        lambda0 = lambda_eval if lambda_eval is not None else getattr(p, 'lambda_single', p.wl_0)
    
    # Check if geometry parameters changed (force recomputation if they did)
    current_geometry = (
        float(getattr(p, 'wg_width', 0)),
        float(getattr(p, 'coupling_gap', 0)),
        float(getattr(p, 'wg_thick', 0)),
        float(getattr(p.medium.SiN, 'permittivity', 1.0)),
        float(getattr(p.medium.SiO2, 'permittivity', 1.0)),
    )
    
    # Get stored geometry from last computation
    last_geometry = getattr(p, '_last_geometry_for_lc', None)
    geometry_changed = (last_geometry != current_geometry)
    
    # Compute coupling_length if:
    # 1. Not set yet, OR
    # 2. freeze_l_c=False (allows recomputation), OR
    # 3. Geometry changed (force recomputation even if freeze_l_c=True)
    if not hasattr(p, 'coupling_length') or p.coupling_length is None or not freeze or geometry_changed:
        # Compute derived coupling_length
        p.coupling_length = compute_Lc_symmetric(
            param=p,
            lambda0=lambda0,
            trim_factor=getattr(p, 'coupling_trim_factor', 0.075),
            cache_dir="data"
        )
        p._last_geometry_for_lc = current_geometry  # Store/update geometry signature
    # else: freeze_l_c=True and geometry unchanged - keep existing value
    
    # Recompute domain sizes (coupling_length is now set)
    p.size_x = 2 * (p.wg_length + p.sbend_length) + p.coupling_length
    p.size_z = 3 * p.wl_0 + p.wg_thick
    p.freq_0 = td.C_0 / p.wl_0
    # Adaptive y padding to keep structures at least ~lambda0/2 from PML with margin.
    # size_y = 2*(sbend_height + wl_0) + wg_width + coupling_gap + max(0, wg_width - wl_0) + pad_extra
    pad_extra = getattr(p, "pad_extra", 0.05)
    p.size_y = (
        2 * (p.sbend_height + p.wl_0)
        + p.wg_width
        + p.coupling_gap
        + max(0.0, float(p.wg_width) - float(p.wl_0))
        + float(pad_extra)
    )

# This function creates the directional coupler geometry.

def create_sbend(p0, sbend_length, sbend_height, wg_width, wg_thick):
    # Creates a s-bend inside a box of dimensions `sbend_length` and `sbend_height`,
    # starting at the point `p0 = [x,y]`. The s-bend waveguides will have `wg_width`
    # width and `wg_thick` thickness.
    
    # S-bend anchor (p1, p4) and control points (p2, p3).
    p1 = p0
    p2 = [p0[0] + sbend_length/2, p0[1]]
    p3 = [p0[0] + sbend_length/2, p0[1] + sbend_height]
    p4 = [p0[0] + sbend_length, p0[1] + sbend_height]
    # S-bend calculation defined as a cubic bezier curve.
    points = []
    uvals = np.linspace(0,1,int(sbend_length/0.01))
    for u in uvals:
        u1 = 1 - u
        u13 = u1*u1*u1
        u3 = u*u*u
        x = u13*p1[0] + 3*u*u1*u1*p2[0] + 3*u*u*u1*p3[0] + u3*p4[0]
        y = u13*p1[1] + 3*u*u1*u1*p2[1] + 3*u*u*u1*p3[1] + u3*p4[1]
        points.append(Point(x,y))
    # Waveguide geometry
    curve = LineString(points)
    wg_geometry = curve.buffer(wg_width/2, cap_style=2)
    # PolySlab object
    wg=td.PolySlab(
        vertices=wg_geometry.exterior.coords, 
        axis=2, 
        slab_bounds=(-wg_thick / 2, wg_thick / 2)
    )
    return wg

def generate_object(param):
    # Directional Coupler parameters.
    cl = param.coupling_length     # DC coupling length.
    gap = param.coupling_gap       # Gap between the upper and lower waveguides.
    sb_length = param.sbend_length # S-bends length.
    sb_height = param.sbend_height # S-bends heigth. 
    wg_w = param.wg_width          # Waveguides width.
    wg_t = param.wg_thick          # Waveguides thickness.
    wg_l = param.wg_length         # Input/output waveguide length.
    mat = param.medium.SiN          # Waveguides material.
    # S-Bend position in y-direction.
    # Starting point for the left s-bends.
    px_l = -param.size_x/2 + wg_l
    py_l = (gap + wg_w)/2 + sb_height
    # Starting point for the right s-bends.
    px_r = -param.size_x/2 + wg_l + sb_length + cl
    py_r = (gap + wg_w)/2
    
    ########### UPPER DC WAVEGUIDE ###############    
    # Group Structure containing all the DC upper waveguide elements.
    upper_wg = td.Structure(
        geometry=td.GeometryGroup(
            geometries=(
                # Input waveguide.
                td.Box(size=(2*wg_l, wg_w, wg_t), 
                       center=(-param.size_x/2, py_l, 0)),
                # Left s-bend.
                create_sbend(p0=[px_l, py_l],
                             sbend_length=sb_length,
                             sbend_height=-sb_height,
                             wg_width=wg_w,
                             wg_thick=wg_t),
                # Coupling waveguide.
                td.Box(size=(cl, wg_w, wg_t), 
                       center=(0, py_r, 0)),
                # Right s-bend.
                create_sbend(p0=[px_r, py_r],
                             sbend_length=sb_length,
                             sbend_height=sb_height,
                             wg_width=wg_w,
                             wg_thick=wg_t),
                # Output waveguide.
                td.Box(size=(2*wg_l, wg_w, wg_t), 
                       center=(param.size_x/2, py_l, 0)),            
            ) 
        ),
        medium=mat,
        name="upper_wg"        
    )
    
    
    ########### LOWER DC WAVEGUIDE ###############    
    # Group Structure containing all the DC upper waveguide elements.
    lower_wg = td.Structure(
        geometry=td.GeometryGroup(
            geometries=(
                # Input waveguide.
                td.Box(size=(2*wg_l, wg_w, wg_t), 
                       center=(-param.size_x/2, -py_l, 0)),
                # Left s-bend.
                create_sbend(p0=[px_l, -py_l],
                             sbend_length=sb_length,
                             sbend_height=sb_height,
                             wg_width=wg_w,
                             wg_thick=wg_t),
                # Coupling waveguide.
                td.Box(size=(cl, wg_w, wg_t), 
                       center=(0, -py_r, 0)),
                # Right s-bend.
                create_sbend(p0=[px_r, -py_r],
                             sbend_length=sb_length,
                             sbend_height=-sb_height,
                             wg_width=wg_w,
                             wg_thick=wg_t),
                # Output waveguide.
                td.Box(size=(2*wg_l, wg_w, wg_t), 
                       center=(param.size_x/2, -py_l, 0)),
            )    
        ),
        medium=mat,
        name="lower_wg"         
    )    
     
    return [upper_wg, lower_wg]


def make_source(param, pol="te"):
    pol_key = pol.lower()
    mode_spec = _resolve_mode_spec(param, pol_key)
    mode_index = None
    if hasattr(param, "mode_indices"):
        indices = param.mode_indices
        if isinstance(indices, dict):
            mode_index = indices.get(pol_key)
        else:
            mode_index = getattr(indices, pol_key, None)

    # Center frequency at lambda0
    f0 = td.C_0 / param.wl_0  # Hz

    # Choose a wavelength span to excite (here ~100 nm around 1.55 µm),
    # then convert to an approximate frequency width: Δf ≈ c * Δλ / λ0^2
    dlam = param.source_dlambda
    df = td.C_0 * dlam / (param.wl_0 ** 2)
    num_freqs = int(param.source_num_freqs)

    source_kwargs = dict(
        name=f"mode_{pol_key}_0",
        center=[-param.size_x/2 + param.wg_length,
                (param.coupling_gap + param.wg_width) / 2 + param.sbend_height, 0],
        size=[0, 4 * param.wg_width, 5 * param.wg_thick],
        source_time=td.GaussianPulse(freq0=f0, fwidth=df),
        num_freqs=num_freqs,
        direction='+',
        mode_spec=mode_spec,
    )
    if mode_index is not None:
        source_kwargs["mode_index"] = int(mode_index)
    return td.ModeSource(**source_kwargs)

def make_monitors(param, pol='te'):
    pol_key = pol.lower()
    mode_spec = _resolve_mode_spec(param, pol_key)
    apod_width = param.wl_0 / td.C_0
    monitor_mode_spec = td.ModeSpec(
        num_modes=3,
        filter_pol=None,
        target_neff=getattr(mode_spec, "target_neff", None),
    )
    lam_start = param.monitor_lambda_start
    lam_stop = param.monitor_lambda_stop
    lam_points = int(param.monitor_lambda_points)
    lam_grid = np.linspace(lam_start, lam_stop, lam_points)
    freqs = td.C_0 / lam_grid
    mon_thru = td.ModeMonitor(
        name=f"monitor_s31_{pol_key}",
        center=[param.size_x / 2 - 2 * param.wl_0,
                (param.coupling_gap + param.wg_width) / 2 + param.sbend_height, 0],
        size=[0, 4 * param.wg_width, 5 * param.wg_thick],
        freqs=freqs,
        apodization=td.ApodizationSpec(width=apod_width),
        mode_spec=monitor_mode_spec
    )
    mon_cross = td.ModeMonitor(
        name=f"monitor_s41_{pol_key}",
        center=[param.size_x / 2 - 2 * param.wl_0,
                -(param.coupling_gap + param.wg_width) / 2 - param.sbend_height, 0],
        size=[0, 4 * param.wg_width, 5 * param.wg_thick],
        freqs=freqs,
        apodization=td.ApodizationSpec(width=apod_width),
        mode_spec=monitor_mode_spec
    )
    field_xy = td.FieldMonitor(
        name=f"field_xy_{pol_key}",
        size=[param.size_x, param.size_y, 0],
        freqs=td.C_0 / param.wl_0,
        apodization=td.ApodizationSpec(width=apod_width)
    )
    return mon_thru, mon_cross, field_xy
