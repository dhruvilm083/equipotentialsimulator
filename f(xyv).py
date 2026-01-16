import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
import sys
import time

# YOUR DATA (EDIT HERE) (sorted by region/group, then voltage, then 2d list of centimeters, and y value)
# THE DATA HERE IS ONLY FOR 6.0V (our lab for 1E03). 
# THIS DATA WILL MESS UP YOUR RESULTS IF YOUR VOLTAGE IS NOT 6
V_supply = (5.96+5.97)/2 #universal supply voltage #average of the flickering values

data_AB = {
    5.0: (
        [0, 4, 8, 12, 13, 14, 14.5, 15],
        [1.85, 1.75, 1.65, 1.60, 1.33, 0.50, -0.85, -2.30]
    ),
    4.0: (
        [0, 4, 8, 12, 13, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21],
        [3.40, 3.40, 3.35, 3.30, 3.20, 2.80, 2.60, 2.33, 2.10, 1.80, 1.33, 1.00, 0.50, 0.00, -0.50, -1.00, -1.50, -2.30, -3.50, -5.00]
    ),
    3.0: (
        [0, 4, 8, 12, 13, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5],
        [5.00, 5.00, 5.00, 5.00, 4.90, 4.85, 4.75, 4.75, 4.75, 4.75, 4.75, 4.70, 4.60, 4.50, 4.50, 4.40, 4.30, 4.20, 4.10, 4.10, 4.00]

    ),
}

data_BC = {
    5.0: (
        [0, 4, 8, 12, 13, 14, 14.5, 15, 15.5, 16],
        [-0.5, -0.5, -0.5, -0.5, -0.6, 2.00, 3.10, 4.90, 7.0, 10.0]
    ),
    4.0: (
        [0, 4, 8, 12, 13, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5],
        [-1.20, -1.10, -1.10, -1.00, -1.00, -0.90, -0.70, -0.50, -0.30, -0.20, -0.20, -0.10, 0.00, 0.00, 0.00]
    ),
    3.0: (
        [0, 4, 8, 12, 13, 14, 14.5, 15, 15.5, 16, 16.5],
        [-1.40, -1.40, -1.40, -1.40, -1.40, -1.20, -1.00, -0.70, -0.40, 0.10, 1.10]
    ),
}

#CONFIGURATION (this is all in board coordinate system, 0,0 for me is start of positive plate)
REGIONS = {"AB": data_AB, "BC": data_BC}

REGION_CFG = {

    "AB": {
        "plate1_V": V_supply,  "plate1_y": 0.0,     "plate1_label": "Plate B (+)",
        "plate2_V": 0.0,       "plate2_y": 10.0,    "plate2_label": "Plate A (-)",
        "x_flat_end": 12.0,
        "board_x_min": 0.0, "board_x_max": 27.0,
        "board_y_min": -5.0, "board_y_max": 15.0,
        "plate_x_start": 0.0, "plate_x_end": 13.0,
    },

    "BC": {
        "plate1_V": 0,       "plate1_y": -3.0,     "plate1_label": "Plate C (-)",
        "plate2_V": V_supply,  "plate2_y": 0.0,     "plate2_label": "Plate B (+)",
        "x_flat_end": 12.0, #x flat end is just before your data starts to diverge
        "board_x_min": 0.0, "board_x_max": 27.0,
        "board_y_min": -5.0, "board_y_max": 15.0,
        "plate_x_start": 0.0, "plate_x_end": 13.0,
    },
}

_SPLINE_CACHE = {}

def build_splines_for_region(region):
    region = region.upper()
    if region in _SPLINE_CACHE:
        return _SPLINE_CACHE[region]

    data = REGIONS[region]
    splines = {}

    for V0, (xs, ys) in data.items():
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)

        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]

        spl = CubicSpline(xs, ys, bc_type="natural", extrapolate=False) 

        splines[float(V0)] = {
            "spl": spl,
            "xmin": float(xs.min()),
            "xmax": float(xs.max())
        }

    _SPLINE_CACHE[region] = splines
    return splines

def lagrange_3point(V, V_pts, Y_pts):
    V1, V2, V3 = V_pts
    y1, y2, y3 = Y_pts
    L1 = (V - V2) * (V - V3) / ((V1 - V2) * (V1 - V3))
    L2 = (V - V1) * (V - V3) / ((V2 - V1) * (V2 - V3))
    L3 = (V - V1) * (V - V2) / ((V3 - V1) * (V3 - V2))
    return y1 * L1 + y2 * L2 + y3 * L3

def y_ideal_flat(region, V):
    cfg = REGION_CFG[region.upper()]
    V1, Y1 = cfg["plate1_V"], cfg["plate1_y"]
    V2, Y2 = cfg["plate2_V"], cfg["plate2_y"]
    return Y1 + (V - V1) * (Y2 - Y1) / (V2 - V1)

def V_mid(region):
    cfg = REGION_CFG[region.upper()]
    return 0.5 * (cfg["plate1_V"] + cfg["plate2_V"])

def Y_mid(region):
    cfg = REGION_CFG[region.upper()]
    return 0.5 * (cfg["plate1_y"] + cfg["plate2_y"])

'''old that works def y_fringe(region, V_target, xs):
    region = region.upper()
    xs = np.asarray(xs, dtype=float)
    Vt = float(V_target)

    spl = build_splines_for_region(region)
    def get_spline(V):
        obj = spl[V]
        if callable(obj):
            return obj
        if isinstance(obj, dict):
            for k in ("spline", "cs", "fn", "f"):
                if k in obj and callable(obj[k]):
                    return obj[k]
            for v in obj.values():
                if callable(v):
                    return v
        raise TypeError(f"spl[{V}] is not callable and no callable found inside it: {type(obj)}")

    Vm = V_mid(region)
    Ym = Y_mid(region)

    # exact midline
    if np.isclose(Vt, Vm, atol=1e-9):
        return np.full_like(xs, Ym, dtype=float)
    
    # reflect below-midline voltages using symmetry
    if Vt < Vm:
        yref = y_fringe(region, 2*Vm - Vt, xs)
        return 2*Ym - yref

    s4 = get_spline(4.0)
    s5 = get_spline(5.0)

    y4 = s4(xs)
    y5 = s5(xs)

    y_out = np.full_like(xs, np.nan, dtype=float)

    m4 = np.isfinite(y4)
    m5 = np.isfinite(y5)

    def lagrange3(V, V0, V1, V2, Y0, Y1, Y2):
        L0 = (V - V1)*(V - V2)/((V0 - V1)*(V0 - V2))
        L1 = (V - V0)*(V - V2)/((V1 - V0)*(V1 - V2))
        L2 = (V - V0)*(V - V1)/((V2 - V0)*(V2 - V1))
        return L0*Y0 + L1*Y1 + L2*Y2

    # If V is between Vm and 4, you only NEED midline + 4V (more stable than using 5V everywhere)
    if Vt <= 4.0:
        alpha = (Vt - Vm) / (4.0 - Vm)
        y_out[m4] = Ym + alpha * (y4[m4] - Ym)

    # If V is above 4, use the 3-point quadratic through (Vm,Ym), (4,y4), (5,y5)
    else:
        mask = m4 & m5
        y_out[mask] = lagrange3(Vt, Vm, 4.0, 5.0, Ym, y4[mask], y5[mask])

    return y_out'''

def _plate_extremes(region):
    cfg = REGION_CFG[region.upper()]
    if cfg["plate1_V"] >= cfg["plate2_V"]:
        return cfg["plate1_V"], cfg["plate1_y"], cfg["plate2_V"], cfg["plate2_y"]
    else:
        return cfg["plate2_V"], cfg["plate2_y"], cfg["plate1_V"], cfg["plate1_y"]

def _knee_weight(xs, x_knee, width=0.06):
    # 0 -> 1 transition around x_knee (small width = sharper 90°-ish turn)
    return 1.0 / (1.0 + np.exp(-(xs - x_knee) / width))

def y_fringe(region, V_target, xs):
    region = region.upper()
    cfg = REGION_CFG[region]
    xs = np.asarray(xs, dtype=float)
    Vt = float(V_target)
    def get_spline(V):
        obj = spl[V]
        if callable(obj):
            return obj
        if isinstance(obj, dict):
            for k in ("spline", "cs", "fn", "f"):
                if k in obj and callable(obj[k]):
                    return obj[k]
            for v in obj.values():
                if callable(v):
                    return v
        raise TypeError(f"spl[{V}] is not callable: {type(obj)}")

    spl = build_splines_for_region(region)

    Vm = V_mid(region)
    Ym = Y_mid(region)

    # 1) midline is hard-flat (ONLY for Vm)
    if np.isclose(Vt, Vm, atol=1e-9):
        return np.full_like(xs, Ym, dtype=float)

    # 2) use symmetry reflection for V below midline (keeps model consistent)
    if Vt < Vm:
        yref = y_fringe(region, 2 * Vm - Vt, xs)
        return 2 * Ym - yref

    # 3) auto anchors from available measured voltages (plus Vm as anchor)
    measured_Vs = sorted(float(v) for v in spl.keys())  # e.g. [3.0,4.0,5.0] or [5.5,7,9,10]
    anchors = {Vm: np.full_like(xs, Ym, dtype=float)}

    for V in measured_Vs:
        s = get_spline(V)
        anchors[V] = s(xs)  # NaN out-of-range if extrapolate=False

    Vs = np.array(sorted(anchors.keys()), dtype=float)

    y_out = np.full_like(xs, np.nan, dtype=float)

    # Helper: safe linear interpolation between two anchor curves
    def interp_between(Vlo, Vhi):
        ylo = anchors[float(Vlo)]
        yhi = anchors[float(Vhi)]
        mask = np.isfinite(ylo) & np.isfinite(yhi)
        if not np.any(mask):
            return None, None
        a = (Vt - Vlo) / (Vhi - Vlo)
        y = np.full_like(xs, np.nan, dtype=float)
        y[mask] = ylo[mask] + a * (yhi[mask] - ylo[mask])
        return y, mask

    # 4) inside measured range -> bracket + linear interpolation
    if (Vt >= Vs[0]) and (Vt <= Vs[-1]):
        k = np.searchsorted(Vs, Vt)
        if k == 0:
            return anchors[float(Vs[0])]
        if k == len(Vs):
            return anchors[float(Vs[-1])]
        Vlo = Vs[k - 1]
        Vhi = Vs[k]
        y, _ = interp_between(Vlo, Vhi)
        return y if y is not None else y_out

    # 5) extrapolation behavior (your “hug plate then sharp knee” request)
    Vpos, Ypos, Vneg, Yneg = _plate_extremes(region)

    # knee location (sharp turn happens just after plate end)
    x_knee = cfg["plate_x_end"] + 0.2
    w = _knee_weight(xs, x_knee, width=0.1)

    # --- Above max anchor (near + plate) ---
    if Vt > Vs[-1]:
        Vmax = float(Vs[-1])
        ymax = anchors[Vmax]

        # if we exceed the physical plate voltage, sit on the plate
        if Vt >= Vpos:
            return np.full_like(xs, Ypos, dtype=float)

        # beta shrinks curve toward the + plate as V approaches Vpos
        denom = (Vpos - Vmax)
        if abs(denom) < 1e-12:
            beta = 0.0
        else:
            beta = (Vpos - Vt) / denom  # in (0..1) when Vmax<Vt<Vpos
        beta = np.clip(beta, 0.0, 1.0) ** 1.6  # stronger hug near plate

        # scaled toward plate
        y_scaled = Ypos + (ymax - Ypos) * beta

        # enforce “hug then knee”: near plate until x_knee, then follow scaled curve
        # (function form can't draw a true vertical line; this makes it near-vertical)
        y = Ypos + (y_scaled - Ypos) * w

        return y

    # --- Below min anchor (near − plate) ---
    if Vt < Vs[0]:
        Vmin = float(Vs[0])
        ymin = anchors[Vmin]

        if Vt <= Vneg:
            return np.full_like(xs, Yneg, dtype=float)

        denom = (Vmin - Vneg)
        if abs(denom) < 1e-12:
            beta = 0.0
        else:
            beta = (Vt - Vneg) / denom  # in (0..1) when Vneg<Vt<Vmin
        beta = np.clip(beta, 0.0, 1.0) ** 1.6

        y_scaled = Yneg + (ymin - Yneg) * beta
        y = Yneg + (y_scaled - Yneg) * w
        return y

    return y_out

def y_piecewise(region, V_target, xs):
    region = region.upper()
    cfg = REGION_CFG[region]
    xs = np.asarray(xs, dtype=float)
    Vt = float(V_target)

    # never draw outside board x-range
    inside_x = (xs >= cfg["board_x_min"]) & (xs <= cfg["board_x_max"])

    # hard constraints for plate voltages (optional, keeps endpoints clean)
    PLATE_V_TOL = 2e-2
    if abs(Vt - cfg["plate1_V"]) <= PLATE_V_TOL:
        y = np.full_like(xs, cfg["plate1_y"], dtype=float)
        y[~inside_x] = np.nan
        return y

    if abs(Vt - cfg["plate2_V"]) <= PLATE_V_TOL:
        y = np.full_like(xs, cfg["plate2_y"], dtype=float)
        y[~inside_x] = np.nan
        return y

    # HARD CONSTRAINT: midline is perfectly flat everywhere
    MID_V_TOL = 1e-3
    Vm = V_mid(region)
    Ym = Y_mid(region)
    if abs(Vt - Vm) <= MID_V_TOL:
        y = np.full_like(xs, Ym, dtype=float)
        y[~inside_x] = np.nan
        return y

    x_flat_end = cfg["x_flat_end"]

    flat_mask = xs <= x_flat_end
    fringe_mask = xs > x_flat_end

    y = np.full_like(xs, np.nan, dtype=float)

    # flat region (idealized)
    y[flat_mask] = y_ideal_flat(region, Vt)

    # fringe region (shifted to match y at x_flat_end)
    if np.any(fringe_mask):
        y0_flat = y_ideal_flat(region, Vt)

        y_fringe_at_x0 = y_fringe(region, Vt, np.array([x_flat_end]))[0]
        if np.isfinite(y_fringe_at_x0):
            delta = y0_flat - y_fringe_at_x0
            y_fr = y_fringe(region, Vt, xs[fringe_mask])
            y[fringe_mask] = y_fr + delta
        else:
            y[fringe_mask] = np.nan

    # stop drawing if outside board y-range (don’t clip — just stop)
    y[(y < cfg["board_y_min"]) | (y > cfg["board_y_max"])] = np.nan

    # stop drawing outside x-range
    y[~inside_x] = np.nan

    return y

def y_at(region, V_target, x0):
    return float(y_piecewise(region, V_target, np.array([x0]))[0])

def x_at(region, V_target, y0, x_min=None, x_max=None, n=4000):
    region = region.upper()
    cfg = REGION_CFG[region]

    if x_min is None:
        x_min = cfg["board_x_min"]
    if x_max is None:
        x_max = cfg["board_x_max"]

    xs = np.linspace(x_min, x_max, n)
    ys = y_piecewise(region, V_target, xs)

    mask = np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]

    if len(xs) == 0:
        return None
    
    idx = np.argmin(np.abs(ys - y0))
    return float(xs[idx])

def cm_to_index(x_cm, x_min_cm, x_max_cm, n):
    """Map cm coordinate to grid index [0..n-1]."""
    t = (x_cm - x_min_cm) / (x_max_cm - x_min_cm)
    idx = int(np.round(t * (n - 1)))
    return max(0, min(n - 1, idx))

def apply_neumann_edges(V):
    """Zero normal derivative on outer boundary: copy interior values to boundary."""
    V[0, :]  = V[1, :]
    V[-1, :] = V[-2, :]
    V[:, 0]  = V[:, 1]
    V[:, -1] = V[:, -2]

def apply_plate_dirichlet(V, fixed_mask, fixed_vals):
    """Apply fixed Dirichlet plate constraints."""
    V[fixed_mask] = fixed_vals[fixed_mask]

def build_plate_constraints(region, nx, ny):
    """
    Returns fixed_mask, fixed_vals for the region.
    Plate segments: y=const for x in [plate_x_start, plate_x_end].
    """
    cfg = REGION_CFG[region]

    x_min, x_max = cfg["board_x_min"], cfg["board_x_max"]
    y_min, y_max = cfg["board_y_min"], cfg["board_y_max"]

    fixed_mask = np.zeros((ny, nx), dtype=bool)
    fixed_vals = np.zeros((ny, nx), dtype=float)

    x0 = cm_to_index(cfg["plate_x_start"], x_min, x_max, nx)
    x1 = cm_to_index(cfg["plate_x_end"],   x_min, x_max, nx)

    y_p1 = cm_to_index(cfg["plate1_y"], y_min, y_max, ny)
    y_p2 = cm_to_index(cfg["plate2_y"], y_min, y_max, ny)

    # plate1
    fixed_mask[y_p1, x0:x1+1] = True
    fixed_vals[y_p1, x0:x1+1] = cfg["plate1_V"]

    # plate2
    fixed_mask[y_p2, x0:x1+1] = True
    fixed_vals[y_p2, x0:x1+1] = cfg["plate2_V"]

    return fixed_mask, fixed_vals

def solve_laplace(region, nx, ny, max_iter, tol, omega):
    """
    Solve ∇²V = 0 using SOR (Gauss-Seidel + relaxation).
    Neumann boundary at paper edges + Dirichlet on plates.
    Spinner updates every iteration.
    Iter + maxΔV prints every 500 iterations.
    """

    region = region.upper()
    cfg = REGION_CFG[region]

    fixed_mask, fixed_vals = build_plate_constraints(region, nx, ny)

    # init guess (linear in y)
    y_coords = np.linspace(cfg["board_y_min"], cfg["board_y_max"], ny)
    V = np.zeros((ny, nx), dtype=float)

    y1, V1 = cfg["plate1_y"], cfg["plate1_V"]
    y2, V2 = cfg["plate2_y"], cfg["plate2_V"]

    if abs(y2 - y1) > 1e-12:
        V_init_1d = V1 + (y_coords - y1) * (V2 - V1) / (y2 - y1)
        V[:, :] = V_init_1d[:, None]

    apply_plate_dirichlet(V, fixed_mask, fixed_vals)
    apply_neumann_edges(V)

    frames = ["|", "/", "-", "\\"]

    for it in range(max_iter):
        V_old = V.copy()

        # update interior
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if fixed_mask[j, i]:
                    continue
                V_new = 0.25 * (V[j, i+1] + V[j, i-1] + V[j+1, i] + V[j-1, i])
                V[j, i] = (1.0 - omega) * V[j, i] + omega * V_new

        apply_neumann_edges(V)
        apply_plate_dirichlet(V, fixed_mask, fixed_vals)

        diff = np.max(np.abs(V - V_old))

        # spinner every iteration
        sys.stdout.write("\r" + frames[it % 4])
        sys.stdout.flush()

        # stats every 500
        if (it + 1) % 1000 == 0:
            sys.stdout.write(f"\riteration {it+1} | goes until convergence or maxΔV < {tol:.3e} | maxΔV = {diff:.3e}\n")
            sys.stdout.flush()

        if diff < tol:
            sys.stdout.write("\n")
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Converged in {it+1} iterations | {elapsed_time:.4f} seconds taken | final maxΔV = {diff:.3e}")
            return V

    sys.stdout.write("\n")
    print(f"Stopped at max_iter={max_iter} | last maxΔV = {diff:.3e}")
    return V

def enforce_no_intersections_with_spacing(region, Vs_plot, xs_dense, Ymat, min_sep_cm=0.12):
    """
    Ymat shape = (len(Vs_plot), len(xs_dense))
    Enforces:
      1) equipotentials don't intersect
      2) adjacent voltages keep a small separation (min_sep_cm)
      3) if a curve collapses onto a neighbor, it gets re-centered using V-fraction midpoint
         (so 5.5 stays between 5.0 and the next higher curve, instead of merging).
    """
    cfg = REGION_CFG[region.upper()]

    Vs = np.asarray(Vs_plot, dtype=float)
    order = np.argsort(Vs)
    Vs_s = Vs[order]
    Y = np.array(Ymat, dtype=float)[order].copy()

    # Determine if y should increase with V (region-aware)
    dV = (cfg["plate2_V"] - cfg["plate1_V"])
    dy = (cfg["plate2_y"] - cfg["plate1_y"])
    y_increases_with_V = (dy * dV) > 0

    nV, nX = Y.shape

    for j in range(nX):
        col = Y[:, j]

        # only work where we actually have values
        finite = np.isfinite(col)
        if finite.sum() < 2:
            continue

        # --- (A) If a curve is collapsing onto neighbor(s), re-center it by V-midpoint ---
        # This is the "put 5.5 in the middle" behavior.
        for i in range(1, nV - 1):
            if not (np.isfinite(col[i-1]) and np.isfinite(col[i]) and np.isfinite(col[i+1])):
                continue

            # if it's too close to either neighbor, push to midpoint-in-V between neighbors
            if (abs(col[i] - col[i-1]) < min_sep_cm) or (abs(col[i+1] - col[i]) < min_sep_cm):
                frac = (Vs_s[i] - Vs_s[i-1]) / (Vs_s[i+1] - Vs_s[i-1])
                target = col[i-1] + frac * (col[i+1] - col[i-1])
                col[i] = target

        # --- (B) Enforce strict ordering with minimum spacing ---
        if y_increases_with_V:
            # y must increase with V
            for i in range(1, nV):
                if not np.isfinite(col[i]) or not np.isfinite(col[i-1]):
                    continue
                if col[i] <= col[i-1] + min_sep_cm:
                    col[i] = col[i-1] + min_sep_cm
        else:
            # y must decrease with V
            for i in range(1, nV):
                if not np.isfinite(col[i]) or not np.isfinite(col[i-1]):
                    continue
                if col[i] >= col[i-1] - min_sep_cm:
                    col[i] = col[i-1] - min_sep_cm

        Y[:, j] = col

    # Clip to board after spacing (keeps it inside)
    Y = np.clip(Y, cfg["board_y_min"], cfg["board_y_max"])

    # undo voltage sorting
    Y_fixed = np.empty_like(Ymat, dtype=float)
    Y_fixed[order] = Y
    return Y_fixed

def plot_solution(region, V_grid, levels=None, show_measured=True):
    region = region.upper()
    cfg = REGION_CFG[region]
    
    ny, nx = V_grid.shape
    x = np.linspace(cfg["board_x_min"], cfg["board_x_max"], nx)
    y = np.linspace(cfg["board_y_min"], cfg["board_y_max"], ny)
    X, Y = np.meshgrid(x, y)

    plt.figure()

    # contour levels
    if levels is None:
        levels = np.linspace(min(cfg["plate1_V"], cfg["plate2_V"]),
                             max(cfg["plate1_V"], cfg["plate2_V"]), 8)

    cs = plt.contour(X, Y, V_grid, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8)

    # draw plates
    plt.hlines(cfg["plate1_y"], cfg["plate_x_start"], cfg["plate_x_end"], linewidth=2, label=cfg["plate1_label"], color="black")
    plt.hlines(cfg["plate2_y"], cfg["plate_x_start"], cfg["plate_x_end"], linewidth=2, label=cfg["plate2_label"], color="black")
    plt.gca().set_xlim(cfg["board_x_min"]-0.5, cfg["board_x_max"]+0.5)
    plt.gca().set_ylim(cfg["board_y_min"]-0.5,cfg["board_y_max"]+0.5)

    # draw board boundary
    plt.gca().add_patch(Rectangle(
        (cfg["board_x_min"], cfg["board_y_min"]),
        cfg["board_x_max"] - cfg["board_x_min"],
        cfg["board_y_max"] - cfg["board_y_min"],
        fill=True,
        color="grey",
        linewidth=2,
        zorder=5,
        label="board",
        alpha=0.1
    ))
    plt.gca().add_patch(Rectangle(
        (cfg["board_x_min"], cfg["board_y_min"]),
        cfg["board_x_max"] - cfg["board_x_min"],
        cfg["board_y_max"] - cfg["board_y_min"],
        fill=False,
        linewidth=2,
        zorder=5,
        label="board boundary",
    ))

    # overlay measured dots
    if show_measured and region in REGIONS:
        data = REGIONS[region]
        for V0, (xs, ys) in sorted(data.items(), key=lambda t: t[0], reverse=True):
            plt.scatter(xs, ys, s=10, alpha=0.8, label=f"measured {V0:.2f}V")
    
    plt.title(f"Laplace Solution Equipotentials ({region})")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")

    # gridlines
    plt.minorticks_on()
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.25))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.25))
    plt.grid(True, which="major")
    plt.grid(True, which="minor", alpha=0.3)
    plt.legend()

def plot_measured(region):
    region = region.upper()
    data = REGIONS[region]
    for V0, (xs, ys) in sorted(data.items(), key=lambda t: t[0], reverse=True):
        plt.scatter(xs, ys, s=10, alpha=0.8, label=f"measured {V0:.2f}V")

def plot_voltages(region, voltages):
    region = region.upper()
    cfg = REGION_CFG[region]

    
    all_x = np.array([x for V0, (xs, ys) in REGIONS[region].items() for x in xs], dtype=float)
    x_min, x_max = all_x.min(), all_x.max()
    xs_dense = np.linspace(
        REGION_CFG[region]["board_x_min"],
        REGION_CFG[region]["board_x_max"],
        900
    )

    plt.figure()
    plot_measured(region)

    for Vt in voltages:
        Vs_plot = voltages_to_plot
        Y_list = []

        for Vt in Vs_plot:
            Y_list.append(y_piecewise(region, Vt, xs_dense))

        Ymat = np.vstack(Y_list)  # (nV, nX)

        # enforce "no intersections"
        Ymat_fixed = enforce_no_intersections_with_spacing(
        region, Vs_plot, xs_dense, Ymat,
        min_sep_cm=0.12  # tweak: 0.08 to 0.20
    )
    ax = plt.gca()
    for i, Vt in enumerate(Vs_plot):
        ax.plot(xs_dense, Ymat_fixed[i], label=f"model {Vt:.3f}V")

    plt.title(f"Equipotential simulation using Lagrangian interpolation, cubic splines, and Lagrange equations | zoomed to 102% board")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.grid(True)
    plt.minorticks_on()
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.25))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.25))
    plt.grid(True, which="major")
    plt.grid(True, which="minor", alpha=0.3)
    ax.add_patch(Rectangle(
        (cfg["board_x_min"], cfg["board_y_min"]),
        cfg["board_x_max"] - cfg["board_x_min"],
        cfg["board_y_max"] - cfg["board_y_min"],
        fill=True,
        color="grey",
        linewidth=2,
        zorder=5,
        label="board",
        alpha=0.1
    ))
    ax.add_patch(Rectangle(
        (cfg["board_x_min"], cfg["board_y_min"]),
        cfg["board_x_max"] - cfg["board_x_min"],
        cfg["board_y_max"] - cfg["board_y_min"],
        fill=False,
        linewidth=2,
        zorder=5,
        label="board boundary",
    ))
    ax.set_xlim(cfg["board_x_min"]-0.5, cfg["board_x_max"]+0.5)
    ax.set_ylim(cfg["board_y_min"]-0.5,cfg["board_y_max"]+0.5)
    ax.hlines(cfg["plate1_y"], cfg["plate_x_start"], cfg["plate_x_end"], linewidth=2, zorder=6, label=cfg["plate1_label"],color="black")
    ax.hlines(cfg["plate2_y"], cfg["plate_x_start"], cfg["plate_x_end"], linewidth=2, zorder=6, label=cfg["plate2_label"],color="black")

    plt.show()

if __name__ == "__main__":
    #config for the solve for laplace, and voltages you want to plot
    region = str(input("region select: "))  

    Vm = V_mid(region)
    voltages_to_plot = [1.0,2.0,2.9,3.0,4.0,5.0,5.5]
    levels = voltages_to_plot.sort()

    start_time = time.perf_counter()
    V = solve_laplace(
        region=region,
        nx=300, ny=250,
        max_iter=10000,
        tol=1e-5,
        omega=1.9
    )
   
    plot_solution(region, V, levels=levels, show_measured=True)
    plot_voltages(region, voltages_to_plot)

    #example of finding a specific y value
    #print(y_at(region,5.0,15))
    #print(x_at(region,5.0,15))

