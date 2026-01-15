import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle

# ---------------------------
# YOUR DATA (EDIT HERE) (sorted by region/group, then voltage, then 2d list of centimeters, and y value)
# ---------------------------

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
        [5.00, 5.00, 5.00, 5.00, 5.10, 5.15, 5.25, 5.25, 5.25, 5.25, 5.25, 5.30, 5.40, 5.50, 5.50, 5.60, 5.70, 5.80, 5.90, 5.90, 6.00]
    ),
}

data_BC = {
    5.0: (
        [0, 4, 8, 12, 13, 14, 14.5, 15, 15.5, 16],
        [2.5, 2.5, 2.5, 2.5, 2.4, 5.00, 6.10, 7.90, 10.0, 13.0]
    ),
    4.0: (
        [0, 4, 8, 12, 13, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5],
        [1.80, 1.90, 1.90, 2.00, 2.00, 2.10, 2.30, 2.50, 2.70, 2.80, 2.80, 2.90, 3.00, 3.00, 3.00]
    ),
    3.0: (
        [0, 4, 8, 12, 13, 14, 14.5, 15, 15.5, 16, 16.5],
        [1.60, 1.60, 1.60, 1.60, 1.60, 1.80, 2.00, 2.30, 2.60, 3.10, 4.10]
    ),
}

#CONFIGURATION (this is all in board coordinate system, 0,0 for me is start of positive plate)
REGIONS = {"AB": data_AB, "BC": data_BC}
REGION_CFG = {

    "AB": {
        "V1": (5.96+5.97)/2,  "Y1": 0.0,     "plate1_label": "Plate B (+)",
        "V2": 0.0,       "Y2": 10.0,    "plate2_label": "Plate A (-)",
        "X_FLAT_END": 12.0,
        "BOARD_X_MIN": 0.0, "BOARD_X_MAX": 27.0,
        "BOARD_Y_MIN": -5.0, "BOARD_Y_MAX": 15.0,
        "PLATE_X_START": 0.0, "PLATE_X_END": 13.0,
    },

    "BC": {
        "V1": 0,       "Y1": 0.0,     "plate1_label": "Plate C (-)",
        "V2": (5.96+5.97)/2,  "Y2": 3.0,     "plate2_label": "Plate B (+)",
        "X_FLAT_END": 12.0,
        "BOARD_X_MIN": 0.0, "BOARD_X_MAX": 27.0,
        "BOARD_Y_MIN": -5.0, "BOARD_Y_MAX": 15.0,
        "PLATE_X_START": 0.0, "PLATE_X_END": 13.0,
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

        spl = CubicSpline(xs, ys, bc_type="natural", extrapolate=True)

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
    V1, Y1 = cfg["V1"], cfg["Y1"]
    V2, Y2 = cfg["V2"], cfg["Y2"]
    return Y1 + (V - V1) * (Y2 - Y1) / (V2 - V1)

def V_mid(region):
    cfg = REGION_CFG[region.upper()]
    return 0.5 * (cfg["V1"] + cfg["V2"])

def Y_mid(region):
    cfg = REGION_CFG[region.upper()]
    return 0.5 * (cfg["Y1"] + cfg["Y2"])

def y_fringe(region, V_target, xs):
    region = region.upper()
    cfg = REGION_CFG[region]
    spl = build_splines_for_region(region)
    xs = np.asarray(xs, dtype=float)
    V_target = float(V_target)
    
    Vm = V_mid(region)
    Ym = Y_mid(region)

    if abs(V_target - Vm) < 1e-6:
        return np.full_like(xs, y_ideal_flat(region, Vm), dtype=float)

    if V_target < Vm:
        V_ref = 2 * Vm - V_target
        y_ref = y_fringe(region, V_ref, xs)
        return 2 * Ym - y_ref

    def eval_safe(V):
        xmin = spl[V]["xmin"]
        xmax = spl[V]["xmax"]
        x_clip = np.clip(xs, xmin, xmax)
        return spl[V]["spl"](x_clip)

    y3 = eval_safe(3.0)
    y4 = eval_safe(4.0)
    y5 = eval_safe(5.0)

    return lagrange_3point(V_target, [3.0, 4.0, 5.0], [y3, y4, y5])

def y_piecewise(region, V_target, xs):

    region = region.upper()
    cfg = REGION_CFG[region]

    xs = np.asarray(xs, dtype=float)
    V_target = float(V_target)

    Vm = V_mid(region)
    flat_end = cfg["X_FLAT_END"]

    # FORCE MIDLINE FLAT FOR ALL X
    if abs(V_target - Vm) < 1e-3:
        return np.full_like(xs, y_ideal_flat(region, Vm), dtype=float)

    # start with ideal flat line everywhere
    y = np.full_like(xs, y_ideal_flat(region, V_target), dtype=float)

    # after flat region, swap in fringe model but shift to match at x=flat_end
    mask = xs > flat_end
    if np.any(mask):
        xs_after = xs[mask]
        y_after = y_fringe(region, V_target, xs_after)

        y12_fringe = float(y_fringe(region, V_target, np.array([flat_end]))[0])
        y12_flat = float(y_ideal_flat(region, V_target))
        y_after = y_after + (y12_flat - y12_fringe)

        y[mask] = y_after

    return y

def y_at(region, V_target, x0):
    return float(y_piecewise(region, V_target, np.array([x0]))[0])

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
    xs_dense = np.linspace(x_min, x_max, 700)

    plt.figure()
    plot_measured(region)

    for Vt in voltages:
        ys = y_piecewise(region, Vt, xs_dense)
        plt.plot(xs_dense, ys, linewidth=2, label=f"model {Vt:.3f}V")

    plt.axvline(cfg["X_FLAT_END"], linestyle="--", linewidth=2, label=f"fringing starts @ x={cfg["X_FLAT_END"]:.1f}cm")
    plt.title(f"Equipotential simulation using Lagrangian interpolation, cubic splines, and Lagrange equations | zoomed to 102% board")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.grid(True)
    plt.legend()
    plt.minorticks_on()
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.25))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.25))
    plt.grid(True, which="major")
    plt.grid(True, which="minor", alpha=0.3)
    ax = plt.gca()
    ax.add_patch(Rectangle(
        (cfg["BOARD_X_MIN"], cfg["BOARD_Y_MIN"]),
        cfg["BOARD_X_MAX"] - cfg["BOARD_X_MIN"],
        cfg["BOARD_Y_MAX"] - cfg["BOARD_Y_MIN"],
        fill=True,
        color="grey",
        linewidth=2,
        zorder=5,
        label="board",
        alpha=0.1
    ))
    ax.add_patch(Rectangle(
        (cfg["BOARD_X_MIN"], cfg["BOARD_Y_MIN"]),
        cfg["BOARD_X_MAX"] - cfg["BOARD_X_MIN"],
        cfg["BOARD_Y_MAX"] - cfg["BOARD_Y_MIN"],
        fill=False,
        linewidth=2,
        zorder=5,
        label="board boundary",
    ))
    ax.set_xlim(cfg["BOARD_X_MIN"]-0.5, cfg["BOARD_X_MAX"]+0.5)
    ax.set_ylim(cfg["BOARD_Y_MIN"]-0.5,cfg["BOARD_Y_MAX"]+0.5)
    ax.hlines(cfg["Y1"], cfg["PLATE_X_START"], cfg["PLATE_X_END"], linewidth=2, zorder=6, label=cfg["plate1_label"],color="black")
    ax.hlines(cfg["Y2"], cfg["PLATE_X_START"], cfg["PLATE_X_END"], linewidth=2, zorder=6, label=cfg["plate2_label"],color="black")
    plt.show()

if __name__ == "__main__":
    region = "BC"  

    Vm = V_mid(region)
    voltages_to_plot = [5.0, 4.0, 3.0, Vm, 2.0, 1.0]

    plot_voltages(region, voltages_to_plot)

    print("Example y_at('AB', 2.0, 10) =", y_at("AB", 5.0, 20.0))
    print(y_at("AB",5.0, 10.0))


