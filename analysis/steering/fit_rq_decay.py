"""Fit piecewise rational quadratic (RQ) validity decay curves to log-odds data.

Models how steering effectiveness decays with increasing coefficient magnitude.
The RQ model captures three regimes: linear, transition, and breakdown.

From Xu et al. "Why Steering Works" (arXiv:2602.02343), Eq. 12-15.

Input:  experiments/{experiment}/analysis/logodds/{trait}/{method}/layer{N}.json
Output: experiments/{experiment}/analysis/rq_curves/{trait}/{method}/layer{N}.json
Usage:
    python analysis/steering/fit_rq_decay.py \
        --experiment wsw_xu_et_al \
        --trait pv_natural/evil_v3 \
        --method probe \
        --layer 11
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.optimize import minimize


# =============================================================================
# RQ decay model
# =============================================================================

def rq_decay(m, m_center, L, p):
    """Rational quadratic decay: D(m) = (1 + (m - m_center)^2 / L)^{-p}"""
    return (1.0 + (m - m_center) ** 2 / L) ** (-p)


def pref_model(m, params):
    """PrefOdds(m) = (alpha * m + beta) * D(m) + b

    Piecewise: D uses different params for m >= 0 vs m < 0.
    """
    alpha, beta, b = params["alpha"], params["beta"], params["b"]
    m_plus, L_plus, p_plus = params["m_plus"], params["L_plus"], params["p_plus"]
    m_minus, L_minus, p_minus = params["m_minus"], params["L_minus"], params["p_minus"]

    if m >= 0:
        D = rq_decay(m, m_plus, L_plus, p_plus)
    else:
        D = rq_decay(m, m_minus, L_minus, p_minus)

    return (alpha * m + beta) * D + b


def util_model(m, params):
    """UtilOdds(m) = beta_u * D(m) + b_u"""
    beta_u, b_u = params["beta_u"], params["b_u"]
    m_plus, L_plus, p_plus = params["m_plus"], params["L_plus"], params["p_plus"]
    m_minus, L_minus, p_minus = params["m_minus"], params["L_minus"], params["p_minus"]

    if m >= 0:
        D = rq_decay(m, m_plus, L_plus, p_plus)
    else:
        D = rq_decay(m, m_minus, L_minus, p_minus)

    return beta_u * D + b_u


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return 1.0 - ss_res / ss_tot


# =============================================================================
# Fitting
# =============================================================================

def fit_pref_curve(coefficients, pref_odds):
    """Fit preference log-odds curve. Returns params dict and R²."""
    m = np.array(coefficients)
    y = np.array(pref_odds)

    # Parameter order: alpha, beta, b, m_plus, L_plus, p_plus, m_minus, L_minus, p_minus
    def objective(x):
        params = {
            "alpha": x[0], "beta": x[1], "b": x[2],
            "m_plus": x[3], "L_plus": x[4], "p_plus": x[5],
            "m_minus": x[6], "L_minus": x[7], "p_minus": x[8],
        }
        pred = np.array([pref_model(mi, params) for mi in m])
        return np.sum((y - pred) ** 2)

    # Continuity at m=0: D(0+) = D(0-)
    # (1 + m_plus^2 / L_plus)^{-p_plus} = (1 + m_minus^2 / L_minus)^{-p_minus}
    def continuity_constraint(x):
        d_plus = (1.0 + x[3] ** 2 / x[4]) ** (-x[5])
        d_minus = (1.0 + x[6] ** 2 / x[7]) ** (-x[8])
        return d_plus - d_minus

    # Initial guess
    x0 = [
        0.1,   # alpha (slope)
        0.0,   # beta (intercept of linear part)
        y[np.argmin(np.abs(m))],  # b (offset, near zero-coefficient value)
        1.0,   # m_plus (center of positive decay)
        10.0,  # L_plus (width)
        1.0,   # p_plus (sharpness)
        -1.0,  # m_minus
        10.0,  # L_minus
        1.0,   # p_minus
    ]

    bounds = [
        (-10, 10),   # alpha
        (-10, 10),   # beta
        (-10, 10),   # b
        (-5, 20),    # m_plus
        (0.01, 100), # L_plus > 0
        (0.01, 10),  # p_plus > 0
        (-20, 5),    # m_minus
        (0.01, 100), # L_minus > 0
        (0.01, 10),  # p_minus > 0
    ]

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds,
        constraints={"type": "eq", "fun": continuity_constraint},
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    params = {
        "alpha": result.x[0], "beta": result.x[1], "b": result.x[2],
        "m_plus": result.x[3], "L_plus": result.x[4], "p_plus": result.x[5],
        "m_minus": result.x[6], "L_minus": result.x[7], "p_minus": result.x[8],
    }

    pred = np.array([pref_model(mi, params) for mi in m])
    r2 = r_squared(y, pred)

    return params, r2, result.success


def fit_util_curve(coefficients, util_odds):
    """Fit utility log-odds curve. Returns params dict and R²."""
    m = np.array(coefficients)
    y = np.array(util_odds)

    # Filter out -inf values
    valid = np.isfinite(y)
    if valid.sum() < 5:
        return None, 0.0, False
    m_valid = m[valid]
    y_valid = y[valid]

    # Parameter order: beta_u, b_u, m_plus, L_plus, p_plus, m_minus, L_minus, p_minus
    def objective(x):
        params = {
            "beta_u": x[0], "b_u": x[1],
            "m_plus": x[2], "L_plus": x[3], "p_plus": x[4],
            "m_minus": x[5], "L_minus": x[6], "p_minus": x[7],
        }
        pred = np.array([util_model(mi, params) for mi in m_valid])
        return np.sum((y_valid - pred) ** 2)

    def continuity_constraint(x):
        d_plus = (1.0 + x[2] ** 2 / x[3]) ** (-x[4])
        d_minus = (1.0 + x[5] ** 2 / x[6]) ** (-x[7])
        return d_plus - d_minus

    x0 = [
        np.max(y_valid) - np.min(y_valid),  # beta_u (amplitude)
        np.min(y_valid),                     # b_u (floor)
        1.0, 10.0, 1.0,                     # m_plus, L_plus, p_plus
        -1.0, 10.0, 1.0,                    # m_minus, L_minus, p_minus
    ]

    bounds = [
        (-50, 50),   # beta_u
        (-50, 50),   # b_u
        (-5, 20),    # m_plus
        (0.01, 100), # L_plus
        (0.01, 10),  # p_plus
        (-20, 5),    # m_minus
        (0.01, 100), # L_minus
        (0.01, 10),  # p_minus
    ]

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds,
        constraints={"type": "eq", "fun": continuity_constraint},
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    params = {
        "beta_u": result.x[0], "b_u": result.x[1],
        "m_plus": result.x[2], "L_plus": result.x[3], "p_plus": result.x[4],
        "m_minus": result.x[5], "L_minus": result.x[6], "p_minus": result.x[7],
    }

    pred = np.array([util_model(mi, params) for mi in m_valid])
    r2 = r_squared(y_valid, pred)

    return params, r2, result.success


def find_breakdown_coefficient(params, threshold=0.5):
    """Find the coefficient where D(m) drops below threshold on the positive side."""
    m_plus = params.get("m_plus", params.get("m_plus", 0))
    L_plus = params.get("L_plus", params.get("L_plus", 10))
    p_plus = params.get("p_plus", params.get("p_plus", 1))

    # D(m) = (1 + (m - m_plus)^2 / L_plus)^{-p_plus} = threshold
    # (m - m_plus)^2 / L_plus = threshold^{-1/p_plus} - 1
    ratio = threshold ** (-1.0 / p_plus) - 1.0
    if ratio < 0:
        return float("inf")
    displacement = (ratio * L_plus) ** 0.5
    return m_plus + displacement


def main():
    parser = argparse.ArgumentParser(description="Fit RQ validity decay curves to log-odds data")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()

    # Load log-odds data
    input_path = Path(f"experiments/{args.experiment}/analysis/logodds/{args.trait}/{args.method}/layer{args.layer}.json")
    if not input_path.exists():
        print(f"ERROR: Log-odds data not found at {input_path}")
        print("Run preference_utility_logodds.py first.")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    coefficients = data["coefficients"]
    pref_odds = data["pref_odds"]
    util_odds = data["util_odds"]

    print(f"Fitting RQ curves for {args.trait}/{args.method}/layer{args.layer}")
    print(f"  {len(coefficients)} coefficient points: [{min(coefficients)}, {max(coefficients)}]")

    # Fit preference curve
    pref_params, pref_r2, pref_ok = fit_pref_curve(coefficients, pref_odds)
    print(f"\n  Preference curve: R²={pref_r2:.4f} (converged={pref_ok})")
    print(f"    Linear slope (alpha): {pref_params['alpha']:.4f}")
    print(f"    Breakdown (D=0.5): coef={find_breakdown_coefficient(pref_params):.1f}")

    # Fit utility curve
    util_params, util_r2, util_ok = fit_util_curve(coefficients, util_odds)
    if util_params:
        print(f"\n  Utility curve: R²={util_r2:.4f} (converged={util_ok})")
        breakdown = find_breakdown_coefficient(util_params)
        print(f"    Breakdown (D=0.5): coef={breakdown:.1f}")
    else:
        print("\n  Utility curve: insufficient valid data points")
        breakdown = None

    # Save
    output_dir = Path(f"experiments/{args.experiment}/analysis/rq_curves/{args.trait}/{args.method}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"layer{args.layer}.json"

    result = {
        "trait": args.trait,
        "method": args.method,
        "layer": args.layer,
        "pref_params": {k: float(v) for k, v in pref_params.items()},
        "pref_r2": float(pref_r2),
        "pref_converged": pref_ok,
        "breakdown_coefficient": float(find_breakdown_coefficient(pref_params)),
    }

    if util_params:
        result["util_params"] = {k: float(v) for k, v in util_params.items()}
        result["util_r2"] = float(util_r2)
        result["util_converged"] = util_ok

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
