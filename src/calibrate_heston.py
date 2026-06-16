"""Heston calibration engine (PLAN.md Work item 2: hardened).

`calibrate_heston(vol_matrix, s, r, g)` fits Heston (theta, kappa, eta, rho, v0) to a
strike x maturity implied-vol surface and returns a dict. Hardening over the prototype:

  - **Box bounds** via `NonhomogeneousBoundaryConstraint` keep parameters economically sane
    (no rho -> +/-1, no exploding eta). Bounds are in `model.params()` order [theta, kappa, eta, rho, v0].
  - **Multiple restarts** from a small data-seeded grid; the lowest-RMSE fit wins (removes the
    dependence on one arbitrary initial guess).
  - **Acceptance gate** replaces the old "did the params move from the guess" sentinel: a fit is
    rejected (params returned as None) if its relative-price RMSE exceeds `RMSE_ACCEPT` or any
    parameter is pinned to a bound. A boundary fit is a non-fit.
  - **Diagnostics** in the return dict: `rmse`, `n_helpers`, `accepted`.

Note: `HestonModelHelper.calibrationError()` is the *relative-price* error in this QuantLib
build (the implied-vol error type is not exposed via SWIG); that is fine for ranking restarts
and for the price-space gate. For interpretable vol-point error use `validate_calibrations.py`.
"""
import numpy as np
import pandas as pd
import QuantLib as ql

# Bounds in model.params() order: [theta, kappa, eta, rho, v0]. rho upper kept slightly positive
# (equity leverage => negative) but not forced; eta capped at 2.0 (SPX vol-of-vol ~0.3-1.2).
LOW = [1e-4, 1e-2, 1e-2, -0.999, 1e-4]
HIGH = [1.0, 20.0, 2.0, 0.5, 1.0]

# Acceptance gate on the engine's relative-price RMSE (HestonModelHelper.calibrationError()).
# Kept deliberately strict. Across 2024-10-07..11 a 0.05 bar rejects ~99% of per-bucket fits, but
# loosening it (e.g. to 0.15) only admits *under-determined* fits: a thin per-bucket surface cannot
# identify five parameters, so a passing RMSE there buys a degenerate, cross-bucket-unstable result,
# not a trustworthy one. The fix is more information per fit (PLAN.md Work item 3 -- one calibration
# per day), not a lower standard. Economic reasonability and the rigorous IV-space (vol-point)
# residual are graded downstream by validate_calibrations.py; an IV-space gate is the planned
# replacement for this price-space proxy -- see PLAN.md "Improving calibration performance".
RMSE_ACCEPT = 0.05    # max relative-price RMSE for an accepted fit (strict; do not loosen -- see above)
BOUND_TOL = 1e-3      # fraction of a bound's span within which a param counts as "pegged"

_FAIL = {k: None for k in ("theta", "kappa", "eta", "rho", "v0", "feller", "rmse")}


def _on_boundary(params):
    """True if any param sits within BOUND_TOL*span of its bound -> the fit hit the wall."""
    for p, lo, hi in zip(params, LOW, HIGH):
        span = hi - lo
        if (p - lo) < BOUND_TOL * span or (hi - p) < BOUND_TOL * span:
            return True
    return False


def _seed_grid(vol_matrix):
    """A small, deterministic set of starting points, seeded from the surface's own level."""
    vols = vol_matrix.to_numpy(dtype=float)
    vols = vols[np.isfinite(vols)]
    var = float(np.median(vols)) ** 2 if vols.size else 0.04
    var = min(max(var, 1e-3), 0.25)
    # each tuple is (v0, kappa, theta, eta, rho) -- HestonProcess constructor order
    return [
        (var, 1.0, var, 0.50, -0.70),
        (var, 3.0, var, 1.00, -0.50),
        (var, 0.5, var, 0.30, -0.90),
        (var, 5.0, var, 0.80, -0.60),
        (var * 0.8, 2.0, var * 1.2, 0.60, -0.75),
        (var, 8.0, var, 1.20, -0.40),
    ]


def _calibrate_once(start, surface, s, r_ts, g_ts, S_handle, constraint):
    """One bounded calibration from `start`. Returns (params_list, rmse, n_helpers)."""
    v0, kappa, theta, eta, rho = start
    process = ql.HestonProcess(r_ts, g_ts, S_handle, v0, kappa, theta, eta, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    helpers = []
    for t in surface.columns:
        for k in surface.index:
            vol = surface.loc[k, t]
            if not pd.isna(vol):
                helper = ql.HestonModelHelper(
                    ql.Period(int(t), ql.Days),
                    ql.UnitedStates(ql.UnitedStates.NYSE),
                    float(s), float(k),
                    ql.QuoteHandle(ql.SimpleQuote(float(vol))),
                    r_ts, g_ts,
                )
                helper.setPricingEngine(engine)
                helpers.append(helper)

    lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    model.calibrate(helpers, lm, ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8), constraint)

    errs = np.array([h.calibrationError() for h in helpers])
    rmse = float(np.sqrt(np.mean(errs ** 2))) if errs.size else np.nan
    return list(model.params()), rmse, len(helpers)


def calibrate_heston(vol_matrix, s, r, g) -> dict:
    calculation_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = calculation_date
    day_count = ql.Actual365Fixed()
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, float(r), day_count))
    g_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, float(g), day_count))
    S_handle = ql.QuoteHandle(ql.SimpleQuote(float(s)))
    constraint = ql.NonhomogeneousBoundaryConstraint(ql.Array(LOW), ql.Array(HIGH))

    best = None  # (params, rmse, n_helpers)
    for start in _seed_grid(vol_matrix):
        try:
            params, rmse, n_helpers = _calibrate_once(
                start, vol_matrix, s, r_ts, g_ts, S_handle, constraint)
        except RuntimeError:
            continue
        if not np.isfinite(rmse):
            continue
        if best is None or rmse < best[1]:
            best = (params, rmse, n_helpers)

    if best is None:
        return {**_FAIL, "n_helpers": 0, "accepted": False}

    params, rmse, n_helpers = best
    theta, kappa, eta, rho, v0 = params
    accepted = (rmse <= RMSE_ACCEPT) and not _on_boundary(params)
    if not accepted:
        # Reject: null the parameters (the pipeline drops null rows) but keep the diagnostics.
        return {**_FAIL, "rmse": rmse, "n_helpers": n_helpers, "accepted": False}

    return {
        "theta": theta, "kappa": kappa, "eta": eta, "rho": rho, "v0": v0,
        "feller": 2 * kappa * theta - eta ** 2,
        "rmse": rmse, "n_helpers": n_helpers, "accepted": True,
    }
