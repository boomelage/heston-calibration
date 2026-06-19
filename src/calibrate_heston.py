"""Heston calibration engine (PLAN.md Work items 2 & 3: hardened, IV-space gate).

`calibrate_heston(vol_matrix, s, r, g)` fits Heston (theta, kappa, eta, rho, v0) to a
strike x maturity implied-vol surface and returns a dict. Hardening over the prototype:

  - **Box bounds** via `NonhomogeneousBoundaryConstraint` keep parameters economically sane
    (no rho -> +/-1, no exploding eta). Bounds are in `model.params()` order [theta, kappa, eta, rho, v0].
  - **Multiple restarts** from a small data-seeded grid; the lowest-**IV-RMSE** fit wins (removes the
    dependence on one arbitrary initial guess).
  - **Switchable in-engine objective** via `objective` ("price" relative-price, default, or "vol"
    IV-space). It only changes what LM minimises per restart; selection and the gate always rank/accept
    on IV-RMSE, so the objective is independent of how a day is chosen and accepted.
  - **Acceptance gate (IV-space).** A fit is rejected (params returned as None) if its
    implied-vol RMSE -- model-implied vol vs market vol, in vol points -- exceeds `IV_RMSE_ACCEPT`,
    or any parameter is pinned to a bound. A boundary fit is a non-fit.
  - **Diagnostics** in the return dict: `iv_rmse` (the gate), `rmse` (relative-price, retained for
    reference), `n_helpers`, `accepted`.

Why IV-space, not price-space (PLAN.md "Improving calibration performance", lever 2). The previous
gate used `HestonModelHelper.calibrationError()`, the *relative-price* RMSE. That denominator is the
option price, so cheap deep-OTM wings inflate it: a pooled day fitting within ~1 vol point everywhere
still scored ~0.07 relative-price RMSE and was wrongly rejected. The vol-point residual is the
natural, interpretable surface-fit metric. We invert each helper's model price back to a Black vol
with `BlackCalibrationHelper.impliedVolatility(modelValue, ...)` (same inversion
`validate_calibrations.py` uses externally) and compare to the market vol that built the helper.
The relative-price RMSE is still computed and returned as `rmse` for continuity, but no longer gates.
"""
import numpy as np
import pandas as pd
import QuantLib as ql

from config import (
    LOW, HIGH, IV_RMSE_ACCEPT, BOUND_TOL,
    IV_ACC, IV_MAXEVAL, IV_LO, IV_HI,
    DEFAULT_OBJECTIVE, SEED_GRID_TEMPLATE, SEED_VAR_FALLBACK, SEED_VAR_LO, SEED_VAR_HI,
)

# String->QuantLib-enum objective map. Kept next to the engine (live ql objects, not serialisable);
# the string names/default live in config. Selection and the gate always run off IV-space RMSE, so
# switching the objective only changes what each restart minimises, not how restarts are ranked.
#   "price" -> RelativePriceError: cheap (one Heston price per residual), the long-standing default.
#   "vol"   -> ImpliedVolError: inverts each model price to a Black vol every LM iteration, so it is
#              more expensive and can throw mid-search (caught per-restart), but weights cells evenly
#              in vol points -- the units the surface and the gate are quoted in.
_ERR = {
    "price": ql.HestonModelHelper.RelativePriceError,
    "vol": ql.HestonModelHelper.ImpliedVolError,
}

_FAIL = {k: None for k in ("theta", "kappa", "eta", "rho", "v0", "feller", "rmse", "iv_rmse")}


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
    var = float(np.median(vols)) ** 2 if vols.size else SEED_VAR_FALLBACK
    var = min(max(var, SEED_VAR_LO), SEED_VAR_HI)
    # expand each template row into (v0, kappa, theta, eta, rho) -- HestonProcess constructor order
    return [(var * v0_mult, kappa, var * theta_mult, eta, rho)
            for v0_mult, kappa, theta_mult, eta, rho in SEED_GRID_TEMPLATE]


def _iv_rmse(helpers, mkt_vols):
    """RMSE between each helper's model-implied Black vol and its market vol, in vol points."""
    resid = []
    for h, mkt in zip(helpers, mkt_vols):
        try:
            model_iv = h.impliedVolatility(h.modelValue(), IV_ACC, IV_MAXEVAL, IV_LO, IV_HI)
        except RuntimeError:
            continue
        if np.isfinite(model_iv):
            resid.append(model_iv - mkt)
    resid = np.asarray(resid)
    return float(np.sqrt(np.mean(resid ** 2))) if resid.size else np.nan


def _calibrate_once(start, surface, s, r_ts, g_ts, S_handle, constraint, error_type):
    """One bounded calibration from `start`. Returns (params_list, iv_rmse, price_rmse, n_helpers)."""
    v0, kappa, theta, eta, rho = start
    process = ql.HestonProcess(r_ts, g_ts, S_handle, v0, kappa, theta, eta, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    helpers, mkt_vols = [], []
    for t in surface.columns:
        for k in surface.index:
            vol = surface.loc[k, t]
            if not pd.isna(vol):
                helper = ql.HestonModelHelper(
                    ql.Period(int(t), ql.Days),
                    ql.UnitedStates(ql.UnitedStates.NYSE),
                    float(s), float(k),
                    ql.QuoteHandle(ql.SimpleQuote(float(vol))),
                    r_ts, g_ts, error_type,
                )
                helper.setPricingEngine(engine)
                helpers.append(helper)
                mkt_vols.append(float(vol))

    lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
    model.calibrate(helpers, lm, ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8), constraint)

    # Relative-price RMSE computed directly from model/market values, so `rmse` keeps the same
    # meaning regardless of `error_type` (h.calibrationError() would otherwise follow the objective).
    errs = np.array([(h.modelValue() - h.marketValue()) / h.marketValue()
                     for h in helpers if h.marketValue() != 0.0])
    price_rmse = float(np.sqrt(np.mean(errs ** 2))) if errs.size else np.nan
    iv_rmse = _iv_rmse(helpers, mkt_vols)
    return list(model.params()), iv_rmse, price_rmse, len(helpers)


def calibrate_heston(vol_matrix, s, r, g, objective=DEFAULT_OBJECTIVE) -> dict:
    error_type = _ERR[objective]
    calculation_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = calculation_date
    day_count = ql.Actual365Fixed()
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, float(r), day_count))
    g_ts = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, float(g), day_count))
    S_handle = ql.QuoteHandle(ql.SimpleQuote(float(s)))
    constraint = ql.NonhomogeneousBoundaryConstraint(ql.Array(LOW), ql.Array(HIGH))

    best = None  # (params, iv_rmse, price_rmse, n_helpers), ranked by iv_rmse
    for start in _seed_grid(vol_matrix):
        try:
            params, iv_rmse, price_rmse, n_helpers = _calibrate_once(
                start, vol_matrix, s, r_ts, g_ts, S_handle, constraint, error_type)
        except RuntimeError:
            continue
        if not np.isfinite(iv_rmse):
            continue
        if best is None or iv_rmse < best[1]:
            best = (params, iv_rmse, price_rmse, n_helpers)

    if best is None:
        return {**_FAIL, "n_helpers": 0, "accepted": False}

    params, iv_rmse, price_rmse, n_helpers = best
    theta, kappa, eta, rho, v0 = params
    accepted = (iv_rmse <= IV_RMSE_ACCEPT) and not _on_boundary(params)
    if not accepted:
        # Reject: null the parameters (the pipeline drops null rows) but keep the diagnostics.
        return {**_FAIL, "iv_rmse": iv_rmse, "rmse": price_rmse,
                "n_helpers": n_helpers, "accepted": False}

    return {
        "theta": theta, "kappa": kappa, "eta": eta, "rho": rho, "v0": v0,
        "feller": 2 * kappa * theta - eta ** 2,
        "iv_rmse": iv_rmse, "rmse": price_rmse, "n_helpers": n_helpers, "accepted": True,
    }
