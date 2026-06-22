"""Bates (1996) calibration engine -- a drop-in alternative to calibrate_heston (see PLAN.md, Bates extension).

Bates is Heston stochastic vol plus Merton lognormal jumps in the log-price, so it adds three
parameters to Heston's five: jump intensity `lambda` (jumps/yr), mean log-jump `nu`, and log-jump std
`delta`. With `lambda = 0` Bates collapses to pure Heston. `calibrate_bates(vol_matrix, s, r, g)` fits
all eight to a strike x maturity implied-vol surface and returns a dict that is a SUPERSET of the
Heston engine's return (same keys + `lambda_, nu, delta`), so the orchestrator reads it unchanged.

Mirrors calibrate_heston.py exactly, swapping the model objects and the parameter handling. The
model-agnostic helpers (`_on_boundary`, `_seed_var`, `_wing_weight`, `_iv_rmse`) are shared via
`_engine_common` so the two engines cannot drift. Selection and the IV-space acceptance gate are
identical; only what each restart minimises (the `objective`) and the parameter vector differ.

THREE DISTINCT ORDERINGS (conflating them mis-bounds the fit):
  1. `BatesModel.params()` returns `[theta, kappa, eta, rho, v0, nu, delta, lambda]` (confirmed live).
     This drives BATES_PARAM_ORDER / BATES_LOW/HIGH and the result unpack below. The first five are
     HestonModel.params() order; the jump triple appends as (nu, delta, lambda).
  2. `BatesProcess(...)` constructor takes `(..., v0, kappa, theta, eta, rho, lambda, nu, delta)`.
     This drives how `_seed_grid` rows are expanded (NOT the params() order).
  3. `vanilla_pricer.bates_price(...)` arg order is handled in `src/pricing`, not here.

Jump params and the acceptance gate. The boundary-pegging gate is checked on the FIVE Heston params
only. `lambda ~ 0` is a legitimate Heston collapse (not a wall-hit), and when `lambda ~ 0` the
`nu`/`delta` are unidentified and may park on a bound without meaning, so gating them would reject
otherwise-good fits. `lambda/nu/delta` are returned for inspection regardless. `feller = 2*kappa*theta
- eta**2` stays the Heston-diffusion quantity (jumps do not enter it); it is reported, never gates.
"""
import numpy as np
import pandas as pd
import QuantLib as ql

from config import (
    BATES_LOW, BATES_HIGH, BATES_JUMP_SEED, IV_RMSE_ACCEPT,
    DEFAULT_OBJECTIVE, SEED_GRID_TEMPLATE,
    WING_WEIGHT_GAIN,
    LM_ARGS, END_CRITERIA_ARGS,
    calendar as _calendar,
)
# Model-agnostic helpers shared with calibrate_heston.py (factored out so the two engines can't drift).
from _engine_common import _on_boundary, _seed_var, _wing_weight, _iv_rmse
# Single home of the QuantLib process/term-structure construction (constructor arg order, day count).
from pricing._quantlib_utils import _quantlib_utils
_qu = _quantlib_utils()

# String->QuantLib-enum objective map. The helper stays HestonModelHelper (ql.BatesHelper does not
# exist in this QuantLib build); only the pricing engine attached to it is a BatesEngine. As in the
# Heston engine, the objective only changes what each restart minimises -- selection and the gate run
# off IV-space RMSE either way.
_ERR = {
    "price": ql.HestonModelHelper.RelativePriceError,
    "vol": ql.HestonModelHelper.ImpliedVolError,
}

_FAIL = {k: None for k in ("theta", "kappa", "eta", "rho", "v0",
                           "nu", "delta", "lambda_", "feller", "rmse", "iv_rmse")}


def _seed_grid(vol_matrix):
    """Starting points in BatesProcess constructor order (v0, kappa, theta, eta, rho, lambda, nu, delta).

    Each Heston template row is expanded as in the Heston engine, then the single fixed jump seed
    BATES_JUMP_SEED = (lambda, nu, delta) is appended so every restart starts from "almost no jumps".
    """
    var = _seed_var(vol_matrix)
    lam, nu, delta = BATES_JUMP_SEED
    return [(var * v0_mult, kappa, var * theta_mult, eta, rho, lam, nu, delta)
            for v0_mult, kappa, theta_mult, eta, rho in SEED_GRID_TEMPLATE]


def _calibrate_once(start, surface, s, r_ts, g_ts, S_handle, constraint, error_type, objective):
    """One bounded Bates calibration from `start` (constructor order).

    Returns (params_list, iv_rmse_sel, iv_rmse_gate, price_rmse, n_helpers); `params_list` is in
    BatesModel.params() order [theta, kappa, eta, rho, v0, nu, delta, lambda]. As in the Heston engine,
    `iv_rmse_sel` is the (optionally wing-weighted) IV-RMSE LM saw for ranking; `iv_rmse_gate` is the
    unweighted IV-RMSE for the gate/reporting. With wing weighting off the two are identical."""
    v0, kappa, theta, eta, rho, lambda_, nu, delta = start
    process = _qu.bates_process(r_ts, g_ts, S_handle, kappa, theta, rho, eta, v0, lambda_, nu, delta)
    model = ql.BatesModel(process)
    engine = ql.BatesEngine(model)

    # Wing weights only meaningful in vol space (see config.py / PLAN Lever B); default-off (GAIN=0).
    apply_wing = (objective == "vol") and (WING_WEIGHT_GAIN > 0.0)

    helpers, mkt_vols, weights = [], [], []
    for t in surface.columns:
        for k in surface.index:
            vol = surface.loc[k, t]
            if not pd.isna(vol):
                helper = ql.HestonModelHelper(
                    ql.Period(int(t), ql.Days),
                    _calendar(),
                    float(s), float(k),
                    ql.QuoteHandle(ql.SimpleQuote(float(vol))),
                    r_ts, g_ts, error_type,
                )
                helper.setPricingEngine(engine)
                helpers.append(helper)
                mkt_vols.append(float(vol))
                weights.append(_wing_weight(k, s) if apply_wing else 1.0)

    lm = ql.LevenbergMarquardt(*LM_ARGS)
    end = ql.EndCriteria(*END_CRITERIA_ARGS)
    if apply_wing:
        # weights must be a plain python list (a DoubleVector); ql.Array does NOT bind this overload.
        model.calibrate(helpers, lm, end, constraint, weights)
    else:
        model.calibrate(helpers, lm, end, constraint)

    errs = np.array([(h.modelValue() - h.marketValue()) / h.marketValue()
                     for h in helpers if h.marketValue() != 0.0])
    price_rmse = float(np.sqrt(np.mean(errs ** 2))) if errs.size else np.nan
    iv_rmse_gate = _iv_rmse(helpers, mkt_vols)
    iv_rmse_sel = _iv_rmse(helpers, mkt_vols, weights) if apply_wing else iv_rmse_gate
    return list(model.params()), iv_rmse_sel, iv_rmse_gate, price_rmse, len(helpers)


def calibrate_bates(vol_matrix, s, r, g, objective=DEFAULT_OBJECTIVE) -> dict:
    error_type = _ERR[objective]
    calculation_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = calculation_date
    r_ts, g_ts = _qu._term_structures(r, g, calculation_date)
    S_handle = _qu._spot_handle(s)
    constraint = ql.NonhomogeneousBoundaryConstraint(ql.Array(BATES_LOW), ql.Array(BATES_HIGH))

    best = None  # (params, iv_rmse_sel, iv_rmse_gate, price_rmse, n_helpers)
    for start in _seed_grid(vol_matrix):
        try:
            params, iv_rmse_sel, iv_rmse_gate, price_rmse, n_helpers = _calibrate_once(
                start, vol_matrix, s, r_ts, g_ts, S_handle, constraint, error_type, objective)
        except RuntimeError:
            continue
        if not np.isfinite(iv_rmse_sel):
            continue
        if best is None or iv_rmse_sel < best[1]:
            best = (params, iv_rmse_sel, iv_rmse_gate, price_rmse, n_helpers)

    if best is None:
        return {**_FAIL, "n_helpers": 0, "accepted": False}

    params, iv_rmse_sel, iv_rmse, price_rmse, n_helpers = best
    # BatesModel.params() order (confirmed live): theta, kappa, eta, rho, v0, nu, delta, lambda.
    theta, kappa, eta, rho, v0, nu, delta, lambda_ = params
    # Pegging gate on the FIVE Heston params only (params[:5]); the jump triple is exempt (see module
    # docstring): lambda~0 is a legitimate Heston collapse, and nu/delta are unidentified when lambda~0.
    accepted = (iv_rmse <= IV_RMSE_ACCEPT) and not _on_boundary(params[:5], BATES_LOW[:5], BATES_HIGH[:5])
    if not accepted:
        return {**_FAIL, "iv_rmse": iv_rmse, "rmse": price_rmse,
                "n_helpers": n_helpers, "accepted": False}

    return {
        "theta": theta, "kappa": kappa, "eta": eta, "rho": rho, "v0": v0,
        "nu": nu, "delta": delta, "lambda_": lambda_,
        "feller": 2 * kappa * theta - eta ** 2,
        "iv_rmse": iv_rmse, "rmse": price_rmse, "n_helpers": n_helpers, "accepted": True,
    }
