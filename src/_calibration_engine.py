"""Unified Heston/Bates calibration engine.

`calibrate(model, vol_matrix, s, r, g)` fits a model ("heston" or "bates") to a strike x maturity
implied-vol surface and returns a dict. It replaces the former `calibrate_heston.py` /
`calibrate_bates.py`, which were ~identical bar the QuantLib classes, the parameter vector and the
pegging-gate slice. Those differences are now data: each model declares its parameter metadata once in
`config.MODELS` (orders, bounds, gate names, jump seed), and this module pairs it with a small per-model
`_WIRING` registry of the live QuantLib builders. `_resolve_spec` assembles the two into a `ModelSpec`,
and the rest of the engine is model-agnostic.

The hardening is unchanged from the prototype engines:
  - **Box bounds** via `NonhomogeneousBoundaryConstraint` (in model.params() order).
  - **Multiple restarts** from a data-seeded grid (`_engine_common._seed_grid`); the lowest-IV-RMSE fit
    wins. The selection score adds the optional soft Feller penalty (config.FELLER_PENALTY) and the
    optional cross-day anchor distance (config.PARAM_ANCHOR_WEIGHT); the gate and the reported iv_rmse
    stay the UNWEIGHTED IV-RMSE.
  - **Switchable in-engine objective** ("vol" IV-space, default, or "price" relative-price) -- it only
    changes what each restart minimises; selection and the gate always rank/accept on IV-RMSE.
  - **Acceptance gate (IV-space).** Reject (params returned as None) if the IV-RMSE exceeds
    IV_RMSE_ACCEPT or any *gated* parameter is pinned to a bound. For Bates the gate covers only the five
    Heston params (spec.gate_names); the jump triple is exempt (lambda~0 is a legitimate Heston collapse,
    and nu/delta are unidentified when lambda~0).

THREE distinct orderings, all sourced from config.MODELS (conflating them mis-bounds the fit):
  1. `params_order` -- model.params() order: drives bounds/low/high, the result unpack, anchor distance.
  2. `ctor_order` -- the *Process* constructor order: drives the seed / warm-start dicts.
  3. the pricing-helper arg order (kappa,theta,rho,eta,v0[,lambda_,nu,delta]) -- applied by the _WIRING
     make_process closures, which read the seed dict BY NAME so no positional order leaks out.
"""
import functools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import QuantLib as ql

import config
from config import (
    IV_RMSE_ACCEPT, DEFAULT_OBJECTIVE,
    WING_WEIGHT_GAIN, FELLER_PENALTY, PARAM_ANCHOR_WEIGHT,
    LM_ARGS, END_CRITERIA_ARGS,
    HESTON_INTEGRATION, BATES_INTEGRATION,
    calendar as _calendar,
)
# Model-agnostic helpers (pure; no QuantLib). _seed_grid/_anchor_seed return name->value dicts.
from _engine_common import (
    _on_boundary, _wing_weight, _iv_rmse, _feller_violation, _anchor_distance,
    _seed_grid, _anchor_seed,
)
# Single home of the QuantLib process/term-structure construction (constructor arg order, day count).
# pricing/ never imports config, so the project's CF-integration accuracy is injected here (and at the
# other two construction sites, _utils._qu and calibrator_prototype.vanp, from the same constants).
from pricing._quantlib_utils import _quantlib_utils
_qu = _quantlib_utils(heston_integration=HESTON_INTEGRATION, bates_integration=BATES_INTEGRATION)

# String->QuantLib-enum objective map. Live ql objects (not serialisable), so kept by the engine, not
# config. Both models use HestonModelHelper (there is no ql.BatesHelper in QuantLib 1.35); only the
# pricing engine attached to it differs. Selection and the gate always run off IV-space RMSE, so the
# objective only changes what each restart minimises.
#   "price" -> RelativePriceError; "vol" -> ImpliedVolError (inverts each model price to a Black vol per
#   LM iteration, more expensive and can throw mid-search -- caught per restart).
_ERR = {
    "price": ql.HestonModelHelper.RelativePriceError,
    "vol": ql.HestonModelHelper.ImpliedVolError,
}

# Per-model live-QuantLib wiring. Kept here (not config) because these are live ql objects. Each
# make_process reads the seed dict BY NAME and passes the params in the pricing-helper arg order
# (kappa,theta,rho,eta,v0[,lambda_,nu,delta]); make_model/make_engine build the model and its pricing
# engine (the engine carries config.{HESTON,BATES}_INTEGRATION, injected into _qu at construction).
_WIRING = {
    "heston": dict(
        make_process=lambda qu, r_ts, g_ts, S, p: qu.heston_process(
            r_ts, g_ts, S, p["kappa"], p["theta"], p["rho"], p["eta"], p["v0"]),
        make_model=lambda process: ql.HestonModel(process),
        make_engine=lambda qu, model: qu.heston_engine_for(model),
    ),
    "bates": dict(
        make_process=lambda qu, r_ts, g_ts, S, p: qu.bates_process(
            r_ts, g_ts, S, p["kappa"], p["theta"], p["rho"], p["eta"], p["v0"],
            p["lambda_"], p["nu"], p["delta"]),
        make_model=lambda process: ql.BatesModel(process),
        make_engine=lambda qu, model: qu.bates_engine_for(model),
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    """Everything the engine needs to calibrate one model: the config parameter data (config.MODELS)
    plus the live QuantLib wiring (_WIRING). Assembled and cached by `_resolve_spec`."""
    name: str
    params_order: tuple          # model.params() order
    ctor_order: tuple            # process constructor order (seeds)
    bounds: dict
    low: tuple                   # bounds[p][0] in params_order
    high: tuple                  # bounds[p][1] in params_order
    gate_names: tuple            # params the pegging gate rejects on
    jump_seed: tuple             # (lambda, nu, delta) ctor-tail, or None
    make_process: Callable
    make_model: Callable
    make_engine: Callable


@functools.lru_cache(maxsize=None)
def _resolve_spec(model):
    """Build (and cache) the ModelSpec for `model` from config.MODELS + the local ql wiring.

    Raises a clear ValueError listing the known models on an unknown name (the old _ENGINES[MODEL]
    dispatch raised a bare KeyError)."""
    try:
        data = config.MODELS[model]
        wiring = _WIRING[model]
    except KeyError:
        raise ValueError(f"unknown model {model!r}; expected one of {tuple(config.MODELS)}") from None
    return ModelSpec(
        name=model,
        params_order=tuple(data["params_order"]),
        ctor_order=tuple(data["ctor_order"]),
        bounds=data["bounds"],
        low=tuple(data["low"]),
        high=tuple(data["high"]),
        gate_names=tuple(data["gate_names"]),
        jump_seed=data["jump_seed"],
        **wiring,
    )


def _fail(spec):
    """The rejection sentinel: every parameter (in params_order) plus feller/rmse/iv_rmse as None."""
    return {**{n: None for n in spec.params_order}, "feller": None, "rmse": None, "iv_rmse": None}


def _calibrate_once(spec, start, surface, s, r_ts, g_ts, S_handle, constraint, error_type, objective):
    """One bounded calibration from the `start` seed dict (name->value).

    Returns (params_list, iv_rmse_sel, iv_rmse_gate, price_rmse, n_helpers). `params_list` is in
    spec.params_order (model.params() order). `iv_rmse_sel` is the (optionally wing-weighted) IV-RMSE the
    LM objective saw, used for restart ranking; `iv_rmse_gate` is the unweighted IV-RMSE used for the gate
    and reporting. With wing weighting off the two are identical."""
    process = spec.make_process(_qu, r_ts, g_ts, S_handle, start)
    model = spec.make_model(process)
    # CF-integration accuracy is config.{HESTON,BATES}_INTEGRATION, injected into every app-side
    # _quantlib_utils/vanilla_pricer from the same constants, so the fit, the calibration_tests
    # repricing and the IV inversion integrate alike.
    engine = spec.make_engine(_qu, model)

    # Wing weights only meaningful in vol space: "price" already up-weights cheap wings via the price
    # denominator, so stacking a wing weight there double-counts (see config.py / PLAN Lever B).
    # Any nonzero GAIN engages the weights: GAIN > 0 up-weights the wings (Lever B), GAIN < 0
    # down-weights them (Lever G, clamped positive by WING_WEIGHT_FLOOR). GAIN == 0 is the exact
    # unweighted baseline path.
    apply_wing = (objective == "vol") and (WING_WEIGHT_GAIN != 0.0)

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
        # Preserve the exact non-weighted call path so GAIN=0 reproduces the committed baselines.
        model.calibrate(helpers, lm, end, constraint)

    # Relative-price RMSE computed directly from model/market values, so `rmse` keeps the same meaning
    # regardless of `error_type` (h.calibrationError() would otherwise follow the objective).
    errs = np.array([(h.modelValue() - h.marketValue()) / h.marketValue()
                     for h in helpers if h.marketValue() != 0.0])
    price_rmse = float(np.sqrt(np.mean(errs ** 2))) if errs.size else np.nan
    iv_rmse_gate = _iv_rmse(helpers, mkt_vols)
    iv_rmse_sel = _iv_rmse(helpers, mkt_vols, weights) if apply_wing else iv_rmse_gate
    return list(model.params()), iv_rmse_sel, iv_rmse_gate, price_rmse, len(helpers)


def calibrate(model, vol_matrix, s, r, g, objective=DEFAULT_OBJECTIVE, anchor=None) -> dict:
    """Calibrate `model` ("heston"|"bates") to a strike x maturity IV surface; return a result dict.

    Multi-restart bounded LM, ranked by a selection score (wing-weighted IV-RMSE + FELLER_PENALTY *
    Feller-violation + PARAM_ANCHOR_WEIGHT * anchor-distance), gated on the UNWEIGHTED IV-RMSE and on
    boundary pegging of spec.gate_names. With wing weighting off, FELLER_PENALTY=0 and no anchor the
    score is the plain IV-RMSE, so ranking/gating reproduce the prototype engines exactly.

    Returns the fitted params keyed by name (model.params() order) plus feller/iv_rmse/rmse/n_helpers/
    accepted. Bates returns a superset (adds lambda_/nu/delta). A rejected fit nulls the params but keeps
    iv_rmse/rmse/n_helpers/accepted=False."""
    spec = _resolve_spec(model)
    error_type = _ERR[objective]
    calculation_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = calculation_date
    r_ts, g_ts = _qu._term_structures(r, g, calculation_date)
    S_handle = _qu._spot_handle(s)
    constraint = ql.NonhomogeneousBoundaryConstraint(ql.Array(list(spec.low)), ql.Array(list(spec.high)))

    # Cross-day anchor (Lever 5): active only when a prior is supplied AND PARAM_ANCHOR_WEIGHT > 0. It
    # adds a warm-start seed at the prior params and a Tikhonov term to the selection score. Off by
    # default (anchor=None), so the seed grid and ranking are exactly the committed baseline.
    use_anchor = anchor is not None and PARAM_ANCHOR_WEIGHT > 0.0
    seeds = list(_seed_grid(vol_matrix, spec.jump_seed))
    if use_anchor:
        warm = _anchor_seed(anchor, spec.ctor_order)
        if warm is not None:
            seeds.append(warm)

    best = None  # (params, iv_rmse_sel, iv_rmse_gate, price_rmse, n_helpers)
    best_score = np.inf
    for start in seeds:
        try:
            params, iv_rmse_sel, iv_rmse_gate, price_rmse, n_helpers = _calibrate_once(
                spec, start, vol_matrix, s, r_ts, g_ts, S_handle, constraint, error_type, objective)
        except RuntimeError:
            continue
        if not np.isfinite(iv_rmse_sel):
            continue
        score = iv_rmse_sel + FELLER_PENALTY * _feller_violation(params)
        if use_anchor:
            score += PARAM_ANCHOR_WEIGHT * _anchor_distance(params, anchor, spec.params_order, spec.bounds)
        if score < best_score:
            best_score = score
            best = (params, iv_rmse_sel, iv_rmse_gate, price_rmse, n_helpers)

    if best is None:
        return {**_fail(spec), "n_helpers": 0, "accepted": False}

    params, iv_rmse_sel, iv_rmse, price_rmse, n_helpers = best
    p = dict(zip(spec.params_order, params))
    # Pegging gate on spec.gate_names only (all five for Heston; the five Heston params for Bates, jump
    # triple exempt). _on_boundary is order-independent, so pairing the named values with their bounds is
    # identical to the old positional params[:5] / bounds-slice gate.
    gate_vals = [p[n] for n in spec.gate_names]
    gate_lo = [spec.bounds[n][0] for n in spec.gate_names]
    gate_hi = [spec.bounds[n][1] for n in spec.gate_names]
    accepted = (iv_rmse <= IV_RMSE_ACCEPT) and not _on_boundary(gate_vals, gate_lo, gate_hi)
    if not accepted:
        # Reject: null the parameters (the pipeline drops null rows) but keep the diagnostics.
        return {**_fail(spec), "iv_rmse": iv_rmse, "rmse": price_rmse,
                "n_helpers": n_helpers, "accepted": False}

    return {
        **p,
        "feller": 2 * p["kappa"] * p["theta"] - p["eta"] ** 2,
        "iv_rmse": iv_rmse, "rmse": price_rmse, "n_helpers": n_helpers, "accepted": True,
    }
