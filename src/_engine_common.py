"""Model-agnostic calibration helpers shared by the Heston and Bates engines.

These were factored out of `calibrate_heston.py` so `calibrate_bates.py` reuses the exact same
boundary/IV-RMSE/wing logic and the two engines cannot drift. Nothing here knows the parameter count
or order: `_on_boundary` takes whatever `low`/`high` it is handed, and `_iv_rmse`/`_wing_weight`
operate on QuantLib helpers and strikes, not on the parameter vector. The Heston engine's behaviour is
unchanged by the move (same functions, same call sites).
"""
import numpy as np

from config import (
    IV_ACC, IV_MAXEVAL, IV_LO, IV_HI,
    SEED_VAR_FALLBACK, SEED_VAR_LO, SEED_VAR_HI,
    SEED_GRID_TEMPLATE, FELLER_SEED_TEMPLATE, FELLER_PENALTY,
    WING_WEIGHT_GAIN, WING_WEIGHT_POWER, WING_WEIGHT_SCALE, WING_WEIGHT_FLOOR,
)


def _on_boundary(params, low, high):
    """True if any param sits within BOUND_TOL*span of its bound -> the fit hit the wall.

    `low`/`high` are passed in (not imported) so a caller can gate a subset of the parameter vector:
    the Bates engine passes only the 5 Heston params + their bounds, leaving the jump triple exempt.
    BOUND_TOL is imported here to keep the tolerance defined in exactly one place.
    """
    from config import BOUND_TOL
    for p, lo, hi in zip(params, low, high):
        span = hi - lo
        if (p - lo) < BOUND_TOL * span or (hi - p) < BOUND_TOL * span:
            return True
    return False


def _feller_violation(params):
    """Feller shortfall max(0, eta^2 - 2*kappa*theta) from a model.params() vector.

    Zero when the Feller condition 2*kappa*theta >= eta^2 holds, otherwise the size of the violation.
    Works for both engines: a HestonModel.params() vector is [theta, kappa, eta, rho, v0] and a
    BatesModel.params() vector is [theta, kappa, eta, rho, v0, nu, delta, lambda]; the first three
    entries are theta, kappa, eta in both, and jumps do not enter Feller. Used by the restart-selection
    soft penalty (config.FELLER_PENALTY) to bias the chosen restart toward Feller-compliant fits.
    """
    theta, kappa, eta = float(params[0]), float(params[1]), float(params[2])
    return max(0.0, eta ** 2 - 2.0 * kappa * theta)


def _anchor_distance(params, anchor, names, bounds):
    """Span-normalised squared distance of a model.params() vector from a prior-day `anchor` dict.

    `params` is the fitted vector, `names` its matching parameter names (the model's params_order from
    config.MODELS), `bounds` the box dict. Only names present (and non-None) in `anchor`
    contribute, so a caller can anchor a subset (e.g. the five Heston params, leaving the noisy jump
    triple free). Each term is ((param - prior) / bound_span) ** 2, so every parameter contributes on a
    comparable 0..1 scale. Zero when the fit equals the prior. Used by the cross-day regularisation
    (config.PARAM_ANCHOR_WEIGHT) to bias restart selection toward the previous day's parameters.
    """
    d = 0.0
    for p, name in zip(params, names):
        a = anchor.get(name)
        if a is None:
            continue
        lo, hi = bounds[name]
        span = (hi - lo) or 1.0
        d += ((float(p) - float(a)) / span) ** 2
    return d


def _seed_var(vol_matrix):
    """The surface's variance level: clip(median(vol)^2, [SEED_VAR_LO, SEED_VAR_HI]).

    Shared by both engines' `_seed_grid`; falls back to SEED_VAR_FALLBACK on an empty surface.
    """
    vols = vol_matrix.to_numpy(dtype=float)
    vols = vols[np.isfinite(vols)]
    var = float(np.median(vols)) ** 2 if vols.size else SEED_VAR_FALLBACK
    return min(max(var, SEED_VAR_LO), SEED_VAR_HI)


def _seed_grid(vol_matrix, jump_seed=None):
    """A small, deterministic set of restart starting points, seeded from the surface's own level.

    Each restart is a {name: value} dict (so the engine builds the process by NAME, not by a fragile
    positional order). The five Heston entries are v0/kappa/theta/eta/rho with v0=var*v0_mult and
    theta=var*theta_mult; the Feller-compliant rows are appended only when FELLER_PENALTY > 0, so with
    the penalty off the grid (and the committed baseline) is unchanged. When `jump_seed` = (lambda, nu,
    delta) is given (Bates) it is appended to every row so each restart starts from "almost no jumps";
    None (Heston) leaves the five Heston params alone. Shared by both models via _calibration_engine.
    """
    var = _seed_var(vol_matrix)
    template = SEED_GRID_TEMPLATE + (FELLER_SEED_TEMPLATE if FELLER_PENALTY > 0.0 else [])
    seeds = []
    for v0_mult, kappa, theta_mult, eta, rho in template:
        seed = {"v0": var * v0_mult, "kappa": kappa, "theta": var * theta_mult, "eta": eta, "rho": rho}
        if jump_seed is not None:
            lam, nu, delta = jump_seed
            seed.update({"lambda_": lam, "nu": nu, "delta": delta})
        seeds.append(seed)
    return seeds


def _anchor_seed(anchor, ctor_order):
    """A warm-start restart at the prior-day params as a {name: value} dict, or None.

    Reads exactly the `ctor_order` names (the process-constructor params) from the `anchor` dict, so the
    returned dict has the same keys a `_seed_grid` row carries and the engine consumes it uniformly.
    Returns None if any required name is missing/unparseable, so a partial or Heston-only prior simply
    contributes no warm seed (matching the old per-engine behaviour). Used only when the cross-day anchor
    is active (config.PARAM_ANCHOR_WEIGHT > 0 and a prior supplied).
    """
    try:
        return {name: float(anchor[name]) for name in ctor_order}
    except (KeyError, TypeError, ValueError):
        return None


def _wing_weight(k, s):
    """LM weight for a cell at strike k against reference spot s: 1 at ATM, shifted into the wings by
    |log(k/s)|. GAIN=0 => 1.0 everywhere (uniform). GAIN>0 up-weights the wings; GAIN<0 down-weights
    them, clamped to WING_WEIGHT_FLOOR so the weight stays positive (QuantLib needs positive weights).
    GAIN=0 makes the clamp inert, so the default path is identical. QuantLib normalises these, so only
    ratios matter."""
    x = abs(np.log(float(k) / float(s)))
    w = 1.0 + WING_WEIGHT_GAIN * (x / WING_WEIGHT_SCALE) ** WING_WEIGHT_POWER
    return max(w, WING_WEIGHT_FLOOR)


def _iv_rmse(helpers, mkt_vols, weights=None):
    """RMSE between each helper's model-implied Black vol and its market vol, in vol points.

    With `weights` (aligned to `helpers`) it returns the weight-normalised RMSE
    sqrt(sum(w*r^2)/sum(w)) -- used for restart *ranking* so a wing-weighted LM fit is ranked on the
    same objective it minimised. Unweighted (weights=None) it is the plain RMSE the acceptance gate
    and the reported `iv_rmse` use, so the gate keeps its "~2 vol points everywhere" meaning."""
    resid, wts = [], []
    for i, (h, mkt) in enumerate(zip(helpers, mkt_vols)):
        try:
            model_iv = h.impliedVolatility(h.modelValue(), IV_ACC, IV_MAXEVAL, IV_LO, IV_HI)
        except RuntimeError:
            continue
        if np.isfinite(model_iv):
            resid.append(model_iv - mkt)
            wts.append(1.0 if weights is None else weights[i])
    resid = np.asarray(resid)
    if not resid.size:
        return np.nan
    wts = np.asarray(wts)
    return float(np.sqrt(np.sum(wts * resid ** 2) / np.sum(wts)))
