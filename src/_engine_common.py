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
    WING_WEIGHT_GAIN, WING_WEIGHT_POWER, WING_WEIGHT_SCALE,
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


def _seed_var(vol_matrix):
    """The surface's variance level: clip(median(vol)^2, [SEED_VAR_LO, SEED_VAR_HI]).

    Shared by both engines' `_seed_grid`; falls back to SEED_VAR_FALLBACK on an empty surface.
    """
    vols = vol_matrix.to_numpy(dtype=float)
    vols = vols[np.isfinite(vols)]
    var = float(np.median(vols)) ** 2 if vols.size else SEED_VAR_FALLBACK
    return min(max(var, SEED_VAR_LO), SEED_VAR_HI)


def _wing_weight(k, s):
    """LM weight for a cell at strike k against reference spot s: 1 at ATM, rising into the wings by
    |log(k/s)|. GAIN=0 => 1.0 everywhere (uniform). QuantLib normalises these, so only ratios matter."""
    x = abs(np.log(float(k) / float(s)))
    return 1.0 + WING_WEIGHT_GAIN * (x / WING_WEIGHT_SCALE) ** WING_WEIGHT_POWER


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
