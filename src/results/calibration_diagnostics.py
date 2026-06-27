"""Per-day calibration diagnostics: curvature mismatch, wing oscillation, kappa-floor pegging.

Read-only. The grading instrument for the Phase-3 mitigation levers (PLAN.md): it scores each
calibrated day on the three failure modes the levers target, so a lever can be A/B-graded on a date
subset instead of by eyeballing smiles.

It reuses `wing_residuals.load_residuals` (the same model-price -> Black-IV inversion the residual
diagnostic and `validate_calibrations.py` use) for the inverted model IV, then per (day, maturity, wing):

  1. CURVATURE / SKEW MISMATCH (Problem 1). Fit a quadratic  iv ~ a + b*lm + c*lm^2  to the market IV
     and to the inverted model IV over that wing's strikes (lm = ln K/S). Report
        d_slope = b_model - b_mkt      (skew mismatch)
        d_curv  = c_model - c_mkt      (curvature mismatch; >0 on the call wing = model too convex)
     A near-linear market wing has small c_mkt, so a large positive d_curv is the overstated curvature.

  2. WING MONOTONICITY / OSCILLATION (Problem 2). Order the inverted model IV from the money outward
     (by |lm|) within each wing and count sign changes of its first difference. A clean wing is
     monotone (0 reversals); >=1 reversal flags an oscillating model smile. Correlated downstream with
     feller/eta from calibrations.csv.

  3. KAPPA-FLOOR PROXIMITY / PEGGING (Problem 3). Per day, the distance of each Heston param to its
     nearer box bound in span-fractions; `near_floor` flags kappa within 5% of its floor. Tallied by
     year alongside the `pegged` rejection rate from rejections.csv.

    python src/results/calibration_diagnostics.py    # MODEL/OBJECTIVE from _results_config

`main(model=None, objective=None, save=True)` returns the per-day frame (notebook-importable, matching
the other src/results/ entry points). With save=True it writes
results/<model>/calibrations/<objective>/diagnostics.csv.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_CODE = Path(__file__).parent.resolve()
SRC = RESULTS_CODE.parent
RESULTS = SRC.parent / "results"

for _p in (str(SRC), str(RESULTS_CODE), str(RESULTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from _results_config import MODEL, OBJECTIVE
from wing_residuals import load_residuals

# A wing needs at least this many distinct-moneyness points to fit a quadratic / judge monotonicity.
MIN_WING_PTS = 3
# A wing whose |lm| span is narrower than this is skipped: the quadratic curvature coefficient
# (~ d2iv/dlm2) explodes on a near-collinear, narrow wing and would swamp the robust aggregates.
MIN_WING_LM_SPAN = 0.02
# kappa within this span-fraction of its floor is flagged near-floor (Problem 3).
NEAR_FLOOR_FRAC = 0.05


def _bounds(model):
    """(param_order, BOUNDS dict) for the model, from config (single source of the box)."""
    if model == "bates":
        return config.BATES_PARAM_ORDER, config.BATES_BOUNDS
    return config.PARAM_ORDER, config.BOUNDS


def _quad(lm, iv):
    """Fit iv ~ a + b*lm + c*lm^2; return (b, c) or (nan, nan) if too few distinct points."""
    lm = np.asarray(lm, dtype=float)
    iv = np.asarray(iv, dtype=float)
    ok = np.isfinite(lm) & np.isfinite(iv)
    lm, iv = lm[ok], iv[ok]
    if lm.size < MIN_WING_PTS or np.unique(lm).size < MIN_WING_PTS:
        return np.nan, np.nan
    if lm.max() - lm.min() < MIN_WING_LM_SPAN:
        return np.nan, np.nan
    try:
        c, b, _a = np.polyfit(lm, iv, 2)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan, np.nan
    return b, c


def _sign_changes(iv_from_atm):
    """Number of first-difference sign changes of an IV series ordered from the money outward."""
    v = np.asarray(iv_from_atm, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < MIN_WING_PTS:
        return 0
    d = np.diff(v)
    d = d[d != 0.0]
    if d.size < 2:
        return 0
    return int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))


def _score_wing(sub):
    """Curvature mismatch + oscillation count for one wing of one (day, maturity).

    `sub` carries lm, market_iv, model_iv for a single wing. Points are ordered from the money outward
    (ascending |lm|) for the monotonicity test; the quadratic is fit on signed lm.
    """
    sub = sub.sort_values("abs_lm")
    if len(sub) < MIN_WING_PTS:
        return None
    b_m, c_m = _quad(sub["lm"], sub["market_iv"])
    b_h, c_h = _quad(sub["lm"], sub["model_iv"])
    return {
        "d_slope": b_h - b_m,
        "d_curv": c_h - c_m,
        "osc": _sign_changes(sub["model_iv"].to_numpy()),
        "n_pts": len(sub),
    }


def _per_day(resid):
    """Aggregate the residual frame into one curvature/oscillation row per day.

    Wings are split by sign of lm (puts lm<0, calls lm>0). Each (date, maturity, wing) with enough
    points contributes a curvature mismatch and an oscillation count; the day row averages over its
    maturities and reports the call/put wings separately (Problem 1 is call-wing-specific).
    """
    rows = []
    for date, day in resid.groupby("date"):
        call_curv, put_curv, slopes, osc_flags, n_mats = [], [], [], 0, 0
        for _t, mat in day.groupby("days_to_maturity"):
            scored_any = False
            for wing, sign in (("call", 1), ("put", -1)):
                w = mat[np.sign(mat["lm"]) == sign]
                s = _score_wing(w)
                if s is None:
                    continue
                scored_any = True
                slopes.append(s["d_slope"])
                (call_curv if wing == "call" else put_curv).append(s["d_curv"])
                if s["osc"] >= 1:
                    osc_flags += 1
            if scored_any:
                n_mats += 1
        if n_mats == 0:
            continue
        n_wing_scores = len(call_curv) + len(put_curv)
        # Median over maturities: the per-wing quadratic coefficient is heavy-tailed, so a robust
        # centre keeps the day score from being set by a single ill-conditioned maturity.
        rows.append({
            "date": date,
            "n_mats_scored": n_mats,
            "d_curv_call": float(np.nanmedian(call_curv)) if call_curv else np.nan,
            "d_curv_put": float(np.nanmedian(put_curv)) if put_curv else np.nan,
            "d_curv_abs": float(np.nanmedian(np.abs(call_curv + put_curv))) if n_wing_scores else np.nan,
            "d_slope": float(np.nanmedian(slopes)) if slopes else np.nan,
            "osc_wing_frac": osc_flags / n_wing_scores if n_wing_scores else np.nan,
        })
    return pd.DataFrame(rows)


def _merge_params(diag, model, objective):
    """Join per-day curvature/oscillation with calibrations.csv params + kappa-floor proximity."""
    calib_path, _rej, _tests = config.calib_paths(model, objective)
    cal = pd.read_csv(calib_path)
    cal["date"] = cal["date"].astype(str)
    diag["date"] = diag["date"].astype(str)
    keep = ["date", "kappa", "eta", "v0", "theta", "rho", "feller"]
    out = diag.merge(cal[keep], on="date", how="left")

    _order, bounds = _bounds(model)
    # span-fraction distance to the NEARER bound for each Heston param present in calibrations.csv.
    for p in ("kappa", "eta", "v0", "theta", "rho"):
        lo, hi = bounds[p]
        span = hi - lo
        out[f"{p}_bound_frac"] = np.minimum(out[p] - lo, hi - out[p]) / span
    out["near_floor"] = (out["kappa"] - bounds["kappa"][0]) / (bounds["kappa"][1] - bounds["kappa"][0]) < NEAR_FLOOR_FRAC
    out["year"] = [int(str(d)[:4]) for d in out["date"]]   # dates are 'YYYY-MM-DD'
    return out


def _pegged_by_year(model, objective):
    """Fraction of attempted days each year rejected for `pegged` (accepts + rejects = attempts)."""
    calib_path, rej_path, _tests = config.calib_paths(model, objective)
    acc = pd.read_csv(calib_path)[["date"]].assign(pegged=False)
    if Path(rej_path).exists():
        r = pd.read_csv(rej_path)
        r = r[["date", "reason"]].assign(pegged=lambda d: d["reason"] == "pegged")[["date", "pegged"]]
    else:
        r = pd.DataFrame({"date": pd.Series(dtype=str), "pegged": pd.Series(dtype=bool)})
    both = pd.concat([acc, r], ignore_index=True)
    both["year"] = [int(str(d)[:4]) for d in both["date"]]   # dates are 'YYYY-MM-DD'
    g = both.groupby("year")["pegged"]
    return pd.DataFrame({"n_attempts": g.count(), "pegged_rate": g.mean()})


def summarize(diag, model, objective):
    """Print the per-year diagnostic summary and return the by-year frame.

    Year-level centres are MEDIANS (the per-wing curvature coefficient is heavy-tailed), plus
    `call_convex_frac` = fraction of days whose call-wing curvature exceeds the market's (the
    interpretable form of Problem 1: a value well above 0.5 means the model is systematically too
    convex on the call wing).
    """
    by_year = diag.groupby("year").agg(
        n_days=("date", "count"),
        d_curv_call=("d_curv_call", "median"),
        d_curv_put=("d_curv_put", "median"),
        call_convex_frac=("d_curv_call", lambda s: float((s > 0).mean())),
        d_slope=("d_slope", "median"),
        osc_frac=("osc_wing_frac", "mean"),
        kappa=("kappa", "mean"),
        eta=("eta", "mean"),
        feller_neg=("feller", lambda s: float((s < 0).mean())),
        kappa_floor_frac=("near_floor", "mean"),
    ).round(4)
    peg = _pegged_by_year(model, objective)
    by_year = by_year.join(peg["pegged_rate"].round(3))

    pd.options.display.float_format = "{:+.4f}".format
    print(f"\n=== calibration diagnostics  model={model} objective={objective} ===")
    print(f"    {len(diag)} scored days\n")
    print("d_curv_* = model - market quadratic curvature (>0 call wing = model too convex)")
    print("osc_frac = fraction of (maturity x wing) with a non-monotone model smile")
    print("kappa_floor_frac = days with kappa within 5% of its floor; pegged_rate from rejections.csv\n")
    print(by_year.to_string())
    print(f"\noverall  d_curv_call(med) {diag['d_curv_call'].median():+.4f}  call_convex_frac "
          f"{(diag['d_curv_call'] > 0).mean():.3f}  osc_frac {diag['osc_wing_frac'].mean():.3f}  "
          f"near_floor {diag['near_floor'].mean():.3f}")
    return by_year


def main(model=None, objective=None, save=True):
    """Score every calibrated day and (optionally) write diagnostics.csv. Returns the per-day frame.

    `model`/`objective` default to the `_results_config` switches when None, so a notebook can grade a
    different run without editing `_results_config`.
    """
    model = model or MODEL
    objective = objective or OBJECTIVE
    resid = load_residuals(objective=objective, model=model)
    diag = _per_day(resid)
    diag = _merge_params(diag, model, objective)
    summarize(diag, model, objective)
    if save:
        out_path = config.calib_paths(model, objective)[0].parent / "diagnostics.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        diag.sort_values("date").to_csv(out_path, index=False)
        print(f"\nwrote {out_path}")
    return diag


if __name__ == "__main__":
    main()
