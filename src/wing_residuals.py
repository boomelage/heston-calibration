"""Residual-by-moneyness diagnostic (PLAN.md Phase 3 Lever B validation, step 2).

Read-only. Quantifies where the calibrated surface mis-fits the smile, stratified by log-moneyness
and maturity, so the wing-weighting lever (`config.WING_WEIGHT_*`) can be judged by whether it
shrinks the *wing* residual without breaking the body. This is the metric a GAIN sweep is graded on.

For each repriced contract in `results/calibrations/<objective>/calibration_tests/*.csv` it inverts
the fitted Heston price back to a Black implied vol (`utils.implied_vol`, the same dividend-consistent
inverter `validate_calibrations.py` uses) and forms the residual

    resid = model_iv - market_iv        (market_iv = the trade's `volatility` column)

so a NEGATIVE resid means the model UNDERESTIMATES IV there. It then reports, across all days:

  1. by signed log-moneyness ln(K/S)  -- puts on the left (negative), calls on the right (positive),
     to expose wing asymmetry;
  2. by |log-moneyness| -- the exact axis `_wing_weight` up-weights, so this table is what the GAIN
     sweep should move;
  3. by maturity band x wing.

    python src/wing_residuals.py            # OBJECTIVE set below (mirrors validate_calibrations.py)

`load_residuals(objective)` returns the per-contract frame and `summarize(df)` the bucket tables, so
the sweep can import and reuse them instead of re-reading the files.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from utils import implied_vol

SRC = Path(__file__).parent.resolve()
RESULTS = SRC.parent / "results"

OBJECTIVE = 'vol'   # which results/calibrations/<objective>/ to grade; mirrors validate_calibrations.py

# Signed log-moneyness ln(K/S) bin edges: puts < 0, calls > 0. ~+/-0.05 ~ 5% OTM.
LM_EDGES = [-np.inf, -0.20, -0.12, -0.07, -0.04, -0.02, 0.02, 0.04, 0.07, 0.12, 0.20, np.inf]
# |log-moneyness| edges: the axis _wing_weight ramps on (0 at ATM, growing into both wings).
ABS_LM_EDGES = [0.0, 0.02, 0.04, 0.07, 0.12, 0.20, np.inf]
# Maturity bands (calendar days).
DTM_EDGES = [0, 45, 90, 180, 10000]


def _tests_dir(objective):
    return RESULTS / "calibrations" / objective / "calibration_tests"


def load_residuals(objective=OBJECTIVE):
    """Invert every repriced contract's Heston price to a Black IV and return the residual frame.

    Columns: date, days_to_maturity, w, spot_price, strike_price, lm (signed ln K/S), abs_lm,
    market_iv, model_iv, resid (= model_iv - market_iv). Rows whose inversion failed are dropped.
    """
    files = sorted(_tests_dir(objective).glob("*.csv"))
    if not files:
        raise SystemExit(f"no calibration_tests files under {_tests_dir(objective)}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        T = df["days_to_maturity"] / 365.0
        df["model_iv"] = [
            implied_vol(p, w, S, K, r, g, t)
            for p, w, S, K, r, g, t in zip(
                df["heston"], df["w"], df["spot_price"], df["strike_price"],
                df["risk_free_rate"], df["dividend_rate"], T,
            )
        ]
        df["date"] = f.stem[-10:]
        frames.append(df)
    a = pd.concat(frames, ignore_index=True)
    a = a.rename(columns={"volatility": "market_iv"})
    a["lm"] = np.log(a["strike_price"] / a["spot_price"])
    a["abs_lm"] = a["lm"].abs()
    a["resid"] = a["model_iv"] - a["market_iv"]
    return a.dropna(subset=["resid"])[
        ["date", "days_to_maturity", "w", "spot_price", "strike_price",
         "lm", "abs_lm", "market_iv", "model_iv", "resid"]
    ].reset_index(drop=True)


def _agg(df, by):
    """count / mean resid / RMSE / mean market IV per bucket (resid in vol points)."""
    g = df.groupby(by, observed=True)["resid"]
    out = g.agg(n="count", mean_resid="mean",
                rmse=lambda x: float(np.sqrt(np.mean(x ** 2))))
    out["mean_mkt_iv"] = df.groupby(by, observed=True)["market_iv"].mean()
    return out


def summarize(df):
    """Print the three stratified residual tables and return them as a dict of frames."""
    df = df.copy()
    df["lm_bucket"] = pd.cut(df["lm"], LM_EDGES)
    df["abs_bucket"] = pd.cut(df["abs_lm"], ABS_LM_EDGES)
    df["dtm_band"] = pd.cut(df["days_to_maturity"], DTM_EDGES)

    by_signed = _agg(df, "lm_bucket")
    by_abs = _agg(df, "abs_bucket")
    by_mat = _agg(df, ["dtm_band", "w"])

    pd.options.display.float_format = "{:+.4f}".format
    n_days = df["date"].nunique()
    print(f"\n=== residual = model_iv - market_iv (vol points); - = model UNDER ===")
    print(f"    {len(df)} contracts across {n_days} days, objective={OBJECTIVE}\n")

    print("--- by signed log-moneyness ln(K/S)  (puts<0, calls>0; exposes wing asymmetry) ---")
    print(by_signed.to_string())
    print("\n--- by |log-moneyness|  (the axis _wing_weight up-weights; GAIN sweep should move this) ---")
    print(by_abs.to_string())
    print("\n--- by maturity band x wing ---")
    print(by_mat.to_string())

    r = df["resid"]
    print(f"\noverall  mean {r.mean():+.4f}  median {r.median():+.4f}  rmse {np.sqrt((r**2).mean()):.4f} "
          f"vol pts  (n={len(r)})")
    return {"signed": by_signed, "abs": by_abs, "maturity": by_mat}


def main():
    df = load_residuals(OBJECTIVE)
    summarize(df)


if __name__ == "__main__":
    main()
