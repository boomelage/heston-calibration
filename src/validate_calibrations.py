"""Read-only validation of Heston calibration outputs (PLAN.md Work items 1 & 3).

Reads the single ``data/calibrations.csv`` (ONE row per trading day, Work item 3) and the matching
per-day ``data/options/calibration_tests/*.csv`` (repriced surface contracts) and emits, per trading
day, a pass/fail report on:

  1. Fit quality   - relative repricing error (heston vs trade_price) and, more rigorously,
                     the IV-space residual (model-implied vol vs market vol, in vol points).
  2. Economic      - two-tier flags (hard reject / suspicious) on (theta, kappa, eta, rho, v0)
     reasonability   against SPX-plausible ranges, plus the Feller condition.
  3. Stability     - spread of the structural params **across days** (cross-day, post Work item 3);
                     these should cluster tightly for one underlying over a short window.

This module touches nothing in the pipeline. Run it before and after the deeper fixes to measure
improvement.

    python src/validate_calibrations.py

All graded rows are written to a single ``data/options/validation/validation.csv``; a per-day
summary and a cross-day stability block are printed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import QuantLib as ql

SRC = Path(__file__).parent.resolve()
DATA = SRC.parent / "data"
OPTIONS = DATA / "options"
CALIBRATIONS_FILE = DATA / "calibrations.csv"   # single one-row-per-day parameters file
TESTS = OPTIONS / "calibration_tests"
OUT = OPTIONS / "validation"

# Tunable acceptance/flag thresholds. "hard" = financially impossible -> reject;
# "susp" (suspicious) = possible but atypical for SPX at these tenors -> flag, don't reject.
THRESHOLDS = dict(
    rho_peg=0.995, rho_lo=-0.999, rho_hi=0.5,      # leverage effect => rho negative
    eta_lo=0.01, eta_hi=2.0, eta_susp=1.5,         # SPX vol-of-vol ~0.3-1.2
    theta_lo=1e-4, theta_hi=1.0, theta_susp=0.25,  # long-run variance (vol>50% suspicious)
    v0_lo=1e-4, v0_hi=1.0, v0_atm_tol=0.05,        # sqrt(v0) should ~ front-month ATM IV
    kappa_lo=0.0, kappa_hi=20.0,                   # mean-reversion speed
    iv_rmse_pts=0.02,                              # accept days fitting within ~2 vol points
)

STRUCTURAL = ["theta", "kappa", "eta", "rho", "v0"]


def implied_vol(price, w, S, K, r, g, T):
    """Invert a Black price to an implied vol (vol points), dividend-consistent via the forward."""
    if not np.isfinite(price) or price <= 0 or T <= 0:
        return np.nan
    F = S * np.exp((r - g) * T)
    disc = np.exp(-r * T)
    opt = ql.Option.Call if w == "call" else ql.Option.Put
    try:
        sd = ql.blackFormulaImpliedStdDev(opt, K, F, price, disc)
        return sd / np.sqrt(T)
    except RuntimeError:
        return np.nan


def day_metrics(test_df):
    """Day-level fit quality from the repriced surface contracts.

    Returns a dict: repricing error (median/p90), IV-space RMSE (vol points), the front-month
    nearest-ATM market IV (reference for the v0 check), and the contract count.
    """
    df = test_df.copy()
    df["rel_err"] = (df["heston"] - df["trade_price"]).abs() / df["trade_price"]
    T = df["days_to_maturity"] / 365.0
    df["model_iv"] = [
        implied_vol(p, w, S, K, r, g, t)
        for p, w, S, K, r, g, t in zip(
            df["heston"], df["w"], df["spot_price"], df["strike_price"],
            df["risk_free_rate"], df["dividend_rate"], T,
        )
    ]
    df["iv_resid"] = df["model_iv"] - df["volatility"]

    tmin = df["days_to_maturity"].min()
    near = df[df["days_to_maturity"] == tmin]
    atm_iv = near.iloc[(near["strike_price"] - near["spot_price"]).abs().to_numpy().argmin()]["volatility"]
    resid = df["iv_resid"].dropna()
    return dict(
        n_contracts=len(df),
        rel_err_median=df["rel_err"].median(),
        rel_err_p90=df["rel_err"].quantile(0.90),
        iv_rmse=float(np.sqrt((resid ** 2).mean())) if len(resid) else np.nan,
        iv_resid_n=len(resid),
        atm_iv=atm_iv,
    )


def grade_day(row, atm_iv, iv_rmse):
    """Two-tier flags for one calibrated day. Returns a dict of booleans + summaries."""
    t = THRESHOLDS
    theta, kappa, eta, rho, v0, feller = (
        row["theta"], row["kappa"], row["eta"], row["rho"], row["v0"], row["feller"],
    )
    hard = dict(
        rho_hard=not (t["rho_lo"] <= rho <= t["rho_hi"]),
        eta_hard=not (t["eta_lo"] <= eta <= t["eta_hi"]),
        theta_hard=not (t["theta_lo"] <= theta <= t["theta_hi"]),
        v0_hard=not (t["v0_lo"] <= v0 <= t["v0_hi"]),
        kappa_hard=not (t["kappa_lo"] < kappa <= t["kappa_hi"]),
        fit_hard=(np.isfinite(iv_rmse) and iv_rmse > t["iv_rmse_pts"]),
    )
    susp = dict(
        rho_pegged=abs(rho) > t["rho_peg"],
        rho_wrong_sign=rho > 0,
        eta_susp=eta > t["eta_susp"],
        theta_susp=theta > t["theta_susp"],
        v0_atm_mismatch=(np.isfinite(atm_iv) and abs(np.sqrt(max(v0, 0)) - atm_iv) > t["v0_atm_tol"]),
        kappa_degenerate=(kappa < 0.1 and theta > 0.5),
        feller_violated=feller < 0,
    )
    out = {**hard, **susp}
    out["hard_fail"] = any(hard.values())
    out["n_suspicious"] = sum(susp.values())
    out["val_accepted"] = not out["hard_fail"]
    return out


def cross_day_stability(cal_all):
    """Spread of structural params across days (post Work item 3 this replaces cross-bucket)."""
    out = {}
    for c in STRUCTURAL:
        x = cal_all[c].dropna()
        if x.empty:
            continue
        out[f"{c}_iqr"] = float(x.quantile(0.75) - x.quantile(0.25))
        out[f"{c}_min"] = float(x.min())
        out[f"{c}_max"] = float(x.max())
        if (x > 0).all():
            out[f"{c}_maxmin"] = float(x.max() / x.min())
    return out


def grade_row(row, test_path):
    """Grade one day's calibration row against its repriced surface contracts."""
    test = pd.read_csv(test_path)
    m = day_metrics(test)
    flags = grade_day(row, m["atm_iv"], m["iv_rmse"])
    record = {**row.to_dict(),
              "n_contracts": m["n_contracts"],
              "rel_err_median": m["rel_err_median"], "rel_err_p90": m["rel_err_p90"],
              "iv_rmse": m["iv_rmse"], "atm_iv": m["atm_iv"], **flags}
    return pd.DataFrame([record])


def print_day(report):
    r = report.iloc[0]
    print(f"\n=== {r['date']}  (1 day,  {int(r['n_contracts'])} contracts,  "
          f"{int(r.get('n_maturities', 0))} maturities x {int(r.get('n_strikes', 0))} strikes) ===")
    print(f"  params : theta={r['theta']:.4f} kappa={r['kappa']:.4f} eta={r['eta']:.4f} "
          f"rho={r['rho']:.4f} v0={r['v0']:.4f}  feller={r['feller']:.4f}")
    print(f"  fit    : engine rmse {r.get('rmse', float('nan')):.4f}   "
          f"repricing rel-err median {r['rel_err_median']:.1%} p90 {r['rel_err_p90']:.1%}   "
          f"IV RMSE {r['iv_rmse']:.4f} vol pts")
    verdict = "PASS" if r["val_accepted"] else "HARD FAIL"
    print(f"  grade  : {verdict}   suspicious flags: {int(r['n_suspicious'])}")
    flagged = [label for col, label in [
        ("rho_pegged", "rho pegged"), ("rho_wrong_sign", "rho>0"), ("eta_susp", "eta>1.5"),
        ("theta_susp", "theta>0.25"), ("v0_atm_mismatch", "sqrt(v0) far from ATM IV"),
        ("kappa_degenerate", "kappa<0.1 & theta>0.5"), ("feller_violated", "Feller<0"),
        ("fit_hard", "IV RMSE>2 vol pts"),
    ] if bool(r.get(col))]
    if flagged:
        print(f"           flags: {', '.join(flagged)}")
    if bool(r.get("high_move")):
        print(f"           note : high intraday move ({r.get('spot_range_pct', float('nan')):.2%}) "
              f"- moneyness normalisation strained")


def print_cross_day(cal_all, stab):
    n = len(cal_all)
    print(f"\n=== CROSS-DAY STABILITY ({n} days; structural params should cluster tightly) ===")
    for c in STRUCTURAL:
        iqr = stab.get(f"{c}_iqr")
        if iqr is None:
            continue
        mm = stab.get(f"{c}_maxmin")
        mm_s = f"  max/min {mm:7.2f}" if mm is not None else ""
        print(f"    {c:6s} IQR {iqr:8.4f}  range [{stab.get(f'{c}_min'):.4f}, "
              f"{stab.get(f'{c}_max'):.4f}]{mm_s}")


def main():
    if not CALIBRATIONS_FILE.exists():
        print(f"No calibrations file at {CALIBRATIONS_FILE}")
        return
    cal = pd.read_csv(CALIBRATIONS_FILE)   # one row per trading day
    all_reports = []
    for _, row in cal.iterrows():
        date = str(row["date"])
        test_path = TESTS / f"cboe_spx_calibration_tests_{date}.csv"
        if not test_path.exists():
            print(f"skipping {date}: no matching test file")
            continue
        report = grade_row(row, test_path)
        print_day(report)
        all_reports.append(report)

    if all_reports:
        combined = pd.concat(all_reports, ignore_index=True)
        n, acc = len(combined), int(combined["val_accepted"].sum())
        OUT.mkdir(exist_ok=True)
        combined.to_csv(OUT / "validation.csv", index=False)
        print_cross_day(combined, cross_day_stability(combined))
        print(f"\n=== ALL DAYS: {acc}/{n} days pass all hard checks ({acc / n:.0%}); "
              f"table written to {OUT / 'validation.csv'} ===")


if __name__ == "__main__":
    main()
