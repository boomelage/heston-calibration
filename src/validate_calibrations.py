"""Read-only validation of Heston calibration outputs (PLAN.md Work item 1).

Reads the existing ``data/options/calibrations/*.csv`` (one row per spot bucket) and the
matching ``data/options/calibration_tests/*.csv`` (one repriced row per contract) and emits,
per spot bucket and per trading day, a pass/fail report on:

  1. Fit quality   - relative repricing error (heston vs trade_price) and, more rigorously,
                     the IV-space residual (model-implied vol vs market vol, in vol points).
  2. Economic      - two-tier flags (hard reject / suspicious) on (theta, kappa, eta, rho, v0)
     reasonability   against SPX-plausible ranges, plus the Feller condition.
  3. Stability     - spread of the structural params across spot buckets within a day; these
                     should cluster tightly (they do not yet -- that is the identification bug).

This module touches nothing in the pipeline. Run it before and after the deeper fixes
(Work items 2 and 3) to measure improvement.

    python src/validate_calibrations.py

Per-bucket tables are written to ``data/options/validation/validation_<date>.csv``; a per-day
summary is printed.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import QuantLib as ql

SRC = Path(__file__).parent.resolve()
OPTIONS = SRC.parent / "data" / "options"
CALIBRATIONS = OPTIONS / "calibrations"
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
    iv_rmse_pts=0.02,                              # accept buckets fitting within ~2 vol points
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


def fit_metrics(test_df):
    """Per spot-bucket fit quality from the repriced test rows.

    Returns a DataFrame indexed by spot_price with repricing error, IV-space RMSE, the
    front-month ATM market IV (reference for the v0 check), and the contract count.
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

    rows = []
    for s, g in df.groupby("spot_price"):
        tmin = g["days_to_maturity"].min()
        near = g[g["days_to_maturity"] == tmin]
        atm_iv = near.iloc[(near["strike_price"] - near["spot_price"]).abs().to_numpy().argmin()]["volatility"]
        resid = g["iv_resid"].dropna()
        rows.append(dict(
            spot_price=s,
            n_contracts=len(g),
            rel_err_median=g["rel_err"].median(),
            rel_err_p90=g["rel_err"].quantile(0.90),
            iv_rmse=float(np.sqrt((resid ** 2).mean())) if len(resid) else np.nan,
            iv_resid_n=len(resid),
            atm_iv=atm_iv,
        ))
    return pd.DataFrame(rows).set_index("spot_price")


def grade_bucket(row, atm_iv, iv_rmse):
    """Two-tier flags for one calibrated bucket. Returns a dict of booleans + summaries."""
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
    out["accepted"] = not out["hard_fail"]
    return out


def day_stability(cal):
    """Spread of structural params across this day's spot buckets (should be tight)."""
    out = {}
    for c in STRUCTURAL:
        x = cal[c].dropna()
        if x.empty:
            continue
        out[f"{c}_iqr"] = float(x.quantile(0.75) - x.quantile(0.25))
        out[f"{c}_min"] = float(x.min())
        out[f"{c}_max"] = float(x.max())
        if (x > 0).all():
            out[f"{c}_maxmin"] = float(x.max() / x.min())
    return out


def validate_file(cal_path, test_path, date):
    cal = pd.read_csv(cal_path)
    test = pd.read_csv(test_path)
    metrics = fit_metrics(test)

    records = []
    for _, row in cal.iterrows():
        s = row["spot_price"]
        m = metrics.loc[s] if s in metrics.index else pd.Series(dtype=float)
        atm_iv = m.get("atm_iv", np.nan)
        iv_rmse = m.get("iv_rmse", np.nan)
        flags = grade_bucket(row, atm_iv, iv_rmse)
        records.append({
            "date": date,
            **row.to_dict(),
            "n_contracts": m.get("n_contracts", np.nan),
            "rel_err_median": m.get("rel_err_median", np.nan),
            "rel_err_p90": m.get("rel_err_p90", np.nan),
            "iv_rmse": iv_rmse,
            "atm_iv": atm_iv,
            **flags,
        })
    report = pd.DataFrame(records)

    OUT.mkdir(exist_ok=True)
    report.to_csv(OUT / f"validation_{date}.csv", index=False)
    return report, day_stability(cal)


def print_summary(date, report, stab):
    n = len(report)
    acc = int(report["accepted"].sum())
    print(f"\n=== {date}  ({n} spot buckets) ===")
    print(f"  accepted (no hard fail) : {acc}/{n} ({acc / n:.0%})")
    print(f"  repricing rel-err       : median {report['rel_err_median'].median():.1%}"
          f"  p90 {report['rel_err_p90'].median():.1%}")
    print(f"  IV-space RMSE (vol pts) : median {report['iv_rmse'].median():.4f}"
          f"  worst {report['iv_rmse'].max():.4f}")
    print("  flag counts:")
    for col, label in [
        ("rho_pegged", "rho pegged (|rho|>0.995)"),
        ("feller_violated", "Feller violated (<0)"),
        ("eta_susp", "eta > 1.5"),
        ("theta_susp", "theta > 0.25"),
        ("rho_wrong_sign", "rho > 0 (wrong sign)"),
        ("v0_atm_mismatch", "sqrt(v0) far from ATM IV"),
        ("kappa_degenerate", "kappa<0.1 & theta>0.5"),
        ("fit_hard", "IV RMSE > 2 vol pts"),
    ]:
        print(f"    {label:32s}: {int(report[col].sum()):3d}/{n}")
    print("  cross-bucket stability (should be tight):")
    for c in STRUCTURAL:
        iqr = stab.get(f"{c}_iqr")
        mm = stab.get(f"{c}_maxmin")
        mm_s = f"  max/min {mm:8.1f}" if mm is not None else ""
        print(f"    {c:6s} IQR {iqr:8.4f}  range [{stab.get(f'{c}_min'):.4f}, {stab.get(f'{c}_max'):.4f}]{mm_s}")


def main():
    cal_files = sorted(CALIBRATIONS.glob("cboe_spx_calibrations_*.csv"))
    if not cal_files:
        print(f"No calibration files in {CALIBRATIONS}")
        return
    all_reports = []
    for cal_path in cal_files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", cal_path.name)
        date = m.group(1) if m else cal_path.stem
        test_path = TESTS / f"cboe_spx_calibration_tests_{date}.csv"
        if not test_path.exists():
            print(f"skipping {date}: no matching test file")
            continue
        report, stab = validate_file(cal_path, test_path, date)
        print_summary(date, report, stab)
        all_reports.append(report)

    if all_reports:
        combined = pd.concat(all_reports, ignore_index=True)
        n, acc = len(combined), int(combined["accepted"].sum())
        print(f"\n=== ALL DAYS: {acc}/{n} buckets accepted ({acc / n:.0%}); "
              f"per-day tables in {OUT} ===")


if __name__ == "__main__":
    main()
