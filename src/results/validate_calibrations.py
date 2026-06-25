"""Read-only validation of Heston/Bates calibration outputs (PLAN.md Work items 1 & 3).

Reads the single ``results/<model>/calibrations/<objective>/calibrations.csv`` (ONE row per trading
day, Work item 3) and the matching per-day
``results/<model>/calibrations/<objective>/calibration_tests/*.csv`` (repriced surface contracts) and
emits, per trading day, a pass/fail report on:

  1. Fit quality   - relative repricing error (model vs trade_price) and, more rigorously,
                     the IV-space residual (model-implied vol vs market vol, in vol points).
  2. Economic      - two-tier flags (hard reject / suspicious) on (theta, kappa, eta, rho, v0)
     reasonability   against SPX-plausible ranges, plus the Feller condition. Under ``MODEL=bates``
                     the jump triple (lambda_, nu, delta) is also reported and flagged (suspicious
                     tier only, matching the engine's gate which exempts the jumps).
  3. Stability     - spread of the structural params **across days** (cross-day, post Work item 3);
                     these should cluster tightly for one underlying over a short window.

This module touches nothing in the pipeline. Run it before and after the deeper fixes to measure
improvement. The model/objective graded come from ``_results_config`` (MODEL/OBJECTIVE), the same
switches the other ``src/results/`` figure scripts read.

    python src/results/validate_calibrations.py

All graded rows are written to a single ``results/<model>/calibrations/<objective>/validation.csv``;
a per-day summary and a cross-day stability block are printed.
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

from _utils import implied_vol
from config import BOUNDS, BATES_BOUNDS, IV_RMSE_ACCEPT, OBJECTIVE_NAMES, MODEL_NAMES, calib_paths
from _results_config import MODEL, OBJECTIVE

if OBJECTIVE not in OBJECTIVE_NAMES:
    raise SystemExit(f"unknown objective {OBJECTIVE!r}; expected one of {OBJECTIVE_NAMES}")
if MODEL not in MODEL_NAMES:
    raise SystemExit(f"unknown model {MODEL!r}; expected one of {MODEL_NAMES}")

# The model/objective the helpers grade against are resolved per-call (default: the _results_config
# switches), so a notebook can pass a different pair without editing _results_config. The builders
# below derive the model-dependent state; module-level defaults keep direct importers of THRESHOLDS /
# STRUCTURAL / PRICE_COL working.


def _jump_params(model):
    """Bates appends the jump triple; grade/stability include them when present."""
    return ["lambda_", "nu", "delta"] if model == "bates" else []


def _thresholds_for(model):
    """Tunable acceptance/flag thresholds for `model`.

    "hard" = financially impossible -> reject; "susp" (suspicious) = possible but atypical for SPX at
    these tenors -> flag, don't reject. The hard-reject ranges that must agree with the engine (the
    box bounds and the IV gate) are pulled from config bounds / config.IV_RMSE_ACCEPT so the two
    cannot drift. The bounds differ by model: Heston uses config.BOUNDS, Bates config.BATES_BOUNDS
    (the five Heston ranges plus the jump triple), so a bates run is graded against the bounds it was
    actually fit under. The remaining knobs (peg/susp tolerances, kappa_lo) are validator-only
    judgement calls and stay local.
    """
    b = BATES_BOUNDS if model == "bates" else BOUNDS
    return dict(
        rho_peg=0.995, rho_lo=b["rho"][0], rho_hi=b["rho"][1],                # leverage => rho negative
        eta_lo=b["eta"][0], eta_hi=b["eta"][1], eta_susp=1.5,                 # SPX vol-of-vol ~0.3-1.2
        theta_lo=b["theta"][0], theta_hi=b["theta"][1], theta_susp=0.25,      # vol>50% suspicious
        v0_lo=b["v0"][0], v0_hi=b["v0"][1], v0_atm_tol=0.05,                  # sqrt(v0) ~ front ATM IV
        kappa_lo=0.0, kappa_hi=b["kappa"][1],                                 # mean-reversion speed
        iv_rmse_pts=IV_RMSE_ACCEPT,                                          # fit within ~2 vol points
        # --- Bates jump triple (gate-exempt in the engine, so suspicious tier only) ---
        jump_peg_frac=0.02,    # |param - bound| < this fraction of the bound's span => "pegged"
        lambda_collapse=0.05,  # lambda below this => jumps off, fit collapsed to pure Heston (a note)
    )


def _structural(model):
    return ["theta", "kappa", "eta", "rho", "v0"] + _jump_params(model)


# Module-level defaults for the _results_config model (keeps direct importers / the CLI working).
JUMP_PARAMS = _jump_params(MODEL)
THRESHOLDS = _thresholds_for(MODEL)
STRUCTURAL = _structural(MODEL)
PRICE_COL = MODEL


def _pegged(val, lo, hi, frac):
    """True if val sits within frac*span of either bound (weak-identification flag for the jumps)."""
    span = hi - lo
    return (val - lo) < frac * span or (hi - val) < frac * span



def day_metrics(test_df, price_col=None):
    """Day-level fit quality from the repriced surface contracts.

    Returns a dict: repricing error (median/p90), IV-space RMSE (vol points), the front-month
    nearest-ATM market IV (reference for the v0 check), and the contract count.
    """
    price_col = price_col or PRICE_COL
    df = test_df.copy()
    # The repriced model-price column is named after the model ('heston' or 'bates').
    df["rel_err"] = (df[price_col] - df["trade_price"]).abs() / df["trade_price"]
    T = df["days_to_maturity"] / 365.0
    df["model_iv"] = [
        implied_vol(p, w, S, K, r, g, t)
        for p, w, S, K, r, g, t in zip(
            df[price_col], df["w"], df["spot_price"], df["strike_price"],
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


def grade_day(row, atm_iv, iv_rmse, thresholds=None, jump_params=None):
    """Two-tier flags for one calibrated day. Returns a dict of booleans + summaries."""
    t = thresholds or THRESHOLDS
    jump_params = JUMP_PARAMS if jump_params is None else jump_params
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
    # Bates jump triple. The engine gate exempts the jumps (lambda~0 is a legitimate Heston collapse,
    # and nu/delta are unidentified when lambda~0), so these stay suspicious-tier and never hard-fail.
    # The two weakly-identified jump params (nu, delta) parking on a bound, or lambda pegged at its
    # cap, is the diagnostic worth surfacing; lambda~0 (jumps switched off) is reported as a note.
    note = {}
    if jump_params:
        lam, nu, delta = row["lambda_"], row["nu"], row["delta"]
        frac = t["jump_peg_frac"]
        susp["lambda_pegged_hi"] = (BATES_BOUNDS["lambda_"][1] - lam) < frac * (
            BATES_BOUNDS["lambda_"][1] - BATES_BOUNDS["lambda_"][0])
        susp["nu_pegged"] = _pegged(nu, *BATES_BOUNDS["nu"], frac=frac)
        susp["delta_pegged"] = _pegged(delta, *BATES_BOUNDS["delta"], frac=frac)
        note["jumps_collapsed"] = lam < t["lambda_collapse"]
    out = {**hard, **susp, **note}
    out["hard_fail"] = any(hard.values())
    out["n_suspicious"] = sum(susp.values())
    out["val_accepted"] = not out["hard_fail"]
    return out


def cross_day_stability(cal_all, structural=None):
    """Spread of structural params across days (post Work item 3 this replaces cross-bucket)."""
    structural = structural or STRUCTURAL
    out = {}
    for c in structural:
        x = cal_all[c].dropna()
        if x.empty:
            continue
        out[f"{c}_iqr"] = float(x.quantile(0.75) - x.quantile(0.25))
        out[f"{c}_min"] = float(x.min())
        out[f"{c}_max"] = float(x.max())
        if (x > 0).all():
            out[f"{c}_maxmin"] = float(x.max() / x.min())
    return out


def grade_row(row, test_path, price_col=None, thresholds=None, jump_params=None):
    """Grade one day's calibration row against its repriced surface contracts."""
    test = pd.read_csv(test_path)
    m = day_metrics(test, price_col=price_col)
    flags = grade_day(row, m["atm_iv"], m["iv_rmse"], thresholds=thresholds, jump_params=jump_params)
    record = {**row.to_dict(),
              "n_contracts": m["n_contracts"],
              "rel_err_median": m["rel_err_median"], "rel_err_p90": m["rel_err_p90"],
              "iv_rmse": m["iv_rmse"], "atm_iv": m["atm_iv"], **flags}
    return pd.DataFrame([record])


def print_day(report, jump_params=None):
    jump_params = JUMP_PARAMS if jump_params is None else jump_params
    r = report.iloc[0]
    print(f"\n=== {r['date']}  (1 day,  {int(r['n_contracts'])} contracts,  "
          f"{int(r.get('n_maturities', 0))} maturities x {int(r.get('n_strikes', 0))} strikes) ===")
    print(f"  params : theta={r['theta']:.4f} kappa={r['kappa']:.4f} eta={r['eta']:.4f} "
          f"rho={r['rho']:.4f} v0={r['v0']:.4f}  feller={r['feller']:.4f}")
    if jump_params:
        print(f"  jumps  : lambda={r['lambda_']:.4f} nu={r['nu']:.4f} delta={r['delta']:.4f}")
    print(f"  fit    : engine rmse {r.get('rmse', float('nan')):.4f}   "
          f"repricing rel-err median {r['rel_err_median']:.1%} p90 {r['rel_err_p90']:.1%}   "
          f"IV RMSE {r['iv_rmse']:.4f} vol pts")
    verdict = "PASS" if r["val_accepted"] else "HARD FAIL"
    print(f"  grade  : {verdict}   suspicious flags: {int(r['n_suspicious'])}")
    flag_labels = [
        ("rho_pegged", "rho pegged"), ("rho_wrong_sign", "rho>0"), ("eta_susp", "eta>1.5"),
        ("theta_susp", "theta>0.25"), ("v0_atm_mismatch", "sqrt(v0) far from ATM IV"),
        ("kappa_degenerate", "kappa<0.1 & theta>0.5"), ("feller_violated", "Feller<0"),
        ("fit_hard", "IV RMSE>2 vol pts"),
    ]
    if jump_params:
        flag_labels += [
            ("lambda_pegged_hi", "lambda pegged at cap"), ("nu_pegged", "nu pegged"),
            ("delta_pegged", "delta pegged"),
        ]
    flagged = [label for col, label in flag_labels if bool(r.get(col))]
    if flagged:
        print(f"           flags: {', '.join(flagged)}")
    if jump_params and bool(r.get("jumps_collapsed")):
        print(f"           note : jumps collapsed (lambda={r.get('lambda_', float('nan')):.4f} ~ 0; "
              f"fit is effectively pure Heston, nu/delta unidentified)")
    if bool(r.get("high_move")):
        print(f"           note : high intraday move ({r.get('spot_range_pct', float('nan')):.2%}) "
              f"- moneyness normalisation strained")


def print_cross_day(cal_all, stab, structural=None):
    structural = structural or STRUCTURAL
    n = len(cal_all)
    print(f"\n=== CROSS-DAY STABILITY ({n} days; structural params should cluster tightly) ===")
    for c in structural:
        iqr = stab.get(f"{c}_iqr")
        if iqr is None:
            continue
        mm = stab.get(f"{c}_maxmin")
        mm_s = f"  max/min {mm:7.2f}" if mm is not None else ""
        print(f"    {c:6s} IQR {iqr:8.4f}  range [{stab.get(f'{c}_min'):.4f}, "
              f"{stab.get(f'{c}_max'):.4f}]{mm_s}")


def main(model=None, objective=None, save=True):
    """Grade a run's calibrations.csv + calibration_tests/ and return the combined report frame.

    `model`/`objective` default to the `_results_config` switches when None, so a notebook can grade
    a different (model, objective) without editing `_results_config`. `save=False` skips writing
    validation.csv (the per-day + cross-day report is still printed and returned), so the notebook can
    inspect the result without touching disk. Returns the combined per-day DataFrame (empty if there
    is nothing to grade).
    """
    model = model or MODEL
    objective = objective or OBJECTIVE
    if objective not in OBJECTIVE_NAMES:
        raise SystemExit(f"unknown objective {objective!r}; expected one of {OBJECTIVE_NAMES}")
    if model not in MODEL_NAMES:
        raise SystemExit(f"unknown model {model!r}; expected one of {MODEL_NAMES}")

    # Built from the same config.calib_paths rule the calibrator uses, so the paths stay in lock-step:
    # results/<model>/calibrations/<objective>/ holds calibrations.csv, the per-day calibration_tests/
    # files, and the validation.csv written here. The repriced model-price column is named after model.
    calibrations_file, _rejections_file, tests = calib_paths(model, objective)
    out = calibrations_file.parent
    writepath = out / "validation.csv"
    price_col = model
    jump_params = _jump_params(model)
    thresholds = _thresholds_for(model)
    structural = _structural(model)

    if not calibrations_file.exists():
        print(f"No calibrations file at {calibrations_file}")
        return pd.DataFrame()
    # float_precision='round_trip': grade_row echoes every calibrations.csv column VERBATIM into the
    # tracked validation.csv (record = {**row.to_dict(), ...}). The default C parser is not
    # correctly-rounded (can land 1 ULP off), so reading the params back any other way would write
    # 1-ULP-shifted copies of theta/kappa/eta/... into validation.csv instead of the source values.
    cal = pd.read_csv(calibrations_file, float_precision='round_trip')   # one row per trading day
    all_reports = []
    for _, row in cal.iterrows():
        date = str(row["date"])
        test_path = tests / f"cboe_spx_calibration_tests_{date}.csv"
        if not test_path.exists():
            print(f"skipping {date}: no matching test file")
            continue
        report = grade_row(row, test_path, price_col=price_col,
                           thresholds=thresholds, jump_params=jump_params)
        print_day(report, jump_params=jump_params)
        all_reports.append(report)

    if not all_reports:
        return pd.DataFrame()

    combined = pd.concat(all_reports, ignore_index=True)
    n, acc = len(combined), int(combined["val_accepted"].sum())
    print_cross_day(combined, cross_day_stability(combined, structural=structural),
                    structural=structural)
    if save:
        out.mkdir(exist_ok=True)
        combined.to_csv(writepath, index=False)
        print(f"\n=== ALL DAYS: {acc}/{n} days pass all hard checks ({acc / n:.0%}); "
              f"table written to {writepath} ===")
    else:
        print(f"\n=== ALL DAYS: {acc}/{n} days pass all hard checks ({acc / n:.0%}); "
              f"not saved (save=False) ===")
    return combined


if __name__ == "__main__":
    main()
