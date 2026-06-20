"""Compare the price- vs vol-objective Heston calibration runs.

Pulls the committed per-objective outputs straight from
``results/calibrations/{price,vol}/`` (calibrations.csv, rejections.csv,
validation.csv) and writes a side-by-side metrics table to
``results/tables/objective_comparison.csv``.

This is meant to be the single source of truth for the acceptance, fit-quality,
parameter-level, and pathology figures quoted in CLAUDE.md, PLAN.md, and
heston-calibration.tex. Re-run it after regenerating either objective and check
the prose against it. Paths resolve from __file__, so it runs from any cwd.

    python src/results/tables/objective_comparison.py

Output: one row per metric, columns ``metric, price, vol, delta_vol_minus_price``.
IV-space RMSE rows are reported in vol points (the stored fraction x100, the unit
the docs quote); repricing error rows are in percent. Counts stay integers.
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

# This script now lives under src/results/tables/, but reads calibrations from and writes its table
# into the repo-level results/ tree. RESULTS routes both to repo/results/.
HERE = Path(__file__).parent.resolve()                 # src/results/tables
REPO = HERE.parents[2]                                  # repo root (tables->results->src->repo)
RESULTS = REPO / "results"
# Compares this MODEL's price vs vol runs; both objectives must exist on disk under
# results/<model>/calibrations/{price,vol}/ (otherwise objective_metrics raises FileNotFoundError).
MODEL = "heston"                                        # 'heston' or 'bates'
TABLES = RESULTS / MODEL / "tables"                     # table dir at repo/results/<model>/tables
TABLES.mkdir(parents=True, exist_ok=True)
CALIB = RESULTS / MODEL / "calibrations"
OBJECTIVES = ("price", "vol")
OUT = TABLES / "objective_comparison.csv"

# Rejection reason categories the orchestrator can emit (calibrator_prototype._skip_day).
REASONS = ["pegged", "iv_miss", "no_trades", "thin", "no_rate", "no_fit"]
# Accepted-set structural parameters; Bates appends the jump triple.
PARAMS = ["theta", "kappa", "eta", "rho", "v0"] + (["lambda_", "nu", "delta"] if MODEL == "bates" else [])
# Validator soft-flag columns (validate_calibrations.grade_day).
SOFT_FLAGS = ["feller_violated", "eta_susp", "rho_pegged",
              "v0_atm_mismatch", "kappa_degenerate", "rho_wrong_sign"]

ETA_SUSP = 1.5      # eta above this is "suspicious" (matches validator threshold)
RHO_PEG = -0.99     # rho at/below this is effectively pegged to its -0.999 floor
KAPPA_CEIL = 19.98  # kappa at/above this sits on its 20.0 box ceiling
RMSE_BLOWUP = 0.2   # relative-price RMSE above this flags a price-space miss


def _read(obj, name):
    p = CALIB / obj / name
    return pd.read_csv(p) if p.exists() else None


def objective_metrics(obj):
    """Return (ordered metric->value dict, set of accepted date strings) for one objective."""
    cal = _read(obj, "calibrations.csv")
    rej = _read(obj, "rejections.csv")
    val = _read(obj, "validation.csv")
    if cal is None:
        raise FileNotFoundError(f"no calibrations.csv for objective {obj!r} under {CALIB / obj}")

    n_acc = len(cal)
    n_rej = 0 if rej is None else len(rej)
    n_att = n_acc + n_rej
    counts = rej["reason"].value_counts().to_dict() if rej is not None else {}
    m = OrderedDict()

    # ---- acceptance ----
    m["attempted_days"] = n_att
    m["accepted_days"] = n_acc
    m["accept_rate"] = n_acc / n_att if n_att else np.nan
    m["rejected_days"] = n_rej
    for r in REASONS:
        m[f"rej_{r}"] = int(counts.get(r, 0))
    m["pegged_share_of_rejections"] = counts.get("pegged", 0) / n_rej if n_rej else np.nan
    m["pegged_share_of_attempted"] = counts.get("pegged", 0) / n_att if n_att else np.nan

    # ---- fit quality: IV-space RMSE over the accepted set, in vol points (x100) ----
    iv = cal["iv_rmse"]
    m["iv_rmse_median_volpts"] = iv.median() * 100
    m["iv_rmse_mean_volpts"] = iv.mean() * 100
    m["iv_rmse_p90_volpts"] = iv.quantile(0.90) * 100
    m["iv_rmse_max_volpts"] = iv.max() * 100

    # ---- relative-price RMSE over the accepted set (the diagnostic that blows up under vol) ----
    m["price_rmse_max"] = cal["rmse"].max()
    m[f"price_rmse_gt_{RMSE_BLOWUP}_count"] = int((cal["rmse"] > RMSE_BLOWUP).sum())

    # ---- parameter levels over the accepted set ----
    for p in PARAMS:
        m[f"{p}_median"] = cal[p].median()
        m[f"{p}_mean"] = cal[p].mean()
        m[f"{p}_p10"] = cal[p].quantile(0.10)
        m[f"{p}_p90"] = cal[p].quantile(0.90)

    # ---- economic pathologies over the accepted set ----
    m["feller_median"] = cal["feller"].median()
    m["feller_lt0_count"] = int((cal["feller"] < 0).sum())
    m["feller_lt0_share"] = float((cal["feller"] < 0).mean())
    m["eta_gt_1p5_count"] = int((cal["eta"] > ETA_SUSP).sum())
    m["eta_gt_1p5_share"] = float((cal["eta"] > ETA_SUSP).mean())
    m["eta_max"] = cal["eta"].max()
    m["rho_pegged_lt_-0p99_count"] = int((cal["rho"] < RHO_PEG).sum())
    m["kappa_at_ceiling_count"] = int((cal["kappa"] >= KAPPA_CEIL).sum())

    # ---- repricing + validator (validation.csv; may be absent on a fresh clone) ----
    if val is not None and len(val):
        m["val_graded_days"] = len(val)
        m["val_pass_count"] = int(val["val_accepted"].sum())
        m["val_pass_rate"] = float(val["val_accepted"].mean())
        # per-day median relative repricing error, summarised across days (percent)
        m["reprice_rel_err_median_pct"] = val["rel_err_median"].median() * 100
        m["reprice_rel_err_p90_pct"] = val["rel_err_median"].quantile(0.90) * 100
        for f in SOFT_FLAGS:
            if f in val.columns:
                m[f"val_flag_{f}"] = int(val[f].sum())

    return m, set(cal["date"].astype(str))


def _delta(pv, vv):
    """vol - price when both are finite numbers, else NaN."""
    if isinstance(pv, (int, float)) and isinstance(vv, (int, float)):
        if np.isfinite(pv) and np.isfinite(vv):
            return vv - pv
    return np.nan


def main():
    metrics, dates = {}, {}
    for obj in OBJECTIVES:
        metrics[obj], dates[obj] = objective_metrics(obj)

    names = list(dict.fromkeys(k for obj in OBJECTIVES for k in metrics[obj]))
    rows = []
    for name in names:
        pv = metrics["price"].get(name, np.nan)
        vv = metrics["vol"].get(name, np.nan)
        rows.append({"metric": name, "price": pv, "vol": vv,
                     "delta_vol_minus_price": _delta(pv, vv)})

    # Cross-objective accepted-date overlap (per-column: the count specific to each objective).
    both = dates["price"] & dates["vol"]
    rows.append({"metric": "accepted_dates_shared", "price": len(both), "vol": len(both),
                 "delta_vol_minus_price": 0})
    rows.append({"metric": "accepted_dates_objective_only",
                 "price": len(dates["price"] - dates["vol"]),
                 "vol": len(dates["vol"] - dates["price"]),
                 "delta_vol_minus_price": np.nan})

    # row dicts are built in column order (metric, price, vol, delta_vol_minus_price)
    out = pd.DataFrame(rows)

    def _fmt(x):
        # render whole numbers (counts/deltas) as ints, keep rates/stats as 6-dp floats
        if isinstance(x, float) and np.isfinite(x):
            return int(x) if x == int(x) else round(x, 6)
        return x

    for c in ("price", "vol", "delta_vol_minus_price"):
        # build an explicit object array so whole numbers survive as ints (pandas would
        # otherwise re-infer the mixed column back to float64 and write "3215.0")
        out[c] = np.array([_fmt(x) for x in out[c].tolist()], dtype=object)
    out.to_csv(OUT, index=False)

    pd.options.display.float_format = "{:.6g}".format
    print(out.to_string(index=False))
    print(f"\nwrote {len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
