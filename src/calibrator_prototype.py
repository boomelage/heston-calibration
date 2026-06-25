"""Heston calibration orchestration (PLAN.md Work item 3: one calibration per trading day).

Replaces the old per-0.5-spot-bucket scheme (`calibrateby_spot`) with `calibrate_by_day`: a
single calibration over one rich, moneyness-normalised, multi-maturity surface per day.

Why per day. ~53 independent 5-parameter fits/day, each on a thin per-bucket slice, left Heston
under-determined (only the product kappa*theta identified -> theta-huge/kappa-tiny degeneracy, and
parameters that swung across adjacent spot buckets). Pooling the day's trades gives the fit the
cross-maturity, cross-strike information it needs.

Handling intraday spot movement. A Heston fit has a single spot S, but the underlying drifts through
the session (~1% on 2024-10-07). Each trade keeps its moneyness m = K / S_row (S_row = the
underlying at trade time) but is re-struck to K* = m * S_ref against one volume-weighted reference
spot S_ref, then snapped to the SPX 5-point strike grid so trades at different intraday spots share
clean surface columns. This re-centres the day under the standard sticky-moneyness assumption
(IV ~stationary in moneyness over a session). It strains on large-move days, which are flagged
(`high_move`) but still written. Heston params are spot-independent, so the repricing diagnostics
below use each trade's *original* spot/strike, not the normalised K*.

Output: the parameters accumulate into a SINGLE `results/calibrations/<objective>/calibrations.csv`
(one row per trading day, keyed by date, recording S_ref, r, g, the five params, feller, rmse,
coverage counts and the intraday spot range). It is APPENDED INCREMENTALLY as each day completes
(streamed back via joblib's `return_as="generator_unordered"` and written from the single main
process, so the file is readable mid-run in worker-completion order), then REWRITTEN SORTED BY DATE
once the run finishes -- so the finished artefact still holds the accepted days, date-sorted, exactly
as before. Ctrl-C (KeyboardInterrupt) does not abort: it stops the loop and still writes the
authoritative calibrations.csv + config_spec.json from the days completed so far. The end-of-run
writes wait-and-retry if the target file is locked (e.g. open in Excel), prompting for Enter rather
than crashing.

RESUME. A run AUTOMATICALLY continues a previous one when EITHER calibrations.csv OR rejections.csv
already exists. Both files are appended incrementally as days complete, so an interrupted run may have
written only accepts, only rejects, or both; either one present marks a prior run to continue. The
days they cover are skipped (matched by DATE -- the completed days are not a contiguous chronological
prefix because workers finish out of order), and this run's new rows are MERGED with the old rows
(dedup by date, this run wins) before the date-sorted rewrite, so old rows are never clobbered. A
resume first aborts if the live config differs from the snapshot in config_spec.json (resuming would
mix incompatible fits), or if that snapshot is missing. The snapshot is written once at the START of a
run (before any incremental output), so it is always present beside a partial calibrations.csv/
rejections.csv. To start fresh, delete the results/<model>/calibrations/<objective>/ files manually.
The bulky per-day repricing diagnostics stay one-file-per-day under
`results/calibrations/<objective>/calibration_tests/`. A day is
written to calibration_tests exactly when it contributes a row, so the two outputs always describe
the same accepted set; a rejected or too-thin day contributes no row and clears its tests file.

Rejection audit: every attempted-but-rejected day contributes one row to a SEPARATE
`results/calibrations/<objective>/rejections.csv` (keyed by date) recording why it was dropped -- a small 
`reason` category (no_trades/no_rate/thin/pegged/iv_miss/no_fit), the human `detail`, the `iv_rmse`
where one exists, and the surface coverage where known. `calibrations.csv` stays accepted-only (it mirrors
calibration_tests/ one-to-one); `rejections.csv` is the complement, so accepted + rejected together
cover every attempted day and the pegged-vs-thin split is auditable. Like calibrations.csv,
rejections.csv is APPENDED INCREMENTALLY as each day is dropped (readable mid-run in worker-completion
order) and then REWRITTEN SORTED BY DATE at the end (merged with any resumed rows); an empty set
removes its file.
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
pd.options.display.float_format = '{:.5f}'.format

SRC = Path(__file__).parent.resolve()
DATA = SRC.parent / "data"
RESULTS = SRC.parent / "results"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pricing.vanilla_pricer import vanilla_pricer
vanp = vanilla_pricer()
from _utils import write_config_spec, _file_date
from prepare_surface import prepare_surface, select_surface, SkipDay
from calibrate_heston import calibrate_heston
from calibrate_bates import calibrate_bates
import config
from config import (
    MIN_MATS, MIN_STRIKES, MIN_CELLS,
    IV_RMSE_ACCEPT, OBJECTIVE_NAMES, MODEL_NAMES, calib_paths, spec_path,
    DEFAULT_MODEL, DEFAULT_OBJECTIVE
)

# Per-model engine, the extra Bates parameter columns, and the repriced model-price column name.
# Heston keeps its 5 params and the `heston` price column; Bates appends (lambda_, nu, delta) and writes
# a `bates` column priced by the Bates wrapper. Everything else in the day routine is model-agnostic.
_ENGINES = {"heston": calibrate_heston, "bates": calibrate_bates}
_EXTRA_PARAMS = {"heston": [], "bates": ["lambda_", "nu", "delta"]}
_PRICE_COL = {"heston": "heston", "bates": "bates"}

if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from get_rg import rg # pyright: ignore[reportMissingImports]

# `rg` is sorted descending (newest first); `asof` needs an ascending index. Sort once here
# instead of per file. `asof(date)` returns the last value on/before `date` regardless of `rg`'s
# ordering, or NaN if `date` precedes all rates.
rg_asc = rg.sort_index()

# Surface coverage / selection knobs and engine bounds/gate live in config.py (single source of
# truth, tuned by PLAN.md Phase 3): the surface-selection knobs MAX_NT/MAX_NK/MIN_MATS/MIN_STRIKES/
# MIN_CELLS and the gate IV_RMSE_ACCEPT are imported above. The filter/normalisation knobs
# (MIN_DTM/MAX_DTM/MAX_MOVE_PCT/STRIKE_GRID) moved with prepare_surface into prepare_surface.py.


def _objective_paths(model, objective):
    """Resolve the (calibrations.csv, rejections.csv, tests-dir) outputs for a (model, objective) pair.

    Thin wrapper over config.calib_paths, the single source of truth for output routing. Uniform
    layout results/<model>/calibrations/<objective>/, holding calibrations.csv, rejections.csv and the
    per-day calibration_tests/ files (basename `cboe_spx_calibration_tests_<date>.csv`).
    validate_calibrations.py rebuilds the same directory from the same rule, so the two stay in lock-step.
    """
    return calib_paths(model, objective)


def _write_blocking(action, target):
    """Run write `action`; if `target` is locked (PermissionError), wait for the user to free it and
    retry. Loops until it succeeds. KeyboardInterrupt still propagates so the user can abort the wait."""
    while True:
        try:
            return action()
        except PermissionError:
            input(f"\n{target} is locked (close it in any program holding it open, e.g. Excel), "
                  f"then press Enter to retry... ")


def _append_row(row, target, header_written):
    """Best-effort mid-run append of one completed day's row to `target` (calibrations.csv or
    rejections.csv). Mirrors the end-of-run columns, so a header written from the first row stays
    valid for the rest of the run. A momentary file lock (PermissionError, e.g. open in Excel) is
    warned and skipped -- the row is still in the in-memory list, so the end-of-run sorted rewrite
    includes it -- rather than stalling the worker loop. Returns the updated header_written flag."""
    try:
        pd.DataFrame([row]).set_index('date').to_csv(target, mode='a', header=not header_written)
        return True
    except PermissionError:
        print(f"WARNING: {target} is locked; skipping mid-run flush (row kept, written at end)")
        return header_written


def _skip_day(test_path, reason, detail, iv_rmse=np.nan,
              n_maturities=np.nan, n_strikes=np.nan, n_cells=np.nan):
    """Drop a day: clear any stale tests file and return a rejection row (one per rejected day).

    The single calibrations.csv is rebuilt from the accepted rows each run, so a dropped day simply
    contributes no row there -- and clearing the matching tests file keeps calibrations.csv and
    calibration_tests/ describing the same accepted set (no desync). Separately, the returned row is
    collected into data/rejections.csv so the rejection cause is auditable: `reason` is a small
    category (no_trades/no_rate/thin/pegged/iv_miss/no_fit), `detail` the human string, with
    `iv_rmse`/coverage filled where that stage reached them (NaN otherwise)."""
    date = test_path[-14:-4]
    if os.path.exists(test_path):
        os.remove(test_path)
        note = "cleared stale tests file"
    else:
        note = "nothing written\n"
    print(f"{date}: {detail}; {note}")
    return {'date': date, 'reason': reason, 'detail': detail, 'iv_rmse': iv_rmse,
            'n_maturities': n_maturities, 'n_strikes': n_strikes, 'n_cells': n_cells}


def calibrate_by_day(filepath, OBJECTIVE, MODEL):
    # Per-day tests file: the directory depends on (MODEL, OBJECTIVE) (results/<model>/calibrations/
    # <objective>/calibration_tests/, or the legacy results/calibrations/<objective>/ for heston);
    # validate_calibrations.py rebuilds the identical name from the same rule. Derive the date from the
    # trailing _<date> token of the raw trades filename (the date is always the last underscore-separated
    # field before .csv), independent of the file's prefix.
    tests_dir = _objective_paths(MODEL, OBJECTIVE)[2]
    filename = os.path.basename(filepath)
    date_str = filename[filename.rfind('_')+1:filename.rfind('.csv')]
    test_path = str(tests_dir / f"cboe_spx_calibration_tests_{date_str}.csv")
    # Read the raw CBOE trades file and clean it in-memory to the OTM snapshot the surface needs
    # (column subset/rename, C/P -> call/put, calendar DTM, OTM-only) via prepare_surface._prepare_options,
    # then build the day's moneyness-normalised trades (IV/DTM filter, S_ref, Kstar) via prepare_surface.
    # prepare_surface raises SkipDay when nothing survives the filter; convert it to a rejection row here
    # (_skip_day owns test_path and the rejections.csv schema). Rate lookup stays below.
    df = pd.read_csv(filepath)
    try:
        df, date, S_ref, spot_min, spot_max, spot_range_pct, high_move = prepare_surface(df)
    except SkipDay as e:
        return _skip_day(test_path, e.reason, e.detail, **e.coverage)
    r = rg_asc['risk_free_rate'].asof(date)
    g = rg_asc['dividend_rate'].asof(date)
    if pd.isna(r) or pd.isna(g):
        return _skip_day(test_path, "no_rate", f"no rate on/before {pd.Timestamp(date).date()}")

    try:
        sel, surf = select_surface(df)
    except SkipDay as e:
        return _skip_day(test_path, e.reason, e.detail, **e.coverage)
    n_strikes, n_mats = surf.shape
    n_cells = int(surf.count().sum())
    if n_mats < MIN_MATS or n_strikes < MIN_STRIKES or n_cells < MIN_CELLS:
        return _skip_day(
            test_path, "thin",
            f"thin surface ({n_strikes} strikes x {n_mats} maturities, {n_cells} cells)",
            n_maturities=n_mats, n_strikes=n_strikes, n_cells=n_cells,
        )

    res = _ENGINES[MODEL](surf, S_ref, r, g, objective=OBJECTIVE)   # ONE calibration for the whole day (hardened engine)
    print(f"{pd.Timestamp(date).date()}  S_ref={S_ref:.1f}  cells={n_cells}  "
          f"iv_rmse={res['iv_rmse']}  price_rmse={res['rmse']}  accepted={res['accepted']}")

    if not res['accepted']:
        # Distinguish the rejection causes. A fit that passes the IV-space gate but is still rejected
        # is boundary-pegged (a param hit a bound -> a non-fit) -- the remaining lever (kappa/rho
        # handling), not an IV-fit-quality problem. iv_rmse is None only when every restart failed.
        iv = res['iv_rmse']
        if iv is not None and iv <= IV_RMSE_ACCEPT:
            reason, detail = "pegged", "calibration rejected (boundary-pegged)"
        elif iv is None:
            reason, detail = "no_fit", "calibration rejected (no finite fit)"
        else:
            reason, detail = "iv_miss", f"calibration rejected (iv_rmse={iv:.4f} > {IV_RMSE_ACCEPT})"
        return _skip_day(test_path, reason, detail,
                         iv_rmse=(iv if iv is not None else np.nan),
                         n_maturities=n_mats, n_strikes=n_strikes, n_cells=n_cells)

    # ---- one calibration row, keyed by date ----
    # Heston's 5 params, plus the Bates jump triple when MODEL=='bates' (appended via _EXTRA_PARAMS).
    params = ['theta', 'kappa', 'rho', 'eta', 'v0'] + _EXTRA_PARAMS[MODEL]
    row = {
        'date': pd.Timestamp(date).date(),
        'spot_price': round(S_ref, 4),
        'risk_free_rate': r, 'dividend_rate': g,
        **{k: res[k] for k in params}, 'feller': res['feller'],
        'iv_rmse': res['iv_rmse'], 'rmse': res['rmse'],
        'n_helpers': res['n_helpers'], 'accepted': res['accepted'],
        'n_maturities': n_mats, 'n_strikes': n_strikes, 'contracts_count': n_cells,
        'total_volume': int(df['trade_size'].sum()),
        'spot_min': spot_min, 'spot_max': spot_max, 'spot_range_pct': spot_range_pct,
        'high_move': high_move,
        'calculation_date': sel['quote_datetime'].max(),
    }
    if res['accepted']:
        for p in (params + ['feller']):
            print(p,res[p],sep=f": {(6-len(p))*' '}")
        print()

    # ---- reprice the surface contracts under the fitted params ----
    # One representative trade per surface cell (the highest-volume, matching the pivot's aggfunc),
    # repriced at its ORIGINAL spot/strike:
    # Heston params are spot-independent, so the honest diagnostic prices at real trade conditions,
    # not the normalised K*/S_ref. The tests file thus mirrors the calibrated surface one-to-one.
    repriced = (sel.drop_duplicates(subset=['Kstar', 'days_to_maturity'], keep='last')
                   .reset_index(drop=True))
    for k in params:
        repriced[k] = res[k]
    repriced['risk_free_rate'] = r
    repriced['dividend_rate'] = g
    repriced = repriced.rename(columns={'trade_iv': 'volatility'})
    try:
        repriced['black_scholes'] = vanp.df_numpy_black_scholes(repriced)
    except Exception:
        repriced['black_scholes'] = np.nan
    # Model price column: `heston` (df_heston_price) or `bates` (df_bates_price, which reads the
    # lambda_/nu/delta columns copied in above). The tests file mirrors the calibrated surface either way.
    price_col = _PRICE_COL[MODEL]
    price_fn = vanp.df_bates_price if MODEL == 'bates' else vanp.df_heston_price
    try:
        repriced[price_col] = price_fn(repriced)
    except Exception:
        repriced[price_col] = np.nan

    tests_dir.mkdir(parents=True, exist_ok=True)
    repriced.dropna(subset=[price_col]).to_csv(test_path, index=False)
    return row


def main():
    parser = argparse.ArgumentParser(description="Attempt per-day calibration of Heston/Bates paramaters off option trades data")
    parser.add_argument("--OBJECTIVE", type=str, default=DEFAULT_OBJECTIVE, choices=list(OBJECTIVE_NAMES),
                        help="Decide whether to minimize residuals of `price` or `vol`")
    parser.add_argument("--MODEL", type=str, default=DEFAULT_MODEL, choices=list(MODEL_NAMES),
                        help="Model to calibrate: `heston` (5 params) or `bates` (Heston + jumps, 8 params)")
    parser.add_argument("--LIMIT", type=int, default=0,
                        help="If >0, calibrate only the LIMIT most recent trading days (by date). 0 = all.")
    parser.add_argument("--MAX_JOBS", type=int, default=max(1, os.cpu_count() // 4),
                        help="Number of threads to use at one (one day's calibration per thread) (1//4 of available threads by default)")
    args = parser.parse_args()

    CALIBRATIONS_FILE, REJECTIONS_FILE, TESTS = _objective_paths(args.MODEL, args.OBJECTIVE)
    TESTS.mkdir(parents=True, exist_ok=True)
    SPEC_FILE = spec_path(args.MODEL, args.OBJECTIVE)

    # ---- Resume support ----
    # When EITHER calibrations.csv OR rejections.csv already exists this run CONTINUES the previous one.
    # Both files are appended incrementally as days complete, so an interrupted run may have produced
    # only accepts, only rejects, or both -- any one present marks a prior run to continue. The days
    # they already cover are SKIPPED (matched by date), and this run's new rows are MERGED into the
    # existing files at the end so old rows are never clobbered. Resume is automatic (no flag).
    existing_accepted = pd.DataFrame()
    existing_rejected = pd.DataFrame()
    processed_dates = set()
    if CALIBRATIONS_FILE.exists() or REJECTIONS_FILE.exists():
        # A resumed run must use the SAME config as the run it continues, or the outputs would mix
        # incompatible fits. Compare the live config to the snapshot the previous run wrote and abort on
        # any mismatch. Both sides are JSON-normalised first (config.as_dict() keeps tuples as tuples,
        # but they serialise to JSON arrays) so the comparison is value-equal, not type-sensitive. The
        # `_run` block (timestamp/commit/tally) is run metadata, not config, so it is dropped. The
        # snapshot is always written at run start (below), so a partial run always has one to verify.
        if not SPEC_FILE.exists():
            raise RuntimeError(
                f"Resuming (partial output exists in {CALIBRATIONS_FILE.parent}) but no {SPEC_FILE} to "
                f"verify the config against. Clean {CALIBRATIONS_FILE.parent} manually to start fresh.")
        saved = json.loads(SPEC_FILE.read_text())
        saved.pop('_run', None)
        current = json.loads(json.dumps(config.as_dict()))
        if saved != current:
            changed = sorted(
                k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
            raise RuntimeError(
                f"Config has changed since the run being resumed (differing keys: {changed}; see "
                f"{SPEC_FILE}). Resuming would mix incompatible calibrations. Revert config.py to "
                f"match, or clean {CALIBRATIONS_FILE.parent} to start fresh.")
        # float_precision='round_trip' makes read-back bit-exact: the default fast C parser is not
        # correctly-rounded (can land 1 ULP off), so re-serializing resumed rows would otherwise churn
        # their shortest-repr (e.g. 0.020857999999999998 -> 0.0208579999999999) on every resume. Read
        # only the file(s) that exist; a missing one stays the empty frame above. Guard `.empty` before
        # indexing 'date' (a header-only file is empty but still carries the column).
        if CALIBRATIONS_FILE.exists():
            existing_accepted = pd.read_csv(CALIBRATIONS_FILE, float_precision='round_trip')
        if REJECTIONS_FILE.exists():
            existing_rejected = pd.read_csv(REJECTIONS_FILE, float_precision='round_trip')
        if not existing_accepted.empty:
            processed_dates |= set(existing_accepted['date'].astype(str))
        if not existing_rejected.empty:
            processed_dates |= set(existing_rejected['date'].astype(str))
        print(f"resuming: {len(existing_accepted)} accepted + {len(existing_rejected)} rejected "
              f"day(s) already done; skipping those dates")

    # Write the run-spec snapshot BEFORE any incremental output, so an interrupted run (even a hard
    # kill) always leaves a config_spec.json beside its partial calibrations.csv/rejections.csv for the
    # resume config-check above. On resume it already exists and was validated (skip); it is refreshed
    # with the final tally at end, and removed there iff the run produced no output at all.
    if not SPEC_FILE.exists():
        write_config_spec(args.MODEL, args.OBJECTIVE, args.LIMIT,
                          len(existing_accepted), len(existing_rejected))

    TRADES = Path(__file__).parent.parent / "data" / "options" / "raw"


    files = [os.path.join(TRADES, f) for f in os.listdir(TRADES) if f.endswith('.csv')]
    # Sort chronologically by the date token (robust to mixed filename prefixes), so --LIMIT selects the
    # most recent trading days. For a full run the order is immaterial.
    files = sorted(files, key=_file_date)
    # Skip already-done days by DATE membership, not by count: workers finish out of order
    # (return_as="generator_unordered"), so an interrupted run's completed days are NOT a contiguous
    # chronological prefix -- a positional files[N:] slice would re-run some days and skip others.
    if processed_dates:
        files = [f for f in files if _file_date(f) not in processed_dates]
    if args.LIMIT and args.LIMIT > 0:
        files = files[-args.LIMIT:]
    files = pd.Series(files).reset_index(drop=True)

    # joblib's default loky backend spawns processes; on Windows the children re-import this module,
    # so the driver MUST live behind `if __name__ == "__main__"` (via main()) -- otherwise each worker
    # re-runs the Parallel call below and recursively spawns process pools.
    from joblib import Parallel, delayed
    
    accepted, rejected = [], []
    # Both calibrations.csv and rejections.csv are appended incrementally as days complete, so each is
    # readable mid-run (in worker-completion order). On a fresh run the first flush for each file
    # creates it with a header; on a resume the file already exists with old rows + header, so suppress
    # the header and let new rows append cleanly beneath them (the end-of-run merge rewrites both sorted
    # by date anyway). A file that did not exist at resume (only accepts, or only rejects, last run)
    # gets its header from this run's first matching row.
    cal_header_written = CALIBRATIONS_FILE.exists()
    rej_header_written = REJECTIONS_FILE.exists()
    # Ctrl-C stops the loop but does NOT abort: we fall through to the end-of-run block and still
    # write the authoritative calibrations.csv + rejections.csv + config_spec.json from the days
    # completed so far. Columns are identical across a (MODEL, OBJECTIVE) run, so a header written from
    # the first row stays valid for the rest; the flush is best-effort and lock-tolerant (_append_row).
    try:
        for r in Parallel(n_jobs=args.MAX_JOBS, return_as="generator_unordered")(
                delayed(calibrate_by_day)(f, args.OBJECTIVE, args.MODEL) for f in files):
            if r is None:
                continue
            if 'reason' in r:
                rejected.append(r)
                rej_header_written = _append_row(r, REJECTIONS_FILE, rej_header_written)
            else:
                accepted.append(r)
                cal_header_written = _append_row(r, CALIBRATIONS_FILE, cal_header_written)
    except KeyboardInterrupt:
        print(f"\nKeyboardInterrupt: stopping after {len(accepted)} accepted / {len(rejected)} "
              f"rejected day(s); finishing writes...")

    # Merge this run's rows with whatever the resumed files already held (empty frames on a fresh run),
    # dedupe by date (this run's row wins if a date somehow recurs), and rewrite sorted by date. On a
    # fresh run this reduces to the old date-sorted write; on a resume it preserves the old rows that
    # are no longer in the in-loop `accepted`/`rejected` lists. The writes block-and-retry on a file
    # lock (see _write_blocking) instead of crashing.
    def _merge(existing, new_rows):
        frames = [df for df in (existing, pd.DataFrame(new_rows)) if not df.empty]
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out['date'] = out['date'].astype(str)
        return out.drop_duplicates(subset='date', keep='last').set_index('date').sort_index()

    merged_accepted = _merge(existing_accepted, accepted)
    merged_rejected = _merge(existing_rejected, rejected)

    have_output = not merged_accepted.empty or not merged_rejected.empty

    if not merged_accepted.empty:
        _write_blocking(lambda: merged_accepted.to_csv(CALIBRATIONS_FILE), CALIBRATIONS_FILE)
        print(f"\nwrote {len(merged_accepted)} accepted day(s) (+{len(accepted)} this run) "
              f"-> {CALIBRATIONS_FILE}")
    else:
        if CALIBRATIONS_FILE.exists():
            _write_blocking(CALIBRATIONS_FILE.unlink, CALIBRATIONS_FILE)
        print(f"\nno accepted days; removed {CALIBRATIONS_FILE}")

    if not merged_rejected.empty:
        _write_blocking(lambda: merged_rejected.to_csv(REJECTIONS_FILE), REJECTIONS_FILE)
        print(f"wrote {len(merged_rejected)} rejected day(s) (+{len(rejected)} this run) "
              f"-> {REJECTIONS_FILE}")
    else:
        if REJECTIONS_FILE.exists():
            _write_blocking(REJECTIONS_FILE.unlink, REJECTIONS_FILE)
        print("no rejected days; removed", REJECTIONS_FILE)

    if have_output:
        attempted = len(merged_accepted) + len(merged_rejected)
        print(f"accept rate {len(merged_accepted)}/{attempted} = {len(merged_accepted) / attempted:.1%}"
              + (f"; rejections by reason: {merged_rejected['reason'].value_counts().to_dict()}"
                 if not merged_rejected.empty else ""))
        # Refresh the config snapshot next to the outputs (Python-readable for the downstream
        # LaTeX-fragment scripts). It was written at run start; rewrite it here with the MERGED on-disk
        # tally (not just this run). Kept whenever EITHER output file is present, removed only when the
        # run produced no output at all (below).
        _write_blocking(
            lambda: write_config_spec(args.MODEL, args.OBJECTIVE, args.LIMIT,
                                      len(merged_accepted), len(merged_rejected)),
            SPEC_FILE)
        print(f"wrote config snapshot -> {SPEC_FILE}")
    elif SPEC_FILE.exists():
        _write_blocking(SPEC_FILE.unlink, SPEC_FILE)
        print(f"no output; removed {SPEC_FILE}")


if __name__ == "__main__":
    main()

