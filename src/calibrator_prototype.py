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

Output: the parameters accumulate into a SINGLE `data/calibrations.csv` (one row per trading day,
keyed by date, recording S_ref, r, g, the five params, feller, rmse, coverage counts and the
intraday spot range) -- fully regenerated each run from the accepted days. The bulky per-day
repricing diagnostics stay one-file-per-day under `data/options/calibration_tests/`. A day is
written to calibration_tests exactly when it contributes a row, so the two outputs always describe
the same accepted set; a rejected or too-thin day contributes no row and clears its tests file.

Rejection audit: every attempted-but-rejected day contributes one row to a SEPARATE
`data/rejections.csv` (keyed by date) recording why it was dropped -- a small `reason` category
(no_trades/no_rate/thin/pegged/iv_miss/no_fit), the human `detail`, the `iv_rmse` where one exists,
and the surface coverage where known. `calibrations.csv` stays accepted-only (it mirrors
calibration_tests/ one-to-one); `rejections.csv` is the complement, so accepted + rejected together
cover every attempted day and the pegged-vs-thin split is auditable. Both files are fully
regenerated each run, and an empty set removes its file.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from quantlib_pricers import vanilla_pricer
vanp = vanilla_pricer()
pd.options.display.float_format = '{:.5f}'.format

SRC = Path(__file__).parent.resolve()
DATA = SRC.parent / "data"
# One calibration per trading day -> one row per day, so the parameters live in a single
# accumulating file, not a file-per-day directory. The bulky per-day repricing diagnostics stay
# under their own directory (one file per day) since they are large, not "parameters".
CALIBRATIONS_FILE = DATA / "calibrations.csv"
# The complement of calibrations.csv: one row per attempted-but-rejected day, recording why it was
# dropped. Lets us quantify the pegged-vs-thin-vs-IV rejection split that calibrations.csv (accepted
# only, by gate construction) cannot show.
REJECTIONS_FILE = DATA / "rejections.csv"
TESTS = SRC.parent / "data" / "options" / "calibration_tests"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from calibrate_heston import calibrate_heston, IV_RMSE_ACCEPT

if str(DATA) not in sys.path:
    sys.path.insert(0, str(DATA))

from get_rg import rg # pyright: ignore[reportMissingImports]

# `rg` is sorted descending (newest first); `asof` needs an ascending index. Sort once here
# instead of per file. `asof(date)` returns the last value on/before `date` regardless of `rg`'s
# ordering, or NaN if `date` precedes all rates.
rg_asc = rg.sort_index()

# Surface coverage / selection knobs. Pooling the whole day (one fit) lets us take more maturities
# than the old per-spot path (was max_nt=7); the surface is built once over the full day.
MAX_NT = 12          # maturities kept, ranked by traded volume
MAX_NK = 8           # strikes kept per wing (highest OTM puts, lowest OTM calls), nearest the money
STRIKE_GRID = 5.0    # SPX near-money strike increment; normalised K* is snapped to this grid
MIN_DTM = 7          # drop ultra-short maturities (< 7 days): Heston fits them poorly and they drive
                     # eta/kappa to extremes (Feller-violating), polluting the pooled fit
MIN_MATS = 3         # require a genuinely multi-maturity surface (identification)
MIN_STRIKES = 5      # require a real strike range
MIN_CELLS = 12       # non-NaN surface cells required (target >= MIN_MATS x MIN_STRIKES)
MAX_MOVE_PCT = 0.03  # intraday spot range above this flags the day (sticky-moneyness strained)
OBJECTIVE = "price"  # in-engine LM objective: "price" (relative-price, default) or "vol" (IV-space).
                     # Only changes what each restart minimises; selection/gate always use IV-RMSE.


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
        note = "nothing written"
    print(f"{date}: {detail}; {note}")
    return {'date': date, 'reason': reason, 'detail': detail, 'iv_rmse': iv_rmse,
            'n_maturities': n_maturities, 'n_strikes': n_strikes, 'n_cells': n_cells}


def _select_surface(df):
    """Pick the day's calibration surface in moneyness-normalised (K*) strike space.

    Top MAX_NT maturities by traded volume; within each, the MAX_NK nearest-the-money strikes per
    wing (highest OTM puts, lowest OTM calls) on K* (already centred on S_ref). Returns the selected
    snapshot rows with original strike/spot retained for repricing, or None if no maturity qualifies.
    """
    byt = df.groupby('days_to_maturity')
    vol_by_t = byt['trade_size'].sum().sort_values(ascending=False)
    T = np.sort(vol_by_t.index[:MAX_NT]).tolist()

    selected = []
    for t in T:
        dft = byt.get_group(t)
        cK = np.sort(dft.loc[dft['w'] == 'call', 'Kstar'].unique())
        pK = np.sort(dft.loc[dft['w'] == 'put', 'Kstar'].unique())
        if len(cK) > 1 and len(pK) > 1:
            keep = list(pK[-min(len(pK), MAX_NK):]) + list(cK[:min(len(cK), MAX_NK)])
            selected.append(dft[dft['Kstar'].isin(keep)])
    if not selected:
        return None
    return pd.concat(selected, ignore_index=True)


def calibrate_by_day(filepath):
    test_path = filepath.replace('otm', 'calibration_tests')
    df = pd.read_csv(filepath)
    df = df[(df['trade_iv'] > 0) & (df['days_to_maturity'] >= MIN_DTM)].copy()
    if df.empty:
        return _skip_day(test_path, "no_trades", "no trades after IV/DTM filter")
    df['quote_datetime'] = pd.to_datetime(df['quote_datetime'])
    date = df['quote_datetime'].dt.floor('D').unique()[0]
    r = rg_asc['risk_free_rate'].asof(date)
    g = rg_asc['dividend_rate'].asof(date)
    if pd.isna(r) or pd.isna(g):
        return _skip_day(test_path, "no_rate", f"no rate on/before {pd.Timestamp(date).date()}")

    # One reference spot for the whole day (volume-weighted). The intraday range that the
    # sticky-moneyness re-centring assumes is mild; a large range strains that assumption.
    S_ref = float(np.average(df['spot_price'], weights=df['trade_size']))
    spot_min, spot_max = float(df['spot_price'].min()), float(df['spot_price'].max())
    spot_range_pct = spot_max / spot_min - 1.0
    high_move = spot_range_pct > MAX_MOVE_PCT
    if high_move:
        print(f"WARNING {pd.Timestamp(date).date()}: intraday spot range {spot_range_pct:.2%} "
              f"> {MAX_MOVE_PCT:.0%}; normalisation to S_ref={S_ref:.1f} may be strained")

    # Moneyness-normalise: each trade keeps m = K / S_row but is re-struck to K* = m * S_ref and
    # snapped to the SPX strike grid, so trades at different intraday spots align on shared columns.
    df['Kstar'] = (df['strike_price'] / df['spot_price']) * S_ref
    df['Kstar'] = (df['Kstar'] / STRIKE_GRID).round() * STRIKE_GRID

    sel = _select_surface(df)
    if sel is None:
        return _skip_day(test_path, "thin", "no usable maturities")
    sel = sel.sort_values('trade_size')   # so pivot aggfunc='last' keeps the highest-volume trade per cell
    surf = sel.pivot_table(index='Kstar', columns='days_to_maturity',
                           values='trade_iv', aggfunc='last')

    n_strikes, n_mats = surf.shape
    n_cells = int(surf.count().sum())
    if n_mats < MIN_MATS or n_strikes < MIN_STRIKES or n_cells < MIN_CELLS:
        return _skip_day(
            test_path, "thin",
            f"thin surface ({n_strikes} strikes x {n_mats} maturities, {n_cells} cells)",
            n_maturities=n_mats, n_strikes=n_strikes, n_cells=n_cells,
        )

    res = calibrate_heston(surf, S_ref, r, g, objective=OBJECTIVE)   # ONE calibration for the whole day (hardened engine)
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
    params = ['theta', 'kappa', 'rho', 'eta', 'v0']
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
    try:
        repriced['heston'] = vanp.df_heston_price(repriced)
    except Exception:
        repriced['heston'] = np.nan

    TESTS.mkdir(parents=True, exist_ok=True)
    repriced.dropna(subset=['heston']).to_csv(test_path, index=False)
    return row


def main():
    # joblib's default loky backend spawns processes; on Windows the children re-import this module,
    # so the driver MUST live behind `if __name__ == "__main__"` (via main()) -- otherwise each worker
    # re-runs the Parallel call below and recursively spawns process pools.
    from joblib import Parallel, delayed
    max_jobs = max(1, os.cpu_count() // 4)
    
    OTM = Path(__file__).parent.parent / "data" / "options" / "otm"
    files = [f for f in os.listdir(OTM) if f.endswith('.csv')]
    files = pd.Series([os.path.join(OTM, f) for f in files]).sort_values(ascending=False).reset_index(drop=True)

    # Every attempted day returns exactly one row: an accepted calibration (no 'reason' key) or a
    # rejection (carries 'reason'). Split them into the two complementary files. The loop covers all
    # OTM files, so both files are fully regenerated each run (no stale rows survive); an empty set
    # removes its file rather than leaving it stale.
    results = [r for r in Parallel(n_jobs=max_jobs)(delayed(calibrate_by_day)(f) for f in files)
               if r is not None]
    accepted = [r for r in results if 'reason' not in r]
    rejected = [r for r in results if 'reason' in r]

    if accepted:
        out = pd.DataFrame(accepted).set_index('date').sort_index()
        out.to_csv(CALIBRATIONS_FILE)
        print(f"\nwrote {len(accepted)} accepted day(s) -> {CALIBRATIONS_FILE}")
    else:
        if CALIBRATIONS_FILE.exists():
            CALIBRATIONS_FILE.unlink()
        print(f"\nno accepted days; removed {CALIBRATIONS_FILE}")

    if rejected:
        rej = pd.DataFrame(rejected).set_index('date').sort_index()
        rej.to_csv(REJECTIONS_FILE)
        attempted = len(accepted) + len(rejected)
        print(f"wrote {len(rejected)} rejected day(s) -> {REJECTIONS_FILE}")
        print(f"accept rate {len(accepted)}/{attempted} = {len(accepted) / attempted:.1%}; "
              f"rejections by reason: {rej['reason'].value_counts().to_dict()}")
    else:
        if REJECTIONS_FILE.exists():
            REJECTIONS_FILE.unlink()
        print("no rejected days; removed", REJECTIONS_FILE)


if __name__ == "__main__":
    main()

