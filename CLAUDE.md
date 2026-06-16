# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Maintaining this file

Keep this document in sync with the code as you work. When a change alters anything described here —
files moved or renamed, column contracts changed, a "Known issue" fixed or a newly found one, run
commands or stages changed — update CLAUDE.md in the **same** change: add what is now true and delete
what is now stale. A stale line here is worse than a missing one. Do not leave fixed issues marked
"done"; remove them.

## Purpose

Calibrate Heston (1993) stochastic-volatility parameters (`v0, kappa, theta, eta, rho`) to
CBOE S&P 500 (SPX) intraday option trades, using QuantLib. `eta` is the vol-of-vol (QuantLib's
`sigma`). The model SDE is documented in `heston-calibration.tex`:

```
dX_t = (r - v_t/2) dt + sqrt(v_t)(rho dW_t + sqrt(1-rho^2) dB_t)
dv_t = kappa(theta - v_t) dt + eta sqrt(v_t) dW_t
```

The README explicitly notes the workflow is convoluted and over-reliant on passing intermediate
CSVs between stages with hard-coded column names. Treat the current scripts as a working
prototype, not a clean design.

## Environment & dependencies

- **Python 3.12**, **QuantLib 1.35**. Also: `pandas`, `numpy`, `scipy`, `joblib`.
- **Two proprietary packages by the repo author** provide the QuantLib convenience wrappers:
  - `model_settings` — exports a ready instance `ms` (used as `ms.df_moneyness(df)`).
  - `quantlib_pricers` — exports the class `vanilla_pricer` (used as `vanp = vanilla_pricer()`).
  - These are **not** on PyPI; they live at `github.com/boomelage/{model_settings,quantlib_pricers}`
    and are currently installed under `E:\Python\Lib\site-packages`. There is no `requirements.txt`,
    `setup.py`, lockfile, or test suite in this repo.

## How to run

The pipeline is four stages. There is no build/lint/test tooling — you run scripts directly.

```bash
# Stage 1: market rates. NOT run standalone -- data/get_rg.py is imported by Stage 3
#          (`from get_rg import rg`) and builds the `rg` rate table on import. Optional
#          sanity check of the rates it will feed the calibrator:
python -c "import sys; sys.path.insert(0,'data'); from get_rg import rg; print(rg[['risk_free_rate','dividend_rate']].head())"

# Stage 2: clean raw trades -> OTM-only snapshots. Resolves paths from __file__,
#          so it runs from any working directory.
python data/extract_otms.py

# Stage 3+4: calibrate per spot level and write params + repricing diagnostics.
#            Safe to run from anywhere (resolves paths from __file__).
python src/calibrator_prototype.py

# Validation (read-only): grade the calibrations/ + calibration_tests/ output for fit quality,
#          economic reasonability, and cross-bucket stability. Writes validation/validation_<date>.csv
#          and prints a per-day summary. Does not modify the pipeline.
python src/validate_calibrations.py
```

There is no single-test command because there are no tests. To exercise just the engine, import
`calibrate_heston(vol_matrix, s, r, g)` from `src/calibrate_heston.py` with a strike×maturity IV
DataFrame.

## Pipeline architecture

Data flows left-to-right through `data/options/` subfolders, one CSV per trading day:

```
raw/  --extract_otms.py-->  otm/  --calibrator_prototype.py-->  calibrations/  + calibration_tests/
```

**Stage 1 — market rates (`data/get_rg.py`).** Imported for its side effect: building a
module-level DataFrame `rg`. Parses two hard-coded filenames in `data/market/`:

- `historical_USGG12M.csv` → US 12M Treasury yield → `risk_free_rate`.
- `historical_SPX_ivols.csv` → SPX `spot_price`, 12M `dividend_rate`, and a term structure of
  `<tenor>_vol` columns.
`rg` is indexed by date, **sorted descending (newest first)**, values divided to decimals.
Downstream only `risk_free_rate` and `dividend_rate` are consumed; the `_vol` columns and `rg`'s
`spot_price` are computed but unused (spot comes from the options data instead).

**Stage 2 — OTM extraction (`data/extract_otms.py`).** For each `raw/UnderlyingOptionsTradesCalcs_*.csv`
(CBOE trade-level data): selects/renames a column subset, uses `underlying_bid` as `spot_price`,
maps `option_type` C/P → `w` call/put, computes `days_to_maturity` (calendar days, `>0` only),
keeps positive IV/spot/strike, then keeps **only OTM** rows via `ms.df_moneyness` (`moneyness < 0`).
Writes `otm/cboe_spx_otm_<lastquotedate>.csv`.

**Stage 3 — orchestration (`src/calibrator_prototype.py`, `calibrateby_spot`).** The non-obvious
core. For each OTM file it does **per-spot-level calibration**, not one snapshot per day:

1. Look up `r`, `g` from `rg` for the file's quote date.
2. Round spot to the nearest 0.5 (`(2*spot).round()//2`) and group trades by this rounded level.
3. For each spot level: rank maturities by traded volume (top `max_nt`=7); for each kept maturity,
   take the `max_nk`=7 strikes nearest the money on each wing (highest OTM puts `pK[-n:]`, lowest
   OTM calls `cK[:n]`) from that maturity's rows only; concat into one snapshot and `pivot_table`
   into a strike×maturity IV surface (`values='trade_iv'`), requiring ≥5 non-NaN cells.
4. Call `calibrate_heston(surface, s, r, g)` **once per spot** over the full multi-maturity surface.
   The engine **rejects** fits it cannot trust (returns `None` params — see Stage 4), so only
   *accepted* spots are kept. Accumulate accepted params **plus the engine diagnostics**
   (`rmse, n_helpers, accepted`) into a per-spot table and write
   `calibrations/cboe_spx_calibrations_<date>.csv` **once** after the spot loop.
5. For diagnostics, reprice **each accepted** spot's snapshot under Black–Scholes
   (`vanp.df_numpy_black_scholes`) and Heston (`vanp.df_heston_price`) with the fitted params;
   accumulate across accepted spots and write `calibration_tests/cboe_spx_calibration_tests_<date>.csv`
   **once** after the spot loop. Both CSVs are written under one decision so they always describe the
   same accepted set; on a day with **zero** accepted fits **both** files are removed (not left stale).

Output routing uses `filepath.replace('otm', ...)`, which rewrites **both** the `otm` directory
segment and the `otm` token in the filename in one call — fragile but intentional.

**Stage 4 — calibration engine (`src/calibrate_heston.py`).** Pure function
`calibrate_heston(vol_matrix, s, r, g) -> dict`, **hardened** (PLAN Work item 2). Builds a QuantLib
`HestonProcess` / `HestonModel` with an `AnalyticHestonEngine` and one `HestonModelHelper` per
non-NaN surface cell (maturity as `Period(days, Days)`, NYSE calendar, `Date.todaysDate()` as eval
date — immaterial under the flat-forward curves used here). It then:

1. **Multiple restarts:** for each of a small data-seeded grid of starting points (`_seed_grid`),
   calibrates with Levenberg–Marquardt under **box bounds**
   (`ql.NonhomogeneousBoundaryConstraint(LOW, HIGH)`), and keeps the fit with the lowest
   relative-price RMSE (`HestonModelHelper.calibrationError()`, the only error type SWIG exposes).
2. **Acceptance gate:** returns the failure sentinel if the best RMSE exceeds `RMSE_ACCEPT`
   (`0.05`, kept strict on purpose — loosening it only admits under-determined per-bucket fits) **or**
   any parameter is pinned within `BOUND_TOL` of a bound (a boundary fit is a non-fit). The old "did
   the params move from the fixed guess" sentinel is **removed**. Note: under this strict gate most
   per-bucket fits are rejected; the fix is PLAN.md Work item 3 (one calibration per day), not a lower
   threshold.

Returns `{theta, kappa, eta, rho, v0, feller, rmse, n_helpers, accepted}` with
`feller = 2*kappa*theta - eta**2` for an accepted fit; a rejected fit returns params/`feller` as
`None` but keeps `rmse`/`n_helpers`/`accepted=False` for diagnostics.
**Param order matters:** `model.params()` returns `[theta, kappa, eta, rho, v0]` — the `LOW`/`HIGH`
bounds arrays follow this exact order (get it wrong and bounds land on the wrong params).

## DataFrame column contracts (the "hard-coded names" the README warns about)

Stages communicate through column names, not typed interfaces. Renaming any of these silently
breaks a downstream stage:

- `otm/*.csv` schema: `quote_datetime, strike_price, w, trade_size, trade_price, trade_iv, spot_price, days_to_maturity`.
- `ms.df_moneyness(df)` needs `w, spot_price, strike_price` (returns `spot-strike` for calls, `strike-spot` for puts).
- `vanp.df_numpy_black_scholes(df)` needs `spot_price, strike_price, days_to_maturity, risk_free_rate, volatility, w`
  (note: `trade_iv` is renamed to `volatility` before this call).
- `vanp.df_heston_price(df)` needs `spot_price, strike_price, days_to_maturity, risk_free_rate, dividend_rate, w, kappa, theta, rho, eta, v0`.

## Known issues & fragility (verify before trusting outputs)

- Output routing uses `filepath.replace('otm', ...)`, which rewrites **both** the `otm` directory
  segment and the `otm` token in the filename in one call — fragile but intentional.
- Spot is rounded to a 0.5 grid (`(2*spot).round()//2`), so a contract that was OTM at its actual
  spot can land with `strike_price == spot_price` (an ATM tie) in the snapshot. Harmless, but a
  strict `K>spot`/`K<spot` OTM check will flag these ties.
- The `data/__pycache__/` holds bytecode for deleted modules (`get_data`, `get_options`, ...) — ignore it.
