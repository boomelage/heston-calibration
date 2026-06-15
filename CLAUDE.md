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
4. Call `calibrate_heston(surface, s, r, g)` **once per spot** over the full multi-maturity surface;
   accumulate params into a per-spot table and write `calibrations/cboe_spx_calibrations_<date>.csv`
   (incrementally, after each spot).
5. For diagnostics, reprice each spot's snapshot under Black–Scholes (`vanp.df_numpy_black_scholes`)
   and Heston (`vanp.df_heston_price`) with the fitted params; accumulate across all spots and write
   `calibration_tests/cboe_spx_calibration_tests_<date>.csv` **once** after the spot loop.

Output routing uses `filepath.replace('otm', ...)`, which rewrites **both** the `otm` directory
segment and the `otm` token in the filename in one call — fragile but intentional.

**Stage 4 — calibration engine (`src/calibrate_heston.py`).** Pure function
`calibrate_heston(vol_matrix, s, r, g) -> dict`. Builds a QuantLib `HestonProcess` / `HestonModel`
with an `AnalyticHestonEngine`, creates one `HestonModelHelper` per non-NaN surface cell (maturity
as `Period(days, Days)`, NYSE calendar, `Date.todaysDate()` as eval date), and calibrates with
Levenberg–Marquardt. Returns `{theta, kappa, eta, rho, v0, feller}` where `feller = 2*kappa*theta - eta**2`.
**Failure sentinel:** if the optimizer never moved from the hard-coded initial guess
(`v0=0.01, kappa=0.2, theta=0.02, rho=-0.75, eta=0.5`), it returns all-`None`.
**Param order matters:** `model.params()` returns `[theta, kappa, eta, rho, v0]`.

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
