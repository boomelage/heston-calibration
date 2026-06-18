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

- **Python 3.12**, **QuantLib 1.35**. Also: `pandas`, `numpy`, `scipy`.
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

# Stage 3+4: calibrate ONCE per trading day. Writes accepted params to the single data/calibrations.csv,
#            one row per REJECTED day (with the cause) to data/rejections.csv, and per-day repricing
#            diagnostics to data/options/calibration_tests/. Prints the accept rate and a
#            rejections-by-reason tally. Resolves paths from __file__, so it runs from any working dir.
python src/calibrator_prototype.py

# Validation (read-only): grade data/calibrations.csv + calibration_tests/ for fit quality, economic
#          reasonability, and cross-day stability. Writes data/options/validation/validation.csv and
#          prints a per-day summary plus a cross-day stability block. Does not modify the pipeline.
python src/validate_calibrations.py
```

**Data not in version control.** `data/options/raw/` (raw CBOE trade files, ~80–90 MB/day — near
GitHub's 100 MB/file limit) and `data/options/otm/` (the OTM snapshots derived from them, ~6 MB/day)
are **git-ignored**; only a `.gitkeep` keeps each folder present. `data/options/calibration_tests/`
(per-day repricing diagnostics, one file per day) is **git-ignored** for the same reason — it grows
with years of data. A fresh clone has none of them — to bootstrap, drop
`UnderlyingOptionsTradesCalcs_*.csv` into `data/options/raw/`, run Stage 2 to materialise
`data/options/otm/`, then Stages 3+4 (which regenerate `calibration_tests/`). Only the small derived
artefacts are tracked (`data/calibrations.csv`, `data/rejections.csv`, `data/options/validation/`,
`data/market/`).

There is no single-test command because there are no tests. To exercise just the engine, import
`calibrate_heston(vol_matrix, s, r, g, objective="price")` from `src/calibrate_heston.py` with a
strike×maturity IV DataFrame. `objective` selects the in-engine LM objective ("price" relative-price,
default, or "vol" IV-space); the orchestrator passes its `OBJECTIVE` constant through.

## Pipeline architecture

Data flows left-to-right. `raw/` and `otm/` hold one CSV per trading day (both **git-ignored** — not
in the repo; see the "Data not in version control" note under How to run); the per-day calibration
parameters accumulate into a **single** `data/calibrations.csv` (one row per day), while the bulky
per-day repricing diagnostics stay one-file-per-day under `data/options/calibration_tests/`:

```
raw/  --extract_otms.py-->  otm/  --calibrator_prototype.py-->  ../calibrations.csv  + calibration_tests/
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

**Stage 3 — orchestration (`src/calibrator_prototype.py`, `calibrate_by_day`).** The non-obvious
core. For each OTM file it does **one calibration per trading day** (PLAN Work item 3), over a
pooled, moneyness-normalised surface — not the old per-0.5-spot-bucket fits:

1. Read trades; keep `trade_iv > 0` **and** `days_to_maturity >= MIN_DTM` (=7). Ultra-short
   maturities are dropped: Heston fits them poorly and they drive `eta`/`kappa` to Feller-violating
   extremes, polluting the pooled fit.
2. Look up `r`, `g` from `rg` for the file's quote date (NaN-guarded).
3. **Reference spot.** Compute one volume-weighted `S_ref` for the day. Record the intraday spot
   range; if it exceeds `MAX_MOVE_PCT` (=3%) set `high_move=True` and warn (the sticky-moneyness
   re-centring below is strained on large-move days) — the day is still written.
4. **Moneyness normalisation.** A Heston fit has a single spot, but trades occur across the
   intraday range. Each trade keeps its moneyness `m = strike / spot_row` but is re-struck to
   `K* = m * S_ref` and **snapped to the SPX 5-point grid** (`STRIKE_GRID`), so trades at different
   intraday spots share clean surface columns (`Kstar`).
5. **Surface.** Rank maturities by traded volume (top `MAX_NT`=12); for each kept maturity take the
   `MAX_NK`=8 nearest-money `Kstar` per wing (highest OTM puts, lowest OTM calls); `pivot_table`
   into a `Kstar`×maturity IV surface (`values='trade_iv'`). When several trades share a cell the
   **highest-volume** trade's IV is kept (`sel` sorted by `trade_size`, then `aggfunc='last'`), not the
   chronologically last — a volume-weighted mean per cell is under consideration (PLAN.md). Require
   richer coverage than before: `>= MIN_MATS`(3) maturities, `>= MIN_STRIKES`(5) strikes, and
   `>= MIN_CELLS`(12) non-NaN cells.
6. Call `calibrate_heston(surface, S_ref, r, g)` **once for the whole day**. The engine **rejects**
   fits it cannot trust (returns `None` params — see Stage 4), printing whether the rejection was a
   thin surface, an IV-RMSE miss, or a **boundary-pegged** param.
7. **On accept** `calibrate_by_day` *returns* the day's **one row keyed by date** (`S_ref` as
   `spot_price`, `r`, `g`, the five params, `feller`, `iv_rmse`, `rmse`, coverage counts, intraday
   spot range, `high_move`) and writes the repriced surface contracts to
   `calibration_tests/cboe_spx_calibration_tests_<date>.csv`. Repricing uses each contract's
   **original** `spot_price`/`strike_price` (Heston params are spot-independent), not `S_ref`/`Kstar`.
   A rejected or too-thin day returns `None` and **removes** any stale per-day tests file.
8. The module-level driver collects the returned rows across **all** OTM files and **splits** them:
   accepted rows (no `reason` key) go to the single `data/calibrations.csv`, rejected rows (each
   carries a `reason`) go to the complementary `data/rejections.csv` — both sorted by date and fully
   **regenerated** each run (no stale rows survive), and an empty set **removes** its file. Accepted +
   rejected together cover every attempted day, so the accept rate and the pegged-vs-thin-vs-IV
   rejection split are auditable directly (the driver also prints them). Because an accepted row is
   returned exactly when a tests file is written, `calibrations.csv` and the per-day tests files always
   describe the same accepted set (no desync).

The per-day **tests** path is derived by `filepath.replace('otm', 'calibration_tests')`, which
rewrites **both** the `otm` directory segment and the `otm` token in the filename in one call —
fragile but intentional. (The calibrations file is the fixed `data/calibrations.csv`, not derived
from the input path.)

**Stage 4 — calibration engine (`src/calibrate_heston.py`).** Pure function
`calibrate_heston(vol_matrix, s, r, g) -> dict`, **hardened** (PLAN Work items 2 & 3). Builds a QuantLib
`HestonProcess` / `HestonModel` with an `AnalyticHestonEngine` and one `HestonModelHelper` per
non-NaN surface cell (maturity as `Period(days, Days)`, NYSE calendar, `Date.todaysDate()` as eval
date — immaterial under the flat-forward curves used here). It then:

1. **Multiple restarts:** for each of a small data-seeded grid of starting points (`_seed_grid`),
   calibrates with Levenberg–Marquardt under **box bounds**
   (`ql.NonhomogeneousBoundaryConstraint(LOW, HIGH)`), and keeps the fit with the lowest
   **IV-space RMSE**. The LM objective itself is switchable via `objective` (`_ERR` maps it to the
   `HestonModelHelper` error type): `"price"` (`RelativePriceError`, default) or `"vol"`
   (`ImpliedVolError`). This only changes what each restart minimises; selection and the gate always
   use IV-RMSE, so it is independent of how a day is chosen/accepted. `"vol"` is more expensive (a
   Black-vol inversion per residual per LM iteration) and can throw mid-search (caught per-restart).
   `rmse` (relative-price) is computed directly from model/market values, so it keeps its meaning
   under either objective.
2. **IV-space error (the gate metric).** Each helper's fitted model price is inverted back to a
   Black vol via `BlackCalibrationHelper.impliedVolatility(modelValue, ...)` and compared to the
   market vol that built it; the RMSE of those residuals is in **vol points**. This replaces the old
   relative-price gate, which deep-OTM wings inflated (a ~1-vol-point fit scored ~0.07 price-RMSE and
   was wrongly rejected). The relative-price RMSE is still computed and returned as `rmse`, but no
   longer gates.
3. **Acceptance gate:** returns the failure sentinel if the best **IV-RMSE** exceeds
   `IV_RMSE_ACCEPT` (`0.02`, ~2 vol points) **or** any parameter is pinned within `BOUND_TOL` of a
   bound (a boundary fit is a non-fit). The old "did the params move from the fixed guess" sentinel
   is **removed**. Note: on the pooled per-day surfaces the genuine fit is excellent — across the full
   multi-year run accepted days have IV-RMSE ~0.5 vol points (median `iv_rmse` 0.0048), so IV-RMSE
   never gates. Accepted days are also **pegging-free by construction** (the gate rejects any
   boundary-pegged param), so a clean `kappa`/`rho` in `calibrations.csv` is *not* evidence pegging is
   solved — it is just what survives the gate. Over **3215** attempted days **1742 (~54%)** accept; the
   1473 rejected never reach `calibrations.csv`, but their cause is logged to `data/rejections.csv`,
   which confirms boundary-pegged `kappa` (→20) / `rho` (→−0.999) as the dominant cause
   (**pegged 1369, iv_miss 103, no_trades 1**) — still the open Phase 3 lever.

Returns `{theta, kappa, eta, rho, v0, feller, iv_rmse, rmse, n_helpers, accepted}` with
`feller = 2*kappa*theta - eta**2` for an accepted fit; a rejected fit returns params/`feller` as
`None` but keeps `iv_rmse`/`rmse`/`n_helpers`/`accepted=False` for diagnostics.
**Param order matters:** `model.params()` returns `[theta, kappa, eta, rho, v0]` — the `LOW`/`HIGH`
bounds arrays follow this exact order (get it wrong and bounds land on the wrong params).

## DataFrame column contracts (the "hard-coded names" the README warns about)

Stages communicate through column names, not typed interfaces. Renaming any of these silently
breaks a downstream stage:

- `otm/*.csv` schema: `quote_datetime, strike_price, w, trade_size, trade_price, trade_iv, spot_price, days_to_maturity`.
- `data/calibrations.csv` schema (**single file, one row per trading day**, keyed by `date`):
  `spot_price` (= `S_ref`), `risk_free_rate, dividend_rate, theta, kappa, rho, eta, v0, feller,
  iv_rmse, rmse, n_helpers, accepted, n_maturities, n_strikes, contracts_count, total_volume,
  spot_min, spot_max, spot_range_pct, high_move, calculation_date`.
- `data/rejections.csv` schema (**single file, one row per rejected trading day**, keyed by `date` —
  the complement of `calibrations.csv`): `reason` (category: `no_trades, no_rate, thin, pegged,
  iv_miss, no_fit`), `detail` (human string), `iv_rmse` (NaN unless calibration ran), `n_maturities,
  n_strikes, n_cells` (NaN unless a surface was built). `date` here is the filename date string.
- `calibration_tests/*.csv`: the day's repriced surface contracts (original `spot_price`/`strike_price`,
  plus `Kstar`, the fitted params, `volatility` (= `trade_iv`), `black_scholes`, `heston`).
- `ms.df_moneyness(df)` needs `w, spot_price, strike_price` (returns `spot-strike` for calls, `strike-spot` for puts).
- `vanp.df_numpy_black_scholes(df)` needs `spot_price, strike_price, days_to_maturity, risk_free_rate, volatility, w`
  (note: `trade_iv` is renamed to `volatility` before this call).
- `vanp.df_heston_price(df)` needs `spot_price, strike_price, days_to_maturity, risk_free_rate, dividend_rate, w, kappa, theta, rho, eta, v0`.

## Known issues & fragility (verify before trusting outputs)

- Output routing uses `filepath.replace('otm', ...)`, which rewrites **both** the `otm` directory
  segment and the `otm` token in the filename in one call — fragile but intentional.
- Moneyness normalisation assumes **sticky-moneyness** (IV ~stationary in `K/S` over a session). It
  is mild on normal days (~1% intraday range) but strained on large-move days; those are flagged
  `high_move` (range > `MAX_MOVE_PCT`=3%) and still written — treat their `S_ref` with suspicion.
- Snapping `Kstar` to the 5-point SPX grid is exact near the money but coarser in the far wings
  (native grid widens to 25/50/100); harmless for QuantLib (any float strike prices) but it slightly
  quantises deep-OTM moneyness.
- **Boundary pegging is still the open Phase 3 lever — and `calibrations.csv` cannot show it.** A
  multi-year run attempted **3215** trading days and accepted **1742 (~54%)**, just under PLAN.md's 60%
  target. The accepted set has 0 pegged `kappa`/`rho`, but that is **tautological**: the gate
  (`_on_boundary`) rejects any boundary-pegged fit, so pegged days never reach the file. The 1473
  rejected days are dropped before write; their cause is logged to `data/rejections.csv`
  (`reason` ∈ `no_trades/no_rate/thin/pegged/iv_miss/no_fit`), and the split is now **measured**:
  **pegged 1369, iv_miss 103, no_trades 1** — so `pegged` is confirmed the dominant cause (93% of
  rejections). None of the Phase 3 levers (A–E) are implemented yet (`MIN_DTM`=7, no `weights`, no
  `fixParameters`, no Feller penalty), so this ~54% is the Phase 2 engine's rate over the long sample,
  not a post-lever result.
- **Feller is the standout issue in the accepted set.** The gate does **not** reject on Feller (it is a
  *suspicious*, not hard-reject, validator flag — short-tenor Heston violates it routinely), so accepted
  days routinely violate it: `feller = 2·kappa·theta − eta² < 0` on **1729/1742 (99%)** accepted days,
  and `eta > 1.5` on ~9% (161 days, max ≈2.0, near its cap). This is PLAN.md Lever D (soft Feller penalty +
  revisit the `eta` cap).
- The `data/__pycache__/` holds bytecode for deleted modules (`get_data`, `get_options`, ...) — ignore it.

## Writing prose (`gpu-options.tex` and other `.tex` documents)

When you write or edit prose in `heston-calibration.tex` or any other `.tex` document here, write it the way a careful human author would, not the way an LLM defaults to. Concretely:

- **Avoid the em dash (`—`) as a sentence connector.** It is the single clearest tell of machine-written prose, and the existing text overuses it. Prefer a period, a comma, a colon, or parentheses, and rephrase so the dash is not needed. Do not replace one em dash with another piece of dashy punctuation (en dash, double hyphen) doing the same job; restructure the sentence instead. (Genuine ranges like `12–31×` and `1…8192` keep their en dash/ellipsis — this is about prose connectors, not numerics.)
- **Keep sentences short and digestible.** One idea per sentence. Break a long sentence into two or three rather than stacking clauses with dashes, semicolons, and nested parentheticals. If a sentence needs more than one comma-separated aside to parse, split it.
- **Prefer plain, direct phrasing over ornate constructions.** Say "the GPU is faster" rather than "the GPU exhibits a marked performance advantage." Cut filler ("it is worth noting that", "importantly", "in order to"), hedging stacks, and rule-of-three flourishes that exist only for rhythm.
- **Match the surrounding voice.** This is a technical paper: declarative, specific, quantitative. State the result and the number; let the data carry the emphasis instead of intensifiers.
- **Read it back as a human.** Before finishing, reread each edited sentence aloud in your head. If it sounds like a generated abstract or could not have been said plainly by a person, rewrite it.

These rules apply to *new and edited* prose. Do not launch a sweeping em-dash-removal pass over untouched paragraphs unless asked, but do clean up the dashes and over-long sentences in any passage you are already editing.