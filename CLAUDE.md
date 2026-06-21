# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git hygiene

Never include Claude-Session links, claude.ai URLs, or any other references that reveal a connection
to Claude or Anthropic tooling in commit messages, comments, or any tracked file. Strip such
references before committing if they appear in generated content.

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

**A Bates (1996) variant is also supported** (Heston stochastic vol + Merton lognormal jumps: the
five Heston params plus jump intensity `lambda_`, mean log-jump `nu`, log-jump std `delta`). It is
selected by the `--MODEL {heston,bates}` flag on `calibrator_prototype.py` (default `heston`) and runs
through the same orchestrator. See `PLAN-Bates.md` for the design and the Bates engine
(`src/calibrate_bates.py`).

## Environment & dependencies

- **Python 3.12**, **QuantLib 1.35**. Also: `pandas`, `numpy`, `scipy`, `joblib`.
- The QuantLib pricing wrapper is **vendored in-repo** at `src/pricing/` (formerly the author's
  external `quantlib_pricers` package, now copied in so the repo is self-contained). It is a plain
  directory with no `__init__.py`, so `pricing` is a namespace package: the single module
  `vanilla_pricer.py` holds the class `vanilla_pricer` (used as `vanp = vanilla_pricer()`). The
  upstream package's `asian_option_pricer` / `barrier_option_pricer` were dropped (this pipeline does
  not price asians or barriers). The class is imported as `from pricing.vanilla_pricer import
  vanilla_pricer` **after** `src` is added to `sys.path`. There is no `requirements.txt`, `setup.py`,
  lockfile, or test suite in this repo.

## How to run

The pipeline is three stages. There is no build/lint/test tooling — you run scripts directly. All
model/calibration constants (surface coverage knobs, box bounds, the acceptance gate, the OTM
filter floor/cutoff, the wing-weight knobs, the seed grid, the Bates bounds/seed) live in one place:
`src/config.py`. Tune there, not in the individual modules. The QuantLib **date conventions** (the
`Actual365Fixed` day count and `UnitedStates.NYSE` calendar) are centralized there too, as
`config.day_count(name=None)` / `config.calendar(name=None)` (the named choices are `DAY_COUNT_NAME`
/ `CALENDAR_NAME`); both engines, both `utils` engine builders, the vendored `pricing.vanilla_pricer`,
and the figure scripts call these instead of rebuilding the literals. The driver calibrates **every** raw file in
`data/options/raw/` by default; `--LIMIT N` restricts to the `N` most recent trading days (by date).
The full multi-year sample is ~3,217 trading days and one Heston `vol` run takes a few hours on this
machine (multi-start LM, ~1,500 cells/day, 8 parallel jobs). A Bates run is ~4.5x slower per day (the
`BatesEngine` is heavier than `AnalyticHestonEngine` and there are 8 params), so the full sample is
~7 hours.

```bash
# Stage 1: market rates. NOT run standalone -- data/get_rg.py is imported by Stage 2
#          (`from get_rg import rg`) and builds the `rg` rate table on import. Optional
#          sanity check of the rates it will feed the calibrator:
python -c "import sys; sys.path.insert(0,'data'); from get_rg import rg; print(rg[['risk_free_rate','dividend_rate']].head())"

# Stage 2+3: calibrate ONCE per trading day. Reads raw trades directly and does the OTM cleaning
#            in-memory (utils._prepare_options), so there is no separate extraction script. Writes
#            accepted params to the single results/<model>/calibrations/<objective>/calibrations.csv, one
#            row per REJECTED day (with the cause) to results/<model>/calibrations/<objective>/rejections.csv,
#            and per-day repricing diagnostics to results/<model>/calibrations/<objective>/calibration_tests/.
#            <model> is `heston` (default) or `bates`, via --MODEL; <objective> is `vol` (default) or
#            `price`, via --OBJECTIVE. --LIMIT N caps to the N most recent days. Prints the accept rate
#            and a rejections-by-reason tally. Resolves paths from __file__, runs from any dir.
python src/calibrator_prototype.py                       # heston, vol, all days
python src/calibrator_prototype.py --MODEL bates --LIMIT 100   # bates, last 100 days

# Validation (read-only): grade results/<model>/calibrations/<objective>/calibrations.csv +
#          calibration_tests/ for fit quality, economic reasonability, and cross-day stability. Prompts
#          for the objective (price/vol), writes results/<model>/calibrations/<objective>/validation.csv,
#          and prints a per-day summary plus a cross-day stability block. Does not modify the pipeline.
#          Model-aware: set the MODEL/OBJECTIVE constants at the top to grade a heston or bates run
#          (bates also grades/reports the jump params lambda_/nu/delta).
python src/validate_calibrations.py

# Wing-residual diagnostic (read-only): invert each repriced contract's Heston price back to a Black
#          IV and report resid = model_iv - market_iv, stratified by signed log-moneyness, by
#          |log-moneyness| (the axis config.WING_WEIGHT_* up-weights), and by maturity x wing. This is
#          the metric a WING_WEIGHT_GAIN sweep is graded on (PLAN Lever B). OBJECTIVE constant at top.
python src/wing_residuals.py
```

**Data not in version control.** `data/options/raw/` (raw CBOE trade files, ~80–90 MB/day — near
GitHub's 100 MB/file limit) is **git-ignored**; only a `.gitkeep` keeps the folder present. The OTM
cleaning is done in-memory by the calibrator (no on-disk OTM snapshots).
`results/*/calibrations/*/calibration_tests/` (per-day repricing diagnostics, one file per day) is
**git-ignored** for the same reason — it grows with years of data. A fresh clone has none of
them — to bootstrap, drop `UnderlyingOptionsTradesCalcs_*.csv` into `data/options/raw/`, then run
Stage 2+3 (which regenerates `calibration_tests/`). Only the
small derived artefacts are tracked: the `{calibrations,rejections,validation}.csv` triple for the
default **`vol`** objective (`results/heston/calibrations/vol/`) plus `data/market/`. The `price`
objective is still selectable (`--OBJECTIVE price`) but its outputs are **not committed on this branch**
(the older narrow-config `price` baseline lives on `master`). Each objective's bulky per-day
`calibration_tests/` stays git-ignored (only a `.gitkeep` is tracked).

**Output routing is namespaced by model AND objective:** `results/<model>/calibrations/<objective>/`
(`<model>` ∈ `heston, bates`). The Heston tree was migrated from the old `results/calibrations/<objective>/`
to `results/heston/calibrations/<objective>/`; Bates lands under `results/bates/...`. The single source
of truth is `config.calib_paths(model, objective)` (and `_objective_paths` wraps it).

There is no single-test command because there are no tests. To exercise just an engine, import
`calibrate_heston(vol_matrix, s, r, g, objective="vol")` from `src/calibrate_heston.py` (or
`calibrate_bates(...)` from `src/calibrate_bates.py`, same signature) with a strike×maturity IV
DataFrame. `objective` selects the in-engine LM objective ("vol" IV-space, the default, or "price"
relative-price); the orchestrator passes its `OBJECTIVE` constant through.

**Downstream figure/table scripts** live under `src/results/` (moved there from `results/`):
`surfaces/example_surface.py` (rebuilds a model IV/price surface from one calibration row),
`surfaces/make_eps.py` (writes the OTM surface/smile EPS + `otm.tex`), `smiles/smiles.py` (per-day
market-vs-model smile EPS + `smiles.tex`), and `tables/objective_comparison.py` (price-vs-vol metrics
table). They resolve `REPO = Path(__file__).parents[2]` and read calibrations from / write figures into
the **repo-level** `results/<model>/` tree, and share the QuantLib helpers (`build_model_engine` ->
`build_heston_engine`/`build_bates_engine`, plus the engine-agnostic `heston_price`,
`heston_implied_vol`) in `src/utils.py`. They are **model-aware**: `MODEL`/`OBJECTIVE` plus every
plotting/grid knob for `example_surface.py`, `make_eps.py` and `smiles.py` live in **one** file,
`src/results/results_config.py` (each of the three adds `src/results` to `sys.path` and imports from
it). Set `MODEL` to `heston` or `bates` there and it picks the engine, the `results/<model>/...`
source/output tree, and the figure labels for all three at once; the grids (`MONEYNESS`,
`MATURITIES_DAYS`), the smile knobs (`TMIN/TMAX`, `NT`, `MARKET_*`, `MKTMONSTEP`, ...) and the surface
view (`SURFACE_ELEV/AZIM`, figsizes) are tuned in the same place. (`objective_comparison.py` is **not**
wired to `results_config.py`; it keeps its own `MODEL` constant.) `example_surface.py` carries the
Bates jump triple in `day_results['params']`, and `smiles.py` shows it in the per-figure caption.
(`objective_comparison.py` needs *both* a price and a vol run for the chosen model on disk.) Run e.g.
`python src/results/surfaces/make_eps.py` then `python src/results/smiles/smiles.py` after a
calibration to refresh the paper's figures.

## Pipeline architecture

Data flows left-to-right. `raw/` holds one CSV per trading day (**git-ignored** — not in the repo;
see the "Data not in version control" note under How to run); the per-day calibration parameters
accumulate into a **single** `results/<model>/calibrations/<objective>/calibrations.csv` (one row per
day), while the bulky per-day repricing diagnostics stay one-file-per-day under
`results/<model>/calibrations/<objective>/calibration_tests/`:

```
raw/  --calibrator_prototype.py --MODEL {heston,bates} (utils._prepare_options cleans to OTM in-memory)-->  results/<model>/calibrations/<objective>/calibrations.csv  + calibration_tests/
```

**Stage 1 — market rates (`data/get_rg.py`).** Imported for its side effect: building a
module-level DataFrame `rg`. Parses two hard-coded filenames in `data/market/`:

- `historical_USGG12M.csv` → US 12M Treasury yield → `risk_free_rate`.
- `historical_SPX_ivols.csv` → SPX `spot_price`, 12M `dividend_rate`, and a term structure of
  `<tenor>_vol` columns.
`rg` is indexed by date, **sorted descending (newest first)**, values divided to decimals.
Downstream only `risk_free_rate` and `dividend_rate` are consumed; the `_vol` columns and `rg`'s
`spot_price` are computed but unused (spot comes from the options data instead).

**OTM cleaning (`src/utils._prepare_options`).** No longer a standalone stage/script (the old
`data/extract_otms.py` is removed). The calibrator calls this in-memory on each raw file: selects/renames
a column subset, uses `underlying_bid` as `spot_price`, maps `option_type` C/P → `w` call/put, computes
`days_to_maturity` (calendar days, `>0` only), keeps positive IV/spot/strike, then keeps the OTM band
via `utils.df_moneyness`: `OTM_MONEYNESS_FLOOR < ratio moneyness < OTM_MONEYNESS_CUTOFF` (0.6 and 0.98
in `config.py`). The CUTOFF drops near-ATM rows (keeps only OTM); the **FLOOR drops the deep-OTM
lottery-ticket tail** (ratio moneyness = `e^-|log(K/S)|`, so 0.6 keeps `|log-moneyness| < ~0.51`, ~40%
OTM). Those far-OTM strikes have extreme prices that peg the fit to its bounds; flooring them recovers
acceptance and tightens IV-RMSE without losing the tradeable wing.

**Stage 2 — orchestration (`src/calibrator_prototype.py`, `calibrate_by_day`).** The non-obvious
core. For each raw trades file it does **one calibration per trading day** (PLAN Work item 3), over a
pooled, moneyness-normalised surface — not the old per-0.5-spot-bucket fits:

1. Read + clean trades (`utils._prepare_options`); keep `trade_iv > 0` **and**
   `MIN_DTM <= days_to_maturity <= MAX_DTM` (`MIN_DTM`=14, `MAX_DTM`=730 in `config.py`). Ultra-short
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
5. **Surface.** Rank maturities by traded volume (top `MAX_NT`=20); for each kept maturity take the
   `MAX_NK`=40 nearest-money `Kstar` per wing (highest OTM puts, lowest OTM calls); `pivot_table`
   into a `Kstar`×maturity IV surface (`values='trade_iv'`). When several trades share a cell the
   **highest-volume** trade's IV is kept (`sel` sorted by `trade_size`, then `aggfunc='last'`), not the
   chronologically last — a volume-weighted mean per cell is under consideration (PLAN.md). Require
   richer coverage than before: `>= MIN_MATS`(3) maturities, `>= MIN_STRIKES`(5) strikes, and
   `>= MIN_CELLS`(12) non-NaN cells.
6. Call the selected engine **once for the whole day** — `_ENGINES[MODEL]`, i.e.
   `calibrate_heston(surface, S_ref, r, g)` or `calibrate_bates(...)` (same signature). The engine
   **rejects** fits it cannot trust (returns `None` params — see Stage 3), printing whether the
   rejection was a thin surface, an IV-RMSE miss, or a **boundary-pegged** param.
7. **On accept** `calibrate_by_day` *returns* the day's **one row keyed by date** (`S_ref` as
   `spot_price`, `r`, `g`, the params — five for Heston, plus `lambda_, nu, delta` for Bates (appended
   via `_EXTRA_PARAMS[MODEL]`) — `feller`, `iv_rmse`, `rmse`, coverage counts, intraday spot range,
   `high_move`) and writes the repriced surface contracts to
   `calibration_tests/cboe_spx_calibration_tests_<date>.csv`. Repricing uses each contract's
   **original** `spot_price`/`strike_price` (Heston params are spot-independent), not `S_ref`/`Kstar`.
   A rejected or too-thin day returns `None` and **removes** any stale per-day tests file.
8. The module-level driver collects the returned rows across the OTM files (all, or the `--LIMIT N`
   most recent) and **splits** them: accepted rows (no `reason` key) go to the single
   `results/<model>/calibrations/<objective>/calibrations.csv`, rejected rows (each carries a `reason`)
   go to the complementary `results/<model>/calibrations/<objective>/rejections.csv` — both sorted by date and fully
   **regenerated** each run (no stale rows survive), and an empty set **removes** its file. Accepted +
   rejected together cover every attempted day, so the accept rate and the pegged-vs-thin-vs-IV
   rejection split are auditable directly (the driver also prints them). Because an accepted row is
   returned exactly when a tests file is written, `calibrations.csv` and the per-day tests files always
   describe the same accepted set (no desync).

The per-day **tests** path is built from `_objective_paths(MODEL, OBJECTIVE)` (a thin wrapper over
`config.calib_paths`): the tests directory `results/<model>/calibrations/<objective>/calibration_tests/`
plus a basename of `cboe_spx_calibration_tests_<date>.csv` (same basename across model/objective — they
are already separated by directory), where `<date>` is sliced out of the OTM filename. The directory
depends on `(MODEL, OBJECTIVE)`, and `validate_calibrations.py` rebuilds the identical path from the
same `calib_paths` rule so the two stay in lock-step. (The calibrations/rejections/validation files are
likewise under `results/<model>/calibrations/<objective>/`: `calibrations.csv` + `rejections.csv` +
`validation.csv`, built from `calib_paths` and not derived from the input path.) The model is selected
by `--MODEL {heston,bates}` (default `heston`) and the objective by `--OBJECTIVE {price,vol}` (default
`vol`, `config.DEFAULT_OBJECTIVE`), both threaded through to the chosen engine.

**Stage 3 — calibration engine (`src/calibrate_heston.py`).** Pure function
`calibrate_heston(vol_matrix, s, r, g) -> dict`, **hardened** (PLAN Work items 2 & 3). Builds a QuantLib
`HestonProcess` / `HestonModel` with an `AnalyticHestonEngine` and one `HestonModelHelper` per
non-NaN surface cell (maturity as `Period(days, Days)`, `config.calendar()` = NYSE, `Date.todaysDate()`
as eval date — immaterial under the flat-forward curves used here; the flat curves use
`config.day_count()` = `Actual365Fixed`). It then:

1. **Multiple restarts:** for each of a small data-seeded grid of starting points (`_seed_grid`),
   calibrates with Levenberg–Marquardt under **box bounds**
   (`ql.NonhomogeneousBoundaryConstraint(LOW, HIGH)`), and keeps the fit with the lowest
   **IV-space RMSE**. The LM objective itself is switchable via `objective` (`_ERR` maps it to the
   `HestonModelHelper` error type): `"vol"` (`ImpliedVolError`, the default — `config.DEFAULT_OBJECTIVE`)
   or `"price"` (`RelativePriceError`). This only changes what each restart minimises; selection and the gate always
   use IV-RMSE, so it is independent of how a day is chosen/accepted. `"vol"` is more expensive (a
   Black-vol inversion per residual per LM iteration) and can throw mid-search (caught per-restart).
   `rmse` (relative-price) is computed directly from model/market values, so it keeps its meaning
   under either objective.
   **Wing weighting (PLAN Lever B, wired but default-off).** `_calibrate_once` can up-weight OTM wing
   cells in the LM objective by `_wing_weight(k, s) = 1 + WING_WEIGHT_GAIN*(|log(k/S_ref)|/SCALE)**POWER`
   (config knobs). It is applied **only under `"vol"`** (the `"price"`/`RelativePriceError` denominator
   already up-weights cheap wings, so stacking there double-counts) and is passed to QuantLib as the 5th
   positional `weights` arg of `model.calibrate(...)` — a **plain python list**, not `ql.Array`. When
   weighting is on, restart **ranking** uses the wing-weighted IV-RMSE (`iv_rmse_sel`, the objective LM
   saw) while the gate and the reported `iv_rmse` stay **unweighted** (`iv_rmse_gate`), so
   `IV_RMSE_ACCEPT` keeps its meaning. `WING_WEIGHT_GAIN=0` (the default) makes both the call path and
   the two metrics identical — an exact no-op vs the pre-lever engine.
2. **IV-space error (the gate metric).** Each helper's fitted model price is inverted back to a
   Black vol via `BlackCalibrationHelper.impliedVolatility(modelValue, ...)` and compared to the
   market vol that built it; the RMSE of those residuals is in **vol points**. This replaces the old
   relative-price gate, which deep-OTM wings inflated (a ~1-vol-point fit scored ~0.07 price-RMSE and
   was wrongly rejected). The relative-price RMSE is still computed and returned as `rmse`, but no
   longer gates.
3. **Acceptance gate:** returns the failure sentinel if the best **IV-RMSE** exceeds
   `IV_RMSE_ACCEPT` (`0.02`, ~2 vol points) **or** any parameter is pinned within `BOUND_TOL` of a
   bound (a boundary fit is a non-fit). The old "did the params move from the fixed guess" sentinel
   is **removed**. Note: on the pooled per-day surfaces the genuine fit is excellent — accepted days
   have IV-RMSE around 0.8 vol points (median), far inside the 2-vol-point gate, so IV-RMSE essentially
   never gates; the binding rejection is boundary pegging. Accepted days are also **pegging-free by
   construction** (the gate rejects any boundary-pegged param), so a clean `kappa`/`rho` in
   `calibrations.csv` is *not* evidence pegging is solved — it is just what survives the gate. For
   acceptance-rate figures and the pegging audit see the boundary-pegging Known-issue bullet.

Returns `{theta, kappa, eta, rho, v0, feller, iv_rmse, rmse, n_helpers, accepted}` with
`feller = 2*kappa*theta - eta**2` for an accepted fit; a rejected fit returns params/`feller` as
`None` but keeps `iv_rmse`/`rmse`/`n_helpers`/`accepted=False` for diagnostics.
**Param order matters:** `model.params()` returns `[theta, kappa, eta, rho, v0]`. The bounds live in
`config.py` as the named `BOUNDS` dict; `LOW`/`HIGH` are derived as `[BOUNDS[p][L/H] for p in
PARAM_ORDER]`, with `PARAM_ORDER = ("theta","kappa","eta","rho","v0")` declaring that order in exactly
one place (get `PARAM_ORDER` wrong and bounds land on the wrong params). The engine imports `LOW`/`HIGH`
/`IV_RMSE_ACCEPT`/the seed-grid template/the `WING_WEIGHT_GAIN` flag from `config.py`; only the `_ERR`
string→QuantLib-enum map (live `ql` objects) stays in `calibrate_heston.py`.

**Shared engine helpers (`src/_engine_common.py`).** The model-agnostic helpers `_on_boundary(params,
low, high)`, `_seed_var`, `_wing_weight`, `_iv_rmse` were factored out of `calibrate_heston.py` so the
Heston and Bates engines reuse identical boundary/IV-RMSE/wing logic and cannot drift. `_on_boundary`
takes its `low`/`high` so a caller can gate a subset of the parameter vector (Bates gates only the 5
Heston params). The factor-out is behaviour-neutral for Heston (the committed `vol` numbers reproduce
to full float precision).

**Stage 3b — Bates engine (`src/calibrate_bates.py`).** Same shape as `calibrate_heston`, swapping
`HestonProcess`/`HestonModel`/`AnalyticHestonEngine` for `BatesProcess`/`BatesModel`/`BatesEngine`. The
helper stays `ql.HestonModelHelper` (there is **no `ql.BatesHelper`** in QuantLib 1.35); only the
pricing engine attached to it is a `BatesEngine`. `calibrate_bates` returns a **superset** of the
Heston dict — the same keys plus `lambda_, nu, delta` — so the orchestrator reads it unchanged.
**THREE distinct orderings (confirmed live, do not conflate):** (1) `BatesModel.params()` returns
`[theta, kappa, eta, rho, v0, nu, delta, lambda]` — Heston's order, then `(nu, delta, lambda)`; this
drives `BATES_PARAM_ORDER`/`BATES_LOW`/`BATES_HIGH` and the unpack. (2) The `BatesProcess(...)`
constructor takes `(..., v0, kappa, theta, eta, rho, lambda, nu, delta)`, driving the seed expansion.
(3) `vanp.bates_price(...)`/`df_bates_price` arg order (handled in `src/pricing`). `feller` stays
the Heston-diffusion quantity (jumps do not enter it; reported, never gates). **Acceptance gate:** IV-RMSE
≤ `IV_RMSE_ACCEPT` and no **Heston** param pegged — the pegging check runs on `params[:5]` only; the
jump triple is **exempt** (`lambda≈0` is a legitimate Heston collapse, and `nu`/`delta` are unidentified
when `lambda≈0`, so they often park on a bound without meaning). Bates bounds/seed are
`BATES_BOUNDS`/`BATES_LOW`/`BATES_HIGH`/`BATES_JUMP_SEED` in `config.py`.

## DataFrame column contracts (the "hard-coded names" the README warns about)

Stages communicate through column names, not typed interfaces. Renaming any of these silently
breaks a downstream stage:

- cleaned OTM snapshot schema (in-memory, from `utils._prepare_options`): `quote_datetime, strike_price, w, trade_size, trade_price, trade_iv, spot_price, days_to_maturity`.
- `results/<model>/calibrations/<objective>/calibrations.csv` schema (**single file, one row per trading day**, keyed by `date`):
  `spot_price` (= `S_ref`), `risk_free_rate, dividend_rate, theta, kappa, rho, eta, v0, feller,
  iv_rmse, rmse, n_helpers, accepted, n_maturities, n_strikes, contracts_count, total_volume,
  spot_min, spot_max, spot_range_pct, high_move, calculation_date`. **Bates** appends three columns
  after `v0`: `lambda_, nu, delta` (inserted by `_EXTRA_PARAMS["bates"]`).
- `results/<model>/calibrations/<objective>/rejections.csv` schema (**single file, one row per rejected trading day**, keyed by `date` —
  the complement of `calibrations.csv`): `reason` (category: `no_trades, no_rate, thin, pegged,
  iv_miss, no_fit`), `detail` (human string), `iv_rmse` (NaN unless calibration ran), `n_maturities,
  n_strikes, n_cells` (NaN unless a surface was built). `date` here is the filename date string.
- `calibration_tests/*.csv`: the day's repriced surface contracts (original `spot_price`/`strike_price`,
  plus `Kstar`, the fitted params, `volatility` (= `trade_iv`), `black_scholes`, and the model price
  column — `heston` (Heston) or `bates` (Bates, which also carries the `lambda_, nu, delta` columns)).
- `utils.df_moneyness(df)` needs `w, spot_price, strike_price` (returns ratio moneyness: `spot/strike` for calls, `strike/spot` for puts; `< 1` => OTM).
- `vanp.df_numpy_black_scholes(df)` needs `spot_price, strike_price, days_to_maturity, risk_free_rate, volatility, w`
  (note: `trade_iv` is renamed to `volatility` before this call).
- `vanp.df_heston_price(df)` needs `spot_price, strike_price, days_to_maturity, risk_free_rate, dividend_rate, w, kappa, theta, rho, eta, v0`.
- `vanp.df_bates_price(df)` needs the `df_heston_price` columns plus `lambda_, nu, delta` (added to
  `src/pricing` for the Bates model price column).

## Known issues & fragility (verify before trusting outputs)

- Output routing comes from `config.calib_paths(model, objective)` (wrapped by `_objective_paths`):
  everything lives under `results/<model>/calibrations/<objective>/` (calibrations.csv, rejections.csv,
  validation.csv, and the per-day `calibration_tests/` files, basename
  `cboe_spx_calibration_tests_<date>.csv` with `<date>` sliced from the OTM filename). The calibrator and
  `validate_calibrations.py` build this path from the same `calib_paths` rule; if the two drift apart the
  validator stops finding the tests files. The Heston tree was migrated from the old
  `results/calibrations/<objective>/` to `results/heston/...`; `validate_calibrations.py` is model-aware
  (its `MODEL` constant feeds the same `calib_paths` rule and reads the model's `heston`/`bates` price
  column).
- Moneyness normalisation assumes **sticky-moneyness** (IV ~stationary in `K/S` over a session). It
  is mild on normal days (~1% intraday range) but strained on large-move days; those are flagged
  `high_move` (range > `MAX_MOVE_PCT`=3%) and still written — treat their `S_ref` with suspicion.
- Snapping `Kstar` to the 5-point SPX grid is exact near the money but coarser in the far wings
  (native grid widens to 25/50/100); harmless for QuantLib (any float strike prices) but it slightly
  quantises deep-OTM moneyness.
- **Coverage was widened and a deep-OTM floor added (current config).** `MAX_NK`=40 (was 8), `MAX_NT`=20
  (was 12), `MIN_DTM`=14 (was 29/7), `MAX_DTM`=730 (was 400), plus `OTM_MONEYNESS_FLOOR`=0.6. Putting the
  wings *into* the calibration (not just near-money) and dropping the deep lottery-ticket tail removed what
  looked like a systematic "model underestimates the wings": on the **full-sample `vol` run**
  (`src/wing_residuals.py`) the per-`|log-moneyness|` residual is small and mixed-sign (bands within ±0.009,
  overall RMSE ~0.011, mean +0.0015 vol pts), so the wing underfit seen in the old smile plots was largely an
  **extrapolation artefact** of near-money-only calibration, not an in-sample bias. The cost: the widening
  lifts `eta` and pushes Feller violation to 100% of accepted days (see the Feller bullet).
- **Boundary pegging is the open Phase 3 lever — and `calibrations.csv` cannot show it.** Current committed
  default (**`vol`** objective, widened config, full sample, `results/heston/calibrations/vol/`): **3217** attempted,
  **1631 (50.7%)** accepted, spanning 2012-01-03..2024-10-15. The accepted set has 0 pegged `kappa`/`rho`, but
  that is **tautological**: the gate (`_on_boundary`) rejects any boundary-pegged fit, so pegged days never
  reach the file. Rejection cause is logged to `results/heston/calibrations/vol/rejections.csv`
  (`reason` ∈ `no_trades/no_rate/thin/pegged/iv_miss/no_fit`); the split is **pegged 1467, iv_miss 116,
  no_trades 3** — pegging the dominant cause (92.5% of rejections, 45.6% of attempted). `MIN_DTM`=14 (Lever A)
  is in; `weights` (Lever B) is wired but default-off and tested null; `fixParameters` and the Feller penalty
  (Levers C/D) remain unimplemented. (For reference, the prior **`price`** baseline at the *old narrow* config
  — `MAX_NK`=8, `MIN_DTM`=29 — accepted 1713/3215 (~53%); not a controlled comparison, the coverage differs.)
- **The objective knob is not a lever; it does not move pegging.** `vol` (the default) and `price` change only
  what LM minimises, not the `(kappa, rho, eta)` degeneracy that pegs — selection and the gate always run off
  IV-RMSE. On the current `vol` run the accepted-set IV-RMSE is tight (median 0.0080, p90 0.0115, max 0.0199
  vol pts). One artefact of the `vol` default: it no longer controls relative price, so the returned `rmse`
  blows up on cheap deep-OTM wing cells (max 120.8; **656 of 1631** accepted days > 0.2). The identification
  fixes (anchor `kappa`, Feller penalty) are needed regardless of objective. `src/results/tables/objective_comparison.py`
  builds a side-by-side `price` vs `vol` metrics table when both objectives are present on disk.
- **Feller is the standout issue in the accepted set.** The gate does **not** reject on Feller (a
  *suspicious*, not hard-reject, validator flag — short-tenor Heston violates it routinely), so accepted days
  violate it. On the current full-sample `vol` run `feller = 2·kappa·theta − eta² < 0` on **all 1631** accepted
  days (median −1.44), and `eta > 1.5` on **596 (36.5%)**, max ≈1.997 (near its 2.0 cap). The widened coverage
  is the driver: fitting the full wings demands more vol-of-vol, so both `eta` and the Feller violation run
  higher than the old near-money `price` baseline (`eta` median ≈0.89, `eta>1.5` ~8.5%, Feller<0 on 99.3%).
  This is PLAN.md Lever D (soft Feller penalty + revisit the `eta` cap).
- **Bates is wired and pilot-verified, not yet a full baseline.** `--MODEL bates` runs end-to-end
  (engine `src/calibrate_bates.py`, `df_bates_price` in `src/pricing`, routing to
  `results/bates/calibrations/<objective>/`). On a **100-day pilot** (`--LIMIT 100`, the 2024-05..10
  window) Bates accepted **94/100** vs Heston **63/100** on the same days, fit **35% tighter** in
  IV-RMSE on the shared days, and roughly **halved `eta`** (1.17 -> 0.52: jumps absorb the tail the
  Heston vol-of-vol was overfitting); Feller stays violated. The weakly-identified `nu`/`delta` park on
  their bounds (gate-exempt), a sign those two bounds are tight. The downstream consumers are now
  **model-aware** (a `MODEL` constant each; `validate_calibrations.py`, `example_surface.py`,
  `smiles.py`, `make_eps.py`, `objective_comparison.py`). **Not yet done:** a full multi-year Bates run,
  and an optional `nu`/`delta` bound widening. See `PLAN-Bates.md`.
- The `data/__pycache__/` holds bytecode for deleted modules (`get_data`, `get_options`, ...) — ignore it.

## Writing prose (`heston-calibration.tex` and other `.tex` documents)

When you write or edit prose in `heston-calibration.tex` or any other `.tex` document here, write it the way a careful human author would, not the way an LLM defaults to. Concretely:

- **Avoid the em dash (`—`) as a sentence connector.** It is the single clearest tell of machine-written prose, and the existing text overuses it. Prefer a period, a comma, a colon, or parentheses, and rephrase so the dash is not needed. Do not replace one em dash with another piece of dashy punctuation (en dash, double hyphen) doing the same job; restructure the sentence instead. (Genuine ranges like `12–31×` and `1…8192` keep their en dash/ellipsis — this is about prose connectors, not numerics.)
- **Keep sentences short and digestible.** One idea per sentence. Break a long sentence into two or three rather than stacking clauses with dashes, semicolons, and nested parentheticals. If a sentence needs more than one comma-separated aside to parse, split it.
- **Prefer plain, direct phrasing over ornate constructions.** Say "the GPU is faster" rather than "the GPU exhibits a marked performance advantage." Cut filler ("it is worth noting that", "importantly", "in order to"), hedging stacks, and rule-of-three flourishes that exist only for rhythm.
- **Match the surrounding voice.** This is a technical paper: declarative, specific, quantitative. State the result and the number; let the data carry the emphasis instead of intensifiers.
- **Read it back as a human.** Before finishing, reread each edited sentence aloud in your head. If it sounds like a generated abstract or could not have been said plainly by a person, rewrite it.

These rules apply to *new and edited* prose. Do not launch a sweeping em-dash-removal pass over untouched paragraphs unless asked, but do clean up the dashes and over-long sentences in any passage you are already editing.