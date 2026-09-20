# CLAUDE.md — `src/`

Mechanism reference for the calibration pipeline that lives under `src/`. It loads when you work
with files in this directory. The repo-root `CLAUDE.md` carries the cross-cutting rules (git
hygiene, environment, how to run, known issues, prose style); `PLAN.md` carries the work tracking.

## Pipeline architecture

Data flows left-to-right. `raw/` holds one CSV per trading day (git-ignored); per-day params accumulate
into a **single** `results/<model>/calibrations/<objective>/calibrations.csv` (one row/day), while the
bulky per-day repricing diagnostics stay one-file-per-day under the sibling `calibration_tests/`:

```plain
raw/  --calibrator_prototype.py --MODEL {heston,bates} (prepare_surface._prepare_options cleans to OTM in-memory)-->  calibrations.csv  + calibration_tests/
```

**Stage 1 — market rates (`data/get_rg.py`).** Imported for its side effect: building a module-level
DataFrame `rg`. Parses two hard-coded filenames in `data/market/`: `historical_USGG12M.csv` → US 12M
Treasury yield → `risk_free_rate`; `historical_SPX_ivols.csv` → SPX `spot_price`, 12M `dividend_rate`,
and a term structure of `<tenor>_vol` columns. `rg` is indexed by date, **sorted descending
(newest first)**, values divided to decimals. Downstream consumes only `risk_free_rate` and
`dividend_rate` (the `_vol` columns and `rg`'s `spot_price` are computed but unused — spot comes from the
options data).

**OTM cleaning (`src/prepare_surface._prepare_options`).** Not a standalone stage; the calibrator calls
it in-memory on each raw file. Selects/renames a column subset, uses `underlying_bid` as `spot_price`,
maps `option_type` C/P → `w`, computes `days_to_maturity` (calendar days, `>0`), keeps positive
IV/spot/strike, then keeps the OTM band via `_utils.df_moneyness`: `OTM_MONEYNESS_FLOOR < ratio moneyness
< OTM_MONEYNESS_CUTOFF` (0.6 and 0.98 in `config.py`). The CUTOFF drops near-ATM rows (keeps only OTM);
the **FLOOR drops the deep-OTM lottery-ticket tail** (ratio moneyness = `e^-|log(K/S)|`, so 0.6 keeps
`|log-moneyness| < ~0.51`). Those far-OTM strikes have extreme prices that peg the fit to its bounds;
flooring them recovers acceptance and tightens IV-RMSE without losing the tradeable wing.

**Stage 2 — orchestration (`src/calibrator_prototype.py`, `calibrate_by_day`).** The non-obvious core.
For each raw file it does **one calibration per trading day** over a pooled, moneyness-normalised surface
(not the old per-0.5-spot-bucket fits):

1. Read + clean trades (`prepare_surface._prepare_options`); keep `trade_iv > 0` **and** `MIN_DTM <=
   days_to_maturity <= MAX_DTM` (`MIN_DTM`=14, `MAX_DTM`=730). Ultra-short maturities are dropped: Heston
   fits them poorly and they drive `eta`/`kappa` to Feller-violating extremes.
2. Look up `r`, `g` from `rg` for the file's quote date (NaN-guarded).
3. **Reference spot.** Compute one volume-weighted `S_ref` for the day. Record the intraday spot range;
   if it exceeds `MAX_MOVE_PCT` (=3%) set `high_move=True` and warn (sticky-moneyness re-centring is
   strained on large-move days) — the day is still written.
4. **Moneyness normalisation.** Each trade keeps its moneyness `m = strike / spot_row` but is re-struck
   to `K* = m * S_ref` and **snapped to the SPX 5-point grid** (`STRIKE_GRID`), so trades at different
   intraday spots share clean surface columns (`Kstar`).
5. **Surface.** Walk maturities in descending traded volume, keep one only if **both** wings carry `>=
   MIN_NK`(2) distinct `Kstar` strikes; stop once `MAX_NT`=20 wing-qualifying maturities are collected (a
   thin-wing maturity no longer consumes a slot). For each kept maturity take the `MAX_NK`=40
   nearest-money `Kstar` per wing; `pivot_table` into a `Kstar`×maturity IV surface (`values='trade_iv'`).
   When several trades share a cell the **highest-volume** trade's IV is kept (`sel` sorted by
   `trade_size`, then `aggfunc='last'`). Require `>= MIN_MATS`(3) maturities, `>= MIN_STRIKES`(5) strikes,
   `>= MIN_CELLS`(12) non-NaN cells.
6. Call the engine **once for the whole day** — `calibrate(MODEL, surface, S_ref, r, g)`
   (`_calibration_engine`, dispatched on the model name). The engine **rejects** fits it
   cannot trust (returns `None` params — see Stage 3), printing whether the rejection was a thin surface,
   an IV-RMSE miss, or a **boundary-pegged** param.
7. **On accept** `calibrate_by_day` *returns* the day's **one row keyed by date** (`S_ref` as
   `spot_price`, `r`, `g`, the params in `config.MODELS[MODEL]["params_order"]` — five for Heston,
   plus `nu, delta, lambda_` for Bates — `feller`, `iv_rmse`, `rmse`, coverage counts, intraday spot range,
   `high_move`) and writes the repriced surface contracts to
   `calibration_tests/cboe_spx_calibration_tests_<date>.csv`. Repricing uses each contract's **original**
   `spot_price`/`strike_price` (Heston params are spot-independent), not `S_ref`/`Kstar`. A rejected or
   too-thin day returns `None` and **removes** any stale per-day tests file.
8. The driver consumes the returned rows (all, or `--LIMIT N` most recent) as workers finish (joblib
   `return_as="generator_unordered"`) and **splits** them: accepted rows (no `reason` key) →
   `calibrations.csv`, rejected rows (each carries a `reason`) → `rejections.csv`. See the write/resume
   model below.

**Incremental write + resume (Stage 2 driver).** Both CSVs are **appended incrementally** from the single
main process (`_append_row`) the moment each day completes, so both are **readable mid-run** (in
worker-completion order); workers never touch the files (they only return the row dict), so the serial
write needs no locking and rows cannot interleave. At run end each file is **rewritten sorted by date**,
**merged** with whatever it already held (dedup by date, this run wins), so old rows are never clobbered;
an empty merged set removes the file. Because an accepted row is returned exactly when a tests file is
written (and the worker writes the tests file *before* returning the row), `calibrations.csv` and the
per-day tests files always describe the same accepted set. **Resume is automatic** when **either**
`calibrations.csv` **or** `rejections.csv` already exists: the driver reads the `date` column of
whichever exists and **skips every raw file whose date is already covered** (matched by date, not a
positional slice — workers finish out of order, so done days are not a contiguous prefix). On resume each
existing file suppresses its CSV header on the mid-run flush; a file absent at resume gets its header from
this run's first matching row. A resume **aborts** (`RuntimeError`) if the live `config.as_dict()` differs
from the `config_spec.json` snapshot (comparison JSON-normalised both sides; the error names the differing
keys), or if that snapshot is missing. It also aborts on this run's **first header-less append** to a
resumed file whose **header column order** differs from what this run writes (`_append_row`'s check,
actual row vs actual on-disk header) — a code-level layout change the config-spec guard cannot see would
otherwise silently misalign appended values against the header; such a file (e.g. one written before the
param columns were unified on `params_order`) must be deleted, not resumed. **To start fresh, delete the
`results/<model>/calibrations/<objective>/` files manually.** The end-of-run writes go through
`_write_blocking`: a locked target (e.g. open in Excel) raises `PermissionError` and the run prompts with
`input()` to retry; the mid-run flush is best-effort by contrast (a momentary lock is warned and skipped,
the row still lands in the end-of-run rewrite).

**Ctrl-C is a two-stage graceful interrupt.** The **first** press sets a shared stop flag (a
`multiprocessing` `Manager().Event()` the workers poll) that keeps any **not-yet-started** day from
beginning, but every **in-flight** worker runs to completion and writes its row, so no day is left
half-done. Workers **ignore `SIGINT`** (`calibrate_by_day` sets `SIG_IGN` when invoked with a
`stop_event`, gated on `multiprocessing.parent_process()` so the inline `MAX_JOBS=1` case stays
interruptible), and the Manager server ignores it too (`_ignore_sigint`), so the flag survives. The
**second** press warns and **force-aborts** whatever is still in flight. Either way execution falls
through to the end-of-run block, so the date-sorted CSVs + `config_spec.json` are still written (merged
with prior rows) from the days completed so far. The skipped days are left for the next resume.

The per-day **tests** path is built from `_objective_paths(MODEL, OBJECTIVE)` (a thin wrapper over
`config.calib_paths`): the tests directory plus basename `cboe_spx_calibration_tests_<date>.csv` (same
basename across model/objective — already separated by directory), `<date>` sliced from the OTM filename.
`validate_calibrations.py` rebuilds the identical path from the same `calib_paths` rule, so the two stay
in lock-step.

**Stage 3 — calibration engine (`src/_calibration_engine.py`).** Pure function `calibrate(model,
vol_matrix, s, r, g) -> dict`, one engine for both models. `_resolve_spec(model)` pairs the
`config.MODELS[model]` parameter data with a per-model `_WIRING` entry (the live QuantLib builders
`make_process`/`make_model`/`make_engine`) into a cached `ModelSpec`; the rest is model-agnostic. It
builds the process/model/engine (Heston: `HestonProcess`/`HestonModel`/`AnalyticHestonEngine`; Bates:
`BatesProcess`/`BatesModel`/`BatesEngine`) and one `HestonModelHelper` per non-NaN surface cell (maturity
as `Period(days, Days)`, `config.calendar()` = NYSE, `Date.todaysDate()` as eval date — immaterial under
the flat-forward curves built by `_qu._term_structures` on `config.day_count()` = `Actual365Fixed`). Then:

1. **Multiple restarts:** for each starting point in a data-seeded grid (`_engine_common._seed_grid`,
   name→value seed dicts), calibrate with Levenberg–Marquardt under **box bounds**
   (`ql.NonhomogeneousBoundaryConstraint(spec.low, spec.high)`), keep the
   fit with the lowest **IV-space RMSE**. The LM objective is switchable via `objective` (`_ERR` maps it
   to the helper error type): `"vol"` (`ImpliedVolError`, default) or `"price"` (`RelativePriceError`).
   This changes only what each restart minimises; selection and the gate always use IV-RMSE. `"vol"` is
   more expensive (a Black-vol inversion per residual per LM iteration) and can throw mid-search (caught
   per-restart). **Wing weighting (Levers B/G, wired default-off):** `_calibrate_once` can re-weight OTM
   wing cells by `_wing_weight(k, s) = 1 + WING_WEIGHT_GAIN*(|log(k/S_ref)|/SCALE)**POWER`, engaged for
   **any nonzero `WING_WEIGHT_GAIN`** (positive up-weights the wings, negative down-weights them, clamped
   by `WING_WEIGHT_FLOOR`) and applied **only under `"vol"`** (the `"price"` denominator already
   up-weights cheap wings). The weights are passed to QuantLib as the 5th positional `weights` arg of
   `model.calibrate(...)` — a **plain python list**, not `ql.Array`. When on, restart ranking uses the
   wing-weighted IV-RMSE (`iv_rmse_sel`) while the gate and reported `iv_rmse` stay **unweighted**
   (`iv_rmse_gate`). `WING_WEIGHT_GAIN=0` (default) is an exact no-op vs the pre-lever engine.
2. **IV-space error (the gate metric).** Each helper's fitted price is inverted back to a Black vol via
   `BlackCalibrationHelper.impliedVolatility(modelValue, ...)` and compared to the market vol that built
   it; the RMSE is in **vol points**. This replaced the old relative-price gate that deep-OTM wings
   inflated. The relative-price RMSE is still computed and returned as `rmse`, but no longer gates.
3. **Acceptance gate:** returns the failure sentinel if the best **IV-RMSE** exceeds `IV_RMSE_ACCEPT`
   (`0.02`, ~2 vol points) **or** any parameter is pinned within `BOUND_TOL` of a bound. On the pooled
   surfaces the genuine fit is excellent (accepted days ~0.8 vol-point median IV-RMSE), so IV-RMSE
   essentially never gates — the binding rejection is **boundary pegging**. Accepted days are pegging-free
   *by construction*, so a clean `kappa`/`rho` in `calibrations.csv` is not evidence pegging is solved (see
   Known issues).

Returns the fitted params keyed by name in `spec.params_order` (`dict(zip(spec.params_order,
model.params()))`) plus `feller = 2*kappa*theta - eta**2`, `iv_rmse`, `rmse`, `n_helpers`, `accepted`; a
rejected fit nulls the params/`feller` but keeps `iv_rmse`/`rmse`/`n_helpers`/`accepted=False`. **Param
order matters:** the orders are declared once per model in `config.MODELS` — `params_order` (QuantLib
`model.params()` order, e.g. Heston `("theta","kappa","eta","rho","v0")`) drives `low`/`high`, the unpack
and the anchor names; `ctor_order` drives the seeds. The engine imports
`IV_RMSE_ACCEPT`/`WING_WEIGHT_GAIN`/the optimizer args (`LM_ARGS`, `END_CRITERIA_ARGS`) from `config.py`
and reads the per-model `bounds`/orders/`jump_seed` from `config.MODELS`; only the `_ERR`
string→QuantLib-enum map and the `_WIRING` builders stay in `_calibration_engine.py`.

**Restart-selection score (Phase-3 levers, all default-off so the baseline reproduces byte-for-byte).**
Each restart is ranked by `iv_rmse_sel + FELLER_PENALTY*_feller_violation(params) +
PARAM_ANCHOR_WEIGHT*_anchor_distance(params, anchor, ...)`; the **gate and reported `iv_rmse` stay the
unweighted IV-RMSE**. `FELLER_PENALTY` (Lever D) biases toward Feller-compliant fits;
`PARAM_ANCHOR_WEIGHT`/`PARAM_ANCHOR_LOOKBACK` (Lever E) softly anchor a day to the median of its last
`PARAM_ANCHOR_LOOKBACK` accepted days, read **in-flight** from the live accepted pool (seeded from
`calibrations.csv` on resume); `PARAM_ANCHOR_WEIGHT > 0` forces a **strictly sequential** run (see Driver
flags). `WING_WEIGHT_GAIN < 0` (Lever G) de-emphasises the wings, clamped positive by
`WING_WEIGHT_FLOOR`. See `PLAN.md` for the lever rationale and sweep findings.

**Shared engine helpers (`src/_engine_common.py`).** Model-agnostic, QuantLib-free helpers the single
engine builds on: `_on_boundary(params, low, high)` (takes its `low`/`high` so a caller can gate a
parameter subset — Bates gates only the 5 Heston params via `spec.gate_names`), `_seed_var`,
`_seed_grid(vol_matrix, jump_seed=None)` (the restart grid as name→value dicts; appends the Bates jump
seed when given), `_anchor_seed(anchor, ctor_order)` (the warm-start dict, or None), `_wing_weight`,
`_iv_rmse`, `_feller_violation(params)` (Feller shortfall `max(0, eta² − 2·kappa·theta)` from a
`params()` vector), and `_anchor_distance(params, anchor, names, bounds)` (span-normalised squared
distance to a prior-day anchor). Behaviour-neutral vs the former two engines (committed `vol` numbers
reproduce to full float precision).

**Stage 3b — Bates specifics (same engine).** `--MODEL bates` runs the same `calibrate`; the `bates`
`_WIRING` swaps `HestonProcess`/`HestonModel`/`AnalyticHestonEngine` for
`BatesProcess`/`BatesModel`/`BatesEngine`. The helper stays `ql.HestonModelHelper` (there is **no
`ql.BatesHelper`** in QuantLib 1.35); only the attached pricing engine is a `BatesEngine`. The result
dict is a **superset** of the Heston one (same keys plus `lambda_, nu, delta`), read unchanged by the
orchestrator. **THREE distinct orderings (declared in `config.MODELS["bates"]`, confirmed live, do not
conflate):** (1) `params_order` = `BatesModel.params()` = `[theta, kappa, eta, rho, v0, nu, delta,
lambda]` — drives `low`/`high` and the unpack; (2) `ctor_order` = `BatesProcess(..., v0, kappa, theta,
eta, rho, lambda, nu, delta)` — drives the seed/jump expansion; (3) `vanp.bates_price(...)`/`df_bates_price`
arg order (handled in `src/pricing`). `feller` stays the Heston-diffusion quantity (jumps do not enter
it; reported, never gates). **Acceptance gate:** IV-RMSE ≤ `IV_RMSE_ACCEPT` and no **Heston** param
pegged — the gate runs on `spec.gate_names` (the five Heston params) only; the jump triple is **exempt**
(`lambda≈0` is a legitimate Heston collapse, and `nu`/`delta` are unidentified when `lambda≈0`). The
Bates `bounds`/`gate_names`/`jump_seed` live in `config.MODELS["bates"]`.

## DataFrame column contracts (the "hard-coded names" the README warns about)

Stages communicate through column names, not typed interfaces. Renaming any of these silently breaks a
downstream stage:

- cleaned OTM snapshot schema (in-memory, from `prepare_surface._prepare_options`): `quote_datetime,
  strike_price, w, trade_size, trade_price, trade_iv, spot_price, days_to_maturity`.
- `calibrations.csv` (**single file, one row per trading day**, keyed by `date`): `spot_price` (=
  `S_ref`), `risk_free_rate, dividend_rate, theta, kappa, eta, rho, v0, feller, iv_rmse, rmse, n_helpers,
  accepted, n_maturities, n_strikes, contracts_count, total_volume, spot_min, spot_max, spot_range_pct,
  high_move, calculation_date`. **Bates** appends three columns after `v0`: `nu, delta, lambda_`. The
  param columns follow `config.MODELS[model]["params_order"]` (QuantLib `model.params()` order); the
  committed baseline CSVs still carry the pre-unification order (`theta, kappa, rho, eta, v0` and Bates
  `lambda_, nu, delta`) and cannot be resumed onto — the header check aborts (delete to regenerate).
- `rejections.csv` (one row per rejected day, keyed by `date` — the complement of `calibrations.csv`):
  `reason` (`no_trades, no_rate, thin, pegged, iv_miss, no_fit`), `detail` (human string), `iv_rmse` (NaN
  unless calibration ran), `n_maturities, n_strikes, n_cells` (NaN unless a surface was built). `date`
  here is the filename date string.
- `calibration_tests/*.csv`: the day's repriced surface contracts (original `spot_price`/`strike_price`,
  plus `Kstar`, the fitted params, `volatility` (= `trade_iv`), `black_scholes`, and the model price
  column — `heston` or `bates` (Bates also carries `lambda_, nu, delta`)).
- `_utils.df_moneyness(df)` needs `w, spot_price, strike_price` (returns ratio moneyness: `spot/strike`
  for calls, `strike/spot` for puts; `< 1` => OTM).
- `vanp.df_numpy_black_scholes(df)` needs `spot_price, strike_price, days_to_maturity, risk_free_rate,
  volatility, w` (note: `trade_iv` is renamed to `volatility` before this call).
- `vanp.df_heston_price(df)` needs `spot_price, strike_price, days_to_maturity, risk_free_rate,
  dividend_rate, w, kappa, theta, rho, eta, v0`.
- `vanp.df_bates_price(df)` needs the `df_heston_price` columns plus `lambda_, nu, delta`.
