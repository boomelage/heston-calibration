# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
It is the **operational** reference (what is where, the column contracts, the gotchas). The **work
tracking** (phases, the Phase-3 mitigation levers, completed-task forensics, baseline result numbers)
lives in `PLAN.md`; the **user-facing** overview (how to run on a fresh clone, why data is untracked) is
in `README.md`. Do not duplicate those here — point to them.

## Git hygiene

Never include Claude-Session links, claude.ai URLs, or any other references that reveal a connection
to Claude or Anthropic tooling in commit messages, comments, or any tracked file. Strip such
references before committing if they appear in generated content.

## Maintaining this file (and `PLAN.md`)

Keep this document in sync with the code as you work. When a change alters anything described here —
files moved or renamed, column contracts changed, a "Known issue" fixed or a newly found one, run
commands or stages changed — update CLAUDE.md in the **same** change: add what is now true and delete
what is now stale. A stale line here is worse than a missing one. Do not leave fixed issues marked
"done"; remove them. Keep it lean — it has a 40K-character budget.

## Purpose

Calibrate Heston (1993) stochastic-volatility parameters (`v0, kappa, theta, eta, rho`) to
CBOE S&P 500 (SPX) intraday option trades, using QuantLib. `eta` is the vol-of-vol (QuantLib's
`sigma`). The model SDE is documented in `skew-calibration.tex` (repo root):

```plain
dX_t = (r - v_t/2) dt + sqrt(v_t)(rho dW_t + sqrt(1-rho^2) dB_t)
dv_t = kappa(theta - v_t) dt + eta sqrt(v_t) dW_t
```

A **Bates (1996) variant** is also supported (Heston stochastic vol + Merton lognormal jumps: the five
Heston params plus jump intensity `lambda_`, mean log-jump `nu`, log-jump std `delta`). It is selected
by the `--MODEL {heston,bates}` flag (default `heston`) and runs through the same orchestrator; engine
`src/calibrate_bates.py`. Its design and pilot results are in `PLAN.md` (Bates extension, PR #12).

Treat the current scripts as a working prototype, not a clean design: the workflow is convoluted and
over-reliant on passing intermediate CSVs between stages with hard-coded column names (see README).

## Environment & dependencies

- **Python 3.12**, **QuantLib 1.35**. Also: `pandas`, `numpy`, `scipy`, `joblib`. No `requirements.txt`,
  `setup.py`, lockfile, or test suite.
- The QuantLib pricing wrapper is **vendored in-repo** at `src/pricing/` (formerly the author's external
  `quantlib_pricers` package). It is a plain directory with no `__init__.py`, so `pricing` is a namespace
  package with three modules: `vanilla_pricer.py` (class `vanilla_pricer`, used as `vanp =
  vanilla_pricer()`), `_quantlib_config.py` (QuantLib date conventions), and `_quantlib_utils.py` (class
  `_quantlib_utils`, the single home of all QuantLib process/engine/option construction). The upstream
  asian/barrier pricers were dropped. Import as `from pricing.vanilla_pricer import vanilla_pricer`
  **after** `src` is added to `sys.path`.

## How to run

The pipeline is three stages; run scripts directly (no build/lint/test tooling).

**Single-sourced config.** All model/calibration constants live in `src/config.py` — tune there, not in
the modules: surface-coverage knobs, box bounds, the acceptance gate, the OTM filter floor/cutoff, the
wing-weight knobs, the seed grid, the Bates bounds/seed, and the Phase-3 mitigation levers
(`HESTON_INTEGRATION`/`BATES_INTEGRATION`, `FELLER_PENALTY`/`FELLER_SEED_TEMPLATE`,
`PARAM_ANCHOR_WEIGHT`/`PARAM_ANCHOR_LOOKBACK`, `WING_WEIGHT_FLOOR`). The QuantLib **date conventions**
(`Actual365Fixed` day count, `UnitedStates.NYSE` calendar) live in `src/pricing/_quantlib_config.py`
(`day_count(name=None)`/`calendar(name=None)`, named choices `DAY_COUNT_NAME`/`CALENDAR_NAME`) and are
**re-exported by `config`**, so `config.day_count`/`config.calendar` keep working as the facade.

**QuantLib construction is centralized** in `src/pricing/_quantlib_utils.py` (`_quantlib_utils`): both
engines build their `HestonProcess`/`BatesProcess` via `_qu.heston_process`/`_qu.bates_process`;
`_utils.build_heston_engine`/`build_bates_engine` are thin wrappers over
`_qu._heston_engine`/`_qu._bates_engine` (which return `(engine, s_handle, ts_r, ts_g, day_count)`); and
`vanilla_pricer` prices through `_qu._{heston,mc_heston,bates}_engine` + `_qu._european_option`. A
constructor-order change is a one-line edit there. The **pricing engine itself** is built in one place:
`_qu.heston_engine_for(model)`/`_qu.bates_engine_for(model)` apply the CF-integration accuracy
(`config.HESTON_INTEGRATION`/`BATES_INTEGRATION`, default = QuantLib order-144 Gauss-Laguerre) and are
used by **both** the calibration fit and the repricing/IV-inversion path, so fit and diagnostics
integrate identically.

**Driver flags** (`src/calibrator_prototype.py`): calibrates **every** raw file in `data/options/raw/`
by default; `--LIMIT N` restricts to the `N` most recent trading days, `--MAX_JOBS N` sets the joblib
worker count (default `max(1, os.cpu_count() // 4)`, one day per worker), `--MODEL {heston,bates}`
(default `heston`), `--OBJECTIVE {price,vol}` (default `vol`, `config.DEFAULT_OBJECTIVE`). A full Heston
`vol` run over the multi-year sample is a few thousand days / a few hours (multi-start LM, ~1,500
cells/day); a Bates run is ~4.5x slower per day.

**Data not in version control** (full rationale + fresh-clone bootstrap in README): `data/options/raw/`
(~80–90 MB/day) and the per-day `results/*/calibrations/*/calibration_tests/` are git-ignored (only
`.gitkeep` kept). Only the small derived artefacts are tracked: for all four model×objective combos
(`results/{heston,bates}/calibrations/{vol,price}/`) the `{calibrations,rejections}.csv` pair plus the
`config_spec.json` run snapshot (the `vol` trees also carry `validation.csv`), plus `data/market/`. The
committed runs are a **prior baseline pending regeneration** under the new spec — treat their cell counts
and result figures as illustrative, not current (numbers caveat repeated wherever results are quoted; the
single authoritative source is each run's `rejections.csv` + the `_run` header of its `config_spec.json`).

**Output routing is namespaced by model AND objective:** `results/<model>/calibrations/<objective>/`
(`<model>` ∈ `heston, bates`). The single source of truth is `config.calib_paths(model, objective)` (and
`_objective_paths` wraps it); the run snapshot's `config.spec_path(model, objective)` is a sibling (kept
separate so `calib_paths`'s positional 3-tuple contract is untouched). The Heston tree was migrated from
the old `results/calibrations/<objective>/` to `results/heston/calibrations/<objective>/`.

**Run-spec snapshot (`config_spec.json`).** Each run writes a Python-readable JSON snapshot of the exact
config it used next to `calibrations.csv` (`_utils.write_config_spec`, from `calibrator_prototype.main`).
Values come from `config.as_dict()` — every JSON-serializable module-level constant, captured by
reflection so new knobs appear automatically (callables and `Path` objects are skipped; tuples round-trip
as arrays). A `_run` header records `timestamp, git_commit, model, objective, limit`, and the
accept/reject tally. Written **at run start** (so a partial run has a spec to verify a resume against) and
**rewritten at the end** with the final tally; kept whenever either CSV is present, removed only when the
run produces no output. Resume aborts if the live config differs from this snapshot (see Resume).
Downstream LaTeX scripts can `json.load` it to recover a run's bounds/coverage/gate.

**Downstream figure/table scripts** live under `src/results/`. They resolve `REPO =
Path(__file__).parents[2]`, read calibrations from / write figures into the **repo-level**
`results/<model>/` tree, and are **model-aware** via `src/results/_results_config.py` (set `MODEL` to
`heston`/`bates` and `OBJECTIVE` there; it picks the engine, the source/output tree, and figure labels
for all of them at once — the grids `MONEYNESS`/`MATURITIES_DAYS`, smile knobs `NT`/`MKTMONSTEP` (each
`None` = every maturity/strike) and `XLO/XHI`, and surface view `SURFACE_ELEV/AZIM`/figsizes live in the
same file). The scripts:

- `surfaces/make_surface.py` — rebuild a model IV/price surface from one calibration row (its richer
  `day_results` dict carries extra `fit` fields `plot_surfaces.py` consumes).
- `surfaces/plot_surfaces.py` — write the OTM surface/smile EPS + `otm.tex`.
- `smiles/smiles.py` — per-day market-vs-model smile EPS + `smiles.tex`. Does **not** read raw trades:
  model lines come from the calibrated params in `calibrations.csv`, the market scatter from that day's
  `calibration_tests/` file (market IV in the `volatility` column). A missing tests file drops the scatter
  (model lines only); a missing `calibrations.csv` row is a hard error. Shows the Bates jump triple in the
  per-figure caption.
- `tables/objective_comparison.py` — price-vs-vol metrics table (**not** wired to `_results_config`; keeps
  its own `MODEL`; needs both a price and a vol run on disk for the chosen model).
- Read-only graders at the `src/results/` top level (each adds `src/results` to `sys.path`, reads
  `MODEL`/`OBJECTIVE` from `_results_config`): `validate_calibrations.py` (grades `calibrations.csv` +
  `calibration_tests/`; under bates grades against `BATES_BOUNDS` and flags the jump triple),
  `wing_residuals.py` (residual-by-moneyness), `calibration_diagnostics.py` (the Phase-3-lever grading
  instrument — call-wing curvature mismatch `d_curv_call`/`call_convex_frac`, wing-oscillation
  `osc_frac`, kappa-floor proximity / `pegged_rate` by year; writes `diagnostics.csv`),
  `_verify_completeness.py` (audits that every raw day resolves to exactly one row across
  `calibrations.csv` + `rejections.csv` and that each accepted row has its tests file; exits non-zero on
  any discrepancy).

**Shared helpers split by config-dependence.** Model-agnostic, **config-free** helpers live in
`src/_utils.py` (so the calibrator can use them and that module never imports `_results_config`): the
QuantLib helpers (`build_model_engine` → `build_heston_engine`/`build_bates_engine`, plus
`model_price`/`model_implied_vol`) and the pure surface-selection helpers (`_normalize_dates`,
`_clip_maturities`, `_sparse_maturities`, `_sparse_strikes` — all take window/count/step explicitly).
The **results-layer** helpers that read `_results_config` knobs or the `results/<model>/` tree live in
`src/results/_results_utils.py`: `load_calibrations_by_date`, `build_day_engine` (engine + Black-inversion
process build, shared by `smiles.py` + `make_surface.py`), `_load_test_scatter`, `_maturity_colors`,
`_day_from_row`.

**Importable for notebooks (`inspect.ipynb`).** Each script's entry point takes optional `model=None,
objective=None` (defaulting to the `_results_config` switches), resolving `(model, objective)`-dependent
state **inside** the call (CLI/`__main__` path unchanged). Writers take `save=True` (set `False` to skip
disk writes); plot scripts (`plot_surfaces.main`, `smiles.main`, `make_surface.make_surface`) take
`show=False` and **return** their figures; graders return their frames. `plot_surfaces.py` selects `Agg`
only under `__main__`, so a notebook's interactive backend survives the import. `inspect.ipynb` enables
`%autoreload 2`.

There is no single-test command (no tests). To exercise just an engine, import `calibrate_heston(vol_matrix,
s, r, g, objective="vol")` from `src/calibrate_heston.py` (or `calibrate_bates(...)`, same signature) with
a strike×maturity IV DataFrame. `objective` selects the in-engine LM objective; the orchestrator passes
its `OBJECTIVE` constant through.

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
6. Call the selected engine **once for the whole day** — `_ENGINES[MODEL]`, i.e.
   `calibrate_heston(surface, S_ref, r, g)` or `calibrate_bates(...)`. The engine **rejects** fits it
   cannot trust (returns `None` params — see Stage 3), printing whether the rejection was a thin surface,
   an IV-RMSE miss, or a **boundary-pegged** param.
7. **On accept** `calibrate_by_day` *returns* the day's **one row keyed by date** (`S_ref` as
   `spot_price`, `r`, `g`, the params — five for Heston, plus `lambda_, nu, delta` for Bates via
   `_EXTRA_PARAMS[MODEL]` — `feller`, `iv_rmse`, `rmse`, coverage counts, intraday spot range,
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
keys), or if that snapshot is missing. **To start fresh, delete the
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

**Stage 3 — calibration engine (`src/calibrate_heston.py`).** Pure function `calibrate_heston(vol_matrix,
s, r, g) -> dict`. Builds a QuantLib `HestonProcess` (via `_qu.heston_process`) / `HestonModel` with an
`AnalyticHestonEngine` and one `HestonModelHelper` per non-NaN surface cell (maturity as `Period(days,
Days)`, `config.calendar()` = NYSE, `Date.todaysDate()` as eval date — immaterial under the flat-forward
curves built by `_qu._term_structures` on `config.day_count()` = `Actual365Fixed`). Then:

1. **Multiple restarts:** for each starting point in a data-seeded grid (`_seed_grid`), calibrate with
   Levenberg–Marquardt under **box bounds** (`ql.NonhomogeneousBoundaryConstraint(LOW, HIGH)`), keep the
   fit with the lowest **IV-space RMSE**. The LM objective is switchable via `objective` (`_ERR` maps it
   to the helper error type): `"vol"` (`ImpliedVolError`, default) or `"price"` (`RelativePriceError`).
   This changes only what each restart minimises; selection and the gate always use IV-RMSE. `"vol"` is
   more expensive (a Black-vol inversion per residual per LM iteration) and can throw mid-search (caught
   per-restart). **Wing weighting (Lever B, wired default-off):** `_calibrate_once` can up-weight OTM
   wing cells by `_wing_weight(k, s) = 1 + WING_WEIGHT_GAIN*(|log(k/S_ref)|/SCALE)**POWER`, applied
   **only under `"vol"`** (the `"price"` denominator already up-weights cheap wings) and passed to
   QuantLib as the 5th positional `weights` arg of `model.calibrate(...)` — a **plain python list**, not
   `ql.Array`. When on, restart ranking uses the wing-weighted IV-RMSE (`iv_rmse_sel`) while the gate and
   reported `iv_rmse` stay **unweighted** (`iv_rmse_gate`). `WING_WEIGHT_GAIN=0` (default) is an exact
   no-op vs the pre-lever engine.
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

Returns `{theta, kappa, eta, rho, v0, feller, iv_rmse, rmse, n_helpers, accepted}` with `feller =
2*kappa*theta - eta**2` for an accepted fit; a rejected fit returns params/`feller` as `None` but keeps
`iv_rmse`/`rmse`/`n_helpers`/`accepted=False`. **Param order matters:** `model.params()` returns
`[theta, kappa, eta, rho, v0]`. Bounds live in `config.py` as `BOUNDS`; `LOW`/`HIGH` derive as
`[BOUNDS[p][L/H] for p in PARAM_ORDER]` with `PARAM_ORDER = ("theta","kappa","eta","rho","v0")` declaring
that order in exactly one place. The engine imports `LOW`/`HIGH`/`IV_RMSE_ACCEPT`/the seed-grid
template/`WING_WEIGHT_GAIN`/the optimizer args (`LM_ARGS`, `END_CRITERIA_ARGS`) from `config.py`; only
the `_ERR` string→QuantLib-enum map stays in `calibrate_heston.py`.

**Restart-selection score (Phase-3 levers, all default-off so the baseline reproduces byte-for-byte).**
Each restart is ranked by `iv_rmse_sel + FELLER_PENALTY*_feller_violation(params) +
PARAM_ANCHOR_WEIGHT*_anchor_distance(params, anchor, ...)`; the **gate and reported `iv_rmse` stay the
unweighted IV-RMSE**. `FELLER_PENALTY` (Lever D) biases toward Feller-compliant fits;
`PARAM_ANCHOR_WEIGHT`/`PARAM_ANCHOR_LOOKBACK` (Lever E) softly anchor a day to a prior run's params via
the two-pass `--PRIOR_FROM` workflow; `WING_WEIGHT_GAIN < 0` (Lever G) de-emphasises the wings, clamped
positive by `WING_WEIGHT_FLOOR`. See `PLAN.md` for the lever rationale and sweep findings.

**Shared engine helpers (`src/_engine_common.py`).** Model-agnostic helpers factored out so the Heston
and Bates engines reuse identical logic and cannot drift: `_on_boundary(params, low, high)` (takes its
`low`/`high` so a caller can gate a parameter subset — Bates gates only the 5 Heston params), `_seed_var`,
`_wing_weight`, `_iv_rmse`, `_feller_violation(params)` (Feller shortfall `max(0, eta² − 2·kappa·theta)`
from a `params()` vector), and `_anchor_distance(params, anchor, names, bounds)` (span-normalised squared
distance to a prior-day anchor). Behaviour-neutral for Heston (committed `vol` numbers reproduce to full
float precision).

**Stage 3b — Bates engine (`src/calibrate_bates.py`).** Same shape as `calibrate_heston`, swapping
`HestonProcess`/`HestonModel`/`AnalyticHestonEngine` for `BatesProcess`/`BatesModel`/`BatesEngine`. The
helper stays `ql.HestonModelHelper` (there is **no `ql.BatesHelper`** in QuantLib 1.35); only the attached
pricing engine is a `BatesEngine`. Returns a **superset** of the Heston dict (same keys plus `lambda_, nu,
delta`), read unchanged by the orchestrator. **THREE distinct orderings (confirmed live, do not
conflate):** (1) `BatesModel.params()` returns `[theta, kappa, eta, rho, v0, nu, delta, lambda]` —
driving `BATES_PARAM_ORDER`/`BATES_LOW`/`BATES_HIGH` and the unpack; (2) the `BatesProcess(...)`
constructor takes `(..., v0, kappa, theta, eta, rho, lambda, nu, delta)`, driving the seed expansion;
(3) `vanp.bates_price(...)`/`df_bates_price` arg order (handled in `src/pricing`). `feller` stays the
Heston-diffusion quantity (jumps do not enter it; reported, never gates). **Acceptance gate:** IV-RMSE ≤
`IV_RMSE_ACCEPT` and no **Heston** param pegged — the check runs on `params[:5]` only; the jump triple is
**exempt** (`lambda≈0` is a legitimate Heston collapse, and `nu`/`delta` are unidentified when
`lambda≈0`). Bates bounds/seed are `BATES_BOUNDS`/`BATES_LOW`/`BATES_HIGH`/`BATES_JUMP_SEED` in
`config.py`.

## DataFrame column contracts (the "hard-coded names" the README warns about)

Stages communicate through column names, not typed interfaces. Renaming any of these silently breaks a
downstream stage:

- cleaned OTM snapshot schema (in-memory, from `prepare_surface._prepare_options`): `quote_datetime,
  strike_price, w, trade_size, trade_price, trade_iv, spot_price, days_to_maturity`.
- `calibrations.csv` (**single file, one row per trading day**, keyed by `date`): `spot_price` (=
  `S_ref`), `risk_free_rate, dividend_rate, theta, kappa, rho, eta, v0, feller, iv_rmse, rmse, n_helpers,
  accepted, n_maturities, n_strikes, contracts_count, total_volume, spot_min, spot_max, spot_range_pct,
  high_move, calculation_date`. **Bates** appends three columns after `v0`: `lambda_, nu, delta` (via
  `_EXTRA_PARAMS["bates"]`).
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

## Known issues & fragility (verify before trusting outputs)

The Phase-3 work (raising the accept rate, the mitigation levers A–G, the result figures) is owned by
`PLAN.md` — read it for the lever rationale, sweep findings, and pass criteria. The standing operational
gotchas:

- **Output routing must stay in lock-step.** The calibrator and `validate_calibrations.py` build the
  `results/<model>/calibrations/<objective>/` paths from the **same** `config.calib_paths` rule; if they
  drift the validator stops finding the tests files. The validator is model-aware (`MODEL`/`OBJECTIVE`
  from `_results_config`); under bates it grades against `BATES_BOUNDS` and flags the jump triple.
- **Moneyness normalisation assumes sticky-moneyness** (IV ~stationary in `K/S` over a session). Mild on
  normal days (~1% intraday range), strained on large-move days, which are flagged `high_move` (range >
  `MAX_MOVE_PCT`=3%) and still written — treat their `S_ref` with suspicion. Snapping `Kstar` to the
  5-point SPX grid is exact near the money but coarser in the far wings (harmless for QuantLib; slightly
  quantises deep-OTM moneyness).
- **Boundary pegging is the open Phase-3 issue — and `calibrations.csv` cannot show it.** The gate
  (`_on_boundary`) rejects any boundary-pegged fit, so the accepted set's 0 pegged `kappa`/`rho` is
  **tautological**; pegged days never reach the file. Rejection cause is in `rejections.csv` and pegging
  is by far the dominant one. The mitigation levers are wired default-off (so the committed baseline
  reproduces) and graded with `calibration_diagnostics.py`. See `PLAN.md`.
- **The objective knob (`vol`/`price`) is not a lever; it does not move pegging.** It changes only what
  LM minimises, not the `(kappa, rho, eta)` degeneracy — selection and the gate always run off IV-RMSE.
  One `vol` artefact: it no longer controls relative price, so the returned `rmse` blows up on cheap
  deep-OTM wing cells. `objective_comparison.py` builds a `price`-vs-`vol` table when both are on disk.
- **Feller is the standout issue in the accepted set.** The gate does **not** reject on Feller (a
  suspicious, not hard-reject, validator flag — short-tenor Heston violates it routinely), so accepted
  days violate it (essentially `feller < 0` on every accepted `vol` day, a minority at `eta > 1.5`). The
  widened coverage is the driver: fitting the full wings demands more vol-of-vol. The soft Feller penalty
  (`FELLER_PENALTY`) is wired but tested weak; for the wing oscillation the high `eta` produces, the safer
  lever is the CF-integration accuracy (`HESTON_INTEGRATION`/`BATES_INTEGRATION`). See `PLAN.md` (Levers
  D/F).
- **Bates runs end-to-end and is committed for all objectives.** `--MODEL bates` runs the full pipeline
  (engine `src/calibrate_bates.py`, `df_bates_price` in `src/pricing`, routing to `results/bates/...`).
  Against Heston on the same days it accepts more, fits tighter in IV-RMSE, and roughly halves `eta`
  (jumps absorb the tail); Feller stays violated. The weakly-identified `nu`/`delta` park on their bounds
  (gate-exempt); `validate_calibrations.py` surfaces this as suspicious-tier flags under bates. A `nu`/`delta`
  bound widening remains open. See `PLAN.md` (Bates extension, PR #12).
- The `data/__pycache__/` holds bytecode for deleted modules (`get_data`, `get_options`, …) — ignore it.

## Writing prose (`skew-calibration.tex` and other `.tex` documents)

When you write or edit prose in `skew-calibration.tex` (repo root) or any other `.tex` document here, write it the way a careful human author would, not the way an LLM defaults to. Concretely:

- **Avoid the em dash (`—`) as a sentence connector.** It is the single clearest tell of machine-written prose, and the existing text overuses it. Prefer a period, a comma, a colon, or parentheses, and rephrase so the dash is not needed. Do not replace one em dash with another piece of dashy punctuation (en dash, double hyphen) doing the same job; restructure the sentence instead. (Genuine ranges like `12–31×` and `1…8192` keep their en dash/ellipsis — this is about prose connectors, not numerics.)
- **Keep sentences short and digestible.** One idea per sentence. Break a long sentence into two or three rather than stacking clauses with dashes, semicolons, and nested parentheticals. If a sentence needs more than one comma-separated aside to parse, split it.
- **Prefer plain, direct phrasing over ornate constructions.** Say "the GPU is faster" rather than "the GPU exhibits a marked performance advantage." Cut filler ("it is worth noting that", "importantly", "in order to"), hedging stacks, and rule-of-three flourishes that exist only for rhythm.
- **Match the surrounding voice.** This is a technical paper: declarative, specific, quantitative. State the result and the number; let the data carry the emphasis instead of intensifiers.
- **Read it back as a human.** Before finishing, reread each edited sentence aloud in your head. If it sounds like a generated abstract or could not have been said plainly by a person, rewrite it.

**Calibration constants are single-sourced via macros (keep in sync with `config.py`).**
`skew-calibration.tex` defines each surface/selection/gate constant once, near `% ---- tab:const`, as a
pair of macros: `\Const<Role>Sym` (the math symbol) and `\Const<Role>Val` (the value), e.g.
`\ConstMinMaturitySym`/`\ConstMinMaturityVal` for `MIN_DTM`. The prose, equations, the algorithm block,
and Table~`tab:const` all reference these macros, never literal numbers. When you change a constant in
`src/config.py` (or rename its symbol), update the matching `\Const…Val` (or `\Const…Sym`) macro in the
**same** change.

These rules apply to *new and edited* prose. Do not launch a sweeping em-dash-removal pass over untouched paragraphs unless asked, but do clean up the dashes and over-long sentences in any passage you are already editing.
