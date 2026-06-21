# PLAN-Bates.md — adding a Bates (Heston + jumps) calibration engine

This file tracks the work to add a **Bates (1996)** calibration engine alongside the existing Heston
engine, reusing `src/calibrator_prototype.py` with minimal change, and routing Bates outputs into a
**model-namespaced** results tree so Heston and Bates results coexist. Keep it in sync with the code
and with `CLAUDE.md` as items land.

The Bates model is Heston stochastic volatility plus Merton lognormal jumps in the log-price. It adds
three parameters to Heston's five: jump intensity `lambda` (jumps per year), mean log-jump `nu`, and
log-jump volatility `delta`. With `lambda = 0` Bates collapses to pure Heston, so the lower bound on
`lambda` is set to exactly `0`.

## Why both models, not a replacement

A Bates fit is **not** a collapsed-Heston fit in practice. With 8 free parameters and `lambda` free in
`[0, hi]`, the optimizer almost never lands at exactly `lambda = 0`; it uses a small jump to absorb
skew, redistributing it across `rho`/`eta` and the jump triple. On the same surface the shared five
parameters come out different, the repriced diagnostic differs, and accept/reject can flip. So the two
runs are genuinely different result sets and neither reconstructs the other. The value of adding Bates
is the **Heston-vs-Bates comparison**, which needs both result sets on disk at once. That is why result
routing is branched by model (see [Result routing](#result-routing-branched-by-model)).

## Decisions (locked)

- **Engine structure.** New `src/calibrate_bates.py`, with the model-agnostic helpers factored into a
  shared `src/_engine_common.py` that both engines import. The committed Heston path and its faster
  `AnalyticHestonEngine` stay behavior-identical (verify against committed outputs).
- **Repricing.** Add `row_bates_price` + `df_bates_price` to the author's own `quantlib_pricers`
  package (edit at source), mirroring the Heston wrappers. The scalar `bates_price` already exists.
- **Result routing.** Branched by model, **uniform** `results/<model>/calibrations/<objective>/`.
  Bates writes under `results/bates/...`, Heston under `results/heston/...`.
- **Migration of the existing Heston tree** to `results/heston/` is **DONE** (the working tree already
  holds the migrated `results/heston/{calibrations,smiles,surfaces,tables}` and `.gitignore` was updated
  to the `results/*/calibrations/*/calibration_tests/*` pattern). `calib_paths` therefore uses the
  uniform layout with no heston special-case.
- **Downstream consumers** (`validate_calibrations.py`, `smiles.py`, `example_surface.py`,
  `objective_comparison.py`) are **deferred** and will be reworked or mirrored **one by one** after the
  calibration works.
- **Jump bounds (equity-skew priors).** `lambda [0, 5]`, `nu [-0.5, 0.2]` (equity jumps skew down),
  `delta [1e-3, 0.5]`. Tunable later like the Heston box.
- **Seeds.** One jump seed appended to each existing Heston seed row (restart count stays 6), including
  a near-zero-`lambda` seed so the fit can collapse to Heston gracefully.

## Verified facts about the installed QuantLib 1.35 / `quantlib_pricers`

Checked live during planning; re-confirm the starred item in code before trusting it.

- **`ql.BatesHelper` does not exist** in this QuantLib 1.35 Python build. Calibration reuses
  `ql.HestonModelHelper` with a `ql.BatesEngine` attached via `helper.setPricingEngine(engine)`. A
  `HestonModelHelper` carrying a `BatesEngine` returns a valid `modelValue()`, and
  `model.calibrate([...helpers...], lm, end, constraint)` drives all 8 Bates parameters (it fails only
  if given fewer helpers than free parameters, same as Heston).
- **`BatesProcess` constructor order** (from `__init__.__doc__`):
  `BatesProcess(riskFreeRate, dividendYield, s0, v0, kappa, theta, sigma, rho, lambda, nu, delta)`.
  The jump triple `(lambda, nu, delta)` appends after the 5 Heston args, whose order is the
  `HestonProcess` order `(v0, kappa, theta, sigma, rho)`.
- **`BatesEngine`**: `ql.BatesEngine(model)` (analytic, default integration order 144), or
  `BatesEngine(model, integrationOrder)`, or `BatesEngine(model, relTolerance, maxEvaluations)`. Use
  the default-order form.
- **`quantlib_pricers.vanilla_pricer` already has a scalar `bates_price`**
  (`bates_price(s, k, t, r, g, w, kappa, theta, rho, eta, v0, lambda_, nu, delta)`), but **no**
  `row_bates_price` and **no** `df_bates_price`. The DataFrame wrapper is the package edit in Phase 1.
- **`BatesModel.params()` ordering — CONFIRMED live (2026-06-20).** Built a `BatesModel` with distinct
  sentinel values and read off `model.params()`. The true order is
  **`[theta, kappa, eta, rho, v0, nu, delta, lambda]`**. The planning hypothesis
  (`[v0, kappa, theta, eta, rho, lambda, nu, delta]`) was **wrong on both counts**: the first five
  follow Heston's `params()` order `(theta, kappa, eta, rho, v0)` (since `BatesModel` extends
  `HestonModel`), **not** the constructor order; and the appended jump triple is `(nu, delta, lambda)`,
  **not** `(lambda, nu, delta)`. So `BATES_PARAM_ORDER = ("theta","kappa","eta","rho","v0","nu","delta","lambda_")`
  and the unpack is `theta, kappa, eta, rho, v0, nu, delta, lambda_ = params`. The
  `BatesProcess` *constructor* order is unchanged `(v0, kappa, theta, eta, rho, lambda, nu, delta)` and
  still drives the seed expansion — it differs from `params()`, which is the whole point of pinning the
  three orderings.

### The three orderings (pin each with a comment in code)

There are three distinct orderings; conflating them is the central Bates gotcha.

1. `BatesModel.params()` return order — `(theta, kappa, eta, rho, v0, nu, delta, lambda)` (confirmed
   live). Drives `BATES_PARAM_ORDER`, `BATES_LOW/HIGH`, `_on_boundary`, and the result unpack.
2. `BatesProcess(...)` constructor order — `(v0, kappa, theta, eta, rho, lambda, nu, delta)` after the
   curve/spot args. Drives how `_seed_grid` rows are expanded.
3. `vanilla_pricer.bates_price(...)` argument order — `(s, k, t, r, g, w, kappa, theta, rho, eta, v0,
   lambda_, nu, delta)`. Drives the `df_bates_price` wrapper. Note `bates_price` lists
   `kappa, theta, rho, eta, v0` and remaps internally; pass arguments by its documented order, not the
   constructor order.

## Phase 1 status — LANDED and verified (2026-06-20)

Implemented and checked end-to-end on `UnderlyingOptionsTradesCalcs_2012-01-03.csv`:

- `src/_engine_common.py`, `src/calibrate_bates.py` added; `calibrate_heston.py`/`config.py`/
  `calibrator_prototype.py` edited; `row_bates_price`/`df_bates_price` added to `quantlib_pricers`.
- **`BatesModel.params()` order confirmed live** = `[theta, kappa, eta, rho, v0, nu, delta, lambda]`
  (both halves of the planning hypothesis were wrong; bounds now follow the real order).
- **Heston parity:** the refactored engine reproduces the committed
  `results/heston/calibrations/vol/calibrations.csv` row to full float precision (the shared-helper
  move is behaviour-neutral).
- **Bates runs and accepts** with all 8 params; on the test day it fit tighter (iv_rmse 0.0088 vs
  Heston 0.0132) at ~4.5x the runtime (~63s vs ~14s/day). `nu`/`delta` pegged at their bounds with
  `lambda`≈0.08 — accepted because the jump triple is exempt from the pegging gate, exactly the
  designed behaviour.
- **Routing:** `python src/calibrator_prototype.py --MODEL bates` writes
  `results/bates/calibrations/<objective>/`; `--MODEL heston` (default) writes `results/heston/...`.

**100-day pilot** (`--LIMIT 100`, 2024-05-23..2024-10-15, `results/bates/calibrations/vol/`):
Bates accepted **94/100** vs Heston **63/100** on the same window (+31 days; **36** of them
Heston-rejected days the jumps rescued from pegging). On the 58 days both accept, Bates iv_rmse is
**35% tighter** (median 0.0043 vs 0.0067, better on all 58) and median **`eta` halves** (1.17 -> 0.52:
jumps absorb the tail the Heston vol-of-vol was overfitting). Feller barely moves (still <0 on 56/58).
Jumps behaved as designed: `lambda` median 0.065, 31/94 near-zero (Heston collapse); the
weakly-identified `nu`/`delta` park on their bounds (`nu` at -0.5 on 33/94, `delta` at 0.5 on 13/94),
harmless because gate-exempt but a sign those two bounds are tight. **Tuning lever:** widen `nu`/`delta`
(e.g. -1.0 / 1.0) so rare-jump days find an interior optimum and the jump params stay interpretable.

Remaining for a full Bates baseline: a complete multi-year run (`--MODEL bates`), then the deferred
downstream consumers and the `CLAUDE.md` sync.

## Phase 1 — get Bates calibrating

### 1. `src/_engine_common.py` (new shared module)

Lift the model-agnostic helpers out of `calibrate_heston.py` so both engines import them and cannot
drift:

- `_on_boundary(params, low, high)` — generalize to take `low`/`high` so it works for 5 or 8 params.
- `_iv_rmse(helpers, mkt_vols, weights=None)` — unchanged (operates on helpers, not parameters).
- `_wing_weight(k, s)` — unchanged.
- the seed-variance proxy logic (`clip(median(vol)^2, [SEED_VAR_LO, SEED_VAR_HI])`) as a small helper
  both `_seed_grid` functions call.

`calibrate_heston.py` then changes **only** its imports; behavior must stay identical. Verify by
re-running an existing day (or the committed sample) and diffing `calibrations.csv` — the Heston
numbers must not move.

### 2. `src/calibrate_bates.py` (new)

`calibrate_bates(vol_matrix, s, r, g, objective=DEFAULT_OBJECTIVE) -> dict`. Structure mirrors
`calibrate_heston.py`:

- `_ERR` map reused (`RelativePriceError` / `ImpliedVolError`); the helper stays `HestonModelHelper`,
  only the engine changes.
- `_FAIL` sentinel adds `lambda_, nu, delta` alongside the five Heston keys and `feller`.
- `_seed_grid(vol_matrix)` builds starting points in **`BatesProcess` constructor order**
  `(v0, kappa, theta, eta, rho, lambda, nu, delta)`: expand each existing Heston template row as today,
  then append the single fixed jump seed. Include one row whose `lambda` seed is near zero.
- `_calibrate_once(...)` swaps the model objects:
  ```
  process = ql.BatesProcess(r_ts, g_ts, S_handle, v0, kappa, theta, eta, rho, lambda_, nu, delta)
  model   = ql.BatesModel(process)
  engine  = ql.BatesEngine(model)
  ```
  Helpers stay `ql.HestonModelHelper(...)` with `helper.setPricingEngine(engine)`. The
  `model.calibrate(...)` call paths (with/without the wing `weights` list) are unchanged. Returns
  `list(model.params())` (length 8).
- `calibrate_bates(...)` unpacks per the confirmed `params()` order:
  `theta, kappa, eta, rho, v0, nu, delta, lambda_ = params`.
- **`feller`** stays the Heston-diffusion quantity `2*kappa*theta - eta**2`; jumps do not enter it. It
  remains a soft diagnostic that never gates, exactly as for Heston. Document this in the docstring.
- The acceptance gate checks IV-RMSE and pegging, but **pegging is checked on the 5 Heston params
  only** (the first five of `params`): `iv_rmse <= IV_RMSE_ACCEPT and not _on_boundary(params[:5],
  BATES_LOW[:5], BATES_HIGH[:5])`. The jump triple (`nu, delta, lambda`) is **exempt from the pegging
  gate**: `lambda ≈ 0` is a legitimate Heston collapse, not a wall-hit, and when `lambda ≈ 0` the
  `nu`/`delta` are unidentified and may park on a bound without meaning. `lambda/nu/delta` are still
  reported for inspection. Selection still ranks restarts on IV-space RMSE.
- Success dict returns all five Heston keys (so the orchestrator's existing reads work unchanged) plus
  `lambda_, nu, delta`.

### 3. `src/config.py` (additive only — nothing removed)

- `BATES_PARAM_ORDER` — the confirmed `model.params()` order:
  `("theta", "kappa", "eta", "rho", "v0", "nu", "delta", "lambda_")`.
- `BATES_BOUNDS` — the five Heston economic ranges (reused) plus the jump bounds:
  `lambda_ (0.0, 5.0)`, `nu (-0.5, 0.2)`, `delta (1e-3, 0.5)`.
- `BATES_LOW` / `BATES_HIGH` — derived as `[BATES_BOUNDS[p][0/1] for p in BATES_PARAM_ORDER]`.
- Bates seed: a single fixed jump triple appended to each Heston seed row (e.g. a weak-jump seed with
  near-zero `lambda`). Keep total restarts at 6.
- `MODEL_NAMES = ("heston", "bates")`.
- The shared gate/IV/wing knobs (`IV_RMSE_ACCEPT`, `BOUND_TOL`, `IV_*`, `WING_WEIGHT_*`,
  `DEFAULT_OBJECTIVE`, `OBJECTIVE_NAMES`) are reused unchanged.
- The path resolver `calib_paths(model, objective)` (see [Result routing](#result-routing-branched-by-model)).

The existing Heston `PARAM_ORDER`, `BOUNDS`, `LOW`, `HIGH`, and seed template are left untouched.

### 4. `quantlib_pricers` (the author's package — edit at source)

Add, mirroring `row_heston_price` / `df_heston_price`:

- `row_bates_price(row)` — read `spot_price, strike_price, days_to_maturity, risk_free_rate,
  dividend_rate, w, kappa, theta, rho, eta, v0, lambda_, nu, delta` and call the existing scalar
  `bates_price` in its documented argument order.
- `df_bates_price(df)` — the vectorized/parallel wrapper, same shape as `df_heston_price`.

The new `calibration_tests/*.csv` column for the Bates model price is named `bates` (Heston runs keep
`heston`).

### 5. `src/calibrator_prototype.py` (minimal, flag-gated)

- Add `from calibrate_bates import calibrate_bates` and a `--MODEL {heston,bates}` argument
  (default `heston`, preserving current behavior).
- Resolve once per run: `engine_fn = calibrate_heston if MODEL=='heston' else calibrate_bates`; the
  param list `params = ['theta','kappa','rho','eta','v0']` for Heston, with `+ ['lambda_','nu','delta']`
  for Bates; and the repricer (`df_heston_price` vs `df_bates_price`) and its output column name
  (`heston` vs `bates`).
- The calibration call becomes `res = engine_fn(surf, S_ref, r, g, objective=OBJECTIVE)` — a one-token
  swap; the signature is unchanged.
- The calibration row's `**{k: res[k] for k in params}` spread already generalizes once `params` is the
  model-appropriate list. `res['feller']` is still present. Bates simply appends three columns.
- The repricing block's `for k in params: repriced[k] = res[k]` already generalizes; only the pricer
  call and the output column name branch on `MODEL`.
- `_objective_paths` becomes a thin wrapper over `calib_paths(MODEL, OBJECTIVE)`.

Thread `MODEL` (and the resolved `engine_fn` / `params`) into `calibrate_by_day`.

### Result routing (branched by model)

Insert `<model>` as the top level under `results/`, covering everything that is model-specific (not
just calibrations), so a future Bates run cannot overwrite Heston figures/tables:

```
results/<model>/calibrations/<objective>/{calibrations,rejections,validation}.csv + calibration_tests/
results/<model>/smiles/figures/        (wired with smiles.py later)
results/<model>/surfaces/{data,plots}/ (wired with example_surface.py later)
results/<model>/tables/                (wired with objective_comparison.py later)
```

The core problem the resolver fixes is that today the `results/calibrations/<objective>/` path is
rebuilt by hand in five places (`calibrator_prototype._objective_paths`, `validate_calibrations.py`,
`smiles.py`, `example_surface.py`, `objective_comparison.py`). Adding a `<model>` dimension multiplies
the places that must agree. Centralize it:

- `calib_paths(model, objective) -> (calibrations.csv, rejections.csv, tests_dir)` in `config.py`.

Phase 1 adds only `calib_paths`. The `figures_dir` / `surfaces_dir` / `tables_dir` resolvers land with
their downstream scripts later. Heston keeps its current path this pass; Bates writes to
`results/bates/...`. Whether to later `git mv` the Heston tree under `results/heston/` for symmetry is
the deferred open decision and is independent of this design.

## To verify during implementation

- **`BatesModel.params()` order** — the starred item; print it from known inputs before wiring bounds.
- `model.calibrate(...)` drives all 8 params with `HestonModelHelper` + `BatesEngine` on a real day.
- `df_bates_price` argument order against the scalar `bates_price` (three orderings, do not conflate).
- `ql.Array(BATES_LOW/HIGH)` with 8 entries feeds `NonhomogeneousBoundaryConstraint` cleanly.
- Per-day wall time under joblib: `BatesEngine` analytic pricing is heavier per LM residual than
  `AnalyticHestonEngine`, and 8 params want more iterations. Confirm acceptable before a full-sample run.
- The Heston path is unchanged: diff `calibrations.csv` after the `_engine_common.py` refactor.

## Expected outcomes and risks

- **Lower accept rate than Heston, more pegging.** 8 parameters on a thin per-day surface leave the
  jump triple weakly identified; expect `_on_boundary` to reject more days. The `lambda` floor at
  exactly 0 lets the fit collapse to Heston gracefully and is the main mitigation. If pegging is severe,
  the second jump seed (12 restarts) is the next dial.
- **Feller** is unchanged conceptually and will still be violated on most accepted days (it never
  gates).
- **Runtime** rises with the heavier engine and the extra dimension.

## Downstream consumers — DONE (model-aware, 2026-06-20)

Each carries a module-level `MODEL` constant and routes through `config.calib_paths` /
`utils.build_model_engine` (`build_heston_engine`/`build_bates_engine`). The pricing/inversion helpers
(`heston_price`, `heston_implied_vol`) were already engine-agnostic, so only the engine builder, the
paths, the model price column, and the figure labels needed changing.

- `src/utils.py` — added `build_bates_engine` + the `build_model_engine(row, calc_date, model)` dispatcher.
- `validate_calibrations.py` — `MODEL`/`OBJECTIVE` constants; grades the model's `heston`/`bates` price
  column; includes the jump triple in the per-day print and cross-day stability. Verified on the
  100-day Bates run (94/94 pass hard checks).
- `example_surface.py` — single source of the `MODEL` constant (the others import it); `build_model_engine`;
  carries the Bates jump triple in `day_results['params']`. Verified for both models.
- `smiles.py` — `build_model_engine`, `results/<model>/smiles/figures/`, model label in legend/caption,
  Bates jumps in the per-figure caption, generated `smiles.tex` include path is model-correct. Verified.
- `make_eps.py` — `results/<model>/surfaces/...`, model-namespaced `otm.tex` include paths and caption.
  Verified for Bates.
- `objective_comparison.py` — reads `results/<model>/calibrations/{price,vol}/`; jump params added to the
  parameter-level rows for Bates. Still requires *both* objectives present on disk for the chosen model.

## Deferred phases (not in this pass)

- **Full multi-year Bates run** for the committed baseline, and an optional `nu`/`delta` bound widening.
- **`heston-calibration.tex`** — the smile include paths are generated by `smiles.py`, so a
  model-aware figures dir makes them model-correct automatically. The only manual references are the
  two commented-out `\input` lines, which gain a `<model>/` segment when re-enabled. A Bates writeup is
  a separate document or section.

## Keep in sync

When Phase 1 lands, update `CLAUDE.md`: the new `src/_engine_common.py` / `src/calibrate_bates.py`,
the `--MODEL` flag and its routing, the `df_bates_price` contract, the `calibration_tests/` `bates`
column, and the model-namespaced results layout. A stale line in `CLAUDE.md` is worse than a missing
one.
