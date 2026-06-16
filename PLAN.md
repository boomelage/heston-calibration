# PLAN.md — Heston calibration: correctness and economic-reasonability plan

This file tracks the work to make the calibrated parameters in `data/calibrations.csv`
**trustworthy**: numerically correct (the optimizer actually fit the surface) and economically
reasonable (the parameters describe a plausible SPX vol process). Keep it in sync with the code, and
keep `CLAUDE.md` in sync with both.

- **Phase 1 — input bugs (done).** Two bugs that corrupted what `calibrate_heston` was fed (stale
  rate lookup, strike-selection slips). See [Completed tasks](#completed-tasks).
- **Phase 2 — identification & honesty (done).** Engine hardening (box bounds, multi-start,
  self-grading), one pooled calibration per trading day over a moneyness-normalised surface, and an
  IV-space acceptance gate. This fixed the per-bucket under-determination: the routine now produces
  one *identified*, cross-day-stable fit per day. See [Completed tasks](#completed-tasks).
- **Phase 3 — acceptance (open, this plan).** The fits are good — days fit to ≤1 vol point — but a
  **minority of days are accepted**: the rest peg a skew parameter to its bound. Phase 3 resolves
  that pegging. See [Current status](#current-status-why-good-fits-still-reject) and the
  [Phase 3 plan](#phase-3-plan-improve-parameter-acceptance).

- **Sample.** The calibration set is no longer the 5-day diagnostic week. `data/options/raw/` now
  holds a multi-year SPX trade history (CBOE `UnderlyingOptionsTradesCalcs_*` from 2012 onward plus
  Hanweck `UnderlyingOptionsTradesCalcsHanweck_*` files spanning 2013 and 2024), ~2700+ trading days.
  Accept-rate targets below are therefore stated as **proportions over the full set**, not "n of 5";
  the 2024-10-07..11 table is retained only as a worked diagnostic example of the pegging mechanism.
- **Specification.** The delivered routine is stated formally in `heston-calibration.tex` (model +
  pricing operators, `S_ref`, `K*`, surface construction, price-space vs IV-space objectives, the
  boundary-pegging gate).

Line numbers in any sketch below drift — match on code, not line numbers.

## Table of Contents

- [Status and scope](#status-and-scope)
- [Current status: why good fits still reject](#current-status-why-good-fits-still-reject)
- [Phase 3 plan: improve parameter acceptance](#phase-3-plan-improve-parameter-acceptance)
- [Sequencing](#sequencing)
- [Done criteria](#done-criteria)
- [Completed tasks](#completed-tasks)

---

## Status and scope

| Item | File(s) | Risk | Status |
|------|---------|------|--------|
| Phase 1 — input bugs (rate lookup, strike selection) | `src/calibrator_prototype.py` | low–med | ✅ done |
| Phase 2 — validation module | `src/validate_calibrations.py` (new) | none (read-only) | ✅ done |
| Phase 2 — engine hardening (bounds, multi-start, gate) | `src/calibrate_heston.py` | medium | ✅ done |
| Phase 2 — one calibration per trading day | `src/calibrator_prototype.py`, `src/calibrate_heston.py` | high (schema) | ✅ done |
| Phase 2 — IV-space acceptance gate | `src/calibrate_heston.py` | medium | ✅ done |
| Write-desync fix | `src/calibrator_prototype.py` | low | ✅ done |
| **Phase 3 — resolve boundary pegging (raise accept rate)** | `src/calibrate_heston.py` (+ knobs in `calibrator_prototype.py`) | medium | ⏳ **open** |

**QuantLib 1.35 API facts** (confirmed in this environment; the plan relies on no non-existent calls):

- `ql.NonhomogeneousBoundaryConstraint(lows, highs)` exists → box bounds (Phase 2, in use).
- `ql.CalibratedModel.calibrate(helpers, method, endCriteria, constraint=…, weights=…, fixParameters=…)`
  — `weights` (Phase 3 lever B) and `fixParameters` (Phase 3 lever C) are both available.
- `HestonModelHelper.calibrationError()` exists, but `setCalibrationErrorType` and the
  `ImpliedVolError` / `RelativePriceError` enums are **not** exposed → the in-engine error is
  relative price; vol-point residuals are computed externally (and, in the engine, via
  `BlackCalibrationHelper.impliedVolatility`).
- `quantlib_pricers.vanilla_pricer` has no implied-vol inverter and `df_numpy_black_scholes` takes no
  dividend → the `black_scholes` column in `calibration_tests/*.csv` is dividend-inconsistent with
  `heston`; never use `heston − black_scholes` as a residual (the validator inverts to IV instead).

---

## Current status: why good fits still reject

The pegging mechanism is clearest on a small, hand-checked slice, so the worked example below is the
`2024-10-07..11` week (5 trading days; `python src/calibrator_prototype.py`, best-fit params shown
even where the day is rejected). The **same** pattern — good IV-RMSE, `theta`/`v0` stable, `kappa`/`rho`
pegging at corners — recurs across the full multi-year set; re-validate against the whole sample (not
this week) when measuring a lever's effect.

| Date | theta | kappa | eta | rho | feller | IV-RMSE | result |
|------|-------|-------|-----|-----|--------|---------|--------|
| 10-07 | 0.0294 | 13.53 | 0.96 | **−0.999** | −0.12 | 0.0082 | reject — `rho` pegged |
| 10-08 | 0.0295 | **20.00** | 1.07 | −0.971 | +0.05 | 0.0070 | reject — `kappa` pegged |
| 10-09 | 0.0308 | 14.97 | 1.53 | −0.837 | −1.41 | 0.0066 | **accept** (but `eta>1.5`, Feller<0) |
| 10-10 | 0.0294 | **20.00** | 1.21 | −0.965 | −0.29 | 0.0098 | reject — `kappa` pegged |
| 10-11 | 0.0288 | 17.88 | 0.82 | **−0.999** | +0.36 | 0.0101 | reject — `rho` pegged |

Reading this:

1. **The fits are good.** Every day's IV-RMSE is **0.66–1.0 vol points**, far inside the
   `IV_RMSE_ACCEPT = 0.02` gate. Fit quality is *not* the blocker.
2. **`theta` and `v0` are already well-identified and cross-day stable.** `theta` ∈ [0.0288, 0.0308]
   (long-run vol √θ ≈ 17.0–17.5%), `v0` ∈ [0.007, 0.026]. Phase 2 solved the identification problem
   for the *level* of the surface.
3. **The skew parameters peg.** `kappa` runs to its 20 ceiling (10-08, 10-10) or sits high (13–18);
   `rho` runs to its −0.999 floor (10-07, 10-11). The one accepted day pegs nothing but lands at
   `eta = 1.53` (> 1.5) with `feller = −1.41`.
4. **It is a degeneracy, not noise.** `kappa` (how fast skew decays with maturity) and `rho`/`eta`
   (the level of skew) are partially interchangeable in matching the surface skew, and the pooled SPX
   surface — including 7-day options whose steep skew single-factor Heston structurally under-fits —
   demands more skew than the model can supply without driving one of them to a corner. The optimizer
   picks whichever corner; `theta`/`v0` are untouched.

So Phase 3 is about **pinning the `(kappa, rho, eta)` skew subspace**, not improving the fit. The two
honest ways to do that: feed the fit less of the skew it cannot match (drop the steepest short
maturities; weight liquid quotes), and add information/priors where the data is silent (anchor
`kappa`; penalise Feller violations). The levers below are ordered by payoff-per-effort and are
independently testable.

---

## Phase 3 plan: improve parameter acceptance

**Goal.** Lift the accept rate (target: a **majority of the full set's days**, ≥ 60%) by stopping the
`kappa → 20` / `rho → −0.999` pegging, **without** lowering `IV_RMSE_ACCEPT` or relaxing the
boundary-rejection — and while keeping `theta`/`v0` in their current tight ranges. Re-run
`validate_calibrations.py` over the **whole sample** after each lever and compare four numbers: accept
rate, pegged-bound share, `eta`/Feller flag counts, and the cross-day `kappa`/`rho` spread. With
thousands of days the metrics are now distributions, not five rows — track shares/percentiles, and
watch for regime dependence (2012–2015 vs 2024, calm vs stressed days) rather than a single rate.

**Policy — do not "fix" this by gaming the gate.** Keep `IV_RMSE_ACCEPT = 0.02` and the
boundary-rejection. Do **not** widen `kappa`'s upper bound merely to turn a peg into a non-peg: a
`kappa` that only fits at 20–30 is the model telling us it is mis-specified for that surface, not a
real estimate. A bound change is allowed only if a lever leaves `kappa` at a *defensible interior*
value and the old bound was the sole obstacle. `rho`'s −0.999 floor stays (it is ≈ −1).

**Lever A — drop the steepest short maturities (cheapest; test first).** Raise `MIN_DTM` from 7 toward
14 (try 14, then 21) in `calibrator_prototype.py`. The < 14-day skew is precisely what single-factor
Heston cannot match and what forces `kappa → 20` and `rho → −1`; removing it should relax both.
*Verify coverage survives* (`MIN_MATS = 3`, `MIN_STRIKES = 5`, `MIN_CELLS = 12`) on every day — if a
day thins out, also raise `MAX_NT`. Measure whether `kappa` leaves 20 and `rho` leaves −0.999. Risk:
low (one knob); cost: discards short-dated information.

**Lever B — weight the objective (vega or volume).** Pass per-helper `weights` to `model.calibrate`.
Vega weighting down-weights deep-OTM wings whose low-vega, high-relative-price-noise quotes over-demand
skew and pull `kappa`/`rho`/`eta` to extremes; volume weighting (`trade_size`, already in the
snapshot) is the cheap proxy. Complements the IV-space gate. Risk: low–medium (engine change, no
schema change). Measure the pegging share.

**Lever C — regularise / anchor `kappa` (attacks the degeneracy at its source).** `theta` and `v0`
are pinned and `kappa` is the parameter hitting its ceiling, so reduce the effective dimensionality of
the skew subspace: either **fix** `kappa` via `calibrate(..., fixParameters=[…])` and fit the other
four, or add a **soft quadratic prior** `λ·(kappa − kappa₀)²` to a custom objective. Choose `kappa₀`
from a cross-day-robust estimate (e.g. the median interior `kappa` once levers A/B are in), not a
guess. This breaks the `kappa ↔ rho` trade-off so `rho` stops pegging. Risk: medium — it changes the
estimator's character, so document that `kappa` is now (partly) imposed, not free. Measure: `rho`
leaves the floor; cross-day `kappa` stabilises.

**Lever D — soft Feller penalty (+ revisit the `eta` cap).** Add `λ·max(0, eta² − 2·kappa·theta)` to
the restart-ranking objective so the optimizer prefers Feller-satisfying corners; this fixes the
accepted day's `feller = −1.41` and keeps `eta` from running toward 2.0. Implement as a post-hoc
penalty in the multi-start selection (cheap) or a custom `ql.CostFunction` (cleaner). Do **not**
hard-reject Feller — short-tenor Heston violates it routinely and it is not always a bad fit. Risk:
medium. Measure: Feller-violation and `eta > 1.5` counts fall.

**Lever E — warm-start + stronger search (stabiliser; last).** Seed each day's Levenberg–Marquardt
from the previous day's accepted params (`kappa`/`rho`/`eta` are persistent across a week) and/or run a
short global pre-search (`ql.DifferentialEvolution`) before LM. Pairs naturally with lever C —
yesterday's `kappa` is the anchor. Keep the argmin-IV-RMSE selection. Risk: low–medium. Measure:
less run-to-run / corner variability.

**Out of scope (future).** Term-structured `r`,`g` curves — flat-forward makes the eval-date
immaterial (Phase 2 confirmed); revisit only if real SPX term structures are introduced. A genuinely
skew-faithful model (a second variance factor / rough vol) is beyond this single-factor-Heston
prototype.

**Sketch (where each lever lands).**

```python
# calibrator_prototype.py
MIN_DTM = 14                                   # lever A: was 7

# calibrate_heston.py — _calibrate_once / calibrate_heston
weights = ql.Array([...])                      # lever B: per-helper vega or trade_size
model.calibrate(helpers, lm, end, constraint, weights)                                   # B
model.calibrate(helpers, lm, end, constraint, weights, [False, True, False, False, False])  # C: fix kappa
# lever D: rank restarts by iv_rmse + lambda*max(0, eta**2 - 2*kappa*theta)
# lever E: prepend previous day's accepted (v0,kappa,theta,eta,rho) to _seed_grid; or a DE pre-search
```

**Verification.**

```bash
python src/calibrator_prototype.py
python src/validate_calibrations.py
# full-set summary (thousands of rows — aggregate, do not dump every day)
python -c "import pandas as pd; d=pd.read_csv('data/calibrations.csv'); print(len(d),'days'); print('accept rate', d['accepted'].mean()); print(d[['kappa','rho','eta','feller','iv_rmse']].describe())"
```

**Pass criteria.** A majority of the full set's days accept (≥ 60%) with **no pegged bound**,
`eta < 1.5`, Feller mostly satisfied, and `theta`/`v0` unchanged in their tight ranges; the cross-day
`kappa`/`rho` distributions stop piling up at corners. Update `CLAUDE.md`'s boundary-pegging "Known
issue" bullet and the Done criteria below in the **same** change as whichever lever lands.

---

## Sequencing

1. Work on a branch off `master` (currently on `test`); do not hand-edit committed CSVs — regenerate
   them via the scripts.
2. Phase 3 levers in order A → B → C → D → E, **committing and re-validating after each** so each
   lever's effect on accept rate / pegging is measured against the prior step. A lever that does not
   move those metrics is reverted, not kept.
3. Update `CLAUDE.md` in the **same** change as whichever lever lands (boundary-pegging bullet, any
   new knob or behaviour), and tick the Done criterion here.
4. **Runtime.** A full run now calibrates ~2700+ days (multi-start LM each), so it is no longer
   seconds. For fast lever iteration, develop against a representative date subset (a calm stretch and
   a stressed one), then confirm the lever on the full set before committing the metrics.

## Done criteria

- [x] **Phase 1:** stale-rate and strike-selection input bugs fixed (see Completed tasks).
- [x] **Phase 2 — validation:** `validate_calibrations.py` runs read-only, reporting fit quality
      (rel-err + IV-space RMSE), two-tier economic flags, and cross-day stability. Pre-fix baseline:
      35/256 per-bucket fits (14%) passed all hard checks — the number Phase 2 had to beat.
- [x] **Phase 2 — engine:** box bounds + multi-start + IV-space gate; boundary/high-RMSE fits
      rejected; `rmse`/`iv_rmse`/`accepted` returned; old "== guess" sentinel removed.
- [x] **Phase 2 — per-day:** one calibration per day over a moneyness-normalised multi-maturity
      surface; single `data/calibrations.csv`, one row/day. Cross-day params tight
      (`theta` 0.029–0.031, `v0` 0.007–0.026, `eta` 0.8–1.5) versus the old cross-bucket `theta`
      0.037 → 11.93 swing; genuine fit ~0.7–1.0 vol points.
- [ ] **Phase 3 — acceptance (open):** ≥ 60% of the full multi-year set's days accept with no pegged
      bound, `eta < 1.5`, Feller mostly satisfied, `theta`/`v0` unchanged. Pursue levers A–E above;
      re-measure over the whole sample after each.

---

## Completed tasks

> Condensed records. Full forensic detail (reproduction snippets, path comparisons) is in git history
> — commits `7481d03`, `fe0ba9d`, `bf55b7c`.

### Phase 1 — input bugs

**Issue 1 — stale rate lookup (✅).** Every calibration was fed the *oldest* rate in `rg` (a 2008
value) instead of the most recent on/before the quote date, because `rg` is sorted newest-first and
the code took `.iloc[-1]` of the `<= date` slice. Fix: module-level `rg_asc = rg.sort_index()` plus
`rg_asc[col].asof(date)` with a NaN guard. Verified: Oct-2024 `risk_free_rate ≈ 0.05` (was the stale
`0.0233`).

**Issue 2 — strike-selection slips (✅).** The per-spot surface was built from the wrong rows (a stale
`dft`; `ct` filtered from the whole-day frame; a `max()` cap that never limited the strike count).
Fix: maturities ranked by traded volume, each kept maturity contributing its own nearest-money strikes
per wing, concatenated into one multi-maturity surface. (Superseded by the Phase 2 per-day
restructure, but the row-selection logic carried over.)

These fixed the **inputs** to `calibrate_heston`; they did not make the output economically
reasonable — that was Phase 2.

### Phase 2 — identification & honesty

**Why (the diagnosis that motivated it).** The old scheme ran ~53 independent 5-parameter fits/day,
one per 0.5-spot bucket, each on a thin slice. Heston needs a rich multi-maturity, multi-strike
surface to separate `kappa` from `theta`; on thin slices only the *product* `kappa·theta` is
identified. The damning symptom: on **one day**, across **adjacent** spot buckets, `theta` ranged
0.037 → **11.93** and `kappa` 0.012 → **18.2** — the structural parameters of one underlying cannot
swing like that. Repricing error was median ≈ 20%, p90 ≈ 54%. That instability was the proof the
per-bucket fit was under-determined.

**Work item 1 — validation module (✅), `src/validate_calibrations.py`.** Read-only grader: fit
quality (relative repricing error **and** an IV-space residual — invert the `heston` price to a Black
vol via `ql.blackFormulaImpliedStdDev` on the forward `F = S·e^{(r−g)T}`, dividend-consistent),
two-tier economic flags (hard-reject vs suspicious) on the five params + Feller, and (post per-day)
**cross-day** stability. It established the 35/256 (14%) baseline. *Latent diagnostic bug recorded:*
the `black_scholes` column omits dividends while `heston` includes `g`, so `heston − black_scholes`
conflates fit error with a dividend mismatch — the validator inverts to IV instead of using that
difference.

**Work item 2 — engine hardening (✅), `src/calibrate_heston.py`.** Replaced the brittle "did the
params move from the guess" sentinel with: **box bounds** (`NonhomogeneousBoundaryConstraint`, order
`[theta, kappa, eta, rho, v0]`), **multiple restarts** from a 6-point data-seeded grid (keep the
lowest-error fit), and an explicit **acceptance gate** (reject high-error or boundary-pegged fits).
Returns `rmse`/`iv_rmse`/`n_helpers`/`accepted`; `calibrator_prototype.py` records the new keys.
Eval-date note: under flat-forward curves and `Period(days, Days)`, the year fraction is `days/365`
regardless of the evaluation date, so `Date.todaysDate()` is immaterial here — do not "fix" it unless
term-structured curves arrive.

**IV-space acceptance gate (✅, pulled forward with the per-day work).** `calibrationError()` is
*relative price*, which deep-OTM wings inflate (a ~1-vol-point fit scored ~0.07 and was wrongly
rejected). The engine now inverts each helper's fitted price back to a Black vol via
`BlackCalibrationHelper.impliedVolatility(modelValue, …)`, computes the vol-point RMSE, **ranks
restarts by it, and gates on `IV_RMSE_ACCEPT = 0.02`**. Relative-price RMSE is retained as the `rmse`
diagnostic. This flipped rejections from "wing-inflated price error" to the genuine remaining issue
(boundary pegging → Phase 3).

**Work item 3 — one calibration per trading day (✅), `calibrate_by_day`.** One calibration over a
pooled, moneyness-normalised surface per day, not per bucket. Each trade keeps `m = K/S_row` but is
re-struck to `K* = m·S_ref` (volume-weighted `S_ref`) and snapped to the 5-pt SPX grid, re-centring
the day under sticky-moneyness; large-move days are flagged `high_move`. `< MIN_DTM = 7`-day
maturities are dropped; coverage gates `MIN_MATS = 3`, `MIN_STRIKES = 5`, `MIN_CELLS = 12`. The output
schema changed to **one row per day** in a single `data/calibrations.csv` (fully regenerated each
run), replacing the old per-day `calibrations/` directory; `calibration_tests/*.csv` still reprices
one-file-per-day at the contract's *original* spot/strike. Result: cross-day params now cluster
tightly (see Done criteria) — the under-determination is fixed. `CLAUDE.md` was updated for the new
schema, engine behaviour, and validation stage.

### Write-desync fix (✅)

The two per-day outputs were written under independent conditions, so a day with no accepted fit left
a stale `calibrations` file beside an emptied `calibration_tests` file. Fix: only accepted fits are
repriced, both outputs are written under one decision, and a zero-accept day removes both — the single
`data/calibrations.csv` is regenerated from accepted rows each run, and the per-day tests file is
cleared by `_skip_day`. Verified on the 2024-10-07..11 run.
