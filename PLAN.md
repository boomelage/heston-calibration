# PLAN.md — Heston calibration: correctness and economic-reasonability plan

This file tracks the work to make the calibrated parameters in
`data/options/calibrations/*.csv` **trustworthy**: both numerically correct (the optimizer
actually fit the surface) and economically reasonable (the parameters describe a plausible
SPX vol process). Keep it in sync with the code, and keep `CLAUDE.md` in sync with both.

- **Phase 1 (done):** two input bugs that corrupted what `calibrate_heston` was fed —
  see [Completed tasks](#completed-tasks).
- **Phase 2 (this plan):** the parameters themselves are still not trustworthy. See
  [Diagnosis](#diagnosis-why-the-parameters-are-not-yet-trustworthy), then the three work
  items below, ordered *measure first, then fix*.

Line numbers in any sketch below refer to the current revision and will drift — match on code,
not line numbers.

## Table of Contents

- [Status and scope](#status-and-scope)
- [Diagnosis: why the parameters are not yet trustworthy](#diagnosis-why-the-parameters-are-not-yet-trustworthy)
- [Work item 1: Validation and diagnostics module](#work-item-1-validation-and-diagnostics-module)
- [Work item 2: Engine hardening](#work-item-2-engine-hardening)
- [Work item 3: One calibration per trading day](#work-item-3-one-calibration-per-trading-day)
- [Improving calibration performance](#improving-calibration-performance)
- [Sequencing](#sequencing)
- [Done criteria](#done-criteria)
- [Completed tasks](#completed-tasks)
  - [Issue 1: Stale rate lookup](#issue-1-stale-rate-lookup)
  - [Issue 2: Strike selection slips](#issue-2-strike-selection-slips)
  - [Write-desync fix](#write-desync-fix)

---

## Status and scope

| Item | File(s) touched | Risk | Status |
|------|-----------------|------|--------|
| Issue 1 — stale rate lookup | `src/calibrator_prototype.py` | low | ✅ done |
| Issue 2 — strike-selection slips | `src/calibrator_prototype.py` | medium | ✅ done |
| Work item 1 — validation/diagnostics | **new** `src/validate_calibrations.py` | none (read-only) | ✅ done |
| Work item 2 — engine hardening | `src/calibrate_heston.py` (+ small prototype edit) | medium | ✅ done (strict gate kept; per-bucket fits capped by under-determination → Item 3) |
| Write-desync fix (newly found bug) | `src/calibrator_prototype.py` | low | ✅ done |
| Work item 3 — one calibration per day | `src/calibrator_prototype.py` (restructure) | high (schema change) | ☐ todo |

**API facts confirmed in this environment** (QuantLib 1.35), so the plan does not rely on
non-existent calls:

- `ql.NonhomogeneousBoundaryConstraint(lows, highs)` **exists** → box bounds for Item 2.
- `ql.CalibratedModel.calibrate(helpers, method, endCriteria, constraint=…, weights=…, fixParameters=…)`
  — the `constraint` and `weights` args are available.
- `HestonModelHelper.calibrationError()` **exists**, but `setCalibrationErrorType` and the
  `ImpliedVolError` / `RelativePriceError` enums are **not** exposed → the in-engine error is the
  default **relative-price** error; vol-point ("IV-space") residuals must be computed externally.
- `quantlib_pricers.vanilla_pricer` has **no implied-vol inverter** (only forward pricers), and
  `df_numpy_black_scholes` takes **no dividend** → the existing `black_scholes` column in
  `calibration_tests/*.csv` is dividend-inconsistent with the `heston` column. Item 1 accounts for this.

---

## Diagnosis: why the parameters are not yet trustworthy

Measured on the regenerated `cboe_spx_calibrations_2024-10-07.csv` (53 spot buckets, one trading day):

| Symptom | Count / 53 | What it means |
|---|---|---|
| `rho` pinned at the boundary (`≤ −0.999`) | 17 | optimizer hit the constraint wall — the data does not pin `rho`; a non-fit. |
| Feller violated (`2κθ − η² < 0`) | 40 | variance can reach zero; `η` (vol-of-vol) implausibly large. |
| `eta > 1.5` | 26 | SPX vol-of-vol is realistically ~0.3–1.2; 2–3 is degenerate. |
| `theta > 1.0` (long-run vol > 100%) | 5 | e.g. spot 5696: `θ=11.93, κ=0.0117` — long-run vol ≈ 345%. |

Repricing (Heston price vs actual trade price, from `calibration_tests/*.csv`):
**median relative error ≈ 20%, 90th percentile ≈ 54%.** That is not a calibrated model.

The single most damning diagnostic: on **one day**, across **adjacent** spot buckets, `theta`
ranges 0.037 → **11.93** and `kappa` 0.012 → **18.2**. The structural parameters of one underlying
on one day should be near-identical from bucket 5687 to 5688. They are not. That instability is the
proof the fit is **under-determined** — the root cause Work item 3 addresses.

Two mechanisms produce this:

1. **Fragmented data (identification).** ~53 independent 5-parameter fits/day, each on a thin
   slice (often few maturities). Heston needs a rich surface — multiple maturities *and* a real
   strike range — to separate `kappa` from `theta`. On thin/short-dated snapshots only the
   *product* `kappa·theta` is identified, which is exactly the `θ` huge × `κ` tiny degeneracy seen above.
2. **Unconstrained optimizer + single fixed start.** One fixed initial guess into an unbounded
   Levenberg–Marquardt lets `rho` run to ±1 and `eta` explode, and the only failure check is
   "did the params move from the guess" — which misses every boundary fit.

The three work items attack this in increasing order of depth and risk.

---

## Work item 1: Validation and diagnostics module

**Goal.** A standalone, **read-only** script — `src/validate_calibrations.py` — that reads the
existing `calibrations/*.csv` and `calibration_tests/*.csv` and emits a per-bucket and per-day
**pass/fail report**. It touches nothing in the pipeline, so it is zero-risk and can run *now* to
establish a baseline, then again after Items 2 and 3 to **measure** the improvement.

**Why first.** Right now "quality" is unmeasured: the pipeline writes numbers and computes prices
but never grades them. Without a metric you cannot tell whether Items 2/3 helped.

**What it checks.**

1. **Fit quality (numerical correctness)** — per bucket, from `calibration_tests/*.csv`:
   - *Primary, computable now:* relative repricing error of `heston` vs `trade_price`
     (`abs(heston − trade_price)/trade_price`), reported as median and p90 per bucket and per day.
   - *Rigorous (recommended): IV-space residual.* Invert the fitted `heston` price back to a
     Black implied vol and compare to the market `volatility` (= `trade_iv`), in **vol points**.
     Because `vanilla_pricer` has no inverter, use QuantLib:
     `ql.blackFormulaImpliedStdDev(type, K, F, price, df)` with forward `F = S·exp((r−g)·T)`,
     discount `df = exp(−r·T)`, then `iv = stddev/sqrt(T)`. This is dividend-consistent and the
     natural surface-fit metric. Flag a bucket if RMSE(IV) exceeds ~1–2 vol points.
   - **Do NOT** use `heston − black_scholes` as the residual: the `black_scholes` column omits
     dividends (its pricer takes no `dividend_rate`) while `heston` includes `g`, so the difference
     conflates fit error with a dividend mismatch. (Record this as a latent diagnostic bug; optionally
     fix later by recomputing BS-with-dividend at market IV via `ql.blackFormula` on the forward.)

2. **Economic reasonability** — per bucket, from `calibrations/*.csv`. Two tiers: a **hard reject**
   range (physically/financially impossible) and a softer **suspicious** flag (possible but
   atypical for SPX at these tenors):

   | Param | Hard reject outside | Suspicious flag | Rationale |
   |---|---|---|---|
   | `rho` | `[−0.999, 0.5]` | `|rho|>0.995` (pegged) or `rho>0` (wrong sign for equities) | leverage effect ⇒ negative |
   | `eta` | `[0.01, 2.0]` | `>1.5` | SPX vol-of-vol ~0.3–1.2 |
   | `theta` | `[1e-4, 1.0]` | `>0.25` (long-run vol > 50%) | long-run variance |
   | `v0` | `[1e-4, 1.0]` | `|sqrt(v0) − atm_iv| > 0.05` | should ≈ front-month ATM IV |
   | `kappa` | `(0, 20]` | `<0.1` **and** `theta>0.5` (the unidentified-product pattern) | mean-reversion speed |
   | `feller` | — | `< 0` (optionally treat as hard) | `2κθ ≥ η²` |

   The `atm_iv` reference for the `v0` check is the shortest-maturity, nearest-ATM `volatility`
   in that day's `calibration_tests` rows — a free sanity bound from the input surface.

3. **Cross-bucket stability (per day)** — the headline metric. For each day compute the spread
   (IQR, and max/min ratio) of `theta, kappa, eta, rho, v0` across spot buckets. Structural
   parameters should cluster tightly; flag a day where, e.g., `theta` IQR is large or `kappa` spans
   orders of magnitude. After Work item 3 this becomes a **cross-day** stability check instead.

**Output.** A per-bucket flagged table (`validation/validation_<date>.csv`) plus a printed per-day
summary: % of buckets passing all hard checks, repricing/IV RMSE distribution, flag counts, and the
stability metrics. Put thresholds in a single `THRESHOLDS` dict at the top of the file so they are tunable.

**Sketch.**

```python
# src/validate_calibrations.py  (read-only; pandas/numpy + QuantLib for the optional IV inversion)
import QuantLib as ql, numpy as np, pandas as pd, glob

THRESHOLDS = dict(
    rho_peg=0.995, rho_lo=-0.999, rho_hi=0.5,
    eta_lo=0.01, eta_hi=2.0, eta_susp=1.5,
    theta_lo=1e-4, theta_hi=1.0, theta_susp=0.25,
    v0_lo=1e-4, v0_hi=1.0, v0_atm_tol=0.05,
    kappa_hi=20.0, iv_rmse_pts=0.02,
)

def implied_vol(price, w, S, K, r, g, T):
    F, df = S*np.exp((r-g)*T), np.exp(-r*T)
    opt = ql.Option.Call if w == 'call' else ql.Option.Put
    try:
        sd = ql.blackFormulaImpliedStdDev(opt, K, F, price, df)
        return sd/np.sqrt(T)
    except RuntimeError:
        return np.nan

def grade_bucket(row): ...          # returns dict of hard/suspicious flags from THRESHOLDS
def day_stability(cal_df): ...      # IQR / max-min ratio of structural params across buckets
# write validation/validation_<date>.csv + print per-day summary
```

**Verification.**

```bash
python src/validate_calibrations.py
```

**Pass criteria.** The script runs read-only and produces, for each existing day, (a) a repricing
RMSE figure, (b) per-bucket hard/suspicious flag counts that reproduce the Diagnosis table
(≈17 pegged `rho`, ≈40 Feller violations on 2024-10-07), and (c) a cross-bucket stability metric.
This is the **baseline** to beat.

---

## Work item 2: Engine hardening

**Goal.** Make `calibrate_heston` return parameters that are bounded, reproducible, and
self-graded — replacing the brittle "did it move from the guess" sentinel with real
bounds, multiple starts, and an explicit acceptance gate.

**File.** `src/calibrate_heston.py` (plus a small coordinating edit in `calibrator_prototype.py`
for the new return keys — see (e)).

**Changes.**

**(a) Box bounds via a constraint.** Pass `ql.NonhomogeneousBoundaryConstraint(lows, highs)` to
`model.calibrate(...)`. **Order must match `model.params()` = `[theta, kappa, eta, rho, v0]`**
(this ordering is already a documented gotcha — get it wrong and bounds land on the wrong params):

```python
#                       theta  kappa   eta    rho    v0
lows  = ql.Array([      1e-4,  1e-2,  1e-2, -0.999, 1e-4])
highs = ql.Array([      1.0,   20.0,   2.0,   0.5,   1.0])
constraint = ql.NonhomogeneousBoundaryConstraint(lows, highs)
model.calibrate(helpers, lm, ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8), constraint)
```

**(b) Multiple restarts.** A single fixed start is fragile. Loop over a small set of starting
points spanning plausible ranges (a fixed grid of ~5–10, or seeded random draws within the bounds),
recalibrate from each, and keep the fit with the lowest objective:

```python
def rmse(helpers):
    e = np.array([h.calibrationError() for h in helpers])   # default: relative-price error
    return float(np.sqrt((e**2).mean()))
# for each start: rebuild process/model/engine, set params, calibrate, record rmse(helpers)
# keep argmin; this also removes the dependence on one arbitrary guess.
```

Note `calibrationError()` is **relative price** here (the IV-error enum is not exposed in this
build); that is fine for *ranking* restarts and for a price-space acceptance gate. For
interpretable vol-point error, rely on Work item 1's external IV inversion.

**(c) Rejection / acceptance gate.** Return the failure sentinel (all-`None`) when any of:
- best-fit relative-price RMSE above `RMSE_ACCEPT`, or
- any parameter within tolerance of its bound (`rho ≤ −0.995`, `eta ≥ 1.99`, etc.) — a boundary
  fit is a non-fit, the exact case the old sentinel missed.

  *Threshold kept strict (deliberate).* `RMSE_ACCEPT` stays at **`0.05`**. Measured across
  2024-10-07..11 the per-bucket relative-price RMSE clusters ~0.07–0.17 (median 0.11, p90 ~0.31), so
  `0.05` rejects ~99% of per-bucket fits — and loosening it (e.g. to `0.15` ≈ p67) was **considered
  and rejected**: it would only admit *under-determined* fits. A thin per-bucket surface cannot
  identify five parameters, so a passing RMSE there buys a degenerate, cross-bucket-unstable result,
  not a trustworthy one. The right response to the low accept rate is **more information per fit**
  (Work item 3, one calibration per day), not a lower bar. Replacing this price-space gate with an
  IV-space (vol-point) one is a separate, orthogonal improvement — see
  [Improving calibration performance](#improving-calibration-performance).

The old check (`v0==0.01 and kappa==0.2 and …`) is **removed**: with randomized starts there is no
single guess to compare against, and it never caught boundary fits anyway.

**(d) Return fit diagnostics.** Augment the dict with `rmse`, `n_helpers`, and an `accepted` bool
so Work item 1 and the CSV can record fit quality, not just the point estimate.

**(e) Coordinate the new return keys with the writer.** `calibrator_prototype.py` does
`sparams.loc[s, parameters.index] = parameters.values` against a fixed column list
`['theta','kappa','rho','eta','v0','feller']`; assigning unknown columns via `.loc` raises
`KeyError`. So extend that column list (and the per-row metadata block) to include `rmse`,
`n_helpers`, `accepted` when adding them to the return dict.

**(f) Eval-date / day-count note (accuracy correction).** With **flat-forward** curves and
`Period(days, Days)` maturities, the year fraction is `days/365` regardless of the evaluation date,
so `Date.todaysDate()` vs the true quote date is **immaterial** here. (Do **not** spend effort
"fixing" it.) It only starts to matter if term-structured curves are introduced later — note it for
that future, don't act now.

**Verification.**

```bash
python src/calibrator_prototype.py
python src/validate_calibrations.py      # compare against the Item 1 baseline
```

**Pass criteria — met (engine), with the expected caveat.** Among *accepted* (written) buckets,
pegged-`rho` is ~0 and `eta>1.5`/`theta>1.0` are excluded **by construction** (the box bounds +
boundary-rejection enforce it); every accepted bucket carries an `rmse ≤ RMSE_ACCEPT`. Rejected
buckets are dropped, not silently written. The engine behaves as designed.

**Caveat — capped by under-determination (the real finding).** Under the strict gate, **almost no
per-bucket fit is accepted** (a full 2024-10-07..11 run accepted ~1 bucket): the boundary check and
`RMSE_ACCEPT=0.05` together reject ~all of them, because per-bucket surfaces are too thin to identify
five parameters (the Diagnosis's root cause, not an engine defect). This near-empty output is itself
the evidence that **Work item 3 (one calibration per day) is the prerequisite** — the gate is correct
to reject these; the data, not the standard, is what must change. We deliberately do **not** lower
`RMSE_ACCEPT` to manufacture acceptances (see (c)): that would only record degenerate,
cross-bucket-unstable fits. Until Item 3, treat the per-bucket output as not yet trustworthy.

**Risk (realized).** Bounds and the acceptance gate reject many currently-"successful" (but
degenerate) fits, so output row counts drop. That is the point; Item 1 quantifies the trade. The
write path was also hardened so a day with **zero** accepted fits clears both output files instead
of leaving a stale `calibrations` file beside an emptied `calibration_tests` file (see the
write-desync fix in [Completed tasks](#completed-tasks)).

---

## Work item 3: One calibration per trading day

**Goal.** Fix the identification problem at the root: calibrate **once per trading day** over a
rich, pooled surface using the **true** spot, instead of ~53 thin per-0.5-spot-bucket fits. This is
the deepest change and the one that should make `theta`/`kappa` stop swinging across buckets.

**File.** `src/calibrator_prototype.py` (restructure `calibrateby_spot` → `calibrate_by_day`).

**Why.** Per-bucket calibration starves each fit of maturities and strikes, leaving `kappa·theta`
only jointly identified (the `θ`-huge/`κ`-tiny degeneracy in the Diagnosis). Pooling the day's
trades into one surface gives Heston the cross-maturity, cross-strike information it needs.

**Design — the key decision (analogous to the earlier Path 1/Path 2 choice): how to handle
intraday spot movement.** A Heston calibration has a single spot `S`, but trades occur across an
intraday range (5687→5739 on 2024-10-07, ≈0.9%).

- **Recommended — moneyness normalization.** Each OTM row already carries the underlying at trade
  time (`spot_price` = `underlying_bid`). Compute per-trade moneyness `m = K / S_row`, pick one
  **reference spot** `S_ref` for the day (volume-weighted or closing), and re-strike every trade to
  `K* = m · S_ref`. Build the IV surface over `(K*, T)`. This re-centers the whole day to one spot
  under the standard sticky-moneyness assumption (IV is ~stationary in moneyness over a session).
  The ≈0.9% move makes the correction mild but correct.
  - *Assumption/limitation:* breaks on days with a large intraday move; flag and optionally split
    such days. Worth a sanity check on the day's spot range before trusting `S_ref`.
- **Alternative — keep absolute strikes at one `S_ref`** (no normalization). Simpler, but mild
  moneyness drift biases the wings. Not recommended once normalization is available.

**Surface construction (per day).** Group by `days_to_maturity`; keep the top maturities by traded
volume (consider raising `max_nt` beyond 7 now that it is once-per-day); per maturity keep the
nearest-money strikes per wing (`max_nk`); pivot to a `K*`×`T` IV surface. Require **richer**
coverage than the old `≥5 cells` — e.g. **≥3 maturities and ≥5 strikes** — so the fit is identified.
Optionally pass `weights` to `model.calibrate` (volume- or vega-weighted) so liquid contracts dominate.

**Output schema change (must be documented).** `calibrations/*.csv` becomes **one row per trading
day** (keyed by date, recording `S_ref`, `r`, `g`, the five params, `feller`, `rmse`, coverage
counts) instead of one row per spot bucket. `calibration_tests/*.csv` still reprices the day's
pooled snapshot. This **breaks the current per-spot CSV contract** — update `CLAUDE.md`'s column
contracts and Stage 3/4 description accordingly, and note the README's "convoluted" framing improves
here. If both granularities are ever wanted, gate per-spot behind a flag, but per-day is primary.

**Sketch.**

```python
def calibrate_by_day(filepath):
    df = read_and_filter(filepath)                  # trade_iv>0, parse dates, rate asof(date) + NaN guard
    S_ref = volume_weighted_spot(df)                # one reference spot for the day
    df['Kstar'] = (df['strike_price'] / df['spot_price']) * S_ref   # moneyness-normalize to S_ref
    snap = select_surface(df, max_nt, max_nk)       # top maturities by volume; nearest-money strikes/wing
    surf = snap.pivot_table(index='Kstar', columns='days_to_maturity', values='trade_iv', aggfunc='last')
    if surf.notna().sum().sum() < MIN_CELLS or surf.shape[1] < 3:
        return                                      # require multi-maturity coverage
    params = calibrate_heston(surf, S_ref, r, g)    # ONE calibration for the whole day (hardened engine)
    # write ONE row keyed by date; reprice snap under params -> calibration_tests
```

**Verification.**

```bash
python src/calibrator_prototype.py
python src/validate_calibrations.py
python -c "
import pandas as pd, glob
for f in sorted(glob.glob('data/options/calibrations/*.csv')):
    d = pd.read_csv(f); print(f.split('/')[-1], 'rows:', len(d))   # expect ~1 per day
"
```

**Pass criteria.** One (or very few) rows per day; parameters economically plausible (Feller mostly
satisfied, `rho` not pegged, `eta < 1.5`, `theta < 0.25`); a full-surface `rmse` reported and below
threshold; and **cross-day** parameter stability far tighter than the old cross-bucket spread.
`CLAUDE.md` updated for the new one-row-per-day schema.

**Risk (high).** Schema change ripples to any consumer of `calibrations/*.csv` and to `CLAUDE.md`;
the moneyness-normalization assumption must be validated on high-move days. Do this last, after
Items 1–2 have stabilized the engine and given a baseline to compare against.

---

## Improving calibration performance

Item 2 made fits *honest* (bounded, self-graded, no silent boundary fits) but did not make many of
them *good*: on per-bucket data ~45% still peg a bound and accepted params still swing across spots.
The levers below raise the share of economically-stable, well-fitting calibrations. They are ordered
by expected payoff per unit effort. **Item 3 (one calibration per day) is the single highest-impact
item — it attacks the root cause (identification); do it first.** The rest sharpen the engine and
the gate and apply equally to per-bucket or per-day surfaces.

1. **Pool the surface — Work item 3 (highest impact).** The boundary pegging and cross-bucket swing
   are symptoms of an under-identified 5-parameter fit on a thin slice. One moneyness-normalized,
   multi-maturity surface per day gives Heston the cross-maturity/cross-strike information it needs.
   Expect boundary-rejection and `eta`/`theta` blow-ups to fall sharply. See Work item 3.

2. **Gate and rank in IV space, not relative price.** `calibrationError()` is relative *price* RMSE,
   which deep-OTM contracts inflate; that is why `RMSE_ACCEPT=0.05` is both strict *and* an imperfect
   metric (it over-penalizes cheap wings). The natural surface-fit metric is the **vol-point
   residual** already implemented in `validate_calibrations.py`
   (`ql.blackFormulaImpliedStdDev` on the forward). Compute IV-RMSE inside `calibrate_heston` for the
   best fit and gate on **~1–2 vol points** instead of price RMSE; also rank restarts by it. This is
   interpretable and dividend-consistent. (Cost: an inversion per helper per accepted fit — cheap.)

3. **Weight the objective.** Pass `weights` to `model.calibrate` so liquid / informative quotes
   dominate: **vega weighting** (down-weights deep-OTM noise, complementing lever 2) or
   **volume weighting** (`trade_size`, already in the snapshot). Reduces the deep-OTM tail that the
   relative-price metric over-penalizes and that pulls `eta` up.

4. **Stronger optimization than one local LM.** The 6-point restart grid helps but is small. Options,
   cheapest first: widen/perturb the seed grid (seeded random draws within bounds); add a short
   global pre-search (`ql.DifferentialEvolution`/simulated annealing) to seed LM; or warm-start each
   day from the previous day's accepted params (parameters are persistent across sessions). Keep the
   argmin-RMSE selection.

5. **Tame the Feller / `eta` degeneracy.** Many short-dated fits violate Feller (`2κθ < η²`). Either
   add a soft Feller penalty to the objective, or tighten the `eta` upper bound once lever 1 reduces
   the genuine need for large vol-of-vol. Do **not** hard-reject Feller violations outright — they
   are a known Heston short-tenor limitation, not always a bad fit.

6. **Two-stage / reduced identification (only if 1–5 are insufficient).** Where only the product
   `κ·θ` is identified, fix or prior-anchor `kappa` (e.g. to a stable cross-day estimate) and fit the
   rest, rather than letting `θ`-huge/`κ`-tiny run free. A pragmatic fallback, not a first move.

7. **Term-structured curves (future, not now).** Flat-forward `r`,`g` make the eval-date immaterial
   (Item 2(f)). If real SPX term structures are introduced later, revisit the eval-date/day-count and
   build per-maturity discount factors. Out of scope until the surface and gate are trustworthy.

**Measurement.** After each lever, re-run `validate_calibrations.py` and compare against the Item 1
baseline (35/256 buckets, 14% accepted) and the post-Item-2 numbers: track accept rate, boundary-
rejection share, IV-RMSE distribution, and the cross-bucket→cross-day stability metric. A lever that
does not move those is not worth keeping.

---

## Sequencing

1. Work on a branch off `master` (currently on `test`); do not edit committed CSVs by hand.
2. **Work item 1** (validation, read-only) first — establish the baseline numbers. Commit.
3. **Work item 2** (engine hardening) — re-run pipeline, then re-run validation; confirm pegged-`rho`
   and Feller-violation counts drop and accepted buckets carry a low `rmse`. Commit.
4. **Decide** the Work item 3 spot-handling approach (moneyness normalization recommended), restructure
   to one calibration per day, re-run pipeline + validation; confirm one row/day and tight cross-day
   stability. Commit.
5. Update `CLAUDE.md` in the **same** change as Item 3: new one-row-per-day schema, the
   `calibrate_heston` bounds/rejection/return-keys, and the new `validate_calibrations.py` stage.
   Remove anything this plan made stale.

## Done criteria

- [x] **Item 1:** `src/validate_calibrations.py` runs read-only and reports per-bucket flags,
      repricing/IV RMSE, and per-day stability; baseline reproduces the Diagnosis table
      (2024-10-07: 17 pegged `rho`, 40 Feller violations, 26 `eta>1.5`). Across all 5 days only
      **35/256 buckets (14%)** pass all hard checks — the baseline to beat.
- [x] **Item 2:** `calibrate_heston` calibrates with box bounds and multiple restarts, rejects
      boundary/high-RMSE fits (old "==guess" sentinel removed), and returns `rmse`/`accepted`;
      `calibrator_prototype.py` records the new keys.
- [x] **Item 2:** among *accepted* buckets, pegged-`rho` and `eta>1.5`/`theta>1.0` are excluded by
      construction (bounds + boundary-rejection). With the strict `RMSE_ACCEPT=0.05`, per-bucket data
      yields almost no accepted fits (~1 across 2024-10-07..11) — the standard is kept strict on
      purpose; the under-determination is what Item 3 fixes, not the threshold. **Remaining lever:**
      swap the price-space gate for the IV-space one (see Improving calibration performance).
- [ ] **Item 3:** one calibration per trading day over a moneyness-normalized multi-maturity surface;
      `calibrations/*.csv` is one row per day; cross-day parameter stability is tight.
- [ ] **Item 3:** `CLAUDE.md` updated to match the new schema, engine behavior, and validation stage.

---

## Completed tasks

> Condensed records. Full pre-fix forensic detail (reproduction snippets, option/path
> comparisons) is in git history — see commits `7481d03`, `fe0ba9d`, `bf55b7c`.

### Issue 1: Stale rate lookup

**Status: ✅ complete (Option B).** Every calibration was being fed the *oldest* rate in
`rg`'s history (a 2008 value) instead of the most recent on/before the quote date, because
`rg` is sorted newest-first and the code took `.iloc[-1]` of the `<= date` slice.

- **Fix:** module-level `rg_asc = rg.sort_index()` plus
  `r = rg_asc['risk_free_rate'].asof(date)` / `g = rg_asc['dividend_rate'].asof(date)`,
  with a NaN guard that skips a file when no rate exists on/before its date.
- **Location:** `src/calibrator_prototype.py` (rate lookup near the top of `calibrateby_spot`).
- **Verified:** regenerated `calibrations/*.csv` show `risk_free_rate ≈ 0.05` for Oct-2024
  (was the stale `0.0233`).

### Issue 2: Strike selection slips

**Status: ✅ complete (Path 2).** The per-spot surface was built from the wrong rows due to
three slips: (A) a stale `dft` (strikes taken from whatever maturity the volume-ranking loop
ended on), (B) `ct` filtered from the whole-day frame instead of the current spot, and (C) a
`max()` cap that never actually limited the strike count.

- **Fix:** maturities ranked by traded volume (top `max_nt`=7); each kept maturity contributes its
  `max_nk`=7 nearest-money strikes per wing (`pK[-n:]` puts, `cK[:n]` calls) from **its own** rows;
  the per-maturity slices are concatenated into one **multi-maturity** surface, calibrated **once
  per spot**, and repriced snapshots are accumulated and written **once** after the spot loop.
- **Location:** `src/calibrator_prototype.py`, the per-spot loop in `calibrateby_spot`.
- **Verified:** Oct-2024 produces multi-maturity surfaces across many spots; the only OTM-check
  "violations" are `strike == rounded_spot` ties from the 0.5 spot grid (no strike on the wrong side).

These two fixes corrected the **inputs** to `calibrate_heston`. They did **not** make the output
economically reasonable — that is Phase 2.

### Write-desync fix

**Status: ✅ complete.** Found while running Item 2. The two per-day outputs were written under
independent conditions: `calibrations/*.csv` only when `sparams.dropna()` was non-empty, but
`calibration_tests/*.csv` whenever any snapshot had been repriced. On a day with **no accepted fit**
(now common under the Item 2 gate) this left a **stale `calibrations` file beside an emptied
`calibration_tests` file** — mutually inconsistent outputs.

- **Fix:** only *accepted* fits are repriced into `test_frames`, so calibrations and
  calibration_tests describe the same bucket set; both are then written under one decision. When a
  day yields zero accepted fits, **both stale files are removed** (and a notice printed) so the pair
  can never desync.
- **Location:** `src/calibrator_prototype.py`, end of `calibrateby_spot` (the write block) and the
  repricing block.
- **Verified:** full 2024-10-07..11 run — 10-07 (1 accepted) writes both files consistently;
  10-08..11 (0 accepted) leave **neither** file. `validate_calibrations.py` skips both-absent days
  cleanly.
