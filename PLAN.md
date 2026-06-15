# PLAN.md — Remediation plan for calibration-correctness bugs

Target file for both fixes: **`src/calibrator_prototype.py`** (function `calibrateby_spot`).
Line numbers below refer to the current revision and will drift once edited — match on code, not line.

## Scope

Fix the two highest-impact "Known issues" from `CLAUDE.md`:

1. **Stale rate lookup** — every calibration is fed the wrong risk-free / dividend rate.
2. **Strike-selection slips** — the per-spot calibration surface is built from the wrong rows.

Both corrupt the *inputs* to `calibrate_heston`, so the engine itself is not touched. There is no
test suite, so each fix ships with a concrete, runnable verification step instead.

**Out of scope (but flagged):** the `filepath.replace('otm', ...)` output routing and the
`vol_count>=5` / `contracts_count>=5` double gate. Leave them unless they block a fix.

**Heads-up:** the committed `data/options/calibrations/*.csv` and `calibration_tests/*.csv` were
produced with both bugs present and are therefore invalid. They will be regenerated when the
pipeline is re-run after the fixes; expect them to change (rates ~0.0233 → ~0.05, surfaces differ).

---

## Issue 1 — Stale rate lookup — ✅ DONE (Option B)

Fixed in `src/calibrator_prototype.py`: a module-level `rg_asc = rg.sort_index()` plus
`r = rg_asc['risk_free_rate'].asof(date)` / `g = rg_asc['dividend_rate'].asof(date)`, with a
NaN guard that skips the file if no rate exists on/before `date`. Verified: the regenerated
`calibrations/*.csv` show `risk_free_rate ≈ 0.042–0.05` for Oct-2024 (was `0.023285`).

### Location
`src/calibrator_prototype.py:32-33`

```python
r = rg[rg.index<=date]['risk_free_rate'].iloc[-1]
g = rg[rg.index<=date]['dividend_rate'].iloc[-1]
```

### Root cause
`rg` (from `data/get_rg.py`) is `sort_index(ascending=False)` — **newest first**. `rg[rg.index<=date]`
keeps every row on/before the quote date but preserves that descending order, so `.iloc[-1]` is the
**oldest row in the entire history**, not the most recent quote on/before `date`.

### Evidence (reproducible now, pre-fix)
```bash
python -c "
import sys; sys.path.insert(0,'data'); from get_rg import rg
import pandas as pd; d = pd.Timestamp('2024-10-07')
sub = rg[rg.index<=d]['risk_free_rate']
print('current .iloc[-1] :', sub.iloc[-1])   # 0.023285  (2008-01-07)  <- WRONG
print('fixed   .iloc[ 0] :', sub.iloc[0])     # 0.049994  (2024-10-07)  <- correct
print('robust  asof      :', rg.sort_index()['risk_free_rate'].asof(d))
"
```
The committed `calibrations/*.csv` show `risk_free_rate=0.023285`, confirming the stale 2008 value
was used in production output.

### Fix — option A (minimal)
Change `.iloc[-1]` → `.iloc[0]` on both lines. Correct because, within a descending frame, the first
row of the `<= date` slice is the largest date on/before `date`.

### Fix — option B (robust, recommended)
Make the lookup independent of `rg`'s sort order with `asof`:

```python
rg_asc = rg.sort_index()                       # ascending; asof requires a sorted index
r = rg_asc['risk_free_rate'].asof(date)
g = rg_asc['dividend_rate'].asof(date)
```

`asof` returns the last value at or before `date`, or `NaN` if none exists — it will not silently
return a wrong row if `rg`'s ordering is ever changed. Hoisting `rg_asc` into `get_rg.py` (export a
pre-sorted frame) avoids re-sorting per file.

### Edge case to add either way
If no rate exists on/before `date` (date precedes all of `rg`), option A raises `IndexError` and
option B yields `NaN` that later breaks QuantLib. Guard it:

```python
if pd.isna(r) or pd.isna(g):
    print(f"skipping {filepath}: no rate on/before {date}")
    return
```

### Verification (post-fix)
```bash
python src/calibrator_prototype.py
python -c "
import pandas as pd, glob
f = sorted(glob.glob('data/options/calibrations/*.csv'))[-1]
print(pd.read_csv(f)[['spot_price','risk_free_rate','dividend_rate']].head())
"
```
Pass criterion: `risk_free_rate ≈ 0.05` (Oct-2024), **not** `0.0233`.

---

## Issue 2 — Strike-selection slips — ✅ DONE (Path 2)

Rewrote the per-spot loop in `calibrateby_spot`: maturities ranked by volume (top `max_nt`=7);
each kept maturity contributes its `max_nk`=7 nearest-money strikes per wing (`pK[-n:]` puts,
`cK[:n]` calls) — fixing (A) stale `dft` (now `byt.get_group(t)` per maturity), (B) wrong frame
(now this spot's `dft` only), and (C) the `max`→`min` cap. The per-maturity slices are concatenated
into one multi-maturity surface, calibrated **once per spot** (Path 2), and the repriced snapshots
are accumulated and written to `calibration_tests/*.csv` **once** after the spot loop. Verified on
Oct-2024: 57 spots, 20 distinct maturities survive; ≤7 strikes per wing; the only OTM-check
"violations" are `strike==rounded_spot` ties from the 0.5 spot grid (no strike on the wrong side).

### Location
`src/calibrator_prototype.py:47-76` (the per-spot loop and the inner maturity loop).

```python
for s in S:
    data = df[df['spot_price']==s]
    ...
    for t in T:                 # volume-ranking loop — leaves `dft` bound to the LAST t
        dft = byt.get_group(t)
        ...
    ...
    if len(T)>0:
        for t in T:             # strike-selection loop
            cK = np.sort(dft[dft['w']=='call']['strike_price'].unique()).tolist()   # (A) stale dft
            pK = np.sort(dft[dft['w']=='put']['strike_price'].unique()).tolist()    # (A) stale dft
            ...
            if ncK>1 and npK>1:
                K = pK[:max(npK,max_nk)] + cK[:max(ncK,max_nk)]                     # (C) max never caps
                ct = df[((df['days_to_maturity']==t)&(df['strike_price'].isin(K)))] # (B) full df, not this spot
                ...
```

### The three named slips
- **(A) Stale `dft`.** The strike loop never reassigns `dft`, so `cK`/`pK` are taken from whatever
  maturity the *volume-ranking* loop happened to end on — not the current `t`. Every maturity in the
  strike loop sees the same arbitrary strike universe.
- **(B) Wrong frame for `ct`.** `ct` is filtered from the whole-day `df` (all rounded spot levels),
  so the per-spot snapshot is polluted with trades that occurred at other underlying levels. Should
  filter from this spot's rows only.
- **(C) Cap never applies.** With `max_nk=7`, `pK[:max(len(pK),7)]` slices to `max(len(pK),7) ≥ len(pK)`,
  i.e. **all** strikes; the intended "keep ≤7" never happens. Must be `min(...)`. Once `min` actually
  caps, *which* strikes are kept starts to matter (see decision below).

### Closely related structural finding — single-maturity "surface" (decide before fixing)
Each `ct` appended to `cals` is a single-maturity slice (`df['days_to_maturity']==t`). In the writer
loop (`:79-111`) every `cal` is pivoted into a one-column `surf` and calibrated separately, each
**overwriting** `sparams.loc[s]` and re-writing the CSVs — so the stored params for a spot come from
only its **last** maturity, and `calibration_tests/*.csv` ends up holding just the final slice of the
final spot. Yet `calibrate_heston` loops over both `T` (columns) and `K` (rows): the engine is built
for a **multi-maturity surface**. So the current per-maturity calibration contradicts the design.

**Decision (one choice drives the rest of this section):**
- **Path 1 — minimal:** fix (A)(B)(C) only; keep one calibration per maturity. Lowest risk, but the
  per-spot result stays single-maturity and the CSV-overwrite behavior remains.
- **Path 2 — recommended:** fix (A)(B)(C) *and* build one multi-maturity surface per spot, calibrate
  once, write once. Matches the engine's design and makes `calibration_tests` a usable per-day diagnostic.

### Strike policy (applies to both paths once (C) is fixed)
`pK`/`cK` are sorted ascending. Recommend keeping the `max_nk` strikes **nearest the money** on each
wing: highest puts (`pK[-n:]`, since OTM puts have `K<spot`) and lowest calls (`cK[:n]`, since OTM
calls have `K>spot`). Confirm this matches the modeling intent; the alternative (deepest-OTM, i.e.
`pK[:n]`) is what a naive `min` fix to the existing head-slice would give for puts.

### Recommended corrected sketch (Path 2)
Replaces the inner loops and the per-`cal` writer with one snapshot + one calibration per spot:

```python
max_nt = 7   # maturities per spot
max_nk = 7   # strikes per wing

for s in S:
    spot_data = df[df['spot_price'] == s]            # renamed from `data` (it was clobbered below)
    total_volume = spot_data['trade_size'].sum()
    byt = spot_data.groupby('days_to_maturity')

    vol_by_t = byt['trade_size'].sum().sort_values(ascending=False)
    T = np.sort(vol_by_t.index[:max_nt]).tolist()    # top maturities by volume

    selected = []
    for t in T:
        dft = byt.get_group(t)                       # (A) current maturity
        cK = np.sort(dft.loc[dft['w'] == 'call', 'strike_price'].unique())
        pK = np.sort(dft.loc[dft['w'] == 'put',  'strike_price'].unique())
        if len(cK) > 1 and len(pK) > 1:
            keep = list(pK[-min(len(pK), max_nk):]) + list(cK[:min(len(cK), max_nk)])  # (C) + nearest-money
            selected.append(dft[dft['strike_price'].isin(keep)])                        # (B) this spot only

    if not selected:
        continue
    snap = (pd.concat(selected, ignore_index=True)
              .drop_duplicates(subset=['strike_price', 'days_to_maturity'], keep='first')
              .dropna())
    surf = snap.pivot_table(index='strike_price', columns='days_to_maturity',
                            values='trade_iv', aggfunc='last')                          # real multi-maturity surface
    if int(surf.count().sum()) < 5:
        continue

    params = pd.Series(calibrate_heston(surf, s, r, g))   # ONE calibration per spot
    # write sparams.loc[s] = params (+ metadata), then reprice `snap` and APPEND to a per-day
    # calibration_tests frame written ONCE after the `for s in S` loop (not per spot/maturity).
```

Notes:
- Rename the per-spot subset (`spot_data`) so it is not overwritten by the later `data = cal...`.
- Move the `calibration_tests` write out of the inner loop; accumulate repriced `snap`s and write
  once at the end so all spots survive.

### Path 1 (minimal) line edits, if Path 2 is rejected
1. Inside the strike loop, add `dft = byt.get_group(t)` as the first statement. (A)
2. `K = pK[-min(npK,max_nk):] + cK[:min(ncK,max_nk)]`. (C + strike policy)
3. `ct = data[(data['days_to_maturity']==t) & (data['strike_price'].isin(K))]...`. (B)

### Verification (post-fix)
```bash
python src/calibrator_prototype.py
python -c "
import pandas as pd, glob
f = sorted(glob.glob('data/options/calibration_tests/*.csv'))[-1]
d = pd.read_csv(f)
print('rows:', len(d), '| spots:', d['spot_price'].nunique(),
      '| maturities:', sorted(d['days_to_maturity'].unique()))
"
```
Pass criteria:
- Path 2: `maturities` has **>1** value and `spots` is **>1** (surface is multi-maturity; not just the
  last slice survived).
- Both paths: spot-check that every strike kept satisfies OTM (`K>spot` for calls, `K<spot` for puts)
  and that no wing exceeds `max_nk` strikes.

---

## Sequencing

1. Work on a branch off `master` (currently on `test`); do not edit committed CSVs by hand.
2. Apply **Issue 1** first (smallest, unblocks correct rates), commit, run the Issue 1 verification.
3. Decide Path 1 vs Path 2 for **Issue 2**, apply, commit, run the Issue 2 verification.
4. Re-run the full pipeline so `calibrations/` and `calibration_tests/` are regenerated from correct
   inputs; sanity-check a couple of spots (rate ≈ 0.05, plausible `feller`, surface coverage).
5. Update `CLAUDE.md`: delete the two "Known issues" once verified fixed (do not leave them marked
   "done"), and drop the `→ see PLAN.md` pointers.

## Done criteria
- [x] `calibrations/*.csv` show `risk_free_rate ≈ 0.05` for the Oct-2024 files.
- [x] `ct`/snapshot rows are sourced from the current spot **and** current maturity only.
- [x] Strike count per wing ≤ `max_nk`; strikes are OTM and follow the agreed near/far policy.
- [x] (Path 2) one calibration per spot over a multi-maturity surface; `calibration_tests` retains all spots.
- [x] `CLAUDE.md` "Known issues" updated to match reality.
