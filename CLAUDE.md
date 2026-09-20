# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
It is the **operational** reference (what is where, how to run it, the gotchas); the pipeline
mechanism and the column contracts live in `src/CLAUDE.md`. The **work
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
commands or stages changed — update CLAUDE.md (or `src/CLAUDE.md`) in the **same** change: add what is now true and delete
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
by the `--MODEL {heston,bates}` flag (default `heston`) and runs through the same orchestrator and the
same engine (`src/_calibration_engine.py`, dispatched on the model name). Its design and pilot results
are in `PLAN.md` (Bates extension, PR #12).

Treat the current scripts as a working prototype, not a clean design: the workflow is convoluted and
over-reliant on passing intermediate CSVs between stages with hard-coded column names (see README).

## Environment & dependencies

- **Python 3.12**, **QuantLib 1.35**. Also: `pandas`, `numpy`, `matplotlib`, `joblib` (and
  `ipywidgets`/`ipython` for `inspect.ipynb`). `requirements.txt` pins them to the versions the
  committed results were produced with, and pins `qlpricing` by commit (see below). No `setup.py`,
  lockfile, or test suite.
- The QuantLib pricing wrapper is an **installed external dependency**, the `qlpricing` package
  (`git+https://github.com/boomelage/qlpricing`). It was vendored in-repo at `src/pricing/` until that copy
  was removed, and before that it was the author's `quantlib_pricers`. `requirements.txt` pins it by
  commit, since it is not on PyPI; `pip install -r requirements.txt` therefore brings it in. For work
  against a local working copy, `pip install -e <path to the qlpricing checkout>` instead. It is named
  `qlpricing`, not `pricing`, because `pricing` on PyPI is an unrelated third-party project. It provides `vanilla_pricer.py` (class
  `vanilla_pricer`, used as `vanp = vanilla_pricer()`), `_quantlib_config.py` (QuantLib date conventions),
  `_quantlib_utils.py` (class `_quantlib_utils`, the single home of all QuantLib process/engine/option
  construction), and the `asian_pricer`/`barrier_pricer` modules this repository does not use. Import as
  `from qlpricing.vanilla_pricer import vanilla_pricer`; it no longer needs `src` on `sys.path`.
  Library defaults live in `_quantlib_config.py` (date conventions, `MC_*`,
  `HESTON_INTEGRATION`/`BATES_INTEGRATION`); the host injects its own values via ctor args on
  `_quantlib_utils`/`vanilla_pricer` (`day_count_name`, the MC knobs,
  `heston_integration`/`bates_integration`), a `None` arg resolving to the library default.
  **Reuse rule: `qlpricing` is shared with other projects and must never import host modules (`config`,
  `_utils`, ...).** Edits to it belong in its own repository, not here — an editable install means a
  change made from this checkout silently alters every other consumer.

## How to run

The pipeline is three stages; run scripts directly (no build/lint/test tooling).

**Single-sourced config.** All model/calibration constants live in `src/config.py` — tune there, not in
the modules: surface-coverage knobs, the acceptance gate, the OTM filter floor/cutoff, the wing-weight
knobs, the seed grid, and the Phase-3 mitigation levers (`HESTON_INTEGRATION`/`BATES_INTEGRATION`,
`FELLER_PENALTY`/`FELLER_SEED_TEMPLATE`, `PARAM_ANCHOR_WEIGHT`/`PARAM_ANCHOR_LOOKBACK`,
`WING_WEIGHT_FLOOR`). **Per-model parameters live in one registry, `config.MODELS`** (keyed by model
name): each entry declares `params_order` (QuantLib `model.params()` order), `ctor_order` (process
constructor order, for seeds), the box `bounds`, the derived `low`/`high`, the pegging `gate_names`
subset, and the Bates `jump_seed`. The private `_model` builder derives `low`/`high` so the order is
declared once; `MODELS` is plain data that `as_dict()` captures for the run snapshot. The QuantLib **date conventions**
(`Actual365Fixed` day count, `UnitedStates.NYSE` calendar) live in the `qlpricing` package's `_quantlib_config.py`
(`day_count(name=None)`/`calendar(name=None)`, named choices `DAY_COUNT_NAME`/`CALENDAR_NAME`) and are
**re-exported by `config`**, so `config.day_count`/`config.calendar` keep working as the facade.

**QuantLib construction is centralized** in the `qlpricing` package's `_quantlib_utils.py` (`_quantlib_utils`): the
engine builds its `HestonProcess`/`BatesProcess` via `_qu.heston_process`/`_qu.bates_process` (selected
by the per-model wiring in `_calibration_engine`);
`_utils.build_heston_engine`/`build_bates_engine` are thin wrappers over
`_qu._heston_engine`/`_qu._bates_engine` (which return `(engine, s_handle, ts_r, ts_g, day_count)`); and
`vanilla_pricer` prices through `_qu._{heston,mc_heston,bates}_engine` + `_qu._european_option`. A
constructor-order change is a one-line edit there. The **pricing engine itself** is built in one place:
`_qu.heston_engine_for(model)`/`_qu.bates_engine_for(model)` apply the CF-integration accuracy
(`config.HESTON_INTEGRATION`/`BATES_INTEGRATION`, default `None` = QuantLib order-144 Gauss-Laguerre).
`pricing/` never imports `config`: the accuracy is **injected** at the three app construction sites —
`_calibration_engine._qu`, `_utils._qu`, `calibrator_prototype.vanp` — each passing the same
`config.{HESTON,BATES}_INTEGRATION` into the `_quantlib_utils`/`vanilla_pricer` ctor, so the
calibration fit and the repricing/IV-inversion path integrate identically. **When adding a new
`_quantlib_utils`/`vanilla_pricer` construction site in app code, pass these constants**, or that
site silently falls back to the library default and can drift from the fit.

**Driver flags** (`src/calibrator_prototype.py`): calibrates **every** raw file in `data/options/raw/`
by default; `--LIMIT N` restricts to the `N` most recent trading days, `--MAX_JOBS N` sets the joblib
worker count (default `max(1, os.cpu_count() // 4)`, one day per worker), `--MODEL {heston,bates}`
(default `heston`), `--OBJECTIVE {price,vol}` (default `vol`, `config.DEFAULT_OBJECTIVE`). A full Heston
`vol` run over the multi-year sample is a few thousand days / a few hours (multi-start LM, ~1,500
cells/day); a Bates run is ~4.5x slower per day. **The cross-day anchor (Lever E) overrides
parallelism:** when `config.PARAM_ANCHOR_WEIGHT > 0` the driver runs **strictly sequentially** in date
order and `--MAX_JOBS` is ignored, because each day anchors on its recent accepted predecessors read
in-flight (an inherently sequential dependency; an anchored run is roughly `MAX_JOBS`× the parallel
wall-clock). Weight 0 (the default) keeps the fast parallel path. The static two-pass `--PRIOR_FROM`
flag was removed.

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
  `calibration_tests/`; under bates grades against the Bates bounds (`config.MODELS["bates"]["bounds"]`)
  and flags the jump triple),
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

There is no single-test command (no tests). To exercise just an engine, import `calibrate(model, vol_matrix,
s, r, g, objective="vol")` from `src/_calibration_engine.py` (`model` ∈ `heston, bates`) with a
strike×maturity IV DataFrame. `objective` selects the in-engine LM objective; the orchestrator passes
its `OBJECTIVE` constant through.

## Pipeline architecture and column contracts

The stage-by-stage mechanism (market rates, OTM cleaning, orchestration, the calibration engine,
Bates specifics) and the DataFrame **column contracts** between stages live in `src/CLAUDE.md`,
which loads automatically whenever you work with files under `src/`. Read it before changing
anything in the pipeline, and keep it in sync the same way you keep this file in sync.

## Known issues & fragility (verify before trusting outputs)

The Phase-3 work (raising the accept rate, the mitigation levers A–G, the result figures) is owned by
`PLAN.md` — read it for the lever rationale, sweep findings, and pass criteria. The standing operational
gotchas:

- **Output routing must stay in lock-step.** The calibrator and `validate_calibrations.py` build the
  `results/<model>/calibrations/<objective>/` paths from the **same** `config.calib_paths` rule; if they
  drift the validator stops finding the tests files. The validator is model-aware (`MODEL`/`OBJECTIVE`
  from `_results_config`); under bates it grades against the Bates bounds and flags the jump triple.
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
  (the unified engine `src/_calibration_engine.py` with the `bates` wiring, `df_bates_price` in the
  `qlpricing` package, routing to `results/bates/...`).
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
