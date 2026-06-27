"""Central tuning constants for the Heston calibration pipeline.

Every model/calibration knob lives here so the orchestrator (`calibrator_prototype.py`), the engine
(`calibrate_heston.py`), the OTM filter (`prepare_surface._prepare_options`) and the validator
(`validate_calibrations.py`) all read one source of truth. PLAN.md Phase 3 tunes these values
(MIN_DTM, the box bounds, a Feller penalty, ...); editing one line here is the whole change.

"""
from pathlib import Path

# Date conventions are defined once in pricing/_quantlib_config.py (alongside the QuantLib engine
# builders that consume them) and re-exported here so the rest of the pipeline keeps a single
# `from config import day_count, calendar` facade. config is the definition site for everything else;
# for the day count / calendar it is a thin pass-through to the canonical source.
from pricing._quantlib_config import (  # noqa: F401  (re-exported)
    day_count, calendar, DAY_COUNT_NAME, CALENDAR_NAME,
)

# ---- Surface selection / coverage (prepare_surface._select_surface / calibrator_prototype.calibrate_by_day) ----
# Pooling the whole day (one fit) lets us take more maturities than the old per-spot path.
MAX_NT = 50          # maturities kept, ranked by traded volume (20 reaches ~485d; volume ranking
                     # caps a top-12 surface at ~394d even when MAX_DTM is larger)
MAX_NK = 40          # strikes kept per wing (highest OTM puts, lowest OTM calls), nearest the money
MIN_NK = 3           # min distinct K* strikes a wing must have for a maturity to be kept (both wings);
                     # a single-strike wing cannot anchor a smile. Maturities failing this are skipped
                     # before the MAX_NT volume cap, so the cap counts only wing-qualifying maturities
STRIKE_GRID = 5.0    # SPX near-money strike increment; normalised K* is snapped to this grid
MIN_DTM = 89         # drop ultra-short maturities: Heston fits them poorly and they drive
                     # eta/kappa to extremes (Feller-violating), polluting the pooled fit.
                     # PLAN.md Phase 3 Lever 1 (Problem 3 = 2019 kappa pegging). The short end is what
                     # IDENTIFIES the mean-reversion speed kappa; raising MIN_DTM to 89 removed it, so in
                     # the calm 2019 regime kappa is under-identified and walks to its floor (the
                     # diagnostic shows ~89% of 2019 days within 5% of the kappa floor). Lowering MIN_DTM
                     # back toward the short end re-anchors kappa: sweep {7, 14, 30} and grade with
                     # src/results/calibration_diagnostics.py (kappa_floor_frac, pegged_rate) and the
                     # validator. Left at 89 here so the committed baseline is unchanged; this is the
                     # primary sweep target for the pegging problem, paired with the Lever 3 short-tenor
                     # Feller behaviour it was originally raised to avoid.
MAX_DTM = 730        # drop very long maturities (thin, stale quotes)
MIN_MATS = 3         # require a genuinely multi-maturity surface (identification)
MIN_STRIKES = 7      # require a real strike range
MIN_CELLS = 12       # non-NaN surface cells required (target >= MIN_MATS x MIN_STRIKES)
MAX_MOVE_PCT = 0.03  # intraday spot range above this flags the day (sticky-moneyness strained)

# ---- OTM filter (prepare_surface._prepare_options) ----
# Keep rows with FLOOR < ratio-moneyness < CUTOFF (see _utils.df_moneyness). The CUTOFF drops near-ATM
# rows (keeps only OTM); the FLOOR drops the deep-OTM tail. Ratio moneyness = e^-|log(K/S)|, so the
# 0.6 floor keeps |log-moneyness| < ~0.51 (~40% OTM): it removes the lottery-ticket strikes (|lm| out
# to ~3.3) whose extreme prices peg the fit to the bounds, while keeping the full tradeable wing
# (95% of fitted cells sit inside |lm| 0.22, and the smile plots only reach ~0.22).
OTM_MONEYNESS_CUTOFF = 0.98
OTM_MONEYNESS_FLOOR = 0.6


# ---- Engine: box bounds (calibrate_heston) ----
# PARAM_ORDER is QuantLib's model.params() order; LOW/HIGH are derived from it so the order is
# declared exactly once. rho upper kept slightly positive (equity leverage => negative) but not
# forced; eta capped at 2.0 (SPX vol-of-vol ~0.3-1.2).
PARAM_ORDER = ("theta", "kappa", "eta", "rho", "v0")
BOUNDS = {
    "theta": (1e-4, 1.0),
    "kappa": (1e-2, 20.0),
    "eta":   (1e-2, 2.0),
    "rho":   (-0.999, 0.5),
    "v0":    (1e-4, 1.0),
}
LOW = [BOUNDS[p][0] for p in PARAM_ORDER]
HIGH = [BOUNDS[p][1] for p in PARAM_ORDER]

# ---- Engine: Bates box bounds (calibrate_bates) ----
# Bates = Heston + Merton lognormal jumps: three extra params lambda (jumps/yr), nu (mean log-jump),
# delta (log-jump std). BATES_PARAM_ORDER is QuantLib's BatesModel.params() order, CONFIRMED live by
# building a model with distinct sentinels: the first five follow HestonModel.params()
# (theta,kappa,eta,rho,v0), then the jump triple appends as (nu, delta, lambda) -- NOT (lambda,nu,delta),
# and the first five are NOT the constructor order. Get this wrong and bounds land on the wrong params.
# The five Heston ranges are reused; jump bounds are equity-skew priors (jumps skew down, so nu allows
# more negative room). lambda floor is exactly 0 so the fit can collapse to pure Heston.
# CURVATURE LEVER (PLAN.md Phase 3 Lever 4, Problem 1). The lognormal jumps add convexity to the smile,
# and the diagnostic (src/results/calibration_diagnostics.py) shows the model call wing is too convex on
# ~87% of days. Tightening the delta (log-jump std) upper bound caps how much wing curvature the jumps
# can manufacture; sweep delta's upper 0.5 -> {0.25, 0.15} (and optionally narrow nu) and grade with the
# diagnostic's d_curv_call / call_convex_frac. Left at the original 0.5 so the committed baseline is
# unchanged; this is a sweep target, not a default change.
BATES_PARAM_ORDER = ("theta", "kappa", "eta", "rho", "v0", "nu", "delta", "lambda_")
BATES_BOUNDS = {
    "theta":   (1e-4, 1.0),
    "kappa":   (1e-2, 20.0),
    "eta":     (1e-2, 2.0),
    "rho":     (-0.999, 0.5),
    "v0":      (1e-4, 1.0),
    "nu":      (-0.5, 0.2),
    "delta":   (1e-3, 0.5),
    "lambda_": (0.0, 5.0),
}
BATES_LOW = [BATES_BOUNDS[p][0] for p in BATES_PARAM_ORDER]
BATES_HIGH = [BATES_BOUNDS[p][1] for p in BATES_PARAM_ORDER]
# One fixed jump seed appended to each Heston seed row (restart count stays 6). lambda near zero so the
# fit can start from "almost no jumps" and grow them only if they help; nu slightly negative (down-jump).
BATES_JUMP_SEED = (0.1, -0.1, 0.1)   # (lambda, nu, delta) in BatesProcess constructor order

# ---- Models ----
MODEL_NAMES = ("heston", "bates")
DEFAULT_MODEL = "heston"

# ---- Engine: in-engine LM objective ----
# Selection and the gate always run off IV-space RMSE; the objective only changes what each restart
# minimises. The string->QuantLib-enum map (`_ERR`) stays next to the engine (live ql objects).
OBJECTIVE_NAMES = ("price", "vol")
DEFAULT_OBJECTIVE = "vol"

# ---- Engine: acceptance gate / tolerances ----
# IV-space RMSE (vol points): model-implied vol vs market vol per helper. ~2 vol points is a tight
# fit and is the metric the surface is quoted in. Boundary-pegged params are rejected separately.
IV_RMSE_ACCEPT = 0.02   # max IV-space RMSE (vol points) for an accepted fit
BOUND_TOL = 1e-3        # fraction of a bound's span within which a param counts as "pegged"

# ---- Engine: IV inversion controls ----
# For BlackCalibrationHelper.impliedVolatility(price, accuracy, maxEval, lo, hi).
IV_ACC, IV_MAXEVAL, IV_LO, IV_HI = 1e-6, 500, 1e-4, 5.0

# ---- Engine: characteristic-function integration accuracy (PLAN.md Phase 3 Lever 2) ----
# Controls how the Heston/Bates analytic engines integrate the characteristic function. A strongly
# violated Feller condition plus elevated eta makes the model implied density non-monotone, so the
# fitted/repriced wing IV can oscillate; a coarser CF integration adds to that wiggle. None keeps the
# QuantLib default (Gauss-Laguerre order 144). An int sets the Gauss-Laguerre order; a
# (relTolerance, maxEvaluations) pair selects the adaptive integrator (Andersen-Piterbarg style).
# Applied at the single construction site in pricing/_quantlib_utils, so EVERY engine build -- the
# calibration fit, the calibration_tests repricing, and the IV inversion -- uses the same accuracy and
# stays consistent. None on both is the exact pre-lever behaviour. The Gauss-Laguerre order is capped
# at 192 by QuantLib, so for more accuracy than that use the adaptive pair, e.g. (1e-8, 10000).
HESTON_INTEGRATION = None
BATES_INTEGRATION = None

# ---- Engine: soft Feller penalty on restart selection (PLAN.md Phase 3 Lever D/3) ----
# Accepted fits violate Feller (2*kappa*theta - eta^2 < 0) on essentially every day, which drives the
# elevated eta and the non-monotone wing density behind the IV oscillation. This biases the *choice*
# among restarts toward Feller-compliant, lower-eta fits by ranking each restart on
#     iv_rmse_sel + FELLER_PENALTY * max(0, eta^2 - 2*kappa*theta)
# instead of iv_rmse_sel alone. It does NOT change the LM objective, the acceptance gate, or the
# reported iv_rmse (those stay the unweighted IV-RMSE), so IV_RMSE_ACCEPT keeps its meaning; a restart
# is only preferred if its small fit cost buys a large Feller improvement. The selection IV-RMSE is
# ~0.005 and a typical violation is ~0.5, so a penalty of ~0.01 makes the two comparable; sweep up
# from there. 0.0 (the default) is the exact pre-lever behaviour (score == iv_rmse_sel). A hard
# in-LM penalty (custom ql.CostFunction) is the heavier follow-up if selection-level proves too weak.
#
# EMPIRICAL CAVEAT (measured on a calm 2012 day, both engines): this lever is WEAK. With the default
# seeds all restarts converge to the same (violating) basin, so the penalty has nothing to choose; with
# the FELLER_SEED_TEMPLATE seeds below it active, a strong penalty selects a compliant fit whose
# IV-RMSE blows through the gate (~0.057 vs the 0.02 cap) and is rejected. On SPX the short-dated data
# genuinely wants eta ~ 1 and a violated Feller, so neither this penalty nor a lower eta cap fixes the
# oscillation for free: they trade away fit or just relocate the peg to the eta ceiling. Prefer
# HESTON/BATES_INTEGRATION (Lever 2) for the oscillation, and grade any nonzero penalty with
# src/results/calibration_diagnostics.py (osc_frac, feller_neg) before trusting it.
FELLER_PENALTY = 0.0

# ---- Engine: cross-day parameter anchor / regularization (PLAN.md Phase 3 Lever 5) ----
# Day-to-day parameter instability (e.g. the 2019 kappa walk toward its floor) is damped by softly
# anchoring each day's fit to a PRIOR day's parameters. This is a TWO-PASS workflow, not a live
# neighbour dependency, so it stays parallelism-safe: Pass 1 is a normal run; Pass 2 reads the Pass-1
# calibrations.csv (via the calibrator's --PRIOR_FROM <path>) and, for each day, anchors to its
# previous accepted day(s) in that file. The anchor enters the engine two ways, BOTH gated on
# PARAM_ANCHOR_WEIGHT > 0 and on an anchor actually being supplied:
#   (1) a warm-start restart seeded at the prior params (so the prior basin is explored), and
#   (2) a Tikhonov term added to the restart-selection score:
#         score += PARAM_ANCHOR_WEIGHT * sum_p ((param_p - prior_p) / bound_span_p) ** 2
#       (span-normalised so every parameter contributes comparably). It does NOT touch the LM
#       objective, the acceptance gate, or the reported iv_rmse. With PARAM_ANCHOR_WEIGHT=0 (default)
#       or no --PRIOR_FROM, the engines add neither the seed nor the term, so the baseline is unchanged.
# PARAM_ANCHOR_LOOKBACK = 1 anchors to the single previous accepted day; N > 1 anchors to the median of
# the last N accepted days (a smoother, more robust prior). The selection IV-RMSE is ~0.005, so a unit
# anchor deviation of a full bound-span is huge: start PARAM_ANCHOR_WEIGHT small (~0.001-0.01) and grade
# day-to-day stability with src/results/calibration_diagnostics.py before trusting it.
PARAM_ANCHOR_WEIGHT = 0.01
PARAM_ANCHOR_LOOKBACK = 5

# ---- Engine: optimizer (Levenberg-Marquardt + EndCriteria) ----
# Both calibration engines (calibrate_heston / calibrate_bates) build their LM optimizer and stopping
# criteria from these. Plain numeric args (not live ql objects), so they live here; the engines
# construct ql.LevenbergMarquardt(*LM_ARGS) and ql.EndCriteria(*END_CRITERIA_ARGS).
#   LevenbergMarquardt(epsfcn, xtol, gtol)
LM_ARGS = (1e-8, 1e-8, 1e-8)
#   EndCriteria(maxIterations, maxStationaryStateIterations, rootEpsilon, functionEpsilon, gradientNormEpsilon)
END_CRITERIA_ARGS = (1000, 100, 1e-8, 1e-8, 1e-8)

# ---- Engine: wing weighting (PLAN.md Phase 3 Lever B) ----
# Up-weight OTM wing cells in the LM objective by |log(Kstar/S_ref)| so the fit stops trading the
# wings away for the body. weight = 1 + GAIN * (|log(K/S)| / SCALE) ** POWER  (1 at ATM, rising into
# both wings). Applied ONLY under the "vol" objective: "price" (RelativePriceError) already implicitly
# up-weights cheap wings via the price denominator, so stacking a wing weight there double-counts and
# over-pulls rho/eta into their bounds. GAIN=0.0 is the exact identity (uniform weights = current
# behaviour) and stays the default until the lever is validated. QuantLib normalises the weights, so
# only their ratios matter.
#
# Sign of GAIN (PLAN.md Phase 3 Lever 4, Problem 1 = call wing too convex). GAIN > 0 UP-weights the
# wings (the original use). GAIN < 0 DOWN-weights them, so on near-linear days the fit is not forced to
# bend the over-convex wing to chase a few wing cells; the result is clamped to WING_WEIGHT_FLOOR so a
# negative gain can never drive a weight to zero or negative (QuantLib needs positive weights). With
# GAIN=0 the floor never binds (weight is 1 everywhere), so the default path is byte-identical.
WING_WEIGHT_GAIN = 0.0      # 0 => uniform (no-op). >0 emphasises wings; <0 de-emphasises them.
WING_WEIGHT_POWER = 1.0     # ramp curvature: 1 linear in |log-moneyness|, 2 far-wing emphasis
WING_WEIGHT_SCALE = 0.05    # reference |log-moneyness| (~5% OTM) at which weight = 1 + GAIN
WING_WEIGHT_FLOOR = 1e-3    # positive clamp so GAIN<0 cannot zero/negate a wing weight

# ---- Engine: seed grid (calibrate_heston._seed_grid) ----
# var is the surface's median-vol^2, clamped to [SEED_VAR_LO, SEED_VAR_HI] (fallback when empty).
# Each template row is (v0_mult, kappa, theta_mult, eta, rho); the function expands v0 = var*v0_mult
# and theta = var*theta_mult into HestonProcess constructor order (v0, kappa, theta, eta, rho).
SEED_VAR_FALLBACK = 0.04
SEED_VAR_LO = 1e-3
SEED_VAR_HI = 0.25
SEED_GRID_TEMPLATE = [
    (1.0, 1.0, 1.0, 0.50, -0.70),
    (1.0, 3.0, 1.0, 1.00, -0.50),
    (1.0, 0.5, 1.0, 0.30, -0.90),
    (1.0, 5.0, 1.0, 0.80, -0.60),
    (0.8, 2.0, 1.2, 0.60, -0.75),
    (1.0, 8.0, 1.0, 1.20, -0.40),
]
# Feller-compliant restart seeds, appended to the seed grid ONLY when FELLER_PENALTY > 0 (so the
# default seed grid -- and the committed baseline -- is byte-for-byte unchanged). The selection-level
# Feller penalty can only PREFER a Feller-compliant fit if some restart actually converges into a
# compliant basin; empirically the 6 default seeds all funnel into the same (Feller-violating) minimum
# on calm days, leaving the penalty nothing to choose. These start with high kappa and low eta so
# 2*kappa*theta >= eta^2 at the start, giving the optimizer a compliant basin to find. Same
# (v0_mult, kappa, theta_mult, eta, rho) template shape as SEED_GRID_TEMPLATE.
FELLER_SEED_TEMPLATE = [
    (1.0, 10.0, 1.0, 0.20, -0.70),
    (1.0,  6.0, 1.5, 0.15, -0.60),
    (0.8,  8.0, 1.2, 0.25, -0.80),
]

# ---- Result routing (model x objective) ----
# Outputs are namespaced by model AND objective (results/<model>/calibrations/<objective>/) so Heston
# and Bates results coexist and the two objectives never clobber each other. This resolver is the
# single source of truth for the per-model, per-objective directory; calibrator_prototype._objective_paths
# is a thin wrapper over it, and the downstream results scripts import it as they are migrated
# (see PLAN.md, Bates extension). The layout is uniform across models -- heston lives under results/heston/, matching
# the migrated tree on disk and the results/*/calibrations/*/ pattern in .gitignore.
REPO = Path(__file__).resolve().parent.parent   # src/config.py -> repo root
# Output root. Defaults to the repo's results/ tree. The HC_RESULTS_DIR env var overrides it, so a run
# (e.g. an A/B lever test) can write to a scratch tree without touching the committed baseline; every
# output path flows through calib_paths/spec_path, which read this at call time. RESULTS is a Path, so
# as_dict() skips it and the run-spec/resume-guard are unaffected.
import os as _os
RESULTS = Path(_os.environ.get("HC_RESULTS_DIR", REPO / "results"))


def calib_paths(model, objective):
    """Resolve (calibrations.csv, rejections.csv, tests_dir) for a (model, objective) pair.

    Uniform layout: results/<model>/calibrations/<objective>/ holds calibrations.csv, rejections.csv
    and the per-day calibration_tests/ directory. No model is special-cased.

    Pure path resolution -- it does NOT create the directory. Read-only callers (validators, the
    figure scripts, a `save=False` notebook run) can resolve a path without leaving an empty tree on
    disk. Every writer mkdirs its own output dir before writing (e.g. the calibrator's TESTS.mkdir).
    """
    base = RESULTS / model / "calibrations" / objective
    return (base / "calibrations.csv",
            base / "rejections.csv",
            base / "calibration_tests")


def spec_path(model, objective):
    """Resolve the run's config snapshot path (config_spec.json), next to calibrations.csv.

    Kept separate from `calib_paths` (whose 3-tuple is unpacked positionally by callers) so adding
    the spec file does not shift that contract. `calibrator_prototype` writes this JSON each run; any
    downstream script (e.g. the figure/table builders) can load it to recover the exact knobs a run
    used without hard-coding values.

    Pure path resolution -- it does NOT create the directory (see `calib_paths`). `_utils.write_config_spec`
    mkdirs the parent before writing.
    """
    base = RESULTS / model / "calibrations" / objective
    return base / "config_spec.json"


def as_dict():
    """The calibration 'specification': every JSON-serializable module-level constant in this config.

    Reflects over this module's namespace and keeps each public name whose value `json` can encode
    (ints, floats, strings, bools, and nested lists/tuples/dicts of them: the bounds dicts, the seed
    grid, the optimizer args, ...). Callables (`day_count`/`calendar`/`calib_paths`/`spec_path`/this
    function) and `Path` objects (`REPO`/`RESULTS`) are skipped. New knobs are captured automatically,
    so the snapshot never drifts from the live config. Tuples round-trip through JSON as lists.
    """
    import json as _json
    spec = {}
    for name, val in globals().items():
        if name.startswith('_') or callable(val):
            continue
        try:
            _json.dumps(val)
        except (TypeError, ValueError):
            continue
        spec[name] = val
    return spec
