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
MAX_NT = 20          # maturities kept, ranked by traded volume (20 reaches ~485d; volume ranking
                     # caps a top-12 surface at ~394d even when MAX_DTM is larger)
MAX_NK = 40          # strikes kept per wing (highest OTM puts, lowest OTM calls), nearest the money
MIN_NK = 2      # min distinct K* strikes a wing must have for a maturity to be kept (both wings);
                     # a single-strike wing cannot anchor a smile. Maturities failing this are skipped
                     # before the MAX_NT volume cap, so the cap counts only wing-qualifying maturities
STRIKE_GRID = 5.0    # SPX near-money strike increment; normalised K* is snapped to this grid
MIN_DTM = 14         # drop ultra-short maturities: Heston fits them poorly and they drive
                     # eta/kappa to extremes (Feller-violating), polluting the pooled fit
MAX_DTM = 730        # drop very long maturities (thin, stale quotes)
MIN_MATS = 3         # require a genuinely multi-maturity surface (identification)
MIN_STRIKES = 5      # require a real strike range
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
WING_WEIGHT_GAIN = 0.0      # 0 => uniform (no-op). Try 0.5, 1.0, 2.0 under --OBJECTIVE vol.
WING_WEIGHT_POWER = 1.0     # ramp curvature: 1 linear in |log-moneyness|, 2 far-wing emphasis
WING_WEIGHT_SCALE = 0.05    # reference |log-moneyness| (~5% OTM) at which weight = 1 + GAIN

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

# ---- Result routing (model x objective) ----
# Outputs are namespaced by model AND objective (results/<model>/calibrations/<objective>/) so Heston
# and Bates results coexist and the two objectives never clobber each other. This resolver is the
# single source of truth for the per-model, per-objective directory; calibrator_prototype._objective_paths
# is a thin wrapper over it, and the downstream results scripts import it as they are migrated
# (see PLAN.md, Bates extension). The layout is uniform across models -- heston lives under results/heston/, matching
# the migrated tree on disk and the results/*/calibrations/*/ pattern in .gitignore.
REPO = Path(__file__).resolve().parent.parent   # src/config.py -> repo root
RESULTS = REPO / "results"


def calib_paths(model, objective):
    """Resolve (calibrations.csv, rejections.csv, tests_dir) for a (model, objective) pair.

    Uniform layout: results/<model>/calibrations/<objective>/ holds calibrations.csv, rejections.csv
    and the per-day calibration_tests/ directory. No model is special-cased.
    """
    base = RESULTS / model / "calibrations" / objective
    base.mkdir(parents=True, exist_ok=True)
    return (base / "calibrations.csv",
            base / "rejections.csv",
            base / "calibration_tests")


def spec_path(model, objective):
    """Resolve the run's config snapshot path (config_spec.json), next to calibrations.csv.

    Kept separate from `calib_paths` (whose 3-tuple is unpacked positionally by callers) so adding
    the spec file does not shift that contract. `calibrator_prototype` writes this JSON each run; any
    downstream script (e.g. the figure/table builders) can load it to recover the exact knobs a run
    used without hard-coding values.
    """
    base = RESULTS / model / "calibrations" / objective
    base.mkdir(parents=True, exist_ok=True)
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
