"""Central tuning constants for the Heston calibration pipeline.

Every model/calibration knob lives here so the orchestrator (`calibrator_prototype.py`), the engine
(`calibrate_heston.py`), the OTM filter (`utils._prepare_options`) and the validator
(`validate_calibrations.py`) all read one source of truth. PLAN.md Phase 3 tunes these values
(MIN_DTM, the box bounds, a Feller penalty, ...); editing one line here is the whole change.

Why a Python module, not JSON: several "constants" are not trivially serialisable or are coupled to
code. The box bounds are only correct in `model.params()` order [theta, kappa, eta, rho, v0]; the
objective map points at live QuantLib enum objects (kept next to the engine); the seed grid is a
template a function expands. Expressing the bounds as a named dict (`BOUNDS`) with `PARAM_ORDER`
turns the positional-order footgun into a single declared mapping.
"""

# ---- Surface selection / coverage (calibrator_prototype._select_surface / calibrate_by_day) ----
# Pooling the whole day (one fit) lets us take more maturities than the old per-spot path.
MAX_NT = 20          # maturities kept, ranked by traded volume (20 reaches ~485d; volume ranking
                     # caps a top-12 surface at ~394d even when MAX_DTM is larger)
MAX_NK = 40          # strikes kept per wing (highest OTM puts, lowest OTM calls), nearest the money
STRIKE_GRID = 5.0    # SPX near-money strike increment; normalised K* is snapped to this grid
MIN_DTM = 14         # drop ultra-short maturities: Heston fits them poorly and they drive
                     # eta/kappa to extremes (Feller-violating), polluting the pooled fit
MAX_DTM = 730        # drop very long maturities (thin, stale quotes)
MIN_MATS = 3         # require a genuinely multi-maturity surface (identification)
MIN_STRIKES = 5      # require a real strike range
MIN_CELLS = 12       # non-NaN surface cells required (target >= MIN_MATS x MIN_STRIKES)
MAX_MOVE_PCT = 0.03  # intraday spot range above this flags the day (sticky-moneyness strained)

# ---- OTM filter (utils._prepare_options) ----
# Keep rows with FLOOR < ratio-moneyness < CUTOFF (see utils.df_moneyness). The CUTOFF drops near-ATM
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

# ---- Engine: acceptance gate / tolerances ----
# IV-space RMSE (vol points): model-implied vol vs market vol per helper. ~2 vol points is a tight
# fit and is the metric the surface is quoted in. Boundary-pegged params are rejected separately.
IV_RMSE_ACCEPT = 0.02   # max IV-space RMSE (vol points) for an accepted fit
BOUND_TOL = 1e-3        # fraction of a bound's span within which a param counts as "pegged"

# ---- Engine: IV inversion controls ----
# For BlackCalibrationHelper.impliedVolatility(price, accuracy, maxEval, lo, hi).
IV_ACC, IV_MAXEVAL, IV_LO, IV_HI = 1e-6, 500, 1e-4, 5.0

# ---- Engine: in-engine LM objective ----
# Selection and the gate always run off IV-space RMSE; the objective only changes what each restart
# minimises. The string->QuantLib-enum map (`_ERR`) stays next to the engine (live ql objects).
OBJECTIVE_NAMES = ("price", "vol")
DEFAULT_OBJECTIVE = "vol"

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
