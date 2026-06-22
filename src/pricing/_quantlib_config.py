import QuantLib as ql

# ---- QuantLib date conventions (single source of truth) ----
# The day count for all flat-forward curves and the calendar for the calibration helpers. These were
# hardcoded as `ql.Actual365Fixed()` / `ql.UnitedStates(ql.UnitedStates.NYSE)` in five places (both
# engines, both utils engine builders, the vanilla pricer, and the figure scripts); they now live here.
# Under the flat-forward curves used throughout, the helper calendar is immaterial to the fit, but it
# is kept consistent so a future non-flat curve does not silently disagree between modules.
# `day_count(name=None)` / `calendar(name=None)` return a FRESH instance (QuantLib value types are cheap
# and safe to rebuild); `name=None` resolves to the canonical choice below.
DAY_COUNT_NAME = "Actual365Fixed"
CALENDAR_NAME = "UnitedStates.NYSE"
_DAY_COUNTS = {
    "Actual365Fixed": lambda: ql.Actual365Fixed(),
    "Thirty360.USA": lambda: ql.Thirty360(ql.Thirty360.USA),
}
_CALENDARS = {
    "UnitedStates.NYSE": lambda: ql.UnitedStates(ql.UnitedStates.NYSE),
}


# ---- Monte Carlo defaults (vanilla_pricer.mc_heston_price -> _quantlib_utils._mc_heston_engine) ----
# Defaults for the MC European Heston engine. A vanilla_pricer ctor arg of None resolves to these, so
# the MC settings live in one place alongside the other QuantLib configuration.
MC_STEPS = 10                   # time steps per path
MC_RNG = "pseudorandom"         # RNG; alternative: "lowdiscrepancy"
MC_NUM_PATHS = 100000           # required samples
MC_SEED = 1312                  # RNG seed


def day_count(name=None):
    """Fresh day-count instance for `name` (defaults to DAY_COUNT_NAME)."""
    name = name or DAY_COUNT_NAME
    try:
        return _DAY_COUNTS[name]()
    except KeyError:
        raise ValueError(f"unsupported day count {name!r}; known: {sorted(_DAY_COUNTS)}")


def calendar(name=None):
    """Fresh calendar instance for `name` (defaults to CALENDAR_NAME)."""
    name = name or CALENDAR_NAME
    try:
        return _CALENDARS[name]()
    except KeyError:
        raise ValueError(f"unsupported calendar {name!r}; known: {sorted(_CALENDARS)}")