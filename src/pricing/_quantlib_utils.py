"""Centralized QuantLib construction for the pricing/calibration stack.

Every QuantLib process, engine and option used across the repo is built here, so a change in
QuantLib's constructor argument order (or the date conventions) is a one-line edit in this module
instead of a hunt through `vanilla_pricer.py`, `utils.py`, `calibrate_heston.py` and
`calibrate_bates.py`. The date conventions themselves live in `_quantlib_config.py`.

Constructor argument orders pinned here (QuantLib 1.35) -- the single home of these orderings:
  - HestonProcess(ts_r, ts_g, S0, v0, kappa, theta, eta(=sigma), rho)
  - BatesProcess (ts_r, ts_g, S0, v0, kappa, theta, eta,        rho, lambda, nu, delta)
The public helpers take the params in (kappa, theta, rho, eta, v0[, lambda_, nu, delta]) order to
match the DataFrame column contracts; the constructor order above is applied internally so callers
never restate it.

The engine builders return a fixed bundle `(engine, s_handle, ts_r, ts_g, day_count)`: the engine for
pricing, the handles/day-count so a caller can build the matching Black process for IV inversion
(`utils.build_heston_engine`) without rebuilding the term structures. Pricers that only need the
engine unpack with `engine, *_ = ...`.
"""
import QuantLib as ql

from pricing._quantlib_config import day_count


class _quantlib_utils:
    def __init__(self, day_count_name=None):
        # Store only the NAME (a string/None), never a live SwigPyObject: a vanilla_pricer holds an
        # instance of this class, and joblib pickles it to its worker processes (df_* pricers). A
        # stored ql.DayCounter is unpicklable ("cannot pickle 'SwigPyObject'"). The day_count property
        # rebuilds a fresh instance on access -- QuantLib value types are cheap and safe to rebuild.
        self.day_count_name = day_count_name

    @property
    def day_count(self):
        # None resolves to the canonical DAY_COUNT_NAME; a name is validated in day_count().
        return day_count(self.day_count_name)

    # ---- primitives -------------------------------------------------------------------------
    def _spot_handle(self, s):
        return ql.QuoteHandle(ql.SimpleQuote(float(s)))

    def _term_structures(self, r, g, calculation_date):
        """Flat-forward r and g curves on the shared day count. Does not touch the eval date."""
        ts_r = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, float(r), self.day_count))
        ts_g = ql.YieldTermStructureHandle(ql.FlatForward(calculation_date, float(g), self.day_count))
        return ts_r, ts_g

    def _option_type(self, w):
        if w == 'call':
            return ql.Option.Call
        if w == 'put':
            return ql.Option.Put
        raise ValueError("call/put flag w should be either 'call' or 'put'")

    def _european_option(self, w, k, t, calculation_date):
        """A European vanilla option struck at k, expiring t calendar days after calculation_date."""
        expiration_date = calculation_date + ql.Period(int(t), ql.Days)
        payoff = ql.PlainVanillaPayoff(self._option_type(w), float(k))
        exercise = ql.EuropeanExercise(expiration_date)
        return ql.VanillaOption(payoff, exercise)

    # ---- processes (the single home of the QuantLib constructor arg order) --------------------
    def heston_process(self, ts_r, ts_g, s_handle, kappa, theta, rho, eta, v0):
        return ql.HestonProcess(ts_r, ts_g, s_handle,
                                float(v0), float(kappa), float(theta), float(eta), float(rho))

    def bates_process(self, ts_r, ts_g, s_handle, kappa, theta, rho, eta, v0, lambda_, nu, delta):
        return ql.BatesProcess(ts_r, ts_g, s_handle,
                               float(v0), float(kappa), float(theta), float(eta), float(rho),
                               float(lambda_), float(nu), float(delta))

    # ---- engines (build term structures + process + engine, return the reusable bundle) ------
    def _heston_engine(self, s, r, g, kappa, theta, rho, eta, v0, calculation_date=None):
        if calculation_date is None:
            calculation_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = calculation_date
        s_handle = self._spot_handle(s)
        ts_r, ts_g = self._term_structures(r, g, calculation_date)
        process = self.heston_process(ts_r, ts_g, s_handle, kappa, theta, rho, eta, v0)
        engine = ql.AnalyticHestonEngine(ql.HestonModel(process))
        return engine, s_handle, ts_r, ts_g, self.day_count

    def _mc_heston_engine(self, s, r, g, kappa, theta, rho, eta, v0,
                          rng, steps, numPaths, seed, calculation_date=None):
        if calculation_date is None:
            calculation_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = calculation_date
        s_handle = self._spot_handle(s)
        ts_r, ts_g = self._term_structures(r, g, calculation_date)
        process = self.heston_process(ts_r, ts_g, s_handle, kappa, theta, rho, eta, v0)
        engine = ql.MCEuropeanHestonEngine(process, rng, steps, requiredSamples=numPaths, seed=seed)
        return engine, s_handle, ts_r, ts_g, self.day_count

    def _bates_engine(self, s, r, g, kappa, theta, rho, eta, v0, lambda_, nu, delta,
                      calculation_date=None):
        if calculation_date is None:
            calculation_date = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = calculation_date
        s_handle = self._spot_handle(s)
        ts_r, ts_g = self._term_structures(r, g, calculation_date)
        process = self.bates_process(ts_r, ts_g, s_handle,
                                     kappa, theta, rho, eta, v0, lambda_, nu, delta)
        engine = ql.BatesEngine(ql.BatesModel(process))
        return engine, s_handle, ts_r, ts_g, self.day_count
