import QuantLib as ql
import numpy as np

def implied_vol(strike, maturity_date, spot, heston_engine, bsm_process, w=None):
    """Price a European option under Heston, invert to a Black vol. NaN if it can't converge.
    If w is None, picks the OTM side (call above spot, put below)."""
    if w is not None:
        payoff_type = ql.Option.Call if w == 'call' else ql.Option.Put
    else:
        payoff_type = ql.Option.Call if strike >= spot else ql.Option.Put
    option = ql.EuropeanOption(ql.PlainVanillaPayoff(payoff_type, strike),
                               ql.EuropeanExercise(maturity_date))
    option.setPricingEngine(heston_engine)
    price = option.NPV()
    try:
        return option.impliedVolatility(price, bsm_process, 1e-6, 500, 1e-4, 5.0)
    except RuntimeError:
        return np.nan

def build_heston_engine(row, calculation_date):
    """Rebuild the Heston model from one calibrations.csv row; return its pricing engine plus the
    spot handle and term structures (reused to build the Black process the inversion runs against)."""
    ql.Settings.instance().evaluationDate = calculation_date
    day_count = ql.Actual365Fixed()
    r_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, float(row['risk_free_rate']), day_count))
    g_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, float(row['dividend_rate']), day_count))
    s_handle = ql.QuoteHandle(ql.SimpleQuote(float(row['spot_price'])))
    # HestonProcess constructor order: (r, g, S0, v0, kappa, theta, sigma=eta, rho)
    process = ql.HestonProcess(
        r_ts, g_ts, s_handle,
        float(row['v0']), float(row['kappa']), float(row['theta']),
        float(row['eta']), float(row['rho'])
    )
    engine = ql.AnalyticHestonEngine(ql.HestonModel(process))
    return engine, s_handle, r_ts, g_ts, day_count

def heston_price(strike, maturity_date, spot, w, heston_engine):
    """Price the European option under Heston. Returns (NPV)."""
    payoff_type = ql.Option.Call if w == 'call' else ql.Option.Put
    option = ql.EuropeanOption(ql.PlainVanillaPayoff(payoff_type, strike),
                               ql.EuropeanExercise(maturity_date))
    option.setPricingEngine(heston_engine)
    return option.NPV()
