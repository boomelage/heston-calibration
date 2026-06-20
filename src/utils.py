import numpy as np
import pandas as pd
import QuantLib as ql

from config import OTM_MONEYNESS_CUTOFF, OTM_MONEYNESS_FLOOR

def df_moneyness(df):
    """Ratio moneyness: spot/strike for calls, strike/spot for puts.

    < 1 => out-of-the-money for either type (call with strike above spot, put with strike below
    spot); > 1 => in-the-money; == 1 => at-the-money.
    """
    return np.where(
        df['w']=='call',
        df['spot_price'] / df['strike_price'],
        df['strike_price'] / df['spot_price']
    )

def _prepare_options(raw):
    """Clean a raw CBOE trades frame and keep only OTM calls and puts.

    Selects/renames the column subset, maps option_type C/P -> w call/put, computes calendar
    `days_to_maturity` (>0 only), and keeps positive IV/spot/strike. Then keeps only the
    out-of-the-money rows (`OTM_MONEYNESS_FLOOR < moneyness < OTM_MONEYNESS_CUTOFF`): OTM calls (strike
    above spot), OTM puts (strike below spot), which together span both wings of the smile. The FLOOR
    drops the deep-OTM lottery-ticket tail (its extreme prices peg the fit to the bounds). Returns the
    cleaned snapshot with the helper `moneyness` column dropped. The calibrator calls this in-memory, so
    it can build a day's surface straight from a raw trades file (there is no separate extraction script).
    """
    raw = raw[
        [
            'underlying_symbol', 'quote_datetime', 
            'sequence_number', 
            'root',
            'expiration', 'strike', 'option_type', 'trade_size',
            'trade_price',
            'best_bid', 'best_ask', 'trade_iv', 'trade_delta', 'underlying_bid',
        ]
    ].copy()
    df = raw.rename(columns={'strike':'strike_price','option_type':'w','underlying_bid':'spot_price'}).copy()
    df['quote_datetime'] = pd.to_datetime(df['quote_datetime'])
    df['expiration'] = pd.to_datetime(df['expiration'],format='%Y-%m-%d')
    df['days_to_maturity'] = (df['expiration'] - df['quote_datetime']) / pd.Timedelta(days=1)
    df['days_to_maturity'] = df['days_to_maturity'].astype(int)
    df = df[df['days_to_maturity']>0]
    df = df[df['spot_price']>0]
    df = df[df['strike_price']>0]
    df = df[df['trade_iv']>0].copy()
    df['w'] = df['w'].replace({'C': 'call', 'P': 'put'})
    df = df[['quote_datetime', 'strike_price', 'w', 'trade_size', 'trade_price','trade_iv', 'spot_price','days_to_maturity']]
    df['moneyness'] = df_moneyness(df)
    df = df[(df['moneyness'] > OTM_MONEYNESS_FLOOR) & (df['moneyness'] < OTM_MONEYNESS_CUTOFF)]
    return df.drop(columns='moneyness').dropna().copy()


def implied_vol(price, w, S, K, r, g, T):
    """Invert a Black price to an implied vol (vol points), dividend-consistent via the forward."""
    if not np.isfinite(price) or price <= 0 or T <= 0:
        return np.nan
    F = S * np.exp((r - g) * T)
    disc = np.exp(-r * T)
    opt = ql.Option.Call if w == "call" else ql.Option.Put
    try:
        sd = ql.blackFormulaImpliedStdDev(opt, K, F, price, disc)
        return sd / np.sqrt(T)
    except RuntimeError:
        return np.nan


# ---- Heston-engine helpers (used by the results/ figure scripts: example_surface, make_eps, smiles).
# Moved here from the former src/results/surfaces/utils.py so there is one shared utils module. These
# build/evaluate a QuantLib Heston engine, distinct from the price-inversion `implied_vol` above:
# `heston_implied_vol` PRICES a strike under the engine and then inverts that model price to a Black
# vol, whereas `implied_vol` inverts a price you already have. The names are kept separate because the
# signatures differ.

def heston_implied_vol(strike, maturity_date, spot, heston_engine, bsm_process, w=None):
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