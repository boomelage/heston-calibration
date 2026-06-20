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