from typing import NamedTuple

import numpy as np
import pandas as pd

from _utils import df_moneyness
from config import (
    MAX_NT, MAX_NK,
    OTM_MONEYNESS_FLOOR, OTM_MONEYNESS_CUTOFF,
    MIN_DTM, MAX_DTM, MAX_MOVE_PCT, STRIKE_GRID,
    MIN_NK
)

class PreparedDay(NamedTuple):
    """What prepare_surface hands back: the moneyness-normalised trades (with the added Kstar column)
    plus the day's date and intraday-spot summary. Rate lookup and the surface pivot stay in the
    orchestrator. A tuple, so the orchestrator can unpack it positionally."""
    df: pd.DataFrame
    date: pd.Timestamp
    S_ref: float
    spot_min: float
    spot_max: float
    spot_range_pct: float
    high_move: bool


class SkipDay(Exception):
    """A day that cannot be turned into a usable surface (raised by prepare_surface).

    Carries the rejection `reason` category (e.g. 'no_trades') and the human `detail`, plus any
    coverage counts known at the point it is raised (passed through as keyword args). The orchestrator
    catches it and converts it into a rejection row via `_skip_day`, which owns `test_path` and the
    rejections.csv schema. Keeping that conversion in the caller is why prepare_surface signals by
    raising rather than returning a sentinel."""

    def __init__(self, reason, detail, **coverage):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.coverage = coverage

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
    df = df[df['strike_price']>0].copy()
    df['w'] = df['w'].replace({'C': 'call', 'P': 'put'})
    df = df[['quote_datetime', 'strike_price', 'w', 'trade_size', 'trade_price','trade_iv', 'spot_price','days_to_maturity']]
    df['moneyness'] = df_moneyness(df)
    df = df[(df['moneyness'] > OTM_MONEYNESS_FLOOR) & (df['moneyness'] < OTM_MONEYNESS_CUTOFF)]
    return df.drop(columns='moneyness').dropna().copy()

def prepare_surface(df):
    """Turn a cleaned OTM snapshot into the day's calibratable, moneyness-normalised trades.

    Keeps `trade_iv > 0` and `MIN_DTM <= days_to_maturity <= MAX_DTM` (ultra-short maturities are
    dropped: Heston fits them poorly and they drive eta/kappa to Feller-violating extremes). Picks one
    volume-weighted reference spot `S_ref` for the day and re-strikes each trade to `K* = (K/S_row) *
    S_ref`, snapped to the SPX 5-point grid (`STRIKE_GRID`), so trades at different intraday spots align
    on shared surface columns under the sticky-moneyness assumption. Flags `high_move` (and warns) when
    the intraday spot range exceeds `MAX_MOVE_PCT`; the day is still returned.

    Raises `SkipDay("no_trades", ...)` if nothing survives the IV/DTM filter. Otherwise returns a
    `PreparedDay` (the df with the added `Kstar` column, the day's `date`, `S_ref`, the intraday spot
    min/max/range and `high_move`). Rate lookup (r, g) and the surface pivot stay in the orchestrator.
    """
    df = _prepare_options(df)
    df = df[(df['trade_iv'] > 0) & (df['days_to_maturity'] >= MIN_DTM)
            & (df['days_to_maturity'] <= MAX_DTM)].copy()
    if df.empty:
        raise SkipDay("no_trades", "no trades after IV/DTM filter")
    df['quote_datetime'] = pd.to_datetime(df['quote_datetime'])
    date = df['quote_datetime'].dt.floor('D').unique()[0]

    # One reference spot for the whole day (volume-weighted). The intraday range that the
    # sticky-moneyness re-centring assumes is mild; a large range strains that assumption.
    S_ref = float(np.average(df['spot_price'], weights=df['trade_size']))
    spot_min, spot_max = float(df['spot_price'].min()), float(df['spot_price'].max())
    spot_range_pct = spot_max / spot_min - 1.0
    high_move = spot_range_pct > MAX_MOVE_PCT
    if high_move:
        print(f"WARNING {pd.Timestamp(date).date()}: intraday spot range {spot_range_pct:.2%} "
              f"> {MAX_MOVE_PCT:.0%}; normalisation to S_ref={S_ref:.1f} may be strained")

    # Moneyness-normalise: each trade keeps m = K / S_row but is re-struck to K* = m * S_ref and
    # snapped to the SPX strike grid, so trades at different intraday spots align on shared columns.
    df['Kstar'] = (df['strike_price'] / df['spot_price']) * S_ref
    df['Kstar'] = (df['Kstar'] / STRIKE_GRID).round() * STRIKE_GRID
    return PreparedDay(df, date, S_ref, spot_min, spot_max, spot_range_pct, high_move)

def _select_surface(df):
    """Pick the day's calibration surface in moneyness-normalised (K*) strike space.

    The trades are OTM calls and puts spanning both wings (see `_prepare_options`). Walk maturities in
    descending traded volume and keep a maturity only if BOTH wings carry at least MIN_NK distinct
    K* strikes (a single-strike wing cannot anchor a smile); within each kept maturity take the MAX_NK
    nearest-the-money strikes per wing on K* (already centred on S_ref): the highest OTM puts (below
    spot) and the lowest OTM calls (above spot). Stop once MAX_NT qualifying maturities are collected,
    so the volume cap is applied to the maturities that survive the wing gate (not consumed by ones that
    fail it). Returns the selected snapshot rows with original strike/spot retained for repricing, or
    None if no maturity qualifies. The public `select_surface` wrapper turns that None into a
    `SkipDay("thin", ...)`.
    """
    byt = df.groupby('days_to_maturity')
    vol_by_t = byt['trade_size'].sum().sort_values(ascending=False)

    selected = []
    for t in vol_by_t.index:
        dft = byt.get_group(t)
        cK = np.sort(dft.loc[dft['w'] == 'call', 'Kstar'].unique())
        pK = np.sort(dft.loc[dft['w'] == 'put', 'Kstar'].unique())
        if len(cK) >= MIN_NK and len(pK) >= MIN_NK:
            keep = list(pK[-min(len(pK), MAX_NK):]) + list(cK[:min(len(cK), MAX_NK)])
            selected.append(dft[dft['Kstar'].isin(keep)])
            if len(selected) >= MAX_NT:
                break
    if not selected:
        return None
    return pd.concat(selected, ignore_index=True)

def select_surface(df):
    """Pick the day's surface and pivot it to a Kstar x maturity IV matrix.

    Wraps `_select_surface` (top-volume wing-qualifying maturities x nearest-money strikes per wing). Raises
    `SkipDay("thin", "no usable maturities")` when no maturity qualifies. Otherwise returns
    `(sel, surf)`: `sel` is the selected snapshot rows sorted by ascending trade_size (the order the
    pivot's `aggfunc='last'` keeps the highest-volume trade per cell, and the same frame the
    orchestrator reprices from), and `surf` is the pivoted IV matrix the engine calibrates against.
    """
    sel = _select_surface(df)
    if sel is None:
        raise SkipDay("thin", "no usable maturities")
    sel = sel.sort_values('trade_size', ascending=True)
    surf = sel.pivot_table(index='Kstar', columns='days_to_maturity',
                           values='trade_iv', aggfunc='last')
    return sel, surf
