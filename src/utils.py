import json
import datetime
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import QuantLib as ql

import config
from config import OTM_MONEYNESS_CUTOFF, OTM_MONEYNESS_FLOOR
from pricing._quantlib_utils import _quantlib_utils

# One shared builder: it carries the canonical day count and is the single place QuantLib engines
# are constructed (see pricing/_quantlib_utils.py).
_qu = _quantlib_utils()

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


# ---- Model-engine helpers (used by the results/ figure scripts: make_surface, plot_surfaces, smiles).
# Moved here from the former src/results/surfaces/utils.py so there is one shared utils module. These
# evaluate whatever QuantLib pricing engine they are handed -- Heston OR Bates, built by
# build_model_engine -- so they are model-agnostic (the engine carries the params). Distinct from the
# price-inversion `implied_vol` above: `model_implied_vol` PRICES a strike under the engine and then
# inverts that model price to a Black vol, whereas `implied_vol` inverts a price you already have. The
# names are kept separate because the signatures differ.

def model_implied_vol(strike, maturity_date, spot, model_engine, bsm_process, w=None):
    """Price a European option under the given engine (Heston or Bates), invert to a Black vol.
    NaN if it can't converge. If w is None, picks the OTM side (call above spot, put below)."""
    if w is not None:
        payoff_type = ql.Option.Call if w == 'call' else ql.Option.Put
    else:
        payoff_type = ql.Option.Call if strike >= spot else ql.Option.Put
    option = ql.EuropeanOption(ql.PlainVanillaPayoff(payoff_type, strike),
                               ql.EuropeanExercise(maturity_date))
    option.setPricingEngine(model_engine)
    price = option.NPV()
    try:
        return option.impliedVolatility(price, bsm_process, 1e-6, 500, 1e-4, 5.0)
    except RuntimeError:
        return np.nan


def build_heston_engine(row, calculation_date):
    """Rebuild the Heston model from one calibrations.csv row; return its pricing engine plus the
    spot handle and term structures (reused to build the Black process the inversion runs against).
    Thin wrapper over the shared builder so the QuantLib construction lives in exactly one place."""
    return _qu._heston_engine(
        s=row['spot_price'], r=row['risk_free_rate'], g=row['dividend_rate'],
        kappa=row['kappa'], theta=row['theta'], rho=row['rho'], eta=row['eta'], v0=row['v0'],
        calculation_date=calculation_date)


def model_price(strike, maturity_date, spot, w, model_engine):
    """Price the European option under the given engine (Heston or Bates). Returns (NPV)."""
    payoff_type = ql.Option.Call if w == 'call' else ql.Option.Put
    option = ql.EuropeanOption(ql.PlainVanillaPayoff(payoff_type, strike),
                               ql.EuropeanExercise(maturity_date))
    option.setPricingEngine(model_engine)
    return option.NPV()


def build_bates_engine(row, calculation_date):
    """Rebuild the Bates model from one calibrations.csv row; return its pricing engine plus the spot
    handle and term structures (same return shape as build_heston_engine). The row must carry the
    jump triple `lambda_, nu, delta` alongside the five Heston params. Thin wrapper over the shared
    builder so the QuantLib construction lives in exactly one place."""
    return _qu._bates_engine(
        s=row['spot_price'], r=row['risk_free_rate'], g=row['dividend_rate'],
        kappa=row['kappa'], theta=row['theta'], rho=row['rho'], eta=row['eta'], v0=row['v0'],
        lambda_=row['lambda_'], nu=row['nu'], delta=row['delta'],
        calculation_date=calculation_date)


def build_model_engine(row, calculation_date, model):
    """Dispatch to the Heston or Bates engine builder by model name. Same return shape either way, so
    the figure scripts (make_surface, smiles) stay model-agnostic. The downstream pricing/inversion
    helpers (`model_price`, `model_implied_vol`) take the engine and work with either."""
    if model == "bates":
        return build_bates_engine(row, calculation_date)
    return build_heston_engine(row, calculation_date)


# ---- Run-specification snapshot (written by calibrator_prototype next to calibrations.csv) ----
# A Python-readable record of the exact config a run used, so downstream scripts that build LaTeX
# fragments (e.g. src/results/smiles/smiles.py) can recover bounds/coverage/gate knobs without
# hard-coding them. The config values come from config.as_dict() (reflection, so new knobs appear
# automatically); write_config_spec wraps them with a small `_run` metadata block.

def _git_commit():
    """Short HEAD hash for the run record, or None outside a git checkout / when git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent), stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def write_config_spec(model, objective, limit, n_accepted, n_rejected):
    """Write config_spec.json next to calibrations.csv: a snapshot of the config the run used plus a
    `_run` metadata block (timestamp, git commit, model/objective/limit, accept-reject tally).

    Config values come from config.as_dict() (every JSON-serializable module-level constant, captured
    by reflection). Downstream consumers `json.load` this to read the run's bounds, coverage knobs and
    gate. Returns the written path. The caller writes it whenever calibrations.csv is written (and
    removes it alongside)."""
    spec = {
        "_run": {
            "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "model": model,
            "objective": objective,
            "limit": limit,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "n_attempted": n_accepted + n_rejected,
        },
    }
    spec.update(config.as_dict())
    path = config.spec_path(model, objective)
    path.write_text(json.dumps(spec, indent=2))
    return path