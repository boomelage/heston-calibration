import os
import json
import datetime
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import QuantLib as ql

import config
from config import OTM_MONEYNESS_CUTOFF, OTM_MONEYNESS_FLOOR
from qlpricing._quantlib_utils import _quantlib_utils

# One shared builder: it carries the canonical day count and is the single place QuantLib engines
# are constructed (see pricing/_quantlib_utils.py). pricing/ never imports config, so the project's
# CF-integration accuracy is injected; this instance must integrate exactly like the calibration fit.
_qu = _quantlib_utils(heston_integration=config.HESTON_INTEGRATION,
                      bates_integration=config.BATES_INTEGRATION)

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
# Moved here from the former src/results/surfaces/_utils.py so there is one shared _utils module. These
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2))
    return path

def _file_date(p):
    """The trailing _<date> token of a raw trades filename, e.g. '2024-10-15' from
    'UnderlyingOptionsTradesCalcs_2024-10-15.csv'. Same string the per-day rows are keyed by, so it
    matches the resume `processed_dates` set."""
    b = os.path.basename(p)
    return b[b.rfind('_') + 1:-4]


# ---- Pure surface-selection helpers (used by the results/ figure scripts: smiles, ...). These are
# config-free on purpose: the figure scripts pass their _results_config knobs (NT/TMIN/TMAX/MKTMONSTEP)
# explicitly, so this module never imports _results_config and stays usable by the calibrator.

def _normalize_dates(dates):
    """Accept a single %Y-%m-%d date string or a list of them; return a list of strings."""
    if isinstance(dates, str):
        return [dates]
    return [str(d) for d in dates]


def _clip_maturities(T, tmin, tmax):
    """Keep only maturities (in days) within the [tmin, tmax] window before sparse selection.
    Either bound is optional: `tmin=None` removes the lower bound, `tmax=None` the upper, and
    both `None` keeps every maturity. Applied to the candidate maturities (calibrated or the
    MATURITIES_DAYS fallback) so the displayed smiles are restricted to the chosen tenor band."""
    lo = -np.inf if tmin is None else tmin
    hi = np.inf if tmax is None else tmax
    return [t for t in sorted(T) if lo <= t <= hi]


def _sparse_maturities(T, nt):
    """Sparsely pick at most `nt` maturities from the sorted list `T`. Always keeps the lowest and
    highest; the remaining nt-2 are spaced as equally as possible across the interior by indexing
    `T` on an evenly spaced grid. `nt=None` (or nt >= len(T)) returns the full sorted list, i.e. draw
    every available maturity."""
    T = sorted(T)
    if nt is None or nt >= len(T) or nt <= 0:
        return T
    if nt == 1:
        return [T[0]]
    idx = np.unique(np.linspace(0, len(T) - 1, nt).round().astype(int))
    return [T[i] for i in idx]


def _sparse_strikes(sub, step):
    """Thin one wing's market points so their `moneyness` is spaced ~`step` apart (percentage terms).
    Selection is done per maturity (`cmat`) so each smile keeps its own evenly spaced subset. From a
    maturity's min moneyness we build a grid at min, min+step, min+2*step, ... up to its max, and for
    each grid node keep the row whose moneyness is nearest. Deduping keeps the lowest and highest
    available moneyness on each smile. `step=None` or `step<=0` (or an empty input) returns `sub`
    unchanged, i.e. draw every point."""
    if step is None or step <= 0 or sub.empty:
        return sub
    keep = []
    for _, grp in sub.groupby('cmat'):
        m = grp['moneyness'].to_numpy()
        lo, hi = m.min(), m.max()
        targets = np.arange(lo, hi + step / 2, step) if hi > lo else np.array([lo])
        idx = np.unique(np.abs(m[:, None] - targets[None, :]).argmin(axis=0))
        keep.append(grp.iloc[idx])
    return pd.concat(keep, ignore_index=True)
