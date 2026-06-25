"""Shared helpers for the downstream figure scripts under ``src/results/``.

This is the results-layer counterpart to ``src/_utils.py``. It may freely import
``_results_config`` (the figure-script knob file) and the ``results/<model>/`` tree
conventions, which ``src/_utils.py`` must not (it is imported by the production
calibrator). The pure, config-free selection helpers (``_normalize_dates``,
``_clip_maturities``, ``_sparse_maturities``, ``_sparse_strikes``) live in
``src/_utils.py`` instead.

Importable once ``src/`` and ``src/results/`` are on ``sys.path`` -- every consuming
script does that bootstrap before importing this module.
"""
import pandas as pd
import QuantLib as ql
import matplotlib.pyplot as plt

from _results_config import MODEL, SMILE_CMAP, INVERSION_PLACEHOLDER_VOL  # type: ignore
from config import calendar as ql_calendar  # type: ignore
from _utils import build_model_engine  # type: ignore


def load_calibrations_by_date(calibrations_file):
    """Load calibrations.csv indexed and sorted by trading date (the ``asof``-ready frame
    smiles/make_surface select a day from)."""
    return (pd.read_csv(calibrations_file, parse_dates=['date'])
              .set_index('date').sort_index())


def build_day_engine(engine_row, date, model, placeholder_vol=INVERSION_PLACEHOLDER_VOL):
    """Rebuild a day's model engine plus a Black process for the price->IV inversion.

    `engine_row` is a mapping (dict or a calibrations.csv Series) carrying `spot_price`,
    `risk_free_rate`, `dividend_rate` and the model params (`kappa, theta, rho, eta, v0`, plus
    `lambda_, nu, delta` for bates). `date` is the row's trading date (anything pd.Timestamp accepts).
    The placeholder Black vol only initialises the term structure: `impliedVolatility` solves for the
    vol that reprices the model NPV, so its value is ignored. Returns `(engine, bsm, calc_date)`."""
    d = pd.Timestamp(date)
    calc_date = ql.Date(d.day, d.month, d.year)
    engine, s_handle, r_ts, g_ts, day_count = build_model_engine(engine_row, calc_date, model)
    bsm = ql.BlackScholesMertonProcess(
        s_handle, g_ts, r_ts,
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            calc_date, ql_calendar(), placeholder_vol, day_count)))
    return engine, bsm, calc_date


def _day_from_row(row, model=MODEL):
    """Assemble the per-day dict the smile figure/caption code consumes from one calibrations.csv row.
    `row` is a pandas Series indexed by the calibrations.csv columns, named by its trading date.

    NB: this is intentionally a SUPERSET-narrower shape than make_surface's `day_results` dict (which
    carries extra `fit` fields -- n_helpers/total_volume/high_move/... -- that plot_surfaces consumes).
    The two are kept separate on purpose; only the engine build and the csv load are shared."""
    date = pd.Timestamp(row.name)
    params = {'kappa': float(row['kappa']), 'theta': float(row['theta']),
              'rho': float(row['rho']), 'eta': float(row['eta']), 'v0': float(row['v0'])}
    if model == 'bates':
        # Carry the jump triple so build_day_engine can rebuild a Bates engine and the caption can show it.
        params.update(lambda_=float(row['lambda_']), nu=float(row['nu']), delta=float(row['delta']))
    return {
        'tag': date.strftime(r'%Y-%m-%d'),
        'date': date,
        'spot': float(row['spot_price']),
        'params': params,
        'market': {'risk_free_rate': float(row['risk_free_rate']),
                   'dividend_rate': float(row['dividend_rate'])},
        'fit': {'iv_rmse': float(row['iv_rmse']), 'rmse': float(row['rmse']),
                'feller': float(row['feller'])},
    }


def _load_test_scatter(tag, tests_dir):
    """Market implied vols for one trading day's *calibrated* contracts, straight from the per-day
    results/<model>/calibrations/<objective>/calibration_tests/ file.

    That file holds exactly the contracts the day was fit on, each with its market IV (`volatility`),
    its original intraday `spot_price`/`strike_price` and `days_to_maturity`. No re-derivation from the
    raw trades, no volume re-weighting: the IV here is the one the calibration actually saw. The
    implied vol at a strike is a property of the strike, not of which side was traded, so (exactly like
    the model lines, which invert the OTM option at every strike) each contract is reparameterised onto
    *both* wings: the call panel at `S/K`, the put panel at `K/S`. That fills the whole window on each
    wing, the OTM half from same-side trades and the in-the-money half (moneyness > 1) from the liquid
    opposite-side OTM trades. Returns None if no tests file exists for `tag` (those files are
    git-ignored, so a fresh clone has none and the figure shows model lines only)."""
    path = tests_dir / f"cboe_spx_calibration_tests_{tag}.csv"
    if not path.exists():
        print(f"  [scatter] no calibration_tests file for {tag}; drawing model lines only")
        return None
    df = pd.read_csv(path, usecols=['strike_price', 'w', 'volatility', 'spot_price',
                                    'days_to_maturity', 'trade_size'])
    df = df[(df['volatility'] > 0) & (df['spot_price'] > 0) & (df['strike_price'] > 0)].copy()
    if df.empty:
        return None
    df = df.rename(columns={'volatility': 'trade_iv', 'strike_price': 'strike',
                            'spot_price': 'spot', 'trade_size': 'volume'})

    # Reparameterise each contract onto both wings (S/K calls, K/S puts) so each wing is populated
    # across the full calibrated span (deep-OTM same-side trades through ITM opposite-side trades).
    call = df.assign(w='call', moneyness=df['spot'] / df['strike'])
    put = df.assign(w='put', moneyness=df['strike'] / df['spot'])
    out = pd.concat([call, put], ignore_index=True)
    return out[['w', 'days_to_maturity', 'strike', 'moneyness', 'trade_iv', 'volume']]


def _maturity_colors(T, cmap_name=SMILE_CMAP):
    """Map each displayed maturity to a distinct color by its RANK in the sorted list `T`, not by its
    day-count value. Spacing by rank spreads the colors across the full palette even when day counts
    cluster, and a qualitative colormap (tab10/tab20/Set1/...) gives categorical hues so adjacent
    maturities stay easy to tell apart. A listed/qualitative map uses its own discrete entries (cycled
    if there are more maturities than colors); a continuous map is sampled at evenly spaced points."""
    T = sorted(T)
    n = len(T)
    cmap = plt.get_cmap(cmap_name)
    base = getattr(cmap, 'colors', None)
    if base is not None:
        colors = [base[i % len(base)] for i in range(n)]
    else:
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    return dict(zip(T, colors))
