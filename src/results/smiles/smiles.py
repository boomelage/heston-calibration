import sys
from pathlib import Path
import numpy as np
import pandas as pd
import QuantLib as ql
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# This script lives under src/results/smiles/ and writes figures into the repo-level results/ tree.
# It no longer reads the raw CBOE trade files: the model smile lines come from the calibrated params
# in results/<model>/calibrations/<objective>/calibrations.csv, and the market scatter comes from the
# per-day calibration_tests/ file for that date (which holds exactly the contracts used in the fit,
# with their market IV in the `volatility` column). A missing tests file just drops the scatter; a
# missing calibrations row is a hard error. SMILES is the moved code dir; RESULTS/REPO route I/O.
# RESULTS_CODE (src/results) holds results_config.py, the central knob file for the figure scripts.
SMILES = Path(__file__).parent.resolve()        # src/results/smiles
SRC = SMILES.parents[1]                           # src/ (shared utils.py, config.py)
RESULTS_CODE = SMILES.parent                      # src/results (results_config.py)
REPO = SMILES.parents[2]                          # repo root (smiles->results->src->repo)
RESULTS = REPO / "results"                        # real results data/figure dir

for _p in (str(SRC), str(RESULTS_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# All tunable parameters live in results_config.py (the central knob file).
from results_config import (  # type: ignore
    MODEL, OBJECTIVE, PLOT_RCPARAMS, INVERSION_PLACEHOLDER_VOL, NT, USE_LEGEND,
    XLO, XHI, MKTMONSTEP, SMILE_M_STEP, SMILE_FIGSIZE, SMILE_CMAP, MATURITIES_DAYS,
    TMIN, TMAX)
from utils import build_model_engine, model_implied_vol # type: ignore
from config import calendar as ql_calendar, calib_paths # type: ignore

plt.rcParams.update(PLOT_RCPARAMS)

MODEL_LABEL = MODEL.capitalize()   # 'Heston' / 'Bates' for figure legends and captions

# Single source of truth for the calibration outputs of this (model, objective): the per-day params
# file we read the fit from, and the calibration_tests/ directory the scatter is loaded from.
CALIBRATIONS_FILE, _REJECTIONS_FILE, TESTS_DIR = calib_paths(MODEL, OBJECTIVE)

FIGURES = RESULTS / MODEL / "smiles" / "figures"
FIGURES.mkdir(parents=True,exist_ok=True)


def _normalize_dates(dates):
    """Accept a single %Y-%m-%d date string or a list of them; return a list of strings."""
    if isinstance(dates, str):
        return [dates]
    return [str(d) for d in dates]


def _day_from_row(row):
    """Assemble the per-day dict the figure/caption code consumes from one calibrations.csv row.
    `row` is a pandas Series indexed by the calibrations.csv columns, named by its trading date."""
    date = pd.Timestamp(row.name)
    params = {'kappa': float(row['kappa']), 'theta': float(row['theta']),
              'rho': float(row['rho']), 'eta': float(row['eta']), 'v0': float(row['v0'])}
    if MODEL == 'bates':
        # Carry the jump triple so _day_engine can rebuild a Bates engine and the caption can show it.
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


def main(dates, use_legend=USE_LEGEND):
    dates = _normalize_dates(dates)
    cal = pd.read_csv(CALIBRATIONS_FILE, parse_dates=['date']).set_index('date').sort_index()

    days = []
    for date in dates:
        ts = pd.Timestamp(pd.to_datetime(date, format=r"%Y-%m-%d"))
        # The model line is rebuilt from the calibrated params, so a row on/before `date` is required.
        row = cal.asof(ts)
        if not isinstance(row, pd.Series) or row.isna().all():
            raise SystemExit(
                f"smiles: no calibration row on/before {date} in {CALIBRATIONS_FILE}")
        day = _day_from_row(row)
        days.append(day)
        _save_day_figure(day, use_legend)

    write_smiles_TeX(days)


def _day_engine(day):
    """Rebuild the day's model engine (and a Black process for the inversion) from the calibrated
    params in `day`. Lets us evaluate the model smile at any strike, not just the strikes the
    calibrated surface happened to sample."""
    row = {
        'spot_price': day['spot'],
        'risk_free_rate': day['market']['risk_free_rate'],
        'dividend_rate': day['market']['dividend_rate'],
        **day['params'],   # kappa, theta, rho, eta, v0 (+ lambda_, nu, delta for bates)
    }
    d = pd.Timestamp(day['date'])
    calc_date = ql.Date(d.day, d.month, d.year)
    engine, s_handle, r_ts, g_ts, day_count = build_model_engine(row, calc_date, MODEL)
    bsm = ql.BlackScholesMertonProcess(
        s_handle, g_ts, r_ts,
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            calc_date, ql_calendar(),
            INVERSION_PLACEHOLDER_VOL, day_count)))
    return engine, bsm, calc_date


def _model_wing_iv(engine, bsm, spot, maturity_date, m_grid, wing):
    """Model Black IV along one wing across the moneyness grid. Plot-convention moneyness:
    S/K for calls (strike = spot/m), K/S for puts (strike = m*spot). The inversion always runs
    off the OTM option at each strike (w=None), so it stays stable across the whole window and is
    a pure function of strike, independent of the wing it is drawn on."""
    strikes = (spot / m_grid) if wing == 'call' else (m_grid * spot)
    return np.array([model_implied_vol(float(k), maturity_date, spot, engine, bsm) for k in strikes])


def _load_test_scatter(tag):
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
    path = TESTS_DIR / f"cboe_spx_calibration_tests_{tag}.csv"
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


def _clip_maturities(T, tmin=TMIN, tmax=TMAX):
    """Keep only maturities (in days) within the [tmin, tmax] window before sparse selection.
    Either bound is optional: `tmin=None` removes the lower bound, `tmax=None` the upper, and
    both `None` keeps every maturity. Applied to the candidate maturities (calibrated or the
    MATURITIES_DAYS fallback) so the displayed smiles are restricted to the chosen tenor band."""
    lo = -np.inf if tmin is None else tmin
    hi = np.inf if tmax is None else tmax
    return [t for t in sorted(T) if lo <= t <= hi]


def _sparse_maturities(T, nt=NT):
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


def _sparse_market_strikes(sub, step=MKTMONSTEP):
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


def _save_day_figure(day, use_legend):
    # Load the calibrated-contract scatter first so each wing's x-axis (and the model line grid) can
    # be framed to that day's calibrated moneyness span. When no tests file exists we draw model lines
    # only, over a default maturity grid and the XLO/XHI fallback window.
    mkt = _load_test_scatter(day['tag'])
    if mkt is not None and len(mkt):
        # The displayed maturities are the calibrated ones, clipped to [TMIN, TMAX]; then sparsely
        # pick NT of them (NT=None => all).
        T = _sparse_maturities(_clip_maturities(mkt['days_to_maturity'].unique().tolist()))
    else:
        T = _sparse_maturities(_clip_maturities(MATURITIES_DAYS))
    if not T:
        print(f"  [{day['tag']}] no maturities to draw; skipping")
        return

    # One distinct color per displayed maturity, keyed by rank (see _maturity_colors). The same map
    # colors the model lines and the market scatter so each mark sits on its matching line's color.
    mat_colors = _maturity_colors(T)
    fig, (ax_put, ax_call) = plt.subplots(1, 2, sharey=True,
                                           figsize=SMILE_FIGSIZE,
                                           layout='constrained')

    engine, bsm, calc_date = _day_engine(day)
    spot = day['spot']

    def _wing_bounds(wing):
        """Moneyness span of the day's *displayed* calibrated data on one wing, or the fallback
        window. Restricted to the displayed maturities `T` (clipped to [TMIN, TMAX] and sparsely
        sampled) so the x-axis covers only where plotted market vols actually exist, not the wider
        span of the maturities that were dropped from the figure."""
        if mkt is not None and len(mkt):
            sub = mkt[(mkt['w'] == wing) & (mkt['days_to_maturity'].isin(T))]
            if not sub.empty:
                return float(sub['moneyness'].min()), float(sub['moneyness'].max())
        return XLO, XHI

    put_lo, put_hi = _wing_bounds('put')
    call_lo, call_hi = _wing_bounds('call')
    put_grid = np.round(np.arange(put_lo, put_hi + 1e-9, SMILE_M_STEP), 4)
    call_grid = np.round(np.arange(call_lo, call_hi + 1e-9, SMILE_M_STEP), 4)

    vols = []
    for t in T:
        maturity_date = calc_date + ql.Period(int(t), ql.Days)
        ivp = _model_wing_iv(engine, bsm, spot, maturity_date, put_grid, 'put')
        ax_put.plot(put_grid, ivp, color=mat_colors[t], zorder=2)
        ivc = _model_wing_iv(engine, bsm, spot, maturity_date, call_grid, 'call')
        ax_call.plot(call_grid, ivc, color=mat_colors[t], label=str(t), zorder=2)
        vols.extend([ivp, ivc])

    drew_market = False
    if mkt is not None and len(mkt):
        # Show only the displayed maturities; color each point with its own (calibrated) maturity so
        # the marks sit on their matching line exactly.
        mkt = mkt[mkt['days_to_maturity'].isin(T)].copy()
        mkt['cmat'] = mkt['days_to_maturity']
        for ax, wing in ((ax_put, 'put'), (ax_call, 'call')):
            sub = mkt[mkt['w'] == wing]
            if sub.empty:
                continue
            # Thin to a sparse, ~MKTMONSTEP-spaced moneyness subset per maturity (MKTMONSTEP=None =>
            # every point) so dense days stay readable.
            sub = _sparse_market_strikes(sub)
            if sub.empty:
                continue
            ax.scatter(sub['moneyness'], sub['trade_iv'],
                       color=[mat_colors[m] for m in sub['cmat']],
                       s=14, edgecolors='0.25', linewidths=0.3, zorder=3)
            drew_market = True

    # Frame the y-axis on the model smile so deep-OTM market outliers do not dominate it.
    finite = np.concatenate(vols)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        pad = 0.1 * (finite.max() - finite.min() + 1e-6)
        ax_put.set_ylim(finite.min() - pad, finite.max() + pad)

    # Each wing's x-axis spans its own calibrated-data range (set above), so the full smile is shown.
    xpad = 0.01
    ax_put.set_xlim(put_lo - xpad, put_hi + xpad)
    ax_call.set_xlim(call_lo - xpad, call_hi + xpad)

    if MODEL == "heston":
        ax_put.set_ylabel(r'Black implied vol $\widehat{\sigma}(\Phi^{\star})$')
    else:
        ax_put.set_ylabel(r'Black implied vol $\widehat{\sigma}(\Theta^{\star})$')
    ax_put.set_xlabel(r'Moneyness $K/S$ (put wing)')
    ax_call.set_xlabel(r'Moneyness $S/K$ (call wing)')
    # Caption left-aligned to the left edge of the plot area (over the put wing), not centred.
    ax_put.set_title(_row_caption(day), loc='left', fontsize=8)

    if use_legend:
        handles, labels = ax_call.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside center right',
                   title='Days to maturity', fontsize=7, title_fontsize=8)
    else:
        # Discrete colorbar: one band per displayed maturity, labeled with its day count.
        listed = mcolors.ListedColormap([mat_colors[t] for t in T])
        bnorm = mcolors.BoundaryNorm(np.arange(len(T) + 1) - 0.5, len(T))
        sm = cm.ScalarMappable(cmap=listed, norm=bnorm)
        cbar = fig.colorbar(sm, ax=(ax_put, ax_call), label='Days to maturity',
                            ticks=np.arange(len(T)))
        cbar.ax.set_yticklabels([str(t) for t in T])

    # When market vols are overlaid, label what the lines vs. the markers are (colors already
    # encode maturity via the key above).
    if drew_market:
        series = [
            Line2D([], [], color='0.25', label=MODEL_LABEL),
            Line2D([], [], color='0.5', marker='o', linestyle='None', markeredgecolor='0.25',
                   markersize=5, label='Market'),
        ]
        ax_put.legend(handles=series, loc='upper right', fontsize=7, framealpha=1.0)

    fig.savefig(FIGURES / f'smiles_{day["tag"]}.eps',
                format='eps', bbox_inches='tight')
    plt.close(fig)


def _row_caption(day):
    """The per-row caption drawn in the figure: only date, S_ref, Phi, Feller, IV-RMSE, RMSE."""
    p, f = day['params'], day['fit']
    phi = (f"{p['theta']:.4f},\\,{p['kappa']:.4f},\\,{p['eta']:.4f},"
           f"\\,{p['rho']:.4f},\\,{p['v0']:.4f}")
    if 'lambda_' in p:
        params = (
            r"$\Theta^{\star}=$"f"$ ({phi}, {p['lambda_']:.4f},\\,"
            f"{p['nu']:.4f},\\,{p['delta']:.4f})$"
        )
    else:
        params = r"$\Phi^{\star}=$"f"$({phi})$"
    return (f"{day['tag']}:  "
            f"$S_{{\\mathrm{{ref}}}}={day['spot']:.2f}$,  "
            f"$\\mathcal{{F}}={f['feller']:.4f}$,  "
            f"IV-RMSE$={f['iv_rmse']:.4f}$,  "
            f"RMSE$={f['rmse']:.4f}$"
            "\n"+params)


def write_smiles_TeX(days):
    """Generate `figures/smiles.tex`: one figure float per day so LaTeX can break across pages.
    Calibration values are drawn in each figure by `_row_caption`."""

    blocks = []
    for day in days:
        date_pretty = day['date'].strftime(r"%B %d, %Y")
        caption = (f"{MODEL_LABEL} implied-volatility smiles for {date_pretty}: "
                   r"put wing (left, $K/S$) and call wing (right, $S/K$), "
                   r"with market trades scattered.")
        label = f"Fig:smiles_{day['tag']}"
        block = (
            r"\begin{figure}[H]" "\n"
            r"    \begin{center}" "\n"
            f"        \\includegraphics[width=\\linewidth,keepaspectratio=false]"
            f"{{../results/{MODEL}/smiles/figures/smiles_{day['tag']}.eps}}\n"
            r"        \captionsetup{font=tiny,skip=-2pt,belowskip=-2pt}" "\n"
            f"        \\caption{{{caption}}}\n"
            f"        \\label{{{label}}}\n"
            r"    \end{center}" "\n"
            r"\end{figure}"
        )
        blocks.append(block)

    (FIGURES / "smiles.tex").write_text("\n".join(blocks) + "\n")


def make_surfaces_for(dates):
    return main(dates=dates)


if __name__ == "__main__":
    cal = pd.read_csv(CALIBRATIONS_FILE)
    cal = cal.sort_values(by='iv_rmse', ascending=True).reset_index(drop=True)[:8].copy()
    dates = cal['date']
    make_surfaces_for(dates=dates)
