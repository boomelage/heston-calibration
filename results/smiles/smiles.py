import sys
from pathlib import Path
import numpy as np
import pandas as pd
import QuantLib as ql
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['cmr10', 'Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,   # tick labels in Computer Modern too
    'axes.unicode_minus': False,           # cmr10 lacks U+2212; avoids missing-glyph warnings
    'font.size': 8,
})

SMILES = Path(__file__).parent
RESULTS = SMILES.parent
REPO = RESULTS.parent
SURFACES = RESULTS / "surfaces"
# SURFACES_DATA = SURFACES / "data"
RAW = REPO / "data" / "options" / "raw"

if str(SURFACES) not in sys.path:
    sys.path.insert(0, str(SURFACES))

from example_surface import make_surface, OBJECTIVE # type: ignore --> Intentional Pylance ingore
from utils import build_heston_engine, implied_vol # type: ignore

FIGURES = SMILES / "figures"
FIGURES.mkdir(parents=True,exist_ok=True)

# Knob for the per-row maturity key: True draws a legend, False (default) draws a colorbar.
USE_LEGEND = True

# Knob: overlay real market implied vols (trade_iv) from data/options/raw/ as a scatter.
# The raw CBOE trade files are git-ignored, so this is a no-op (with a printed warning) on a
# fresh clone that has not been bootstrapped.
ENRICH_MARKET = True

# Market-scatter window. OTM market moneyness (S/K for calls, K/S for puts) is always in (0, 1];
# we drop the deep wing below MARKET_M_MIN and clip the IV outliers the deep-OTM corner throws.
MARKET_M_MIN = 0.5
MARKET_IV_MAX = 2.0
# Both wings share one moneyness window (S/K calls, K/S puts). The model curve is evaluated
# straight off the Heston engine on this grid, so each wing spans the full window instead of
# stopping where the saved surface's K/S grid ran out (the call wing only reached S/K ~= 0.91).
XLO, XHI = MARKET_M_MIN, 1.75

def _normalize_dates(dates):
    """Accept a single %Y-%m-%d date string or a list of them; return a list of strings.
    `make_surface` parses each with format="%Y-%m-%d"."""
    if isinstance(dates, str):
        return [dates]
    return list(dates)


def main(dates, OUT=None, use_legend=USE_LEGEND, enrich=ENRICH_MARKET):
    dates = _normalize_dates(dates)

    days = []
    for date in dates:
        surface, day_results = make_surface(target_date=date, OUT=OUT, SAVE=False)
        d = day_results['date']
        tag = str(d.strftime(r"%Y-%m-%d"))
        T = surface['maturity_days'].unique().tolist()
        surface = surface[surface['maturity_days'].isin(T)]
        days.append({'tag': tag, 'date': d, 'surface': surface, 'T': T,
                     'spot': day_results['spot'], 'params': day_results['params'],
                     'market': day_results['market'], 'fit': day_results['fit']})

    cmap = cm.jet
    for day in days:
        _save_day_figure(day, cmap, use_legend, enrich)

    write_smiles_TeX(days)


def _day_engine(day):
    """Rebuild the day's Heston engine (and a Black process for the inversion) from the calibrated
    params in `day`. Lets us evaluate the model smile at any strike, not just the strikes the
    saved surface grid happened to sample."""
    row = {
        'spot_price': day['spot'],
        'risk_free_rate': day['market']['risk_free_rate'],
        'dividend_rate': day['market']['dividend_rate'],
        **day['params'],   # kappa, theta, rho, eta, v0
    }
    d = pd.Timestamp(day['date'])
    calc_date = ql.Date(d.day, d.month, d.year)
    engine, s_handle, r_ts, g_ts, day_count = build_heston_engine(row, calc_date)
    bsm = ql.BlackScholesMertonProcess(
        s_handle, g_ts, r_ts,
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            calc_date, ql.UnitedStates(ql.UnitedStates.NYSE), 0.20, day_count)))
    return engine, bsm, calc_date


def _model_wing_iv(engine, bsm, spot, maturity_date, m_grid, wing):
    """Model Black IV along one wing across the moneyness grid. Plot-convention moneyness:
    S/K for calls (strike = spot/m), K/S for puts (strike = m*spot). The inversion always runs
    off the OTM option at each strike (w=None), so it stays stable across the whole window and is
    a pure function of strike, independent of the wing it is drawn on."""
    strikes = (spot / m_grid) if wing == 'call' else (m_grid * spot)
    return np.array([implied_vol(float(k), maturity_date, spot, engine, bsm) for k in strikes])


def _load_market_vols(tag, T):
    """Fetch real market implied vols for one trading day from data/options/raw/.

    Mirrors `data/extract_otms.py`: parses the CBOE trade file, computes calendar
    `days_to_maturity`, keeps positive-IV OTM trades, then collapses each (maturity, strike) to a
    single **volume-weighted** point (weights = `trade_size`). The implied vol at a strike is a
    property of the strike, not of which side was traded, so (exactly like the model lines, which
    invert the OTM option at every strike) each point is reparameterised onto *both* wings: the
    call panel at `S/K`, the put panel at `K/S`. That fills the whole window on each wing, the OTM
    half from same-side trades and the in-the-money half (moneyness > 1) from the liquid
    opposite-side OTM trades. `cmat` is the maturity snapped to the nearest model maturity in `T`,
    so each mark takes the color of its line. Returns None if no raw file exists for `tag` (the
    raw files are git-ignored)."""
    candidates = [RAW / f"UnderlyingOptionsTradesCalcs_{tag}.csv", *sorted(RAW.glob(f"*{tag}.csv"))]
    raw_path = next((p for p in candidates if p.exists()), None)
    if raw_path is None:
        print(f"  [market] no raw file for {tag}; skipping scatter")
        return None

    cols = ['quote_datetime', 'expiration', 'strike', 'option_type',
            'trade_size', 'trade_iv', 'underlying_bid']
    df = pd.read_csv(raw_path, usecols=cols)
    df['quote_datetime'] = pd.to_datetime(df['quote_datetime'])
    df['expiration'] = pd.to_datetime(df['expiration'], format='%Y-%m-%d')
    df['days_to_maturity'] = ((df['expiration'] - df['quote_datetime']) / pd.Timedelta(days=1)).astype(int)
    df = df[(df['days_to_maturity'] > 0) & (df['trade_iv'] > 0) & (df['trade_size'] > 0)
            & (df['underlying_bid'] > 0) & (df['strike'] > 0)].copy()
    df['w'] = df['option_type'].map({'C': 'call', 'P': 'put'})

    spot, strike = df['underlying_bid'], df['strike']
    otm = (((df['w'] == 'call') & (strike > spot)) | ((df['w'] == 'put') & (strike < spot)))
    df = df[otm].copy()

    lo, hi = min(T), max(T)
    df = df[(df['days_to_maturity'] >= lo) & (df['days_to_maturity'] <= hi)]
    df = df[df['trade_iv'] <= MARKET_IV_MAX]
    df = df[['days_to_maturity', 'strike', 'trade_iv', 'underlying_bid', 'trade_size']].dropna()
    if df.empty:
        cols_out = ['w', 'days_to_maturity', 'strike', 'moneyness', 'trade_iv', 'volume', 'cmat']
        return pd.DataFrame(columns=cols_out)

    # Volume-weighted average per (maturity, strike): one IV and one reference spot (the intraday
    # spot moves across a strike's trades) per displayed point.
    df['_iv_w'] = df['trade_iv'] * df['trade_size']
    df['_s_w'] = df['underlying_bid'] * df['trade_size']
    agg = df.groupby(['days_to_maturity', 'strike'], as_index=False).agg(
        _iv_w=('_iv_w', 'sum'), _s_w=('_s_w', 'sum'), volume=('trade_size', 'sum'))
    agg['trade_iv'] = agg['_iv_w'] / agg['volume']
    agg['spot'] = agg['_s_w'] / agg['volume']

    # Reparameterise each point onto both wings (S/K calls, K/S puts), then keep what falls in the
    # plotted window so each wing is populated across the full bound.
    call = agg.assign(w='call', moneyness=agg['spot'] / agg['strike'])
    put = agg.assign(w='put', moneyness=agg['strike'] / agg['spot'])
    out = pd.concat([call, put], ignore_index=True)
    out = out[(out['moneyness'] >= XLO) & (out['moneyness'] <= XHI)].copy()
    if out.empty:
        return out[['w', 'days_to_maturity', 'strike', 'moneyness', 'trade_iv', 'volume']].assign(cmat=[])

    # Snap each point's maturity to the nearest model maturity so its color matches that line.
    T_arr = np.array(sorted(T))
    nearest = np.abs(out['days_to_maturity'].to_numpy()[:, None] - T_arr[None, :]).argmin(axis=1)
    out['cmat'] = T_arr[nearest]
    return out[['w', 'days_to_maturity', 'strike', 'moneyness', 'trade_iv', 'volume', 'cmat']]


def _save_day_figure(day, cmap, use_legend, enrich):
    T = [
        30, 60, 90, 180, # 270, 350, 540
    ]#sorted(day['T'])
    norm = mcolors.Normalize(vmin=min(T), vmax=max(T))
    fig, (ax_put, ax_call) = plt.subplots(1, 2, sharey=True,
                                           figsize=(8, 2.7),
                                           layout='constrained')

    
    m_grid = np.round(np.arange(XLO, XHI + 1e-9, 0.005), 4)
    engine, bsm, calc_date = _day_engine(day)
    spot = day['spot']

    vols = []
    for t in T:
        maturity_date = calc_date + ql.Period(int(t), ql.Days)
        ivp = _model_wing_iv(engine, bsm, spot, maturity_date, m_grid, 'put')
        ax_put.plot(m_grid, ivp, color=cmap(norm(t)), zorder=2)
        ivc = _model_wing_iv(engine, bsm, spot, maturity_date, m_grid, 'call')
        ax_call.plot(m_grid, ivc, color=cmap(norm(t)), label=str(t), zorder=2)
        vols.extend([ivp, ivc])

    drew_market = False
    if enrich:
        mkt = _load_market_vols(day['tag'], T)
        if mkt is not None and len(mkt):
            for ax, wing in ((ax_put, 'put'), (ax_call, 'call')):
                sub = mkt[mkt['w'] == wing]
                if sub.empty:
                    continue
                # Color each market point with its snapped maturity, matching that line exactly.
                ax.scatter(sub['moneyness'], sub['trade_iv'],
                           color=cmap(norm(sub['cmat'].to_numpy())),
                           s=14, edgecolors='0.25', linewidths=0.3, zorder=3)
                drew_market = True

    # Frame the y-axis on the model smile so deep-OTM market outliers do not dominate it.
    finite = np.concatenate(vols)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        pad = 0.1 * (finite.max() - finite.min() + 1e-6)
        ax_put.set_ylim(finite.min() - pad, finite.max() + pad)

    xpad = 0.01
    ax_put.set_xlim(XLO - xpad, XHI + xpad)
    ax_call.set_xlim(XLO - xpad, XHI + xpad)

    ax_put.set_ylabel(r'Black implied vol $\widehat{\sigma}(\Phi^{\star})$')
    ax_put.set_xlabel(r'Moneyness $K/S$ (put wing)')
    ax_call.set_xlabel(r'Moneyness $S/K$ (call wing)')
    fig.suptitle(_row_caption(day), fontsize=8)

    if use_legend:
        handles, labels = ax_call.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside center right',
                   title='Days to maturity', fontsize=7, title_fontsize=8)
    else:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=(ax_put, ax_call), label='Days to maturity')

    # When market vols are overlaid, label what the lines vs. the markers are (colors already
    # encode maturity via the key above).
    if drew_market:
        series = [
            Line2D([], [], color='0.25', label='Heston'),
            Line2D([], [], color='0.5', marker='o', linestyle='None', markeredgecolor='0.25',
                   markersize=5, label='Market (vol-weighted)'),
        ]
        ax_put.legend(handles=series, loc='upper right', fontsize=7, framealpha=1.0)

    fig.savefig(FIGURES / f'smiles_{day["tag"]}.eps',
                format='eps', bbox_inches='tight')
    plt.close(fig)


def _row_caption(day):
    """The per-row caption drawn in the figure: only date, S_ref, Phi, Feller, IV-RMSE, RMSE."""
    p, f = day['params'], day['fit']
    phi = (f"({p['theta']:.4f},\\,{p['kappa']:.4f},\\,{p['eta']:.4f},"
           f"\\,{p['rho']:.4f},\\,{p['v0']:.4f})")
    return (f"{day['tag']}:  "
            f"$S_{{\\mathrm{{ref}}}}={day['spot']:.2f}$,  "
            r"$\Phi^{\star}=$"f"${phi}$,  "
            f"$\\mathcal{{F}}={f['feller']:.4f}$,  "
            f"IV-RMSE$={f['iv_rmse']*100:.2f}$,  "
            f"RMSE$={f['rmse']:.4f}$")


def write_smiles_TeX(days):
    """Generate `figures/smiles.tex`: one figure float per day so LaTeX can break across pages.
    Calibration values are drawn in each figure by `_row_caption`."""

    blocks = []
    for day in days:
        date_pretty = day['date'].strftime(r"%B %d, %Y")
        caption = (f"Heston implied-volatility smiles for {date_pretty}: "
                   r"put wing (left, $K/S$) and call wing (right, $S/K$), "
                   r"with market trades scattered.")
        label = f"Fig:smiles_{day['tag']}"
        block = (
            r"\begin{figure}[H]" "\n"
            r"    \begin{center}" "\n"
            f"        \\includegraphics[width=\\linewidth,keepaspectratio=true]"
            f"{{results/smiles/figures/smiles_{day['tag']}.eps}}\n"
            # f"        \\caption{{{caption}}}\n"
            f"        \\label{{{label}}}\n"
            r"    \end{center}" "\n"
            r"\end{figure}"
        )
        blocks.append(block)

    (FIGURES / "smiles.tex").write_text("\n".join(blocks) + "\n")


def make_surfaces_for(dates):
    return main(dates=dates, OUT=None)


if __name__ == "__main__":
    CALIBRATIONS_FILE = SURFACES.parent / "calibrations" / OBJECTIVE / "calibrations.csv"
    cal = pd.read_csv(CALIBRATIONS_FILE)
    cal = cal.sort_values(by='iv_rmse',ascending=True).reset_index(drop=True)
    dates = cal['date'][:24].copy()
    make_surfaces_for(dates=dates)
