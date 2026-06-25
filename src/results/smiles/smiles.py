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
# RESULTS_CODE (src/results) holds _results_config.py, the central knob file for the figure scripts.
SMILES = Path(__file__).parent.resolve()        # src/results/smiles
SRC = SMILES.parents[1]                           # src/ (shared _utils.py, config.py)
RESULTS_CODE = SMILES.parent                      # src/results (_results_config.py)
REPO = SMILES.parents[2]                          # repo root (smiles->results->src->repo)
RESULTS = REPO / "results"                        # real results data/figure dir

for _p in (str(SRC), str(RESULTS_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# All tunable parameters live in _results_config.py (the central knob file).
from _results_config import (  # type: ignore
    MODEL, OBJECTIVE, PLOT_RCPARAMS, NT, USE_LEGEND,
    XLO, XHI, MKTMONSTEP, SMILE_M_STEP, SMILE_FIGSIZE, MATURITIES_DAYS,
    TMIN, TMAX)
from _utils import (  # type: ignore
    model_implied_vol, _normalize_dates, _clip_maturities, _sparse_maturities, _sparse_strikes)
from _results_utils import (  # type: ignore
    load_calibrations_by_date, build_day_engine, _load_test_scatter, _maturity_colors, _day_from_row)
from config import calib_paths # type: ignore

plt.rcParams.update(PLOT_RCPARAMS)

# Module-level defaults for the _results_config model (used by the CLI / __main__). main() and the
# helpers resolve their own (model, objective) per call, so a notebook can pass a different pair.
MODEL_LABEL = MODEL.capitalize()   # 'Heston' / 'Bates' for figure legends and captions
# Single source of truth for the calibration outputs of this (model, objective): the per-day params
# file we read the fit from, and the calibration_tests/ directory the scatter is loaded from.
CALIBRATIONS_FILE, _REJECTIONS_FILE, TESTS_DIR = calib_paths(MODEL, OBJECTIVE)
FIGURES = RESULTS / MODEL / "smiles" / "figures"


def _figures_dir(model):
    return RESULTS / model / "smiles" / "figures"


def main(dates, model=None, objective=None, save=True, show=False, use_legend=USE_LEGEND):
    """Render per-day market-vs-model smile figures.

    `model`/`objective` default to the `_results_config` switches when None, so a notebook can plot a
    different run without editing `_results_config`. `save=True` writes one EPS per day plus smiles.tex
    under results/<model>/smiles/figures/; `save=False` skips disk. `show=True` leaves the figures open
    (does not close them) so the caller can render them; the rendering itself is the caller's job (see
    inspect.ipynb). Returns a dict {tag: Figure}.
    """
    model = model or MODEL
    objective = objective or OBJECTIVE
    calibrations_file, _rej, tests_dir = calib_paths(model, objective)
    FIGURES = _figures_dir(model)
    if save:
        FIGURES.mkdir(parents=True, exist_ok=True)

    dates = _normalize_dates(dates)
    cal = load_calibrations_by_date(calibrations_file)

    days, figs = [], {}
    for date in dates:
        ts = pd.Timestamp(pd.to_datetime(date, format=r"%Y-%m-%d"))
        # The model line is rebuilt from the calibrated params, so a row on/before `date` is required.
        row = cal.asof(ts)
        if not isinstance(row, pd.Series) or row.isna().all():
            raise SystemExit(
                f"smiles: no calibration row on/before {date} in {calibrations_file}")
        day = _day_from_row(row, model=model)
        days.append(day)
        figs[day['tag']] = _save_day_figure(day, use_legend, model=model, tests_dir=tests_dir,
                                            figures=FIGURES, save=save, show=show)
    if save:
        write_smiles_TeX(days, model=model, figures_dir=FIGURES)
    return figs


def _model_wing_iv(engine, bsm, spot, maturity_date, m_grid, wing):
    """Model Black IV along one wing across the moneyness grid. Plot-convention moneyness:
    S/K for calls (strike = spot/m), K/S for puts (strike = m*spot). The inversion always runs
    off the OTM option at each strike (w=None), so it stays stable across the whole window and is
    a pure function of strike, independent of the wing it is drawn on."""
    strikes = (spot / m_grid) if wing == 'call' else (m_grid * spot)
    return np.array([model_implied_vol(float(k), maturity_date, spot, engine, bsm) for k in strikes])


def _save_day_figure(day, use_legend, model=None, tests_dir=None, figures=None, save=True, show=False):
    model = model or MODEL
    model_label = model.capitalize()
    figures = figures or _figures_dir(model)
    # Load the calibrated-contract scatter first so each wing's x-axis (and the model line grid) can
    # be framed to that day's calibrated moneyness span. When no tests file exists we draw model lines
    # only, over a default maturity grid and the XLO/XHI fallback window.
    mkt = _load_test_scatter(day['tag'], tests_dir=tests_dir)
    if mkt is not None and len(mkt):
        # The displayed maturities are the calibrated ones, clipped to [TMIN, TMAX]; then sparsely
        # pick NT of them (NT=None => all).
        T = _sparse_maturities(_clip_maturities(mkt['days_to_maturity'].unique().tolist(), TMIN, TMAX), NT)
    else:
        T = _sparse_maturities(_clip_maturities(MATURITIES_DAYS, TMIN, TMAX), NT)
    if not T:
        print(f"  [{day['tag']}] no maturities to draw; skipping")
        return

    # One distinct color per displayed maturity, keyed by rank (see _maturity_colors). The same map
    # colors the model lines and the market scatter so each mark sits on its matching line's color.
    mat_colors = _maturity_colors(T)
    fig, (ax_put, ax_call) = plt.subplots(1, 2, sharey=True, 
                                          figsize=SMILE_FIGSIZE, 
                                          layout='constrained')

    # Engine row from the calibrated params, so the model smile can be evaluated at any strike (not
    # just the calibrated ones). build_day_engine is the shared (with make_surface) construction.
    engine_row = {
        'spot_price': day['spot'],
        'risk_free_rate': day['market']['risk_free_rate'],
        'dividend_rate': day['market']['dividend_rate'],
        **day['params'],   # kappa, theta, rho, eta, v0 (+ lambda_, nu, delta for bates)
    }
    engine, bsm, calc_date = build_day_engine(engine_row, day['date'], model)
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
            sub = _sparse_strikes(sub, MKTMONSTEP)
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

    if model == "heston":
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
            Line2D([], [], color='0.25', label=model_label),
            Line2D([], [], color='0.5', marker='o', linestyle='None', markeredgecolor='0.25',
                   markersize=5, label='Market'),
        ]
        ax_put.legend(handles=series, loc='upper right', fontsize=7, framealpha=1.0)

    if save:
        figures.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures / f'smiles_{day["tag"]}.eps',
                    format='eps', bbox_inches='tight')
    if not show:
        plt.close(fig)
    return fig


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


def write_smiles_TeX(days, model=None, figures_dir=None):
    """Generate `figures/smiles.tex`: one figure float per day so LaTeX can break across pages.
    Calibration values are drawn in each figure by `_row_caption`. Returns the .tex string; writes it
    only when `save`."""
    model = model or MODEL
    model_label = model.capitalize()
    figures_dir = figures_dir or _figures_dir(model)

    blocks = []
    for day in days:
        date_pretty = day['date'].strftime(r"%B %d, %Y")
        caption = (f"{model_label} implied-volatility smiles for {date_pretty}: "
                   r"put wing (left, $K/S$) and call wing (right, $S/K$), "
                   r"with market trades scattered.")
        label = f"Fig:{model}_smiles_{day['tag']}"
        block = (
            r"\begin{figure}[H]" "\n"
            r"    \begin{center}" "\n"
            f"        \\includegraphics[width=\\linewidth,keepaspectratio=false]"
            f"{{results/{model}/smiles/figures/smiles_{day['tag']}.eps}}\n"
            r"        \captionsetup{font=tiny,skip=-2pt,belowskip=-2pt}" "\n"
            f"        \\caption{{{caption}}}\n"
            f"        \\label{{{label}}}\n"
            r"    \end{center}" "\n"
            r"\end{figure}"
        )
        blocks.append(block)

    tex = "\n".join(blocks) + "\n"
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "smiles.tex").write_text(tex)
    return tex


def make_smiles_for(dates):
    return main(dates=dates)


if __name__ == "__main__":
    cal = pd.read_csv(CALIBRATIONS_FILE)
    cal = cal.sort_values(by='iv_rmse', ascending=True).reset_index(drop=True)[:8].copy()
    dates = cal['date']
    make_smiles_for(dates=dates)
