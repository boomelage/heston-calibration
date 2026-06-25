"""Render the Heston option-price surface as EPS figures for LaTeX.

Reads `results/example_price_surface.csv` (from `example_price_surface.py`; columns
strike, maturity_days, moneyness, w, price) and draws MATLAB-style 3D surface plots -- jet
colormap, meshed facets, strike x maturity x price axes -- saved as vector EPS, the same kind of
figure embedded in `results/example-surface-rendering/`.

Three figures are produced (calls wing, puts wing, full OTM surface) plus a `price_surface.tex`
that includes them like the example's `analysis_090924.tex`. EPS embeds in LaTeX via
`\\includegraphics`; with pdflatex, convert first (`epstopdf *.eps`) or compile the provided .tex
with `latex price_surface.tex` (the classic dvips route) -- or just `pdflatex` after epstopdf.

Run:  python src/results/surfaces/plot_surfaces.py
Out:  results/price_surface_calls.eps, price_surface_puts.eps, price_surface_both.eps
      results/price_surface.tex
"""
import numpy as np
import pandas as pd
import matplotlib
# NB: do not force the 'Agg' backend at import time -- that would override an interactive backend
# (e.g. ipympl's 'widget') a notebook selects with `%matplotlib widget` before importing this module.
# The CLI (__main__) selects 'Agg' itself for headless file writing.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the '3d' projection; also the ax type below
from pathlib import Path

# This script now lives under src/results/surfaces/, but reads/writes the repo-level results/ tree.
# `from make_surface import ...` resolves from this dir; SURFACES routes figure I/O to repo/results/.
# RESULTS_CODE (src/results) holds _results_config.py, the central knob file for the figure scripts.
import sys
HERE = Path(__file__).parent.resolve()                 # src/results/surfaces
SRC = HERE.parents[1]                                   # src/ (shared _utils.py, config.py)
RESULTS_CODE = HERE.parent                              # src/results (_results_config.py)
REPO = HERE.parents[2]                                  # repo root (surfaces->results->src->repo)
RESULTS = REPO / "results"
for _p in (str(HERE), str(SRC), str(RESULTS_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# All tunable parameters live in _results_config.py. MODEL/OBJECTIVE pick the engine, the
# calibrations source, and the results/<model>/surfaces/ figure tree.
from _results_config import (  # type: ignore
    MODEL, OBJECTIVE, PLOT_RCPARAMS, SURFACE_ELEV, SURFACE_AZIM, SURFACE_FIGSIZE)
from config import calib_paths  # type: ignore

# Match the default LaTeX font (Computer Modern serif) so the axis text blends with the surrounding
# document. Uses matplotlib's bundled Computer Modern (cmr10) -- no LaTeX/usetex toolchain required.
plt.rcParams.update(PLOT_RCPARAMS)
ELEV, AZIM = SURFACE_ELEV, SURFACE_AZIM


def _texdir(model):
    """Figure/tex output dir for a model: repo/results/<model>/surfaces/plots/tex."""
    return RESULTS / model / "surfaces" / "plots" / "tex"


def plot_surface(grid, out_path=None, title=None, invert_K=False, invert_T=False, AZIM_ADJUST=0.0,
                 xlabel=r'moneyness ($K/S$)', zlabel=r'price', save=True, show=False):
    """Draw one moneyness x maturity x z surface (grid: index=moneyness, columns=maturity_days).

    Returns the Figure. Saves an EPS to `out_path` only when `save` and `out_path` are set; closes the
    figure afterwards unless `show` (so a notebook keeps it live to render, e.g. under ipympl)."""

    strikes = grid.index.to_numpy(dtype=float)
    maturities = grid.columns.to_numpy(dtype=float) / 365.0      # days -> years, like the example
    X, Y = np.meshgrid(strikes, maturities)                      # (n_mat, n_strike)
    Z = grid.to_numpy(dtype=float).T                             # (n_mat, n_strike)

    fig = plt.figure(figsize=SURFACE_FIGSIZE)
    plt.style.use('fast')
    # computed_zorder=False so axis ticks/labels always draw on top of the surface; the default
    # (True) depth-sorts the white-faced surface in front of the z-axis label once the view is
    # rotated, hiding the label behind the facets.
    ax: Axes3D = fig.add_subplot(111, projection='3d', computed_zorder=False)  # type: ignore[assignment]
    surf = ax.plot_surface(X, Y, Z, rstride=3, cstride=3,
                    color="white", edgecolor="black",
                    linewidth=0.4, shade=False, antialiased=True)
    
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor("0.7")
        pane.pane.set_linewidth(0.5)
 
    ax.grid(False)
    ax.tick_params(labelsize=9, colors="black")
    ax.view_init(elev=ELEV, azim=AZIM+AZIM_ADJUST)
    ax.set_xlabel(xlabel)
    if invert_K:
        ax.invert_xaxis()
    if invert_T:
        ax.invert_yaxis()
    ax.set_ylabel(r'maturity in years ($T$)')
    ax.set_zlabel(zlabel)
    ax.set_zlim(np.nanmin(Z), np.nanmax(Z))
    if title:
        ax.set_title(title)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()
    if save and out_path is not None:
        # pad_inches: bbox_inches='tight' under-counts the rotated 3D z-axis label in mplot3d and
        # crops it off (the smile view rotates the z-axis title out past the tight box); the pad keeps
        # it in.
        fig.savefig(out_path, format='eps', bbox_inches='tight', pad_inches=0.2)
        print(f"wrote {out_path.name}")
    if not show:
        plt.close(fig)
    return fig


def write_surfaces_TeX(spot, date, params, market, fit, model=None, texdir=None):
    model = model or MODEL
    model_label = model.capitalize()
    texdir = texdir or _texdir(model)

    TeX = \
r"""
\textbf{<MODEL_LABEL> model prices} produced by the market\!\,\footnote{
The fit uses <nhelpers> calibration cells across <nmats> maturities and <nstrikes> strikes,
backed by <volume> contracts of traded volume.
The reference spot was $S_{\mathrm{ref}} = <spot>$,
priced under a risk-free rate of <r>\% and a dividend rate of <q>\%.
The intraday spot range was <rangepct>\%.<movenote>
Fit quality is <ivrmse> vol points of implied-volatility RMSE,
with a relative-price RMSE of <rmse>.
The Feller condition <fellersign> at this calibration,
with $2\kappa\theta - \eta^2 = <feller>$.
}
calibrated pricing operator:
\begin{center}
    <operator>~\eqref{eq:accept}.
\end{center}
\begin{figure}[H]
    \begin{center}
        \includegraphics[width=6.25cm,keepaspectratio=true]{results/<MODEL>/surfaces/plots/tex/price_surface_puts.eps}
        \includegraphics[width=6.25cm,keepaspectratio=true]{results/<MODEL>/surfaces/plots/tex/price_surface_calls.eps}
        \caption{<MODEL_LABEL> option prices for $S_{\mathrm{ref}}$ <spot> on <date> with <parameters>: puts wing (left) and calls wing (right).}
        \label{fig:<MODEL>-wings}
    \end{center}
    \begin{center}
        \includegraphics[width=9cm,keepaspectratio=true]{results/<MODEL>/surfaces/plots/tex/smile_surface.eps}
        \caption{\emph{Out of the money} implied volatilites from Figure~\ref{fig:<MODEL>-wings}}
    \end{center}
\end{figure}

"""

    hestonparams = r'$\Phi^{\star} = (<theta>,\ <kappa>,\ <eta>,\ <rho>,\ <v0>)$'
    batesparams = r'$\Theta^{\star} = (<theta>,\ <kappa>,\ <eta>,\ <rho>,\ <v0>, \ <lambda>, \ <nu>, \ <delta>)$'
    paramstr = hestonparams if model == 'heston' else batesparams
    paramstr = paramstr.replace('<theta>', str(round(params['theta'], 4)))
    paramstr = paramstr.replace('<kappa>', str(round(params['kappa'], 4)))
    paramstr = paramstr.replace('<eta>', str(round(params['eta'], 4)))
    paramstr = paramstr.replace('<rho>', str(round(params['rho'], 4)))
    paramstr = paramstr.replace('<v0>', str(round(params['v0'], 4)))
    if model == 'bates':
        paramstr = paramstr.replace('<lambda>', str(round(params['lambda_'], 4)))
        paramstr = paramstr.replace('<nu>', str(round(params['nu'], 4)))
        paramstr = paramstr.replace('<delta>', str(round(params['delta'], 4)))
    hestoneq = r'$C_{\mathrm{H}}(\Phi^{\star}; S,K,\tau,w)$~\eqref{eq:heston-price}'
    bateseq = r'$C_{\mathrm{Bates}}(\Theta^{\star}; S,K,\tau,w)$~\eqref{eq:bates-cf}'
    TeX = TeX.replace('<parameters>', paramstr)
    TeX = TeX.replace('<operator>', hestoneq if model == 'heston' else bateseq)
    TeX = TeX.replace('<spot>', str(spot))
    TeX = TeX.replace('<date>', str(date.strftime(r"%B %d, %Y")))
    TeX = TeX.replace('<r>', f"{market['risk_free_rate']*100:.2f}")
    TeX = TeX.replace('<q>', f"{market['dividend_rate']*100:.2f}")
    TeX = TeX.replace('<ivrmse>', f"{fit['iv_rmse']*100:.2f}")
    TeX = TeX.replace('<rmse>', str(round(fit['rmse'], 4)))
    TeX = TeX.replace('<feller>', str(round(fit['feller'], 4)))
    TeX = TeX.replace('<nhelpers>', str(fit['n_helpers']))
    TeX = TeX.replace('<nmats>', str(fit['n_maturities']))
    TeX = TeX.replace('<nstrikes>', str(fit['n_strikes']))
    TeX = TeX.replace('<volume>', f"{fit['total_volume']:,}")
    TeX = TeX.replace('<rangepct>', f"{fit['spot_range_pct']*100:.2f}")
    TeX = TeX.replace('<fellersign>', 'violates' if fit['feller'] < 0 else 'satisfies')
    TeX = TeX.replace('<movenote>',
        r' The intraday range exceeded the 3\% threshold, so treat $S_{\mathrm{ref}}$ with caution.'
        if fit['high_move'] else '')
    # Model-namespace the figure include paths and the caption label (Heston / Bates).
    TeX = TeX.replace('<MODEL>', str(model))
    TeX = TeX.replace('<MODEL_LABEL>', str(model_label))

    texdir.mkdir(parents=True, exist_ok=True)
    tex_path = texdir / r"surfaces.tex"
    tex_path.write_text(TeX)

def price_grid(df, side):
    surface = df[df['w'] == side].copy()
    surface['moneyness'] = np.where(
        surface['w'] == 'call',
        surface['s_ref'] / surface['strike'],
        surface['strike'] / surface['s_ref']
    )
    surface = surface[surface['moneyness']<=1.15]
    return surface.pivot(index='moneyness', columns='maturity_days', values='price')

def smile_for(df):
    # Single signed log-moneyness axis ln(K/S) (NOT the per-wing ratio that always stays <1). Keep
    # OTM only: puts below spot (ln(K/S) < 0), calls above (ln(K/S) > 0). One contract per axis point,
    # so the two wings form a continuous smile instead of folding on top of each other (the old <1
    # ratio collided a put at K/S=m with a call at S/K=1/m onto the same value, which made it zig-zag).
    surface = df.copy()
    surface['moneyness'] = np.log(surface['strike'] / surface['s_ref'])
    otm = surface[((surface['w'] == 'put') & (surface['moneyness'] < 0)) |
                  ((surface['w'] == 'call') & (surface['moneyness'] > 0))]
    return otm.pivot(index='moneyness', columns='maturity_days', values='implied_vol')
    
def main(model=None, objective=None, target_date=None, save=True, show=False):
    """Render the puts/calls/smile OTM surfaces for one calibrated day.

    `model`/`objective` default to the `_results_config` switches when None (pass them to inspect a
    different run). `target_date` picks the day (default: the lowest-IV-RMSE day). `save=True` writes
    the three EPS + surfaces.tex under results/<model>/surfaces/plots/tex/; `save=False` skips disk.
    `show=True` leaves the figures open (does not close them) so the caller can render them; the
    rendering itself is the caller's job (see inspect.ipynb). Returns `(figs, day_results)` where
    `figs` is a dict keyed 'calls'/'puts'/'smile'.
    """
    from make_surface import make_surface  # type: ignore
    model = model or MODEL
    objective = objective or OBJECTIVE
    texdir = _texdir(model)
    if save:
        texdir.mkdir(parents=True, exist_ok=True)

    calibrations_file = calib_paths(model, objective)[0]
    cal = pd.read_csv(calibrations_file)
    cal = cal.sort_values(by='iv_rmse', ascending=True).reset_index(drop=True)
    if target_date is None:
        target_date = cal['date'][0]

    df, day_results = make_surface(target_date=target_date, model=model, objective=objective)
    date = day_results['date']
    spot = day_results['spot']
    params = day_results['params']
    market = day_results['market']
    fit = day_results['fit']

    figs = {
        'calls': plot_surface(price_grid(df, 'call'), texdir / "price_surface_calls.eps",
                              invert_K=True, xlabel=r"moneyness ($S/K$)", zlabel="call price",
                              save=save, show=show),
        'puts': plot_surface(price_grid(df, 'put'), texdir / "price_surface_puts.eps",
                             invert_K=True, zlabel="put price", save=save, show=show),
        'smile': plot_surface(smile_for(df), texdir / "smile_surface.eps", AZIM_ADJUST=-10,
                              invert_T=True, xlabel=r'log-moneyness ($\ln(K/S)$)',
                              zlabel=r'implied volatility ($\sigma^{\mathrm{mod}}$)',
                              save=save, show=show),
    }
    
    if save:
        write_surfaces_TeX(spot, date, params, market, fit, model=model, texdir=texdir)
        
    return figs, day_results

if __name__ == "__main__":
    matplotlib.use('Agg')   # headless: write files, never open a window
    main()
