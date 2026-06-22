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
matplotlib.use('Agg')               # headless: write files, never open a window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the '3d' projection; also the ax type below
from pathlib import Path

# This script now lives under src/results/surfaces/, but reads/writes the repo-level results/ tree.
# `from make_surface import ...` resolves from this dir; SURFACES routes figure I/O to repo/results/.
# RESULTS_CODE (src/results) holds results_config.py, the central knob file for the figure scripts.
import sys
HERE = Path(__file__).parent.resolve()                 # src/results/surfaces
SRC = HERE.parents[1]                                   # src/ (shared utils.py, config.py)
RESULTS_CODE = HERE.parent                              # src/results (results_config.py)
REPO = HERE.parents[2]                                  # repo root (surfaces->results->src->repo)
RESULTS = REPO / "results"
for _p in (str(HERE), str(SRC), str(RESULTS_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# All tunable parameters live in results_config.py. MODEL/OBJECTIVE pick the engine, the
# calibrations source, and the results/<model>/surfaces/ figure tree.
from results_config import (  # type: ignore
    MODEL, OBJECTIVE, PLOT_RCPARAMS, SURFACE_ELEV, SURFACE_AZIM, SURFACE_FIGSIZE)
from config import calib_paths  # type: ignore
MODEL_LABEL = MODEL.capitalize()
SURFACES = RESULTS / MODEL / "surfaces"                 # data/figure dir at repo/results/<model>/surfaces
SURFACE_CSV = SURFACES / "data" / "example_surface.csv"
DAY_RESULTS = SURFACES / "data" / "day_results.pkl"
TEXDIR = SURFACES / "plots" / "tex"
TEXDIR.mkdir(parents=True, exist_ok=True)

# Match the default LaTeX font (Computer Modern serif) so the axis text blends with the surrounding
# document. Uses matplotlib's bundled Computer Modern (cmr10) -- no LaTeX/usetex toolchain required.
plt.rcParams.update(PLOT_RCPARAMS)
ELEV, AZIM = SURFACE_ELEV, SURFACE_AZIM


def plot_surface(grid, out_path, title=None, invert_K=False):
    """Draw one strike x maturity x price surface (grid: index=strike, columns=maturity_days)."""

    strikes = grid.index.to_numpy(dtype=float)
    maturities = grid.columns.to_numpy(dtype=float) / 365.0      # days -> years, like the example
    X, Y = np.meshgrid(strikes, maturities)                      # (n_mat, n_strike)
    Z = grid.to_numpy(dtype=float).T                             # (n_mat, n_strike)

    fig = plt.figure(figsize=SURFACE_FIGSIZE)
    plt.style.use('fast')
    ax: Axes3D = fig.add_subplot(111, projection='3d')  # type: ignore[assignment]
    surf = ax.plot_surface(X, Y, Z, rstride=3, cstride=3,
                    color="white", edgecolor="black",
                    linewidth=0.4, shade=False, antialiased=True)
    
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor("0.7")
        pane.pane.set_linewidth(0.5)
 
    ax.grid(False)
    ax.tick_params(labelsize=9, colors="black")
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel(r'strike ($K$)')
    if invert_K:
        ax.invert_xaxis()
    ax.set_ylabel(r'maturity in years ($T$)')
    ax.set_zlabel(r'price')
    ax.set_zlim(bottom=0)
    if title:
        ax.set_title(title)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.tight_layout()
    fig.savefig(out_path, format='eps', bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path.name}")


def write_otm_TeX(spot, date, params, market, fit):

    TeX = \
r"""
\section{Example option prices} Produced by the market\!\,\footnote{
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
calibrated pricing operator <operator>~\eqref{eq:accept}.
\begin{figure}[H]
    \begin{center}
        \includegraphics[width=6.25cm,keepaspectratio=true]{results/<MODEL>/surfaces/plots/tex/price_surface_puts.eps}
        \includegraphics[width=6.25cm,keepaspectratio=true]{results/<MODEL>/surfaces/plots/tex/price_surface_calls.eps}
        \caption{<MODEL_LABEL> option prices for $S_{\mathrm{ref}}$ <spot> on <date> with <parameters>: puts wing (left) and calls wing (right).}
        \label{Fig:wings}
    \end{center}
    \begin{center}
        \includegraphics[width=10cm,keepaspectratio=true]{results/<MODEL>/surfaces/plots/tex/call_smile.eps}
        \caption{\emph{Out of the money} implied volatilites from Figure~\ref{Fig:wings}}
    \end{center}
\end{figure}

"""

    hestonparams = r'$\Phi^{\star} = (<theta>,\ <kappa>,\ <eta>,\ <rho>,\ <v0>)$'
    batesparams = r'$\Theta^{\star} = (<theta>,\ <kappa>,\ <eta>,\ <rho>,\ <v0>, \ <lambda>, \ <nu>, \ <delta>)$'
    paramstr = hestonparams if MODEL == 'heston' else batesparams
    paramstr = paramstr.replace('<theta>', str(round(params['theta'], 4)))
    paramstr = paramstr.replace('<kappa>', str(round(params['kappa'], 4)))
    paramstr = paramstr.replace('<eta>', str(round(params['eta'], 4)))
    paramstr = paramstr.replace('<rho>', str(round(params['rho'], 4)))
    paramstr = paramstr.replace('<v0>', str(round(params['v0'], 4)))
    if MODEL == 'bates':
        paramstr = paramstr.replace('<lambda>', str(round(params['lambda_'], 4)))
        paramstr = paramstr.replace('<nu>', str(round(params['nu'], 4)))
        paramstr = paramstr.replace('<delta>', str(round(params['delta'], 4)))
    hestoneq = r'$C_{\mathrm{H}}(\Phi^{\star}; S,K,\tau,w)$~\eqref{eq:heston-price}'
    bateseq = r'$C_{\mathrm{Bates}}(\Theta^{\star}; S,K,\tau,w)$~\eqref{eq:bates-cf}'
    TeX = TeX.replace('<parameters>', paramstr)
    TeX = TeX.replace('<operator>', hestoneq if MODEL == 'heston' else bateseq)
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
    TeX = TeX.replace('<MODEL>', str(MODEL))
    TeX = TeX.replace('<MODEL_LABEL>', str(MODEL_LABEL))

    tex_path = TEXDIR / r"surfaces.tex"
    tex_path.write_text(TeX)

def otm_grid(df, side):
    surface = df[df['w'] == side].copy()
    surface['moneyness'] = np.where(
        surface['w'] == 'call',
        surface['s_ref'] / surface['strike'],
        surface['strike'] / surface['s_ref']
    )
    surface = surface[surface['moneyness']<=1.15]
    return surface.pivot(index='strike', columns='maturity_days', values='price')

def smile_for(df):
    surface = df[df['w'] == 'call'].copy()
    df['moneyness'] = np.where(
        df['w'] == 'call',
        df['s_ref'] / df['strike'],
        df['strike'] / df['s_ref']
    )
    surface = df[df['moneyness']<1].copy().reset_index(drop=True)
    return surface.pivot(index='strike', columns='maturity_days', values='implied_vol')
    
def main():
    from make_surface import make_surface  # type: ignore (MODEL/OBJECTIVE imported at module top)
    CALIBRATIONS_FILE = calib_paths(MODEL, OBJECTIVE)[0]
    cal = pd.read_csv(CALIBRATIONS_FILE)
    cal = cal.sort_values(by='iv_rmse',ascending=True).reset_index(drop=True)

    target_date = cal['date'][0]
    df, day_results = make_surface(target_date=target_date)
    date = day_results['date']
    spot = day_results['spot']
    params = day_results['params']
    market = day_results['market']
    fit = day_results['fit']

    plot_surface(otm_grid(df, 'call'), TEXDIR / "price_surface_calls.eps")
    plot_surface(otm_grid(df, 'put'), TEXDIR / "price_surface_puts.eps", invert_K=True)
    plot_surface(smile_for(df), TEXDIR / "call_smile.eps")
    write_otm_TeX(spot, date, params, market, fit)
    
if __name__ == "__main__":
    main()
