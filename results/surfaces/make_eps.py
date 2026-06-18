"""Render the Heston option-price surface as EPS figures for LaTeX.

Reads `results/example_price_surface.csv` (from `example_price_surface.py`; columns
strike, maturity_days, moneyness, w, price) and draws MATLAB-style 3D surface plots -- jet
colormap, meshed facets, strike x maturity x price axes -- saved as vector EPS, the same kind of
figure embedded in `results/example-surface-rendering/`.

Three figures are produced (calls wing, puts wing, full OTM surface) plus a `price_surface.tex`
that includes them like the example's `analysis_090924.tex`. EPS embeds in LaTeX via
`\\includegraphics`; with pdflatex, convert first (`epstopdf *.eps`) or compile the provided .tex
with `latex price_surface.tex` (the classic dvips route) -- or just `pdflatex` after epstopdf.

Run:  python results/make_price_surface_eps.py
Out:  results/price_surface_calls.eps, price_surface_puts.eps, price_surface_both.eps
      results/price_surface.tex
"""
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')               # headless: write files, never open a window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the '3d' projection; also the ax type below
from pathlib import Path
import QuantLib as ql

RESULTS = Path(__file__).parent
SURFACE_CSV = RESULTS / "data" / "example_surface.csv"
DAY_RESULTS = RESULTS / "data" / "day_results.pkl"
TEXDIR = RESULTS / "plots" / "tex"
TEXDIR.mkdir(parents=True, exist_ok=True)

# Match the default LaTeX font (Computer Modern serif) so the axis text blends with the surrounding
# document. Uses matplotlib's bundled Computer Modern (cmr10) -- no LaTeX/usetex toolchain required.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['cmr10', 'Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,   # tick labels in Computer Modern too
    'axes.unicode_minus': False,           # cmr10 lacks U+2212; avoids missing-glyph warnings
    'font.size': 8,
})
ELEV, AZIM = 25, -60


def plot_surface(grid, out_path, title=None, invert_K=False):
    """Draw one strike x maturity x price surface (grid: index=strike, columns=maturity_days)."""

    strikes = grid.index.to_numpy(dtype=float)
    maturities = grid.columns.to_numpy(dtype=float) / 365.0      # days -> years, like the example
    X, Y = np.meshgrid(strikes, maturities)                      # (n_mat, n_strike)
    Z = grid.to_numpy(dtype=float).T                             # (n_mat, n_strike)

    fig = plt.figure(figsize=(5.0, 4.0))
    ax: Axes3D = fig.add_subplot(111, projection='3d')  # type: ignore[assignment]
    ax.plot_surface(X, Y, Z, cmap='jet', rstride=1, cstride=1,
                    linewidth=0.2, edgecolors='k', antialiased=False)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel(r'strike ($K$)')
    if invert_K:
        ax.invert_xaxis()
    ax.set_ylabel(r'maturity in years ($T$)')
    ax.set_zlabel(r'price')
    ax.set_zlim(bottom=0)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, format='eps', bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path.name}")

def _load_data():
    df = pd.read_csv(SURFACE_CSV)
    with open(DAY_RESULTS, 'rb') as file:
        day_results = pickle.load(file)
    return (df, day_results)

def write_otm_TeX(spot, date, params, market, fit):

    TeX = \
r"""\begin{figure}[H]
    \begin{center}
        \includegraphics[width=6.25cm,keepaspectratio=true]{results/surfaces/plots/tex/price_surface_puts.eps}
        \includegraphics[width=6.25cm,keepaspectratio=true]{results/surfaces/plots/tex/price_surface_calls.eps}
        \caption{Heston OTM option prices for $S_{\mathrm{ref}}$ <spot> on <date> with $\Phi = (<theta>,\ <kappa>,\ <eta>,\ <rho>,\ <v0>)$: puts wing (left) and calls wing (right).}
        \label{Fig:wings}
    \end{center}
    \begin{center}
        \includegraphics[width=6.25cm,keepaspectratio=true]{results/surfaces/plots/tex/put_smile.eps}
        \includegraphics[width=6.25cm,keepaspectratio=true]{results/surfaces/plots/tex/call_smile.eps}
        \caption{All put (left) and call (right) options from Figure~\ref{Fig:wings}}
    \end{center}
\end{figure}

\noindent The surface above is the Heston model's own implied volatility on <date>.
The fit uses <nhelpers> calibration cells across <nmats> maturities and <nstrikes> strikes,
backed by <volume> contracts of traded volume.
The reference spot was $S_{\mathrm{ref}} = <spot>$,
priced under a risk-free rate of <r>\% and a dividend rate of <q>\%.
The intraday spot range was <rangepct>\%.<movenote>
Fit quality is <ivrmse> vol points of implied-volatility RMSE,
with a relative-price RMSE of <rmse>.
The Feller condition <fellersign> at this calibration,
with $2\kappa\theta - \eta^2 = <feller>$.
    """
    TeX = TeX.replace('<spot>', str(spot))
    TeX = TeX.replace('<date>', str(date.strftime(r"%B %d, %Y")))
    TeX = TeX.replace('<theta>', str(round(params['theta'], 4)))
    TeX = TeX.replace('<kappa>', str(round(params['kappa'], 4)))
    TeX = TeX.replace('<eta>', str(round(params['eta'], 4)))
    TeX = TeX.replace('<rho>', str(round(params['rho'], 4)))
    TeX = TeX.replace('<v0>', str(round(params['v0'], 4)))
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

    tex_path = TEXDIR / r"otm.tex"
    tex_path.write_text(TeX)

def grid_for(df, side):
    surface = df[df['w'] == side].copy()
    surface['moneyness'] = np.where(
        surface['w'] == 'call',
        surface['s_ref'] / surface['strike'],
        surface['strike'] / surface['s_ref']
    )
    surface = surface[surface['moneyness']<=1]
    return surface.pivot(index='strike', columns='maturity_days', values='price')

def smile_for(df, side):
    surface = df[df['w'] == side].copy()
    return surface.pivot(index='strike', columns='maturity_days', values='price')
    
def main():
    from example_surface import make_surface
    make_surface(target_date=r'2020-03-16')
    if not SURFACE_CSV.exists():
        sys.exit(f"{SURFACE_CSV} not found -- run `python results/example_surface.py` first.")
    df, day_results = _load_data()
    date = day_results['date']
    spot = day_results['spot']
    params = day_results['params']
    market = day_results['market']
    fit = day_results['fit']

    plot_surface(grid_for(df, 'call'), TEXDIR / "price_surface_calls.eps")
    plot_surface(grid_for(df, 'put'), TEXDIR / "price_surface_puts.eps", invert_K=True)
    plot_surface(smile_for(df, 'call'), TEXDIR / "call_smile.eps")
    plot_surface(smile_for(df, 'put'), TEXDIR / "put_smile.eps", invert_K=True)
    write_otm_TeX(spot, date, params, market, fit)
    
if __name__ == "__main__":
    main()
