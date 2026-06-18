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
import os
import sys
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


def plot_surface(grid, out_path, title=None):
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
    ax.set_ylabel(r'maturity in years ($T$)')
    ax.set_zlabel(r'price')
    ax.set_zlim(bottom=0)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, format='eps', bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path.name}")

def _load_surface(SURFACE_CSV):

    df = pd.read_csv(SURFACE_CSV)
    date = pd.Timestamp(df['date'][0]).strftime("%B %d, %Y")
    spot = float(df['s_ref'].unique()[0]) 
    return (spot, date, df)

def write_otm_TeX(spot, date):

    TeX = \
r"""\begin{figure}[H]
    \begin{center}
        \includegraphics[width=6.25cm,keepaspectratio=true]{results/surfaces/plots/tex/price_surface_puts.eps}
        \includegraphics[width=6.25cm,keepaspectratio=true]{results/surfaces/plots/tex/price_surface_calls.eps}
        \caption{Heston OTM option prices for $S_{\mathrm{ref}}$ <spot> on <date>: puts wing (left) and calls wing (right).}
        \label{Fig:wings}
    \end{center}
\end{figure}
    """
    TeX = TeX.replace('<spot>',str(spot))
    TeX = TeX.replace('<date>',str(date))
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
    print(surface)
    return surface.pivot(index='strike', columns='maturity_days', values='price')

def main():
    if not SURFACE_CSV.exists():
        sys.exit(f"{SURFACE_CSV} not found -- run `python results/example_surface.py` first.")
    spot, date, df = _load_surface(SURFACE_CSV)
    print(date)

    plot_surface(grid_for(df, 'call'), TEXDIR / "price_surface_calls.eps")
    plot_surface(grid_for(df, 'put'), TEXDIR / "price_surface_puts.eps")
    write_otm_TeX(spot, date)
    
if __name__ == "__main__":
    main()
