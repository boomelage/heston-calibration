"""Render the Heston option-price surface as EPS figures for LaTeX -- gnuplot edition.

A drop-in twin of `make_eps.py`: same read paths (`data/example_surface.csv`,
`data/day_results.pkl`, the `price/calibrations.csv` pick) and the same write paths
(`plots/tex/{price_surface_calls,price_surface_puts,call_smile,put_smile}.eps` plus `otm.tex`),
but the 3D surfaces are drawn with `py-gnuplot` (gnuplot's pm3d) instead of matplotlib.

Each strike x maturity x price grid is shipped to gnuplot as an inline datablock with blank lines
separating the strike scanlines, then drawn `with pm3d` under a jet palette and black facet borders
to keep the MATLAB-style meshed look of the matplotlib version. EPS embeds in LaTeX via
`\\includegraphics`; with pdflatex, convert first (`epstopdf *.eps`) or compile the provided .tex
with `latex` (the classic dvips route).

Run:  python results/surfaces/gnu.py
Out:  results/surfaces/plots/tex/{price_surface_calls,price_surface_puts,call_smile,put_smile}.eps
      results/surfaces/plots/tex/otm.tex
"""
import os
import shutil
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# py-gnuplot shells out to a `gnuplot` binary that is installed here but not on PATH on this box.
# Inject the bin dir before the first Gnuplot() is constructed (the subprocess inherits os.environ).
_GNUPLOT_BIN = r"C:\Program Files\gnuplot\bin"
if shutil.which("gnuplot") is None and os.path.isdir(_GNUPLOT_BIN):
    os.environ["PATH"] = _GNUPLOT_BIN + os.pathsep + os.environ["PATH"]

from pygnuplot import gnuplot

SURFACES = Path(__file__).parent
SURFACE_CSV = SURFACES / "data" / "example_surface.csv"
DAY_RESULTS = SURFACES / "data" / "day_results.pkl"
TEXDIR = SURFACES / "plots" / "tex"
TEXDIR.mkdir(parents=True, exist_ok=True)

ELEV, AZIM = 25, -60
# gnuplot's `set view rot_x, rot_z` reproduces matplotlib's view_init(elev, azim): equating the
# camera eye-vectors gives rot_x = 90 - elev and rot_z = 90 - azim (verified to match exactly).
ROT_X, ROT_Z = 25, -60

# Jet-like palette (dark blue -> cyan -> green -> yellow -> dark red), matching matplotlib's 'jet'.
_JET = '(0 0 0 0.5, 1 0 0 1, 2 0 1 1, 3 0.5 1 0.5, 4 1 1 0, 5 1 0 0, 6 0.5 0 0)'


def _nice_step(raw):
    """Round a raw tic step up to the nearest 1/2/5 x 10^n, so labels land on round strikes."""
    if raw <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def plot_surface(grid, out_path, title=None, invert_K=False):
    """Draw one strike x maturity x price surface (grid: index=strike, columns=maturity_days)."""

    strikes = grid.index.to_numpy(dtype=float)
    maturities = grid.columns.to_numpy(dtype=float) / 365.0      # days -> years, like the example
    Z = grid.to_numpy(dtype=float)                               # (n_strike, n_mat)

    # Pick a strike-tic step that yields ~5 labels regardless of the plot's strike span, so the
    # narrow OTM-wing surfaces and the wider full-smile surfaces share the same label density.
    xtic_step = _nice_step((strikes.max() - strikes.min()) / 5.0)

    # Exact data extents in every direction, so the plot box hugs the surface (no gnuplot padding
    # to the next tic, no floor floated below the data).
    finite = Z[np.isfinite(Z)]
    zmin, zmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    smin, smax = float(strikes.min()), float(strikes.max())
    tmin, tmax = float(maturities.min()), float(maturities.max())

    # Build a gnuplot grid datablock: one scanline (block) per strike, blank line between blocks so
    # pm3d reads it as a surface. Missing cells are written as 'nan' and flagged via datafile missing.
    rows = []
    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            z = Z[i, j]
            rows.append(f"{K:.10g} {T:.10g} {'nan' if np.isnan(z) else format(z, '.10g')}")
        rows.append("")
    data = "\n".join(rows)

    g = gnuplot.Gnuplot(log=False)
    g.cmd('set terminal postscript eps enhanced color font "Times-Roman,18" size 5in,4in')
    g.cmd(f'set output "{out_path.as_posix()}"')
    g.cmd('set datafile missing "nan"')
    g.cmd('unset key')
    g.cmd(f'set view {ROT_X}, {ROT_Z}')
    g.cmd(f'set palette defined {_JET}')
    g.cmd('unset colorbox')
    g.cmd('set pm3d depthorder')
    g.cmd('set pm3d border lc rgb "black" lw 0.4')
    g.cmd('set style fill solid')
    # Big fonts collide with the axes at gnuplot's default offsets. Thin the strike tics (every 200,
    # else they overrun each other along the receding x-axis), push each tic series and label outward
    # (offsets are in character widths/heights), run the maturity label parallel to its axis, and
    # widen the margins so nothing clips the EPS bounding box.
    g.cmd('set lmargin 8')
    g.cmd('set rmargin 12')
    g.cmd('set tics font "Times-Roman,15"')
    g.cmd(f'set xtics {xtic_step:g} offset 0,-0.4')
    g.cmd('set ytics 0.4 offset 1.2,0')
    g.cmd('set ztics offset -0.3,0')
    g.cmd('set xlabel "strike (K)" font "Times-Roman,20" offset 0,-1.5 rotate parallel')
    g.cmd('set ylabel "maturity in years (T)" font "Times-Roman,20" offset -4,-1.8 rotate parallel')
    g.cmd('set zlabel "price" font "Times-Roman,20" offset 1,0 rotate by 90')
    g.cmd(f'set yrange [{tmin:g}:{tmax:g}]')
    g.cmd(f'set zrange [{zmin:g}:{zmax:g}]')
    g.cmd(f'set xyplane at {zmin:g}')   # base at the data floor (gnuplot floats it half a z-range below by default)
    if invert_K:
        g.cmd(f'set xrange [{smax:g}:{smin:g}]')
    else:
        g.cmd(f'set xrange [{smin:g}:{smax:g}]')
    if title:
        g.cmd(f'set title "{title}"')
    # cmd() drops blank lines, so feed the datablock through the raw writer to preserve scanline gaps.
    g.__call__(f'$SURF << EOD\n{data}\nEOD')
    g.cmd('splot $SURF using 1:2:3 with pm3d')
    g.close()           # flush: quitting gnuplot triggers the EPS write
    print(f"wrote {out_path.name}")

def _load_data():
    df = pd.read_csv(SURFACE_CSV)
    with open(DAY_RESULTS, 'rb') as file:
        day_results = pickle.load(file)
    return (df, day_results)

def write_otm_TeX(spot, date, params, market, fit):

    TeX = \
r"""
\subsubsection{Example option prices} Produced by the market\!\,\footnote{
The surface above is the Heston model's own implied volatility on <date>.
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
calibrated pricing operator $C_{\mathrm{H}}(\Phi^\star)$~\eqref{eq:heston-price}~\eqref{eq:accept}.
\begin{figure}[H]
    \centering
    \includegraphics[width=6.25cm,keepaspectratio=true]{results/surfaces/plots/tex/price_surface_puts.eps}%
    \includegraphics[width=6.25cm,keepaspectratio=true]{results/surfaces/plots/tex/price_surface_calls.eps}
    \caption{Heston OTM option prices for $S_{\mathrm{ref}}$ <spot> on <date> with $\Phi = (<theta>,\ <kappa>,\ <eta>,\ <rho>,\ <v0>)$: puts wing (left) and calls wing (right).}
    \label{Fig:wings}
    \par\vspace{-0.5ex}
    \includegraphics[width=6.25cm,keepaspectratio=true]{results/surfaces/plots/tex/put_smile.eps}%
    \includegraphics[width=6.25cm,keepaspectratio=true]{results/surfaces/plots/tex/call_smile.eps}
    \caption{All put (left) and call (right) options from Figure~\ref{Fig:wings}}
\end{figure}
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
    CALIBRATIONS_FILE = SURFACES.parent / "calibrations" / 'price' / "calibrations.csv"
    cal = pd.read_csv(CALIBRATIONS_FILE)
    cal = cal[cal['feller']>=0].copy().sort_values(by='rmse',ascending=True).reset_index(drop=True)
    target_date = cal['date'][1]
    make_surface(target_date=target_date)
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
