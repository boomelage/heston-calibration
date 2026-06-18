import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
SURFACES = RESULTS / "surfaces"
SURFACES_DATA = SURFACES / "data"

if str(SURFACES) not in sys.path:
    sys.path.insert(0, str(SURFACES))

from example_surface import make_surface # type: ignore --> Intentional Pylance ingore

FIGURES = SMILES / "figures"
FIGURES.mkdir(parents=True,exist_ok=True)

def main(date, OUT):
    surface, day_results = make_surface(target_date=date, OUT=OUT, SAVE=False)
    date = day_results['date']
    tag = str(date.strftime(r"%Y-%m-%d"))
    T = surface['maturity_days'].unique().tolist()

    surface = surface[surface['maturity_days'].isin(T)]

    norm = mcolors.Normalize(vmin=min(T), vmax=max(T))
    cmap = cm.jet

    # Puts and calls share one figure so they share the y axis, x label and colorbar.
    fig, (ax_call, ax_put) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    for t in T:
        df = surface[surface['maturity_days'] == t]
        dfc = df[df['w'] == 'call'].sort_values(by='strike')
        ax_call.plot(dfc['strike'], dfc['price'], color=cmap(norm(t)))
        dfp = df[df['w'] == 'put'].sort_values(by='strike')
        ax_put.plot(dfp['strike'], dfp['price'], color=cmap(norm(t)))

    ax_put.set_title('Puts')
    ax_call.set_title('Calls')
    ax_call.set_ylabel('Price')
    fig.supxlabel('Strike')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=(ax_put, ax_call), label='Days to maturity')

    fig.savefig(FIGURES/f'smiles_{tag}.eps', format='eps', bbox_inches='tight')
    plt.close(fig)

    write_smiles_TeX(tag, day_results['spot'], date, day_results['params'],
                     day_results['market'], day_results['fit'])


def write_smiles_TeX(tag, spot, date, params, market, fit):
    """Generate `figures/smiles.tex` from the day's calibration, the same token-replacement
    principle as `surfaces/make_eps.py::write_otm_TeX`: all caption/prose values come from
    `day_results`, so the figure and its description never drift from the calibrated day."""

    TeX = \
r"""\begin{figure}[H]
    \begin{center}
        \includegraphics[width=\linewidth]{results/smiles/figures/smiles_<tag>.eps}
        \caption{Heston OTM option prices for $S_{\mathrm{ref}}$ <spot> on <date> with
        $\Phi = (<theta>,\ <kappa>,\ <eta>,\ <rho>,\ <v0>)$: put wing (left) and call wing
        (right), sharing a common price axis, strike axis, and maturity colorbar.}
        \label{Fig:smiles}
    \end{center}
\end{figure}

\noindent The smiles above are the Heston model's fitted OTM prices on <date>.
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
    TeX = TeX.replace('<tag>', tag)
    TeX = TeX.replace('<spot>', str(round(spot, 4)))
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

    (FIGURES / "smiles.tex").write_text(TeX)


def make_surface_for(date):
    tag = f"{date[0]}-{date[1]}-{date[2]}"
    return tag, main(date=date, OUT=None)#FIGURES/f"surface_{tag}")

if __name__ == "__main__":
    make_surface_for(date=(2020,3,16))
