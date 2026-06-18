import sys
from pathlib import Path
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


def _normalize_dates(dates):
    """Accept a single %Y-%m-%d date string or a list of them; return a list of strings.
    `make_surface` parses each with format="%Y-%m-%d"."""
    if isinstance(dates, str):
        return [dates]
    return list(dates)


def main(dates, OUT=None):
    dates = _normalize_dates(dates)

    # One pass per day: pull its surface + calibration, keep what the plot and the TeX need.
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
    n = len(days)
    # Each day is its own subfigure row: a (puts | calls) pair that shares its price (y) axis,
    # plus its own caption and its own colorbar (each day spans a different maturity range).
    fig = plt.figure(figsize=(8, 2.7 * n), layout='constrained')
    subfigs = fig.subfigures(n, 1)
    if n == 1:
        subfigs = [subfigs]
    rows = []
    for sf, day in zip(subfigs, days):
        T = day['T']
        norm = mcolors.Normalize(vmin=min(T), vmax=max(T))
        ax_call, ax_put  = sf.subplots(1, 2, sharey=True)
        for t in T:
            df = day['surface'][day['surface']['maturity_days'] == t]
            dfc = df[df['w'] == 'call'].sort_values(by='strike')
            ax_call.plot(dfc['strike'], dfc['price'], color=cmap(norm(t)))
            dfp = df[df['w'] == 'put'].sort_values(by='strike')
            ax_put.plot(dfp['strike'], dfp['price'], color=cmap(norm(t)))
        ax_call.set_ylabel('Price')
        sf.suptitle(_row_caption(day), fontsize=8)
        lbl = sf.supxlabel('Strike')
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sf.colorbar(sm, ax=(ax_call, ax_put), label='Days to maturity')
        rows.append((ax_call, ax_put, lbl))

    # `supxlabel` centers on the whole subfigure, but the colorbar steals right-side space, so the
    # two panels' midpoint is left of that. Resolve the layout, freeze it, then re-centre each
    # 'Strike' on the actual span of its two panels. (Stacked subfigures are full width, so axes
    # x-fractions and the subfigure-relative `supxlabel` x coincide.)
    fig.draw_without_rendering()
    fig.set_layout_engine('none')
    for ax_call, ax_put, lbl in rows:
        bc, bp = ax_call.get_position(), ax_put.get_position()
        lbl.set_x((min(bc.x0, bp.x0) + max(bc.x1, bp.x1)) / 2)

    tag_all = "_".join(day['tag'] for day in days)
    fig.savefig(FIGURES / f'smiles_{tag_all}.eps', format='eps', bbox_inches='tight')
    plt.close(fig)

    write_smiles_TeX(tag_all, days)


def _row_caption(day):
    """The per-row caption drawn in the figure: only date, S_ref, Phi, Feller, IV-RMSE, RMSE."""
    p, f = day['params'], day['fit']
    phi = (f"({p['theta']:.4f},\\,{p['kappa']:.4f},\\,{p['eta']:.4f},"
           f"\\,{p['rho']:.4f},\\,{p['v0']:.4f})")
    return (f"{day['tag']}:  "
            f"$S_{{\\mathrm{{ref}}}}={day['spot']:.2f}$,  "
            f"$\\Phi={phi}$,  "
            f"$\\mathcal{{F}}={f['feller']:.4f}$,  "
            f"IV-RMSE$={f['iv_rmse']*100:.2f}$,  "
            f"RMSE$={f['rmse']:.4f}$")


def write_smiles_TeX(tag_all, days):
    """Generate `figures/smiles.tex`: just the tiled figure float. The per-row calibration values
    (S_ref, Phi, Feller, IV-RMSE, RMSE) are drawn in the figure itself by `_row_caption`, so no
    prose is emitted here."""

    date_list = ", ".join(day['date'].strftime(r"%B %d, %Y") for day in days)
    caption = (f"Heston option prices for {date_list}: put wing (left) and call wing (right) "
               r"per day.")

    TeX = \
r"""\begin{figure}[H]
    \begin{center}
        \includegraphics[width=\linewidth,keepaspectratio=true]{results/smiles/figures/smiles_<tag>.eps}
        \caption{<caption>}
        \label{Fig:smiles}
    \end{center}
\end{figure}
"""
    TeX = TeX.replace('<tag>', tag_all).replace('<caption>', caption)
    (FIGURES / "smiles.tex").write_text(TeX)


def make_surfaces_for(dates):
    return main(dates=dates, OUT=None)


if __name__ == "__main__":
    CALIBRATIONS_FILE = SURFACES.parent / "calibrations" / 'price' / "calibrations.csv"
    import pandas as pd
    cal = pd.read_csv(CALIBRATIONS_FILE)
    cal = cal[cal['feller']>=0]
    dates = cal['date'][:4]
    make_surfaces_for(dates=dates)