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

# Knob for the per-row maturity key: True draws a legend, False (default) draws a colorbar.
USE_LEGEND = True


def _normalize_dates(dates):
    """Accept a single %Y-%m-%d date string or a list of them; return a list of strings.
    `make_surface` parses each with format="%Y-%m-%d"."""
    if isinstance(dates, str):
        return [dates]
    return list(dates)


def main(dates, OUT=None, use_legend=USE_LEGEND):
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
        _save_day_figure(day, cmap, use_legend)

    write_smiles_TeX(days)


def _save_day_figure(day, cmap, use_legend):
    T = [
        30, 60, 90, 180, 270, 350, 540 
    ]#sorted(day['T'])
    norm = mcolors.Normalize(vmin=min(T), vmax=max(T))

    fig, (ax_put, ax_call) = plt.subplots(1, 2, sharey=True,
                                           figsize=(8, 2.7),
                                           layout='constrained')
    for t in T:
        df = day['surface'][day['surface']['maturity_days'] == t]
        dfp = df[df['w'] == 'put'].sort_values(by='strike')
        ax_put.plot(dfp['strike'], dfp['price'], color=cmap(norm(t)))
        dfc = df[df['w'] == 'call'].sort_values(by='strike')
        ax_call.plot(dfc['strike'], dfc['price'], color=cmap(norm(t)), label=str(t))
    ax_put.set_ylabel(r'Price: $C_{\mathrm{H}}(\Phi^{\star})$')
    fig.suptitle(_row_caption(day), fontsize=8)
    lbl = fig.supxlabel('Strike ($K$)')

    if use_legend:
        handles, labels = ax_call.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside center right',
                   title='Days to maturity', fontsize=7, title_fontsize=8)
    else:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=(ax_put, ax_call), label='Days to maturity')

    fig.draw_without_rendering()
    fig.set_layout_engine('none')
    bc, bp = ax_call.get_position(), ax_put.get_position()
    lbl.set_x((min(bc.x0, bp.x0) + max(bc.x1, bp.x1)) / 2)

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
        caption = (f"Heston option prices for {date_pretty}: "
                   r"put wing (left) and call wing (right).")
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
    CALIBRATIONS_FILE = SURFACES.parent / "calibrations" / 'price' / "calibrations.csv"
    import pandas as pd
    cal = pd.read_csv(CALIBRATIONS_FILE)
    cal = cal[cal['feller']>=0].copy().reset_index(drop=True)
    dates = cal['date']
    make_surfaces_for(dates=dates)