import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
    fig, ax = plt.subplots()
    for t in T:
        df = surface[surface['maturity_days']==t]
        df = df[df['w']=='call']
        df = df.sort_values(by='strike')
        ax.plot(df['strike'], df['price'], color=cmap(norm(t)))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label='Days to maturity')
    fig.savefig(FIGURES/f'calls_{tag}.eps', format='eps', bbox_inches='tight')
    
    norm = mcolors.Normalize(vmin=min(T), vmax=max(T))
    cmap = cm.jet
    fig, ax = plt.subplots()
    for t in T:
        df = surface[surface['maturity_days']==t]
        df = df[df['w']=='put']
        df = df.sort_values(by='strike')
        ax.plot(df['strike'], df['price'], color=cmap(norm(t)))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label='Days to maturity')
    fig.savefig(FIGURES/f'puts_{tag}.eps', format='eps', bbox_inches='tight')
    
def make_surface_for(date):
    tag = f"{date[0]}-{date[1]}-{date[2]}"
    return tag, main(date=date, OUT=None)#FIGURES/f"surface_{tag}")

if __name__ == "__main__":
    make_surface_for(date=(2020,3,16))
