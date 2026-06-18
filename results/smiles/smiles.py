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

DATA = SMILES / "data"
DATA.mkdir(parents=True,exist_ok=True)

def main(date, OUT):
    tag = f"{date[0]}-{date[1]}-{date[2]}"
    surface, day_results = make_surface(target_date=date, OUT=OUT, SAVE=False)

    T = surface['maturity_days'].unique().tolist()

    surface = surface[surface['maturity_days'].isin(T)]
    
    for t in T:
        plot = surface[surface['maturity_days']==t].copy().reset_index(drop=True)
        plot = plot[['strike','price']][plot['w']=='call']
        plot.to_csv(OUT/f"plot_calls_{t}-day_{tag}.csv" , index=False)
    
    for t in T:
        plot = surface[surface['maturity_days']==t].copy().reset_index(drop=True)
        plot = plot[['strike','price']][plot['w']=='put']
        plot.to_csv(OUT/f"plot_puts_{t}-day_{tag}.csv" , index=False)

    # norm = mcolors.Normalize(vmin=min(T), vmax=max(T))
    # cmap = cm.viridis
    # fig, ax = plt.subplots()
    # for t in T:
    #     df = surface[surface['maturity_days']==t]
    #     df = df[df['w']=='call']
    #     df = df.sort_values(by='strike')
    #     ax.plot(df['strike'], df['price'], color=cmap(norm(t)))
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # fig.colorbar(sm, ax=ax, label='Days to maturity')
    # plt.show()

def make_surface_for(date):
    tag = f"{date[0]}-{date[1]}-{date[2]}"
    return tag, main(date=date, OUT=DATA/f"surface_{tag}")

if __name__ == "__main__":
    make_surface_for(date=(2020,3,16))
