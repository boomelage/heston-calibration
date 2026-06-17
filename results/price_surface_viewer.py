"""Interactive 3D view of the example Heston option-price surface.

Reads the grid produced by `example_price_surface.py` (`results/example_price_surface.csv`,
long-form: strike, maturity_days, moneyness, w, price), optionally restricts to one wing, pivots to
a strike x maturity mesh, and renders it as an interactive Plotly surface -- rotate, zoom, and hover
for the option price at each node.

The surface prices the OTM wing (calls above spot, puts below), so `--side calls` / `--side puts`
shows that half of the strike range; `both` (default) shows the full OTM smile peaking near ATM.

Run:  python results/price_surface_viewer.py [--side calls|puts|both]
      (run example_price_surface.py first to (re)generate the CSV)
"""
import argparse
import sys
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

RESULTS = Path(__file__).parent.resolve()
SURFACE_CSV = RESULTS / "example_price_surface.csv"


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D Heston price surface.")
    parser.add_argument('--side', choices=['calls', 'puts', 'both'], default='calls',
                        help="which OTM wing to display (default both)")
    args = parser.parse_args()

    if not SURFACE_CSV.exists():
        sys.exit(f"{SURFACE_CSV} not found -- run `python results/example_price_surface.py` first.")

    surface = pd.read_csv(SURFACE_CSV)
    if args.side != 'both':
        wing = 'call' if args.side == 'calls' else 'put'
        surface = surface[surface['w'] == wing]

    # Pivot to a strike (rows) x maturity (cols) mesh of prices. x=maturity, y=strike, z=price.
    grid = surface.pivot(index='strike', columns='maturity_days', values='price')
    x = grid.columns.to_numpy()            # maturities in days
    y = grid.index.to_numpy()              # strikes
    z = grid.to_numpy()                    # option price

    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=z,
        colorscale='Viridis',
        colorbar=dict(title='price'),
        hovertemplate=('maturity: %{x} days<br>strike: %{y:.0f}<br>'
                       'price: %{z:.2f}<extra></extra>'),
        contours={'z': {'show': True, 'usecolormap': True, 'project_z': True}},
    )])

    label = {'calls': 'OTM calls', 'puts': 'OTM puts', 'both': 'OTM wing'}[args.side]
    fig.update_layout(
        title=f'Heston option-price surface ({label})',
        scene=dict(
            xaxis_title='maturity (days)',
            yaxis_title='strike',
            zaxis_title='price',
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.show()


if __name__ == "__main__":
    main()
