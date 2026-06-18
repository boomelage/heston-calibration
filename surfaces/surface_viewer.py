"""Interactive 3D view of the example Heston implied-volatility surface.

Reads the grid produced by `example_surface.py` (`results/example_surface.csv`, long-form:
strike, maturity_days, moneyness, implied_vol), pivots it to a strike x maturity mesh, and renders
it as an interactive Plotly surface -- rotate, zoom, and hover for the implied vol at each node.
Opens in the browser and also writes a self-contained `results/example_surface.html`.

Run:  python results/surface_viewer.py
      (run example_surface.py first to (re)generate the CSV)
"""
import sys
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

SURFACES = Path(__file__).parent.resolve()
SURFACE_CSV = SURFACES / "data" / "example_surface.csv"

SURFACE_TYPE = "price"
if SURFACE_TYPE == "vol":
    HTML_OUT = SURFACES / "plots" / "example_surface.html"
    ZNAME = "implied vol (%)"
    KEY = "implied_vol"

if SURFACE_TYPE == "price":
    HTML_OUT = SURFACES / "plots" / "example_price_surface.html"
    ZNAME = "Heston price"
    KEY = "price"


def main():
    if not SURFACE_CSV.exists():
        sys.exit(f"{SURFACE_CSV} not found -- run `python results/example_surface.py` or `python results/example_price_surface.py` first.")

    surface = pd.read_csv(SURFACE_CSV)

    # Pivot to a strike (rows) x maturity (cols) mesh of implied vols. x=maturity, y=strike, z=IV.
    grid = surface.pivot(index='strike', columns='maturity_days', values=KEY)
    x = grid.columns.to_numpy()            # maturities in days
    y = grid.index.to_numpy()              # strikes
    z = grid.to_numpy() * 100.0 if SURFACE_TYPE == 'vol' else grid.to_numpy()           # implied vol in percent for readable axis ticks

    fig = go.Figure(data=[go.Surface(
        x=x, y=y, z=z,
        colorscale='Viridis',
        colorbar=dict(title='IV (%)' if SURFACE_TYPE == 'vol' else 'price'),
        hovertemplate=('maturity: %{x} days<br>strike: %{y:.0f}<br>'
                       +ZNAME+': %{z:.2f}%<extra></extra>'),
        contours={'z': {'show': True, 'usecolormap': True, 'project_z': True}},
    )])

    fig.update_layout(
        title='Heston implied-volatility surface',
        scene=dict(
            xaxis_title='maturity (days)',
            yaxis_title='strike',
            zaxis_title=ZNAME,
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(HTML_OUT, include_plotlyjs='cdn')
    print(f"wrote interactive surface -> {HTML_OUT}")
    fig.show()


if __name__ == "__main__":
    main()
