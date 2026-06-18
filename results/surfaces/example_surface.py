"""Generate an example Heston implied-volatility surface from one calibrated day.

Takes the FIRST row of `data/calibrations.csv` (the earliest trading day, since the file is
sorted ascending by date), rebuilds the Heston model from its `spot_price`, `risk_free_rate`,
`dividend_rate` and the five fitted parameters (`v0, kappa, theta, eta, rho`), and evaluates the
Black implied vol it implies across a strike x maturity grid.

For each grid point we price the OTM European option under Heston (AnalyticHestonEngine) and invert
that price back to a Black implied vol -- the model's own implied-vol surface. We price the OTM wing
(call above spot, put below) so the inversion works off time value, not deep intrinsic, and we go
through the explicit price-then-invert path rather than `HestonBlackVolSurface`: its internal
inversion returns spurious roots in the short-dated far-OTM corner (prices ~0), whereas the explicit
`impliedVolatility` call raises there and we record an honest NaN instead of a bogus vol.

We sample on a moneyness grid (strikes = m * spot) and a maturity grid (in days), and write the
result both long-form (one row per grid point) and as a strike x maturity pivot for inspection.

Run:  python results/example_surface.py
Out:  results/example_surface.csv        (long: strike, maturity_days, moneyness, implied_vol)
      results/example_surface_grid.csv   (pivot: index=strike, columns=maturity_days)
"""
import numpy as np
import pandas as pd
import QuantLib as ql
from pathlib import Path

SURFACES = Path(__file__).parent.resolve()
OBJECTIVE = input("Validate `vol` or `price` calibrations? ").strip().lower()

CALIBRATIONS_FILE = SURFACES.parent / "calibrations" / OBJECTIVE / "calibrations.csv"

# Grid the surface is sampled on. Moneyness K/S around the money; maturities in calendar days
# spanning the range the calibration actually sees (>= MIN_DTM=7 up to ~1y).
MONEYNESS = np.round(np.arange(0.75, 1.205, 0.01), 4)   # 0.80 .. 1.20
MATURITIES_DAYS = np.arange(30, 365, 5).tolist()

from utils import implied_vol, build_heston_engine, heston_price

def main():
    calibrations = pd.read_csv(CALIBRATIONS_FILE)
    row = calibrations.iloc[0]
    spot = float(row['spot_price'])

    # Evaluation date: the row's own trading date, so day-count maturities line up with reality.
    date = pd.Timestamp(row['date'])
    calculation_date = ql.Date(date.day, date.month, date.year)

    heston_engine, s_handle, r_ts, g_ts, day_count = build_heston_engine(row, calculation_date)
    # A Black process for the inversion. Its vol quote is a placeholder -- impliedVolatility solves
    # for the vol that reprices the Heston NPV, ignoring whatever sits here.
    bsm_process = ql.BlackScholesMertonProcess(
        s_handle, g_ts, r_ts,
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
            calculation_date, ql.UnitedStates(ql.UnitedStates.NYSE), 0.20, day_count)))

    print(f"Heston example surface for {row['date']}  (spot={spot:.2f}, "
          f"r={row['risk_free_rate']:.4f}, q={row['dividend_rate']:.4f})")
    print(f"  v0={row['v0']:.4f}  kappa={row['kappa']:.4f}  theta={row['theta']:.4f}  "
          f"eta={row['eta']:.4f}  rho={row['rho']:.4f}\n")

    records = []
    for days in MATURITIES_DAYS:
        maturity_date = calculation_date + ql.Period(days, ql.Days)
        for m in MONEYNESS:
            strike = m * spot
            iv = implied_vol(strike, maturity_date, spot, heston_engine, bsm_process)
            w, price = heston_price(strike, maturity_date, spot, heston_engine)
            records.append({
                'strike': round(strike, 4),
                'maturity_days': days,
                'moneyness': m,
                'w': w,
                'implied_vol': iv,
                'price': price
            })

    surface = pd.DataFrame(records)
    long_path = SURFACES / "data" /"example_surface.csv"    
    surface.to_csv(long_path, index=False)
    
    price_grid_path = SURFACES / "data" / "example_price_surface_grid.csv"
    price_grid = surface.pivot(index='strike', columns='maturity_days', values='price')
    price_grid.to_csv(price_grid_path)

    vol_grid_path = SURFACES / "data" / "example_surface_grid.csv"
    vol_grid = surface.pivot(index='strike', columns='maturity_days', values='implied_vol')
    vol_grid.to_csv(vol_grid_path)

    with pd.option_context('display.float_format', '{:.4f}'.format,
                           'display.max_columns', None, 'display.width', 200):
        print(vol_grid, '\n', price_grid)
    print(f"\nwrote {len(surface)} grid points                -> {long_path}")
    print(f"wrote strike x maturity price pivot   -> {price_grid_path}")
    print(f"wrote strike x maturity vol pivot     -> {vol_grid_path}")


if __name__ == "__main__":
    main()
