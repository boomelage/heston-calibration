"""Generate an example Heston option-price surface from one calibrated day.

Sibling of `example_surface.py`: same model build and OTM grid, but each grid point keeps the
Heston *price* (option NPV) instead of inverting it to an implied vol. Takes a GIVEN row of
`data/calibrations.csv` (index via `--row`, default 0 = earliest day; or `--date YYYY-MM-DD`),
rebuilds the Heston model from its `spot_price`, `risk_free_rate`, `dividend_rate` and the five
fitted parameters (`v0, kappa, theta, eta, rho`), and prices the OTM European option (call above
spot, put below) under `AnalyticHestonEngine` across a strike x maturity grid.

We price the OTM wing so each quote is mostly time value -- the deep-ITM mirror would just add
intrinsic and obscure the surface shape. Result is written long-form (one row per grid point) and
as a strike x maturity pivot.

Run:  python results/example_price_surface.py [--row N | --date YYYY-MM-DD]
Out:  results/example_price_surface.csv        (long: strike, maturity_days, moneyness, w, price)
      results/example_price_surface_grid.csv   (pivot: index=strike, columns=maturity_days)
"""
import argparse
import numpy as np
import pandas as pd
import QuantLib as ql
from pathlib import Path

RESULTS = Path(__file__).parent.resolve()
CALIBRATIONS_FILE = RESULTS.parent / "data" / "calibrations.csv"

# Grid the surface is sampled on. Moneyness K/S around the money; maturities in calendar days
# spanning the range the calibration sees (>= MIN_DTM=7) out to a 2y tail.
MONEYNESS = np.round(np.arange(0.75, 1.25, 0.025), 10)
MATURITIES_DAYS = np.arange(30,365,10).tolist() # [30, 60, 90, 120, 180, 270, 365]


def build_heston_engine(row, calculation_date):
    """Rebuild the Heston model from one calibrations.csv row; return its pricing engine."""
    ql.Settings.instance().evaluationDate = calculation_date
    day_count = ql.Actual365Fixed()
    r_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, float(row['risk_free_rate']), day_count))
    g_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, float(row['dividend_rate']), day_count))
    s_handle = ql.QuoteHandle(ql.SimpleQuote(float(row['spot_price'])))
    # HestonProcess constructor order: (r, g, S0, v0, kappa, theta, sigma=eta, rho)
    process = ql.HestonProcess(
        r_ts, g_ts, s_handle,
        float(row['v0']), float(row['kappa']), float(row['theta']),
        float(row['eta']), float(row['rho']),
    )
    return ql.AnalyticHestonEngine(ql.HestonModel(process))


def heston_price(strike, maturity_date, spot, heston_engine):
    """Price the OTM European option (call above spot, put below) under Heston. Returns (w, NPV)."""
    w = 'call' if strike >= spot else 'put'
    payoff_type = ql.Option.Call if w == 'call' else ql.Option.Put
    option = ql.EuropeanOption(ql.PlainVanillaPayoff(payoff_type, strike),
                               ql.EuropeanExercise(maturity_date))
    option.setPricingEngine(heston_engine)
    return w, option.NPV()


def select_row(calibrations, args):
    """Pick the calibrations.csv row to price: by --date if given, else by --row index."""
    if args.date is not None:
        match = calibrations.index[calibrations['date'] == args.date]
        if len(match) == 0:
            raise SystemExit(f"no calibration row for date {args.date}")
        return calibrations.loc[match[0]]
    return calibrations.iloc[args.row]


def main():
    parser = argparse.ArgumentParser(description="Heston option-price surface from one calibrated day.")
    parser.add_argument('--row', type=int, default=0,
                        help="row index into calibrations.csv (default 0 = earliest day)")
    parser.add_argument('--date', type=str, default=None,
                        help="trading date YYYY-MM-DD (overrides --row)")
    args = parser.parse_args()

    calibrations = pd.read_csv(CALIBRATIONS_FILE)
    row = select_row(calibrations, args)
    spot = float(row['spot_price'])

    # Evaluation date: the row's own trading date, so day-count maturities line up with reality.
    date = pd.Timestamp(row['date'])
    calculation_date = ql.Date(date.day, date.month, date.year)

    heston_engine = build_heston_engine(row, calculation_date)

    print(f"Heston example price surface for {row['date']}  (spot={spot:.2f}, "
          f"r={row['risk_free_rate']:.4f}, q={row['dividend_rate']:.4f})")
    print(f"  v0={row['v0']:.4f}  kappa={row['kappa']:.4f}  theta={row['theta']:.4f}  "
          f"eta={row['eta']:.4f}  rho={row['rho']:.4f}\n")

    records = []
    for days in MATURITIES_DAYS:
        maturity_date = calculation_date + ql.Period(days, ql.Days)
        for m in MONEYNESS:
            strike = m * spot
            w, price = heston_price(strike, maturity_date, spot, heston_engine)
            records.append({
                'strike': round(strike, 4),
                'maturity_days': days,
                'moneyness': m,
                'w': w,
                'price': price,
            })

    surface = pd.DataFrame(records)
    long_path = RESULTS / "example_price_surface.csv"
    surface.to_csv(long_path, index=False)

    grid = surface.pivot(index='strike', columns='maturity_days', values='price')
    grid_path = RESULTS / "example_price_surface_grid.csv"
    grid.to_csv(grid_path)

    with pd.option_context('display.float_format', '{:.4f}'.format,
                           'display.max_columns', None, 'display.width', 200):
        print(grid)
    print(f"\nwrote {len(surface)} grid points -> {long_path}")
    print(f"wrote strike x maturity pivot     -> {grid_path}")


if __name__ == "__main__":
    main()
