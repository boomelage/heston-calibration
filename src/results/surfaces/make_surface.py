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

Run:  python src/results/surfaces/make_surface.py
Out:  results/example_surface.csv        (long: strike, maturity_days, moneyness, implied_vol)
      results/example_surface_grid.csv   (pivot: index=strike, columns=maturity_days)
"""
# import pickle
import sys
import numpy as np
import pandas as pd
import QuantLib as ql
from pathlib import Path

# This script now lives under src/results/surfaces/, but reads/writes the repo-level results/ tree.
# HERE is the script dir; SRC holds the shared _utils/config; RESULTS routes data I/O to repo/results/.
# RESULTS_CODE (src/results) holds _results_config.py, the central knob file for the figure scripts.
HERE = Path(__file__).parent.resolve()                 # src/results/surfaces
SRC = HERE.parents[1]                                   # src/ (shared _utils.py, config.py)
RESULTS_CODE = HERE.parent                              # src/results (_results_config.py)
REPO = HERE.parents[2]                                  # repo root (surfaces->results->src->repo)
for _p in (str(SRC), str(RESULTS_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
RESULTS = REPO / "results"
# All tunable parameters live in _results_config.py. MODEL also picks the QuantLib engine
# (Heston vs Bates) and the data output tree (results/<model>/surfaces/). They are the per-call
# defaults; make_surface(model=..., objective=...) overrides them for a notebook.
from _results_config import (  # type: ignore
    MODEL, OBJECTIVE, MONEYNESS, MATURITIES_DAYS)

from config import calib_paths  # type: ignore

from _utils import model_implied_vol, model_price
from _results_utils import load_calibrations_by_date, build_day_engine  # type: ignore

def make_surface(target_date=None, model=None, objective=None, save=False, out=None):
    """Rebuild one calibrated day's model IV/price surface.

    `model`/`objective` default to the `_results_config` switches; pass them to inspect a different
    run without editing `_results_config`. `save=False` (the default) returns the frame without
    touching disk; `save=True` writes the long-form CSV under `out` (default
    results/<model>/surfaces/data/). Returns `(surface_df, day_results)`.
    """
    model = model or MODEL
    objective = objective or OBJECTIVE
    calibrations_file = calib_paths(model, objective)[0]
    data = RESULTS / model / "surfaces" / "data"
    out = out or data
    if save:
        out.mkdir(parents=True, exist_ok=True)
        data.mkdir(parents=True, exist_ok=True)
    calibrations = load_calibrations_by_date(calibrations_file)
    if target_date is not None:
        ts = pd.Timestamp(pd.to_datetime(target_date, format=r"%Y-%m-%d"))
        row = calibrations.asof(ts)
    else:
        row = calibrations.iloc[0]
    row['date'] = row.name
    spot = float(row['spot_price'])

    # Evaluation date: the row's own trading date, so day-count maturities line up with reality.
    date = pd.Timestamp(row['date'])
    # Engine + a Black process for the price->IV inversion (placeholder vol is ignored by
    # impliedVolatility). build_day_engine returns (engine, bsm, calc_date) and is the one shared
    # home of this construction (also used by smiles).
    engine, bsm_process, calculation_date = build_day_engine(row, date, model)
    kappa, theta, rho, eta, v0 = row['kappa'], row['theta'], row['rho'], row['eta'], row['v0']
    print(f"Option surface for {row['date']}  (model={model}, spot={spot:.2f}, "
          f"r={row['risk_free_rate']:.4f}, q={row['dividend_rate']:.4f})")
    print(f"  v0={v0:.4f}  kappa={kappa:.4f}  theta={theta:.4f}  "
          f"eta={eta:.4f}  rho={rho:.4f}  feller={row['feller']:.4f}  "
          f"rmse={row['rmse']:.4f}  iv_rmse={row['iv_rmse']:.8f}")
    if model == "bates":
        print(f"  lambda={float(row['lambda_']):.4f}  nu={float(row['nu']):.4f}  "
              f"delta={float(row['delta']):.4f}")
    print()

    records = []
    for days in MATURITIES_DAYS:
        maturity_date = calculation_date + ql.Period(days, ql.Days)
        for m in MONEYNESS:
            strike = m * spot
            for w in ('call', 'put'):
                iv = model_implied_vol(strike, maturity_date, spot, engine, bsm_process, w=w)
                price = model_price(strike, maturity_date, spot, w, engine)
                records.append({
                    's_ref':spot,
                    'strike': round(strike, 4),
                    'maturity_days': days,
                    'moneyness': m,
                    'w': w,
                    'implied_vol': iv,
                    'price': price,
                })

    surface = pd.DataFrame(records)
    if save:
        long_path = out / r"example_surface.csv"
        surface.to_csv(long_path, index=False)
    high_move = row['high_move']
    if not isinstance(high_move, (bool, np.bool_)):
        high_move = str(high_move).strip().lower() == 'true'
    params = {"kappa": kappa, "theta": theta, "rho": rho, "eta": eta, "v0": v0}
    if model == "bates":
        # Carry the jump triple so a consumer (e.g. plot_surfaces) can rebuild a Bates engine from
        # day_results['params']. (smiles.py rebuilds its engine straight from calibrations.csv now.)
        params.update(lambda_=float(row['lambda_']), nu=float(row['nu']), delta=float(row['delta']))
    day_results = {
        "params": params,
        "model": model,
        "spot": spot,
        "date": date,
        "market": {
            "risk_free_rate": float(row['risk_free_rate']),
            "dividend_rate": float(row['dividend_rate']),
        },
        "fit": {
            "iv_rmse": float(row['iv_rmse']),
            "rmse": float(row['rmse']),
            "feller": float(row['feller']),
            "n_helpers": int(row['n_helpers']),
            "n_maturities": int(row['n_maturities']),
            "n_strikes": int(row['n_strikes']),
            "total_volume": int(row['total_volume']),
            "spot_range_pct": float(row['spot_range_pct']),
            "high_move": bool(high_move),
        },
    }
    # if SAVE:
    #     with open(OUT/'day_results.pkl', 'wb') as file:
    #         pickle.dump(day_results, file)
    
    return (surface, day_results)
