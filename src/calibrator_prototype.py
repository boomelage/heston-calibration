import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from quantlib_pricers import vanilla_pricer
vanp = vanilla_pricer()
pd.options.display.float_format = '{:.5f}'.format

SRC = Path(__file__).parent.resolve()
DATA = SRC.parent / "data"
CALIBRATIONS = SRC.parent / "data" / "options" / "calibrations"
TESTS = SRC.parent / "data" / "options" / "calibration_tests"

if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

from calibrate_heston import calibrate_heston

if str(DATA) not in sys.path:
    sys.path.insert(0,str(DATA))

from get_rg import rg # pyright: ignore[reportMissingImports]

# `rg` is sorted descending (newest first); `asof` needs an ascending index. Sort once here
# instead of per file. `asof(date)` returns the last value on/before `date` regardless of `rg`'s
# ordering, or NaN if `date` precedes all rates.
rg_asc = rg.sort_index()

def calibrateby_spot(filepath):
    df = pd.read_csv(filepath)
    df = df[df['trade_iv']>0]
    df['quote_datetime'] = pd.to_datetime(df['quote_datetime'])
    date = df['quote_datetime'].copy().dt.floor('D').unique()[0]
    r = rg_asc['risk_free_rate'].asof(date)
    g = rg_asc['dividend_rate'].asof(date)
    if pd.isna(r) or pd.isna(g):
        print(f"skipping {filepath}: no rate on/before {date}")
        return
    df['spot_price'] = (2*df['spot_price']).round()//2
    S = df['spot_price'].copy().drop_duplicates().sort_values().reset_index(drop=True)
    bys = df.groupby('spot_price')

    # -------- Volumes filter -------
    # volumes = pd.Series(np.tile(np.nan,len(S)),index=S)
    # for s in S:
    #     volumes[s] = np.sum(bys.get_group(s)['trade_size'])
    # volumes = volumes.sort_values(ascending=False).iloc[:20]
    # df = df[df['spot_price'].isin(volumes.index)].reset_index(drop=True)

    sparams = pd.DataFrame(np.tile(np.nan,(max(len(S),1),6)),index=S,columns = ['theta','kappa','rho','eta','v0','feller'])

    max_nt = 7   # maturities per spot, ranked by traded volume
    max_nk = 7   # strikes kept per wing, nearest the money
    test_frames = []   # repriced snapshots, accumulated across spots and written once

    for s in S:
        spot_data = df[df['spot_price']==s]
        total_volume = sum(spot_data['trade_size'])
        byt = spot_data.groupby('days_to_maturity')

        # top maturities for this spot, ranked by traded volume
        vol_by_t = byt['trade_size'].sum().sort_values(ascending=False)
        T = np.sort(vol_by_t.index[:max_nt]).tolist()

        selected = []
        for t in T:
            dft = byt.get_group(t)   # (A) strikes from the CURRENT maturity
            cK = np.sort(dft.loc[dft['w']=='call','strike_price'].unique())
            pK = np.sort(dft.loc[dft['w']=='put', 'strike_price'].unique())
            if len(cK)>1 and len(pK)>1:
                # (C) cap each wing with min(); nearest-money: highest OTM puts, lowest OTM calls
                keep = list(pK[-min(len(pK),max_nk):]) + list(cK[:min(len(cK),max_nk)])
                selected.append(dft[dft['strike_price'].isin(keep)])   # (B) this spot's rows only

        if not selected:
            continue
        snap = (pd.concat(selected,ignore_index=True)
                  .drop_duplicates(subset=['strike_price','days_to_maturity'],keep='first')
                  .dropna()
                  .reset_index(drop=True))
        surf = snap.pivot_table(index='strike_price',columns='days_to_maturity',
                                values='trade_iv',aggfunc='last')   # multi-maturity surface
        contracts_count = int(surf.count().sum())
        if contracts_count<5:
            continue

        lastquote_time = np.sort(snap['quote_datetime'].unique())[-1]
        res = calibrate_heston(surf,s,r,g)   # ONE calibration per spot
        print(pd.Series(res))
        # Pricing/feller params go into sparams' fixed columns; the engine's diagnostics
        # (rmse, n_helpers, accepted) are written scalar-wise. Rejected fits return null
        # params -> the row is dropped by sparams.dropna() below (not written).
        params = pd.Series({k: res[k] for k in ['theta','kappa','rho','eta','v0','feller']})
        sparams.loc[s,params.index] = params.values
        sparams.loc[s,'rmse'] = res['rmse']
        sparams.loc[s,'n_helpers'] = res['n_helpers']
        sparams.loc[s,'accepted'] = res['accepted']
        sparams.loc[s,'calculation_date'] = lastquote_time
        sparams.loc[s,'contracts_count'] = contracts_count
        sparams.loc[s,'total_volume'] = total_volume
        sparams.loc[s,'risk_free_rate'] = r
        sparams.loc[s,'dividend_rate'] = g

        # Only repriced *accepted* fits feed the tests file, so calibrations and
        # calibration_tests describe the same set of buckets (no desync). Rejected fits
        # return null params -- repricing under them is meaningless (and was dropped anyway).
        if res['accepted']:
            repriced = snap.copy()
            repriced[params.index] = np.tile(params.values,(repriced.shape[0],1))
            repriced['risk_free_rate'] = r
            repriced['dividend_rate'] = g
            repriced = repriced.rename(columns={'trade_iv':'volatility'})
            try:
                repriced['black_scholes'] = vanp.df_numpy_black_scholes(repriced)
            except Exception:
                repriced['black_scholes'] = np.nan
            try:
                repriced['heston'] = vanp.df_heston_price(repriced)
            except Exception:
                repriced['heston'] = np.nan
            test_frames.append(repriced)

    # `calibrated` (accepted buckets) and `test_frames` (repriced accepted buckets) describe the
    # same set, so write/skip both together. Writing only one -- the old bug -- left a stale
    # calibrations file beside an emptied tests file whenever a day produced no accepted fit.
    calibrated = sparams.dropna()
    cal_path = filepath.replace('otm','calibrations')
    test_path = filepath.replace('otm','calibration_tests')

    if calibrated.empty:
        # Nothing accepted this day: clear any stale outputs so the pair never desyncs.
        for stale in (cal_path, test_path):
            if os.path.exists(stale):
                os.remove(stale)
        print(f"no accepted fit for {date}: cleared stale outputs")
        return

    CALIBRATIONS.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(cal_path)
    TESTS.mkdir(parents=True, exist_ok=True)
    pd.concat(test_frames,ignore_index=True).dropna().to_csv(test_path,index=False)


OTM = Path(__file__).parent.parent / "data" / "options" / "otm"
files = [f for f in os.listdir(OTM) if f.endswith('.csv')]
files = pd.Series([os.path.join(OTM,f) for f in files]).sort_values(ascending=False).reset_index(drop=True)
for f in files: calibrateby_spot(f)


# max_jobs = os.cpu_count() // 2
# max_jobs = max(1,max_jobs)
# Parallel(n_jobs=max_jobs)(delayed(calibrateby_spot)(f) for f in files)