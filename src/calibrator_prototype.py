import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
# from heston_model_calibration import calibrate_heston
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

def calibrateby_spot(filepath):
    df = pd.read_csv(filepath)
    df = df[df['trade_iv']>0]
    df['quote_datetime'] = pd.to_datetime(df['quote_datetime'])
    date = df['quote_datetime'].copy().dt.floor('D').unique()[0]
    r = rg[rg.index<=date]['risk_free_rate'].iloc[-1]
    g = rg[rg.index<=date]['dividend_rate'].iloc[-1]
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
    
    for s in S:
        data = df[df['spot_price']==s]
        total_volume = sum(data['trade_size'])
        T = data['days_to_maturity'].drop_duplicates()
        byt = data.groupby('days_to_maturity')
        cals = []
        new_T = pd.Series(index=T)

        for t in T:
            dft = byt.get_group(t)
            t_volume = sum(dft['trade_size'])
            new_T[t] = t_volume
        new_T = new_T.sort_values(ascending=False)
        max_nt = 7
        tloc = min(len(new_T),max_nt)
        T = np.sort(np.unique(new_T.iloc[:tloc].index)).tolist()

        if len(T)>0:
            for t in T:
                cK = np.sort(dft[dft['w']=='call']['strike_price'].unique()).tolist()
                pK = np.sort(dft[dft['w']=='put']['strike_price'].unique()).tolist()
                ncK = len(cK)
                npK = len(pK)
                max_nk = 7
                if ncK>1 and npK>1:
                    K = pK[:max(npK,max_nk)] + cK[:max(ncK,max_nk)]
                    ct = df[((df['days_to_maturity']==t)&(df['strike_price'].isin(K)))].sort_values('trade_iv',ascending=False)
                    vol_count = len(ct['trade_iv'].copy().dropna().values)
                    if vol_count>=5:
                        cals.append(ct)

        if len(cals)>0:
            for cal in cals:
                data = cal.drop_duplicates(subset=['strike_price','days_to_maturity'],keep='first').copy().dropna().reset_index(drop=True)
                surf = data.pivot_table(index='strike_price',columns='days_to_maturity',values='trade_iv',aggfunc='last')
                contracts_count = sum(surf.count())
                if contracts_count>=5:
                    lastquote_time = np.sort(data['quote_datetime'].unique())[-1]
                    parameters = pd.Series(calibrate_heston(surf,s,r,g))
                    print(parameters)
                    sparams.loc[s,parameters.index] = parameters.values
                    sparams.loc[s,'calculation_date'] = lastquote_time
                    sparams.loc[s,'contracts_count'] = contracts_count
                    sparams.loc[s,'total_volume'] = total_volume
                    sparams.loc[s,'risk_free_rate'] = r
                    sparams.loc[s,'dividend_rate'] = g
                    if os.path.exists(CALIBRATIONS)==False:
                        os.mkdir(CALIBRATIONS)                   
                    sparams.dropna().to_csv(filepath.replace('otm','calibrations'))

                    data[parameters.index] = np.tile(parameters.values,(data.shape[0],1))
                    data['risk_free_rate'] = r
                    data['dividend_rate'] = g
                    data = data.rename(columns = {'trade_iv':'volatility'})
                    try:
                        data['black_scholes'] = vanp.df_numpy_black_scholes(data)
                    except Exception:
                        data['black_scholes'] = np.nan
                    try:
                        data['heston'] = vanp.df_heston_price(data)
                    except Exception:
                        data['heston'] = np.nan
                    if os.path.exists(TESTS) == False:
                        os.mkdir(TESTS)
                    data.dropna().to_csv(filepath.replace('otm','calibration_tests'))


OTM = Path(__file__).parent.parent / "data" / "options" /"otm"
if not os.path.exists(OTM):
    os.mkdir(OTM)
files = [f for f in os.listdir(OTM) if f.endswith('.csv')]
files = pd.Series([os.path.join(OTM,f) for f in files]).sort_values(ascending=False).reset_index(drop=True)
for f in files: calibrateby_spot(f)


# max_jobs = os.cpu_count() // 2
# max_jobs = max(1,max_jobs)
# Parallel(n_jobs=max_jobs)(delayed(calibrateby_spot)(f) for f in files)