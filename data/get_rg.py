import pandas as pd
from pathlib import Path

SRC = Path(__file__).parent.resolve()

# ------- Interest rate retrieval -------
f_us12 = SRC / "market" / "historical_USGG12M.csv"
us12 = pd.read_csv(f_us12).iloc[4:,:2].reset_index(drop=True)
us12.columns = us12.iloc[0]
us12 = us12.iloc[1:,:]
us12 = us12.rename(columns={'Dates':'date','LAST_PRICE':'risk_free_rate'})
us12['risk_free_rate'] = pd.to_numeric(us12['risk_free_rate'],errors='coerce')
us12['date'] = pd.to_datetime(us12['date'],format="mixed",errors='coerce')
us12 = us12.set_index('date').dropna().squeeze().sort_index(ascending=False)/100

# ------- Implied volatility and dividend rate retrieval -------
f_spx_ivols_g = SRC / "market" / "historical_SPX_ivols.csv"
rg = pd.read_csv(f_spx_ivols_g).iloc[3:,:].reset_index(drop=True)
rg.columns = rg.iloc[0,:].values
rg = rg.iloc[1:]
rg = rg.rename(columns={'Dates':'date','PX_LAST':'spot_price','EQY_DVD_YLD_12M':'dividend_rate'})
rg['date'] = pd.to_datetime(rg['date'],format='mixed',errors='coerce')
for col in rg.columns[1:]:
	rg[col] = pd.to_numeric(rg[col],errors='coerce')
rg = rg.dropna().set_index('date').sort_index(ascending=False)
vol_cols = rg.columns[2:].tolist()
vol_cols = [c[:c.find('_',0)-2]+'_vol' for c in vol_cols]
rg.columns = rg.columns[:2].tolist() + vol_cols
rg[vol_cols] = rg[vol_cols]/100
rg['dividend_rate'] = rg['dividend_rate']/100
rg['risk_free_rate'] = us12
rg = rg.dropna()