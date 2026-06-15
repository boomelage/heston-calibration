# Heston (1993) stochastic volatility parameter calibration

This repository has quite convoluted logic at the moment and relies a bit too much on storing various files as csvs which makes the workflow less intuitive

**How to use:**

1. Ensure the correct format of CBOE trades data is present in [`data/options/raw`](data/options/raw) (e.g., [`UnderlyingOptionsTradesCalcs_2024-10-07.csv`](data/options/raw/UnderlyingOptionsTradesCalcs_2024-10-07.csv))
2. Ensure you have the same format of dividend and interest rates as present in [`data/market/`](data/market/). Namely, [`historical_SPX_ivols.csv`](data/market/historical_SPX_ivols.csv) and [`historical_USGG12M.csv`](data/market/historical_USGG12M.csv). These two file names are hard-coded into the logic of [`get_rg.py`](data/get_rg.py) so the filenames must match.
3. Run [`extract_otms.py`](data/extract_otms.py) to clean the data and extract only out-of-the-money options
4. Run [`calibrator_prototype.py`](src/calibrator_prototype.py) to attempt calibration on various spots by its own logic and the calibration engine [`calibrate_heston.py`](src/calibrate_heston.py)

As you can see the whole setup is quite dependent on a rather convoluted file structure with very specific data formats and the way the data is called in [`calibrator_prototype.py`](src/calibrator_prototype.py) is through hardcoded calls to dataframe column names.
