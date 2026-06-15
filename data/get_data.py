import os
import sys
from pathlib import Path
import pandas as pd

ROOT = Path().resolve()
SRC = ROOT/"src"
DATA = ROOT/"data"

if str(SRC) not in sys.path:
    sys.path.insert(0,str(SRC))

if str(DATA) not in sys.path:
    sys.path.insert(0,str(DATA))

CSVS = [DATA/f for f in os.listdir(DATA) if f.endswith('.csv')]

chain = {
    str(s)[str(s).rfind('_')+1:-4] : pd.read_csv(s) for s in CSVS
}
dates = list(chain.keys())