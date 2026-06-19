import numpy as np, pandas as pd
import gnu as G
from example_surface import make_surface
from pygnuplot import gnuplot
import sys

cal = pd.read_csv(G.SURFACES.parent/'calibrations'/'price'/'calibrations.csv')
cal = cal[cal['feller']>=0].copy().sort_values(by='rmse').reset_index(drop=True)
make_surface(target_date=cal['date'][1])
df,_ = G._load_data()
grid = G.grid_for(df,'call')
strikes = grid.index.to_numpy(float); mats = grid.columns.to_numpy(float)/365.0
Z = grid.to_numpy(float)
rows=[]
for i,K in enumerate(strikes):
    for j,T in enumerate(mats):
        z=Z[i,j]; rows.append(f'{K:.10g} {T:.10g} '+('nan' if np.isnan(z) else format(z,".10g")))
    rows.append('')
data='\n'.join(rows)

def render(setup, out):
    g=gnuplot.Gnuplot(log=False)
    g.cmd('set terminal pngcairo enhanced font "Times-Roman,18" size 750,600')
    g.cmd(f'set output "{out}"')
    g.cmd('set datafile missing "nan"'); g.cmd('unset key')
    g.cmd('set palette defined '+G._JET); g.cmd('unset colorbox')
    g.cmd('set pm3d depthorder'); g.cmd('set pm3d border lc rgb "black" lw 0.4')
    g.cmd('set style fill solid')
    for c in setup: g.cmd(c)
    g.cmd('set zrange [0:*]'); g.cmd('set xyplane at 0')
    g.__call__('$S << EOD\n'+data+'\nEOD')
    g.cmd('splot $S using 1:2:3 with pm3d')
    g.close()

setup = [
 'set tics font "Times-Roman,15"',
 'set xtics 200 offset 0,-0.4',
 'set ytics 0.4 offset 1.2,0',
 'set ztics offset -0.3,0',
 'set xlabel "strike (K)" font "Times-Roman,20" offset 0,-1.5',
 'set ylabel "maturity in years (T)" font "Times-Roman,20" offset 0,-1.2 rotate parallel',
 'set zlabel "price" font "Times-Roman,20" offset -3,0 rotate by 90',
 'set rmargin 10',
]
render(setup, "_check.png")
print("done")
