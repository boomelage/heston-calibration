"""Central configuration for the downstream figure/table scripts under ``src/results/``.

Every tunable knob for the figure generators lives here so a surface/smile can be
re-shaped from one place instead of editing the individual scripts. The two master
switches are ``MODEL`` (picks the QuantLib engine -- Heston vs Bates -- AND the
``results/<model>/`` output tree) and ``OBJECTIVE`` (picks which calibration run to
read, ``results/<model>/calibrations/<objective>/``). Everything else is per-script
grid/plot parameters, grouped by the script that consumes it.

Imported by ``surfaces/make_surface.py``, ``smiles/smiles.py`` and
``surfaces/plot_surfaces.py`` (each adds ``src/results`` to ``sys.path`` then
``import _results_config``).
"""
import numpy as np

# ------- shared across all results scripts

# MODEL picks the QuantLib engine (Heston vs Bates) AND the results/<model>/ output tree.
# OBJECTIVE picks which calibration run to read (results/<model>/calibrations/<objective>/).
# This module is the single source for both -- the other scripts import them from here.
MODEL = "bates"      # 'heston' or 'bates'
OBJECTIVE = "vol"     # 'vol' or 'price'

# Matplotlib styling shared by smiles.py and plot_surfaces.py. Computer Modern serif to match the
# LaTeX document; cmr10 lacks U+2212 so unicode_minus is disabled to avoid missing-glyph warnings.
PLOT_RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['cmr10', 'Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,   # tick labels in Computer Modern too
    'axes.unicode_minus': False,           # cmr10 lacks U+2212; avoids missing-glyph warnings
    'font.size': 8,
}

# Placeholder Black vol seeded into BlackConstantVol for the price->IV inversion. impliedVolatility
# solves for the vol that reprices the model NPV, so this value is ignored; it only initialises the
# term structure. Used by make_surface.make_surface and smiles._day_engine.
INVERSION_PLACEHOLDER_VOL = 0.20


# ------- `surfaces/make_surface.py` parameters

# Moneyness grid (strike = m * spot) the model surface is sampled on. K/S around the money.
MONEYNESS = np.round(np.arange(0.75, 1.4, 0.005), 4).tolist()   # 0.75 .. 1.245

# Maturity grid in calendar days the model surface is sampled on (>= MIN_DTM up to ~MAX_DTM).
MATURITIES_DAYS = np.arange(start=60, stop=750, step=7).tolist()


# ------- `smiles/smiles.py` parameters
#
# smiles.py reads the calibrated params from results/<model>/calibrations/<objective>/calibrations.csv
# (the model smile lines) and the market scatter from that day's calibration_tests/ file (the exact
# contracts the day was fit on, with their market IV). It does not read the raw CBOE trades. The full
# calibrated contract set is plotted; the two knobs below only thin it for readability.

# Number of maturities drawn per figure. The plotted set always includes the lowest and highest
# available maturity; the remaining NT-2 are spaced as equally as possible. Set NT >= the number of
# available maturities, or NT = None, to draw them all.
NT = 6

# Maturity window (calendar days) the displayed smiles are clipped to before the NT sparse pick.
# Either bound may be None to disable it: TMIN=None drops the lower bound, TMAX=None the upper,
# both None draws every available maturity.
TMIN, TMAX = 30, 750

# Per-row maturity key: True draws a legend, False draws a colorbar.
USE_LEGEND = True

# Fallback moneyness window for the model lines / x-axis (S/K calls, K/S puts), used only when a day
# has no calibration_tests file to frame on. With the scatter present each wing is framed to that
# day's calibrated moneyness span instead.
XLO, XHI = 0.8, 1.15

# Market-scatter thinning. Keep a sparse subset spaced ~MKTMONSTEP apart in moneyness (percentage
# terms). 0.05 => ~5% gaps. Set to 0 or None to disable thinning (draw every calibrated point).
MKTMONSTEP = 0.05

# Moneyness step for the model smile lines (put_grid/call_grid resolution).
SMILE_M_STEP = 0.005

# Figure size (inches) for the two-panel smile figure, and the maturity colormap name. Each displayed
# maturity is colored by its RANK (not its day-count value), so a qualitative colormap with distinct
# categorical hues -- e.g. "tab10" (10 colors), "tab20" (20), "Set1", "Dark2" -- keeps adjacent
# maturities easy to tell apart. A continuous map ("jet", "viridis") still works: it is sampled at
# evenly spaced points by rank, but its neighboring hues are inherently closer.
SMILE_FIGSIZE = (8, 2.7)
SMILE_CMAP = "tab10"


# ------- `surfaces/plot_surfaces.py` parameters

# 3D view angle (elevation, azimuth) for the price-surface renders.
SURFACE_ELEV, SURFACE_AZIM = 25, -60

# Figure size (inches) for each 3D surface render.
SURFACE_FIGSIZE = (5.0, 4.0)


# ------- `tables/objective_comparison.py` parameters
