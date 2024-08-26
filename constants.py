import os

import cftime
import numpy as np
from matplotlib.pyplot import cm
import matplotlib as mpl
mpl.rc("font", family="serif", size="10")
mpl.rc("figure", facecolor="w")
mpl.rc("text", usetex=False)

from utils import dates_to_years


# ----- CONSTANT PARAMETERS -----

# data frequency
FREQ = "monthly"

# data locations
DATADIR = "./data"
INFILE_BASE = "{varname}_{force}Tg_ens{ens}_glob_" + f"{FREQ}.nc"
FIGDIR = "./figs"

if not os.path.isdir(FIGDIR):
    os.mkdir(FIGDIR)

# variable names
VARNAMES = [
    "FLNT",
    "FSDS",
    "T050",
    "TREFHT",
]
UNITS = {
    "SO2":    "Tg",
    "FSDS":   "W/m$^2$",
    "FLNT":   "W/m$^2$",
    "TREFHT": "K",
    "T050":   "K",
}

FORCE_VAR = "SO2"
FORCE_UNITS = UNITS[FORCE_VAR]

# for plotting posteriors
PATHNAMES_PLOT = {
    "surf-single-f-direct": "Surface single-step",
    "surf-multi-f-direct": "Surface multi-step",
    "strat-single-f-direct": "Strat. single-step",
    "strat-multi-f-direct": "Strat. multi-step",
    "combo-single-f-direct": "Combined single-step",
    "combo-multi-f-direct": "Combined multi-step",
}

# ensemble specifiers
ENSLIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
FORCELIST = [0, 1, 3, 5, 7, 10, 13, 15]
FORCE_OBSERVED = 10

# expected sign of peak deviation
EXPECTED_DIRECTIONS = {
    "FLNT": -1,
    "FSDS": -1,
    "T050": 1,
    "TREFHT": -1,
}

# line plot parameters
colormap = cm.viridis_r
colors = colormap(np.linspace(0, 1, len(FORCELIST)))
linewidths = np.linspace(0.5, 3, len(FORCELIST))
PLOTCOLORS = {}
LINEWIDTHS = {}
for force_idx, force in enumerate(FORCELIST):
    PLOTCOLORS[str(force)] = colors[force_idx, :]
    LINEWIDTHS[str(force)] = linewidths[force_idx]

# fontsize parameters
TITLE_FONTSIZE = 18
AXIS_FONTSIZE = 16
TICKLABELS_FONTSIZE = 14
LEGEND_FONTSIZE = 14

# time axis label parameters
TIMEAXIS_LABELS = [
    "JUL 1991",
    "OCT 1991",
    "JAN 1992",
    "APR 1992",
    "JUL 1992",
    "OCT 1992",
    "JAN 1993",
    "APR 1993",
]
# NOTE: need to offset by a month for monthly dates
timeaxis_ticks = [
    "1991-08-01",
    "1991-11-01",
    "1992-02-01",
    "1992-05-01",
    "1992-08-01",
    "1992-11-01",
    "1993-02-01",
    "1993-05-01",
]
timeaxis_vals = [[int(val) for val in tick.split("-")] for tick in timeaxis_ticks]
timeaxis_datetimes = [cftime.datetime(val[0], val[1], val[2], calendar="noleap", has_year_zero=True) for val in timeaxis_vals]
from_monthly = True if (FREQ == "monthly") else False
TIMEAXIS_TICKS = dates_to_years(timeaxis_datetimes, from_monthly=from_monthly)
