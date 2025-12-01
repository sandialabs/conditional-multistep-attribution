import os
import itertools

import cftime
import numpy as np
from matplotlib.pyplot import cm
import matplotlib as mpl
mpl.rc("font", family="serif", size="10")
mpl.rc("figure", facecolor="w", dpi=300)
mpl.rc("text", usetex=False)

from utils import dates_to_years


# ----- START USER SETTINGS -----

DATADIR_BASE = "./data"
FIGDIR_BASE = "./figs"

# data frequency
FREQ = "monthly"

# data locations
INFILE_BASE = "{varname}_{force}Tg_ens{ens}_avg_" + f"{FREQ}.nc"

REGIONS = [
    "glob",
    # "nh",
    # "na",
]

PERIODS = [
    "all",
    # "jja",
    # "jfm"
]

# variable names
VARNAMES = [
    "FSNT",
    "TREFHT",
]
VARLABELS = {
    "FSNT": "FSNT",
    "TREFHT": "TREFHT",
    "SO2": r"SO$_2$",
}
UNITS = {
    "SO2":    "Tg",
    "FSNT":   "W/m$^2$",
    "TREFHT": "K",
}

FORCE_VAR = "SO2"
FORCE_UNITS = UNITS[FORCE_VAR]

# for plotting posteriors
PATHNAMES_PLOT = {
    "surf-single": "Single-step",
    "surf-inter": "Intermediate",
    "surf-multi": "Multi-step",
}

# ensemble specifiers
ENSLIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
FORCELIST = [0, 1, 3, 5, 7, 10, 13, 15]
FORCE_OBSERVED = 10
NFORCE_SAMP = 301

# line plot parameters
colormap = cm.viridis_r
colors = colormap(np.linspace(0.1, 1, len(FORCELIST)))
linewidths = np.linspace(1.0, 3.5, len(FORCELIST))

# fontsize parameters
TITLE_FONTSIZE = 20
AXIS_FONTSIZE = 18
TICKLABELS_FONTSIZE = 16
LEGEND_FONTSIZE = 16

# time axis label parameters
TIMEAXIS_LABELS = [
    "JUL 1991",
    "JAN 1992",
    "JUL 1992",
    "JAN 1993",
    "JUL 1993",
    "JAN 1994",
]
# NOTE: need to offset by a month for monthly dates
timeaxis_ticks = [
    "1991-07-15",
    "1992-01-15",
    "1992-07-15",
    "1993-01-15",
    "1993-07-15",
    "1994-01-15",
]

LATEX = False

# ----- END USER SETTINGS -----

PLOTCOLORS = {}
LINEWIDTHS = {}
for force_idx, force in enumerate(FORCELIST):
    PLOTCOLORS[str(force)] = colors[force_idx, :]
    LINEWIDTHS[str(force)] = linewidths[force_idx]

SPACETIMES = list(itertools.product(REGIONS, PERIODS))

# timebounds
TIMEBOUNDS = {
    "all": {
        "timebounds": ["1991-06-02", "1994-06-01"],
        "label": "Three-year",
    },
    "jja": {
        "timebounds": ["1992-06-02", "1992-09-02"],
        "label": "1992 JJA",
    },
    "jfm": {
        "timebounds": ["1992-01-02", "1992-04-02"],
        "label": "1992 JFM",
    },
}

# lat/lonbounds
REGIONBOUNDS = {
    "glob": {
        "label": "Global",
        "latbounds": [-66, 66],
        "lonbounds": [0, 360],
    },
    "nh": {
        "label": "Northern Hemisphere",
        "latbounds": [0, 66],
        "lonbounds": [0, 360],
    },
    "na": {
        "label": "North America",
        "latbounds": [25, 66],
        "lonbounds": [190, 300],
    },
}

if not os.path.isdir(FIGDIR_BASE):
    os.mkdir(FIGDIR_BASE)

for region in REGIONS:
    for period in PERIODS:
        casename = f"{region}-{period}"
        datadir = os.path.join(DATADIR_BASE, f"data-{casename}")
        if not os.path.isdir(datadir):
            os.mkdir(datadir)
        figdir  = os.path.join(FIGDIR_BASE, f"figs-{casename}")
        if not os.path.isdir(figdir):
            os.mkdir(figdir)

timeaxis_vals = [[int(val) for val in tick.split("-")] for tick in timeaxis_ticks]
timeaxis_datetimes = [cftime.datetime(val[0], val[1], val[2], calendar="noleap", has_year_zero=True) for val in timeaxis_vals]
TIMEAXIS_TICKS = dates_to_years(timeaxis_datetimes)
