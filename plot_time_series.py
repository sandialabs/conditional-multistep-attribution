import os
from copy import copy

import numpy as np
import matplotlib.pyplot as plt

from utils import load_glob_data, calc_peak_values, dates_to_years
from constants import DATADIR, INFILE_BASE, FIGDIR
from constants import VARNAMES, ENSLIST, FORCELIST, EXPECTED_DIRECTIONS
from constants import UNITS, FORCE_UNITS
from constants import PLOTCOLORS, LINEWIDTHS
from constants import AXIS_FONTSIZE, LEGEND_FONTSIZE, TICKLABELS_FONTSIZE
from constants import TIMEAXIS_TICKS, TIMEAXIS_LABELS


# ----- START USER INPUTS -----

# y-limits
axis_lims = {
    "FLNT": [-3.5, 0.5],
    "FSDS": [-5.0, 1.0],
    "TREFHT": [-0.5, 0.1],
    "T050": [-0.5, 3.0],
}

# plot legend, according to order in VARNAMES
plot_legend = [True, True, False, False]

# ----- END USER INPUTS -----

# make output directory
outdir = os.path.join(FIGDIR, "ts_peaks_glob")
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# load ensemble mean global time series data
data_dict = load_glob_data(DATADIR, INFILE_BASE, VARNAMES, FORCELIST, ENSLIST)

# get time for plotting
dates = data_dict[VARNAMES[0]]["da_mean_list"][0].time.values
time = dates_to_years(dates, from_monthly=True)

for var_idx, var in enumerate(VARNAMES):

    fig, ax = plt.subplots(1, 1)
    artist_list = []

    # compute peak value and index
    da_mean_list = data_dict[var]["da_mean_list"]
    _, peak_idx_list = calc_peak_values(da_mean_list, EXPECTED_DIRECTIONS[var])

    for force_idx, force in enumerate(FORCELIST):

        # don't display counterfactual
        if force == 0:
            continue

        # extract values and peak
        vals = da_mean_list[force_idx].values
        peak_idx = peak_idx_list[force_idx]

        # plot line plot, save artist for legend
        artist, = ax.plot(time, vals, color=PLOTCOLORS[str(force)], linewidth=LINEWIDTHS[str(force)])
        artist_list.append(copy(artist))

        # plot peak point
        ax.plot(time[peak_idx], vals[peak_idx], color=PLOTCOLORS[str(force)], marker="o", markersize=8)

    # plot zero-impact line
    ax.plot(time, np.zeros(time.size), color="k", linestyle="--")

    # format
    ax.set_xlabel("Date", fontsize=AXIS_FONTSIZE)
    ax.set_xticks(TIMEAXIS_TICKS, labels=TIMEAXIS_LABELS, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel(f"{var} impact ({UNITS[var]})", fontsize=AXIS_FONTSIZE)
    ax.set_xlim([np.amin(time), np.amax(time)])
    ax.set_ylim(axis_lims[var])
    if plot_legend[var_idx]:
        ax.legend(artist_list, [f"{force} {FORCE_UNITS}" for force in FORCELIST], loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=1.0)
    ax.tick_params(axis="both", which="major", labelsize=TICKLABELS_FONTSIZE)
    plt.tight_layout()

    # save
    outfile = os.path.join(outdir, f"{var}_glob_ts.png")
    print(f"Saving image to {outfile}")
    plt.savefig(outfile)
    plt.close(fig)

print("Finished")