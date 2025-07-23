import os
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
import cftime

from utils import load_avg_data, dates_to_years
from constants import DATADIR_BASE, FIGDIR_BASE, TIMEBOUNDS, SPACETIMES, INFILE_BASE
from constants import VARNAMES, VARLABELS, ENSLIST, FORCELIST
from constants import UNITS, FORCE_UNITS, FORCE_OBSERVED
from constants import PLOTCOLORS, LINEWIDTHS
from constants import AXIS_FONTSIZE, LEGEND_FONTSIZE, TICKLABELS_FONTSIZE
from constants import TIMEAXIS_TICKS, TIMEAXIS_LABELS


# ----- START USER INPUTS -----

# y-limits
axis_lims = {
    "FSNT": [None, None],
    "TREFHT": [None, None],
    "CLDTOT": [None, None],
}

plot_timebounds = True
boundcolors = ["crimson", "orange"]

# plot legend
plot_legend = [True] + [False]*5
legend_loc = "lower right"

# ----- END USER INPUTS -----


plotcount = 0
for region, period in SPACETIMES:

    # only plotting "all" period
    if period != "all":
        continue

    casename = f"{region}-all"
    figdir = os.path.join(FIGDIR_BASE, f"figs-{casename}")
    datadir = os.path.join(DATADIR_BASE, f"data-{casename}")

    # make output directory
    outdir = os.path.join(figdir, "ts")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # load ensemble mean global time series data
    data_dict = load_avg_data(datadir, INFILE_BASE, VARNAMES, FORCELIST, ENSLIST)

    # get time for plotting
    dates = data_dict[VARNAMES[0]]["da_mean_list"][0].time.values
    time = dates_to_years(dates, from_monthly=True)

    for var_idx, var in enumerate(VARNAMES):

        fig, ax = plt.subplots(1, 1)
        artist_list = []

        # compute peak value and index
        da_mean_list = data_dict[var]["da_mean_list"]

        for force_idx, force in enumerate(FORCELIST):

            # don't display counterfactual
            if force == 0:
                continue

            vals = da_mean_list[force_idx].values

            if force == FORCE_OBSERVED:
                linestyle = "--"
            else:
                linestyle = "-"

            # plot line plot, save artist for legend
            artist, = ax.plot(time, vals, color=PLOTCOLORS[str(force)], linewidth=LINEWIDTHS[str(force)], linestyle=linestyle)
            artist_list.append(copy(artist))

        # plot zero-impact line
        ax.axhline(0.0, color="k", linestyle=":", linewidth=2)

        # format
        ax.set_xlabel("Date", fontsize=AXIS_FONTSIZE)
        ax.set_xticks(TIMEAXIS_TICKS, labels=TIMEAXIS_LABELS, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylabel(f"{VARLABELS[var]} impact ({UNITS[var]})", fontsize=AXIS_FONTSIZE)
        ax.set_xlim([np.amin(time), np.amax(time)])
        ax.set_ylim(axis_lims[var])
        if plot_legend[plotcount]:
            ax.legend(artist_list, [f"{force} {FORCE_UNITS}" for force in FORCELIST if force != 0], loc=legend_loc, fontsize=LEGEND_FONTSIZE, framealpha=1.0)
        ax.tick_params(axis="both", which="major", labelsize=TICKLABELS_FONTSIZE)

        if plot_timebounds:
            boundcount = 0
            calendar = dates[0].calendar
            for bound_name, bound_dict in TIMEBOUNDS.items():
                if bound_name == "all":
                    continue
                timebounds = bound_dict["timebounds"]
                yr_st, mo_st, da_st = timebounds[0].split("-")
                yr_fn, mo_fn, da_fn = timebounds[1].split("-")
                date_st = cftime.datetime(int(yr_st), int(mo_st), int(da_st), calendar=calendar)
                date_fn = cftime.datetime(int(yr_fn), int(mo_fn), int(da_fn), calendar=calendar)
                times_bnd = dates_to_years([date_st, date_fn])

                ax.axvline(times_bnd[0], color=boundcolors[boundcount], linestyle=":", linewidth=2)
                ax.axvline(times_bnd[1], color=boundcolors[boundcount], linestyle=":", linewidth=2)
                boundcount += 1

        plt.tight_layout()

        # save
        outfile = os.path.join(outdir, f"{var}-{region}-{period}-ts.png")
        print(f"Saving image to {outfile}")
        plt.savefig(outfile)
        plt.close(fig)

        plotcount += 1

print("Finished")
