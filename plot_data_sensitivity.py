import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from utils import load_avg_data, calc_pvals
from constants import DATADIR_BASE, FIGDIR_BASE, SPACETIMES, INFILE_BASE
from constants import VARNAMES, VARLABELS, ENSLIST, FORCELIST, TIMEBOUNDS, REGIONBOUNDS
from constants import FORCE_VAR, FORCE_OBSERVED, FORCE_UNITS
from constants import LEGEND_FONTSIZE, AXIS_FONTSIZE, TICKLABELS_FONTSIZE, TITLE_FONTSIZE


# ----- START USER INPUTS -----

ubound = 1.0
path_name = "surf-multi"
pathlist = ["SO2", "FSNT", "TREFHT"]

calc_ens_sens = True
ens_counts = [2, 5, 10, 15]
nrandom_ens = 50

calc_force_sens = True
forcelists_select = [
    [0, 5, 10, 15],
    [0, 1, 3, 5, 7, 10, 13, 15],
]

mc_evals = 1000000

plotcolors = ["#648FFF", "#FE6100", "#DC267F", "#FFB000"]
plot_legend = [True] + [False] * (len(SPACETIMES) - 1)
legend_loc = "upper left"
pbounds = [0.001, 0.01, 0.05, 0.1]

# ----- END USER INPUTS -----

rng = np.random.default_rng(seed=10)
norm_dist = norm
norm_dist.random_state = rng

# for convenience
minval = 1.0 / mc_evals
null_forces_full = [force for force in FORCELIST if force != FORCE_OBSERVED]
nnull_full = len(null_forces_full)

# temporarily remove forcing from variable list
varnames = [var for var in pathlist if var != FORCE_VAR]

assert all([var in VARNAMES for var in varnames]), "Invalid variable in varnames"

for region_idx, (region, period) in enumerate(SPACETIMES):

    casename = f"{region}-{period}"
    figdir = os.path.join(FIGDIR_BASE, f"figs-{casename}")
    datadir = os.path.join(DATADIR_BASE, f"data-{casename}")

    print(f"\n******* {casename} *******")

    timelabel   = TIMEBOUNDS[period]["label"]
    regionlabel = REGIONBOUNDS[region]["label"]

    # make output directory
    outdir = os.path.join(figdir, "data_sensitivity")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # ----- downselecting ensemble size -----

    if calc_ens_sens:

        print("Testing ensemble size...")
        fig, ax = plt.subplots(1, 1)
        ax.set_yscale("log")
        barwidth = 0.9 / len(ens_counts)
        for ens_count_idx, ens_count in enumerate(ens_counts):
            print(f"  Ensemble count: {ens_count}")

            # randomly select from full ensemble range
            if ens_count == len(ENSLIST):
                nrandom = 1
            else:
                nrandom = nrandom_ens
            for rand_idx in range(nrandom):
                print(f"    Sample {rand_idx+1:>3} / {nrandom}")

                enslist_rand = rng.permuted(ENSLIST)[:ens_count]

                # load global time series data
                # TODO: kind of slow to reload this every time, but everything is buried in a dict...
                data_dict = load_avg_data(datadir, INFILE_BASE, varnames, FORCELIST, enslist_rand)

                pvals = calc_pvals(
                    pathlist,
                    data_dict,
                    FORCELIST,
                    enslist_rand,
                    FORCE_VAR,
                    FORCE_OBSERVED,
                    null_forces_full,
                    mc_evals,
                    norm_dist,
                )

                if rand_idx == 0:
                    pvals_mean = pvals.copy()
                else:
                    pvals_mean += pvals

            pvals_mean /= nrandom

            # for thresholding to minval
            pvals_plot = [max(minval, pval) for pval in pvals_mean]

            offset = -0.45 + barwidth * (1 + 2 * ens_count_idx) / 2
            bars = ax.bar(
                np.arange(nnull_full) + offset,
                pvals_plot,
                width=barwidth,
                color=plotcolors[ens_count_idx],
                edgecolor="k"
            )

        ax.set_xlabel(f"{VARLABELS[FORCE_VAR]} impact ({FORCE_UNITS})", fontsize=AXIS_FONTSIZE)
        ax.set_ylabel("p-value", fontsize=AXIS_FONTSIZE)
        ax.set_xticks(np.arange(nnull_full), [f"{force}" for force in null_forces_full])
        ax.set_ylim([minval, ubound])
        ax.tick_params(axis="both", which="major", labelsize=TICKLABELS_FONTSIZE)
        ax.tick_params(axis="y", which="minor", left=True)
        ax.set_title(f"{regionlabel} {timelabel}", fontsize=TITLE_FONTSIZE)

        if plot_legend[region_idx]:
            legend_labels = [r"N$_e$ = " + str(ens_count) for ens_count in ens_counts]
            ax.legend(legend_labels, fontsize=LEGEND_FONTSIZE, loc=legend_loc)

        plt.tight_layout()
        outfile = os.path.join(outdir, f"{region}-{period}-pval-enscount.png")
        print(f"Saving image to {outfile}")
        plt.savefig(outfile)
        plt.close(fig)

    # ----- end ensemble size downselect -----

    # ----- downselect forcing levels -----

    if calc_force_sens:

        print("Testing forcing downselect...")
        fig, ax = plt.subplots(1, 1)
        ax.set_yscale("log")
        barwidth = 0.9 / len(forcelists_select)
        for forcelist_idx, forcelist in enumerate(forcelists_select):

            print(f"  forcelist: {forcelist}")

            null_forces = [force for force in forcelist if force != FORCE_OBSERVED]

            data_dict = load_avg_data(datadir, INFILE_BASE, varnames, forcelist, ENSLIST)

            pvals = calc_pvals(
                pathlist,
                data_dict,
                forcelist,
                ENSLIST,
                FORCE_VAR,
                FORCE_OBSERVED,
                null_forces_full,
                mc_evals,
                norm_dist,
                experimental=True
            )

            # for thresholding to minval
            pvals_plot = [max(minval, pval) for pval in pvals]

            offset = -0.45 + barwidth * (1 + 2 * forcelist_idx) / 2
            bars = ax.bar(
                np.arange(nnull_full) + offset,
                pvals_plot,
                width=barwidth,
                color=plotcolors[forcelist_idx],
                edgecolor="k"
            )

        ax.set_xlabel(f"{VARLABELS[FORCE_VAR]} impact ({FORCE_UNITS})", fontsize=AXIS_FONTSIZE)
        ax.set_ylabel("p-value", fontsize=AXIS_FONTSIZE)
        ax.set_xticks(np.arange(nnull_full), [f"{force}" for force in null_forces_full])
        ax.set_ylim([minval, ubound])
        ax.tick_params(axis="both", which="major", labelsize=TICKLABELS_FONTSIZE)
        ax.tick_params(axis="y", which="minor", left=True)
        ax.set_title(f"{regionlabel} {timelabel}", fontsize=TITLE_FONTSIZE)

        if plot_legend[region_idx]:

            legend_labels = [
                r"f $\in$ {" + ", ".join([str(force) for force in forcelist]) + "} Tg" \
                    for forcelist in forcelists_select
            ]
            ax.legend(legend_labels, fontsize=LEGEND_FONTSIZE, loc=legend_loc)

        plt.tight_layout()
        outfile = os.path.join(outdir, f"{region}-{period}-pval-forceselect.png")
        print(f"Saving image to {outfile}")
        plt.savefig(outfile)
        plt.close(fig)