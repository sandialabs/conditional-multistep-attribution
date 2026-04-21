import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from utils import load_avg_data, calc_pvals
from constants import DATADIR_BASE, FIGDIR_BASE, SPACETIMES, INFILE_BASE, LATEX
from constants import VARNAMES, VARLABELS, ENSLIST, FORCELIST, TIMEBOUNDS, REGIONBOUNDS
from constants import FORCE_VAR, FORCE_OBSERVED, FORCE_UNITS
from constants import LEGEND_FONTSIZE, AXIS_FONTSIZE, TICKLABELS_FONTSIZE, TITLE_FONTSIZE


# ----- START USER INPUTS -----

ubound = 1.0
# path_name = "surf-single"
# pathlist = ["SO2", "TREFHT"]
path_name = "surf-multi"
pathlist = ["SO2", "FSNT", "TREFHT"]

calc_ens_sens = True
ens_counts = [2, 5]
nrandom_ens = 25

calc_force_sens = True
forcelists_select = [
    [0, 5, 10, 15],
]

mc_evals = 1000000
nforce_samp = 151

plotcolors = ["#648FFF", "#FE6100", "#DC267F", "#FFB000"]
plot_legend = [True] + [False] * (len(SPACETIMES) - 1)
legend_loc = "upper left"

pval_thresh = [0.05]

# ----- END USER INPUTS -----

rng = np.random.default_rng(seed=10)
norm_dist = norm
norm_dist.random_state = rng

# for convenience
minval = 1.0 / mc_evals
null_forces = list(np.linspace(
    min(FORCELIST),
    max(FORCELIST),
    nforce_samp,
))
obs_idx = null_forces.index(FORCE_OBSERVED)  # for excluding observation
null_forces = null_forces[:obs_idx] + null_forces[obs_idx+1:]
nnull_full = len(null_forces)

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
        artist_list = []
        for ens_count_idx, ens_count in enumerate(ens_counts):
            print(f"  Ensemble count: {ens_count}")

            # randomly select from full ensemble range
            if ens_count == len(ENSLIST):
                nrandom = 1
            else:
                nrandom = nrandom_ens
            for rand_idx in range(nrandom):

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
                    null_forces,
                    mc_evals,
                )

                if rand_idx == 0:
                    pvals_agg = pvals.copy()
                else:
                    # pvals_agg += pvals_agg
                    pvals_agg = np.maximum(pvals, pvals_agg)

            # pvals_agg /= nrandom

            if LATEX:
                # confidence intervals
                for idx, pthresh in enumerate(pval_thresh):
                    # upper and lower bound indices
                    thresh_bools = list(pvals_agg > pthresh)
                    thresh_bools_rev = thresh_bools[::-1]
                    cl_idx = thresh_bools.index(True)
                    cu_idx = len(thresh_bools) - 1 - thresh_bools_rev.index(True)

                    # actual forcing values
                    cl = null_forces[cl_idx]
                    cu = null_forces[cu_idx]

                    print(f"    CI {pthresh:>4.2f}: {cl:>4.1f} & {cu:>4.1f} \n")

            artist, = ax.plot(
                null_forces[:obs_idx],
                pvals_agg[:obs_idx],
                color=plotcolors[ens_count_idx],
                linewidth=2,
            )
            ax.plot(
                null_forces[obs_idx:],
                pvals_agg[obs_idx:],
                color=plotcolors[ens_count_idx],
                linewidth=2,
            )
            artist_list.append(artist)

        ax.set_xlabel(f"{VARLABELS[FORCE_VAR]} impact ({FORCE_UNITS})", fontsize=AXIS_FONTSIZE)
        ax.set_ylabel("p-value", fontsize=AXIS_FONTSIZE)
        ax.set_xticks(FORCELIST, [f"{force}" for force in FORCELIST])
        ax.set_ylim([minval, ubound])
        ax.tick_params(axis="both", which="major", labelsize=TICKLABELS_FONTSIZE)
        ax.tick_params(axis="y", which="minor", left=True)
        ax.set_title(f"{regionlabel} {timelabel}", fontsize=TITLE_FONTSIZE)

        if plot_legend[region_idx]:
            legend_labels = [r"N$_e$ = " + str(ens_count) for ens_count in ens_counts]
            ax.legend(artist_list, legend_labels, fontsize=LEGEND_FONTSIZE, loc=legend_loc)

        plt.tight_layout()
        outfile = os.path.join(outdir, f"{region}-{period}-{path_name}-pval-enscount.png")
        print(f"Saving image to {outfile}")
        plt.savefig(outfile)
        plt.close(fig)

    # ----- end ensemble size downselect -----

    # ----- downselect forcing levels -----

    if calc_force_sens:

        print("Testing forcing downselect...")
        fig, ax = plt.subplots(1, 1)
        ax.set_yscale("log")
        artist_list = []
        for forcelist_idx, forcelist in enumerate(forcelists_select):

            print(f"  forcelist: {forcelist}")

            data_dict = load_avg_data(datadir, INFILE_BASE, varnames, forcelist, ENSLIST)

            pvals = calc_pvals(
                pathlist,
                data_dict,
                forcelist,
                ENSLIST,
                FORCE_VAR,
                FORCE_OBSERVED,
                null_forces,
                mc_evals,
            )

            artist, = ax.plot(
                null_forces[:obs_idx],
                pvals[:obs_idx],
                color=plotcolors[forcelist_idx],
                linewidth=2,
            )
            ax.plot(
                null_forces[obs_idx:],
                pvals[obs_idx:],
                color=plotcolors[forcelist_idx],
                linewidth=2,
            )
            artist_list.append(artist)

            if LATEX:
                # confidence intervals
                for idx, pthresh in enumerate(pval_thresh):
                    # upper and lower bound indices
                    thresh_bools = list(pvals > pthresh)
                    thresh_bools_rev = thresh_bools[::-1]
                    cl_idx = thresh_bools.index(True)
                    cu_idx = len(thresh_bools) - 1 - thresh_bools_rev.index(True)

                    # actual forcing values
                    cl = null_forces[cl_idx]
                    cu = null_forces[cu_idx]

                    print(f"    CI {pthresh:>4.2f}: {cl:>4.1f} & {cu:>4.1f} \n")

        ax.set_xlabel(f"{VARLABELS[FORCE_VAR]} impact ({FORCE_UNITS})", fontsize=AXIS_FONTSIZE)
        ax.set_ylabel("p-value", fontsize=AXIS_FONTSIZE)
        ax.set_xticks(FORCELIST, [f"{force}" for force in FORCELIST])
        ax.set_ylim([minval, ubound])
        ax.tick_params(axis="both", which="major", labelsize=TICKLABELS_FONTSIZE)
        ax.tick_params(axis="y", which="minor", left=True)
        ax.set_title(f"{regionlabel} {timelabel}", fontsize=TITLE_FONTSIZE)

        if plot_legend[region_idx]:

            legend_labels = [
                r"f $\in$ {" + ", ".join([str(force) for force in forcelist]) + "} Tg" \
                    for forcelist in forcelists_select
            ]
            ax.legend(artist_list, legend_labels, fontsize=LEGEND_FONTSIZE, loc=legend_loc)

        plt.tight_layout()
        outfile = os.path.join(outdir, f"{region}-{period}-{path_name}-pval-forceselect.png")
        print(f"Saving image to {outfile}")
        plt.savefig(outfile)
        plt.close(fig)