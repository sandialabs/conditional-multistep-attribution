import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from utils import load_avg_data, calc_pvals
from constants import DATADIR_BASE, FIGDIR_BASE, SPACETIMES, INFILE_BASE
from constants import VARNAMES, VARLABELS, ENSLIST, FORCELIST, TIMEBOUNDS, REGIONBOUNDS
from constants import FORCE_VAR, FORCE_OBSERVED, FORCE_UNITS, NFORCE_SAMP
from constants import LEGEND_FONTSIZE, AXIS_FONTSIZE, TICKLABELS_FONTSIZE, TITLE_FONTSIZE, PATHNAMES_PLOT, LATEX


# ----- START USER INPUTS -----

ubound = 1.0
pathdicts = {
    "surf-single": ["SO2", "TREFHT"],
    "surf-multi": ["SO2", "FSNT", "TREFHT"],
}

mc_evals = 1000000

plotcolors = ["royalblue", "darkorange"]
# plot_legend = [False] * 4 + [True] + [False] * 4
plot_legend = [False] * 20
legend_loc = "upper left"
pbounds = [0.001, 0.01, 0.05, 0.1]

pval_thresh = [0.05, 0.1]

# ----- END USER INPUTS -----

# for TeX formatting
letters = [chr(i) for i in range(ord('a'), ord('z')+1)]
letters = np.array(letters[:len(pbounds)+1])

legend_labels = [PATHNAMES_PLOT[path_name] for path_name in pathdicts.keys()]

null_forces = list(np.linspace(
    min(FORCELIST),
    max(FORCELIST),
    NFORCE_SAMP,
))
obs_idx = null_forces.index(FORCE_OBSERVED)  # for excluding observation
null_forces = null_forces[:obs_idx] + null_forces[obs_idx+1:]

rng = np.random.default_rng(seed=10)
norm_dist = norm
norm_dist.random_state = rng

# for convenience
minval = 1.0 / mc_evals

# temporarily remove forcing from variable list
varnames = []
for keys, vals in pathdicts.items():
    path_vars = [var for var in vals if var != FORCE_VAR]
    varnames += path_vars
varnames = list(set(varnames))

assert all([var in VARNAMES for var in varnames]), "Invalid variable in varnames"

for region_idx, (region, period) in enumerate(SPACETIMES):

    casename = f"{region}-{period}"
    figdir = os.path.join(FIGDIR_BASE, f"figs-{casename}")
    datadir = os.path.join(DATADIR_BASE, f"data-{casename}")

    print(f"\n******* {casename} *******")

    timelabel   = TIMEBOUNDS[period]["label"]
    regionlabel = REGIONBOUNDS[region]["label"]

    # make output directory
    outdir = os.path.join(figdir, "pvals")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # load global time series data
    data_dict = load_avg_data(datadir, INFILE_BASE, varnames, FORCELIST, ENSLIST)

    fig, ax = plt.subplots(1, 1)
    ax.set_yscale("log")

    artist_list = []
    for path_idx, (path_name, pathlist) in enumerate(pathdicts.items()):

        print(f"Path: {"-".join(pathlist)}")

        pvals = calc_pvals(
            pathlist,
            data_dict,
            FORCELIST,
            ENSLIST,
            FORCE_VAR,
            FORCE_OBSERVED,
            null_forces,
            mc_evals,
        )

        if LATEX:
            # downselect at forcelist
            pvals_tex = []
            for force in FORCELIST:
                if force == FORCE_OBSERVED:
                    continue
                force_idx = null_forces.index(force)
                pvals_tex.append(pvals[force_idx])

            # table of p values
            texstrs = []
            texletters = letters[np.searchsorted(pbounds, pvals_tex)]
            for idx, pval in enumerate(pvals_tex):
                if pval == 0.0:
                    texstr = f"$<$ {minval:#.2e}".replace("e-0", "e-").replace("e+00", "")
                else:
                    texstr = f"{pval:#.2e}".replace("e-0", "e-").replace("e+00", "")
                texstrs.append(f"\\tc{texletters[idx]} {texstr}")
            print("PVALUES: " + " & ".join(texstrs) + "\n")

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

                print(f"CI {pthresh:>4.2f}: {cl:>4.1f} & {cu:>4.1f} \n")

        artist, = ax.plot(
            null_forces[:obs_idx],
            pvals[:obs_idx],
            color=plotcolors[path_idx],
            linewidth=2,
        )
        ax.plot(
            null_forces[obs_idx:],
            pvals[obs_idx:],
            color=plotcolors[path_idx],
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
        ax.legend(artist_list, legend_labels, fontsize=LEGEND_FONTSIZE, loc=legend_loc)

    plt.tight_layout()
    outfile = os.path.join(outdir, f"{region}-{period}-pval.png")
    print(f"Saving image to {outfile}")
    plt.savefig(outfile)
    plt.close(fig)