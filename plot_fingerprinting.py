import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import t

from utils import load_avg_data, calc_ols
from constants import DATADIR_BASE, FIGDIR_BASE, INFILE_BASE
from constants import VARNAMES, ENSLIST
from constants import FORCE_UNITS, FORCE_OBSERVED
from constants import AXIS_FONTSIZE, TICKLABELS_FONTSIZE, TITLE_FONTSIZE, LEGEND_FONTSIZE


# ----- START USER INPUTS -----

varlists = [
    ["TREFHT"],
    ["FSNT", "TREFHT"],
]

# forcelist = [0, 10]
# colors = ["#0C7BDC", "m"]
# ylims = [-1.0, 3.0]

forcelist = [0, 7, 10]
colors = ["#0C7BDC", "#FFC20A", "m"]
ylims = [-2.0, 4.0]

confidence = 0.95
gap = 3

# ----- END USER INPUTS -----

perc = 1.0 - (1.0 - confidence) / 2.0

# only doing this for glob-all
figdir = os.path.join(FIGDIR_BASE, f"figs-glob-all")
datadir = os.path.join(DATADIR_BASE, f"data-glob-all")

# make output directory
outdir = os.path.join(figdir, "fingerprinting")
if not os.path.isdir(outdir):
    os.mkdir(outdir)

nforce = len(forcelist)
assert len(colors) == nforce
nens = len(ENSLIST)
obs_idx = forcelist.index(FORCE_OBSERVED)

varnames_regress = sum(varlists, [])
varnames_regress = list(set(varnames_regress))
assert all([var in VARNAMES for var in varnames_regress]), "Invalid variable in varnames"

# load global time series data
data_dict = load_avg_data(datadir, INFILE_BASE, varnames_regress, forcelist, ENSLIST)

# calculate normalization constants, normalize
norm_mins = []
norm_maxs = []
for var in varnames_regress:
    varmin = np.inf
    varmax = -np.inf
    da_list = data_dict[var]["da_list"]
    for force_idx in range(nforce):
        for ens_idx in range(nens):
            ensmin = float(np.min(da_list[force_idx][ens_idx].values))
            ensmax = float(np.max(da_list[force_idx][ens_idx].values))
            varmin = min(varmin, ensmin)
            varmax = max(varmax, ensmax)
    norm_mins.append(varmin)
    norm_maxs.append(varmax)

    # normalize
    for force_idx in range(nforce):
        for ens_idx in range(nens):
            da_list[force_idx][ens_idx] = -1 + (da_list[force_idx][ens_idx].values - varmin) * 2 / (varmax - varmin)

ntime = da_list[0][0].shape[0]

for list_idx, varnames in enumerate(varlists):
    nvars = len(varnames)
    nsamps = ntime * nvars

    beta_vals = [None for _ in range(nens)]
    attrib_check = [[] for _ in range(nens)]
    ci_vals = [None for _ in range(nens)]

    # regress observation onto data
    for loo_idx in range(nens):

        rem_arr = np.zeros((nsamps, nforce), dtype=np.float64)
        loo_arr = np.zeros((nsamps, 1), dtype=np.float64)

        for var_idx, var in enumerate(varnames):

            da_list = data_dict[var]["da_list"]

            for force_idx, force in enumerate(forcelist):
                # extract remainder mean
                force_vals_list = [da_list[force_idx][ens_idx] for ens_idx in range(nens) if ens_idx != loo_idx]
                force_vals = np.mean(force_vals_list, axis=0)
                rem_arr[var_idx*ntime:(var_idx+1)*ntime, force_idx] = force_vals[:]

            # extract LOO data
            loo_arr[var_idx*ntime:(var_idx+1)*ntime, 0] = da_list[obs_idx][loo_idx][:]

        # record forced beta
        betas, _, _, stand_errs, _, _, _, _ = calc_ols(
            rem_arr,
            loo_arr,
            intercept=True,
            stats=True,
        )
        betas = np.squeeze(betas)
        beta_vals[loo_idx] = betas[1:].copy()

        # check for attribution
        tmult = t.ppf(perc, nsamps - 2)
        ci_vals[loo_idx] = tmult * stand_errs[1:]

        for ci_idx, ci_val in enumerate(ci_vals[loo_idx]):
            beta_check = betas[ci_idx + 1]
            if (beta_check + ci_val > 1.0) and \
               (beta_check - ci_val < 1.0) and \
               (beta_check - ci_val > 0.0):
                attrib_check[loo_idx].append(True)
            else:
                attrib_check[loo_idx].append(False)


    # plot contour
    fig, ax = plt.subplots(1, 1)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=2)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=2)

    for x, y, err, attrib in zip(ENSLIST, beta_vals, ci_vals, attrib_check):
        for force_idx, force in enumerate(forcelist):
            if attrib[force_idx]:
                alpha = 1.0
            else:
                alpha = 0.25
            xplot = x + (nens + gap) * force_idx
            ax.errorbar(
                xplot,
                y[force_idx],
                err[force_idx],
                marker="o",
                markersize=5,
                lw=2,
                capsize=0,
                capthick=2,
                color=colors[force_idx],
                alpha=alpha
            )
            # repeat scatter for solid dots
            ax.scatter(
                xplot,
                y[force_idx],
                marker="o",
                s=25,
                color=colors[force_idx],
            )

    for force_idx, force in enumerate(forcelist):
        center = np.mean(np.array(ENSLIST) + (nens + gap) * force_idx)
        ax.annotate(
            f"{force} {FORCE_UNITS}",
            xy=(center, ylims[1] * 0.9),
            fontsize=LEGEND_FONTSIZE,
            c=colors[force_idx],
            ha='center',
            va='center',
            bbox=dict(
                boxstyle="round",
                fc="white",
            ),
        )
        if force_idx != 0:
            vline_coord = (nens + gap) * force_idx - (gap - 1) / 2
            ax.axvline(vline_coord, color="gray", linestyle=":")

    # format
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylim(ylims)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(f"LOO ensembles", fontsize=AXIS_FONTSIZE)
    ax.set_ylabel(r"$\widehat{\beta}$", fontsize=AXIS_FONTSIZE)
    if nvars == 1:
        title_str = "univariate"
    else:
        title_str = "multivariate"
    ax.set_title(f"{", ".join(varnames)} ({title_str})", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICKLABELS_FONTSIZE)

    # save
    plt.tight_layout()
    outfile = os.path.join(outdir, f"{"-".join(varnames)}-{"-".join([str(force) for force in forcelist])}-betas.png")
    print(f"Saving image to {outfile}")
    plt.savefig(outfile)
    plt.close(fig)
