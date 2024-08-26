import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from utils import load_glob_data, calc_peak_values, calc_ovl_binormal
from constants import DATADIR, INFILE_BASE, FIGDIR
from constants import VARNAMES, ENSLIST, FORCELIST, EXPECTED_DIRECTIONS
from constants import FORCE_OBSERVED, FORCE_UNITS
from constants import TITLE_FONTSIZE, AXIS_FONTSIZE, TICKLABELS_FONTSIZE


# ----- START USER INPUTS -----

# fonts may appear a little small in contour plots
fontsize_scaling = 1.2

# contour levels and skip in colorbar
nlevels = 41
levelskip = 4

# ----- END USER INPUTS -----

# make output directory
outdir = os.path.join(FIGDIR, "ovl")
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# meshgrid for plotting contours
nforce = len(FORCELIST)
xvals = np.zeros(nforce+1)
for i in range(1, nforce):
    xvals[i] = (FORCELIST[i] + FORCELIST[i-1]) / 2.0
xvals[0] = FORCELIST[0] - (xvals[1] - FORCELIST[0])
xvals[-1] = FORCELIST[-1] + (FORCELIST[-1] - xvals[-2])
X, Y = np.meshgrid(xvals, xvals)

# load global time series data
data_dict = load_glob_data(DATADIR, INFILE_BASE, VARNAMES, FORCELIST, ENSLIST)

for var_idx, var in enumerate(VARNAMES):

    da_list = data_dict[var]["da_list"]

    ovl_arr = np.eye(nforce, dtype=np.float64)
    for force_idx_1 in range(nforce):

        # extract peak impacts for forcing level 1
        da_list_1 = da_list[force_idx_1]
        delta_list_1, _ = calc_peak_values(da_list_1, EXPECTED_DIRECTIONS[var])

        # compute sample mean and standard deviation
        mu_1 = np.mean(delta_list_1)
        s_1 = np.std(delta_list_1)

        # relationship is symmetric
        for force_idx_2 in range(force_idx_1+1, nforce):

            # extract peak impacts for forcing level 2, compute statistics
            da_list_2 = da_list[force_idx_2]
            delta_list_2, _ = calc_peak_values(da_list_2, EXPECTED_DIRECTIONS[var])
            mu_2 = np.mean(delta_list_2)
            s_2 = np.std(delta_list_2)

            # compute OVL
            ovl = calc_ovl_binormal(mu_1, mu_2, s_1, s_2)
            ovl_arr[force_idx_1, force_idx_2] = ovl
            ovl_arr[force_idx_2, force_idx_1] = ovl

    # plotting

    # contour parameters
    levels = np.linspace(0, 1, nlevels)
    norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)
    ticks = levels[::levelskip]

    # indices for counterfactual and "observed" dataset
    idx_cf = FORCELIST.index(0)
    idx_obs = FORCELIST.index(FORCE_OBSERVED)

    # plot contour
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    cs = ax.pcolor(xvals, xvals, ovl_arr, cmap="viridis", norm=norm, edgecolors="k", shading="flat")

    # format
    ax.set_xticks(FORCELIST)
    ax.set_yticks(FORCELIST)
    ax.set_xlabel(f"Forcing mass 1 (Tg)", fontsize=AXIS_FONTSIZE*fontsize_scaling)
    ax.set_ylabel(f"Forcing mass 2 (Tg)", fontsize=AXIS_FONTSIZE*fontsize_scaling)
    ax.set_title(f"{var} impact OVL, 0 vs {FORCE_OBSERVED} {FORCE_UNITS}: {ovl_arr[idx_cf, idx_obs]:#.3f}", fontsize=TITLE_FONTSIZE*fontsize_scaling)
    ax.tick_params(axis="both", which="major", labelsize=TICKLABELS_FONTSIZE*fontsize_scaling)
    fig.subplots_adjust(right=0.875)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='vertical')
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=TICKLABELS_FONTSIZE*fontsize_scaling)

    # save
    outfile = os.path.join(outdir, f"{var}_ovl.png")
    print(f"Saving image to {outfile}")
    plt.savefig(outfile)
    plt.close(fig)