import os
from math import copysign

import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

from utils import load_glob_data, collect_linregress_data
from utils import calc_ols
from constants import DATADIR, INFILE_BASE, FIGDIR
from constants import VARNAMES, ENSLIST, FORCELIST, EXPECTED_DIRECTIONS
from constants import UNITS, FORCE_VAR, FORCE_OBSERVED
from constants import PLOTCOLORS
from constants import TITLE_FONTSIZE, AXIS_FONTSIZE, TICKLABELS_FONTSIZE


# ----- START USER INPUTS -----

# each entry is a single regression
# the LAST variable in each sublist is the target variable
regress_lists = [
    ["SO2", "FLNT"],
    ["SO2", "FSDS"],
    ["SO2", "FLNT", "T050"],
    ["SO2", "FSDS", "TREFHT"],
    ["SO2", "T050"],
    ["SO2", "TREFHT"],
]

axis_lims = {
    "SO2": [-1, 16],
    "FLNT": [-4.0, 0.0],
    "FSDS": [-6.0, 0.0],
    "TREFHT": [-0.7, 0.05],
    "T050": [0.0, 3.0],
}

# confidence interval, e.g., 90% confidence interval -> 0.9
confidence = 0.9

# legend text size is sometimes too large
legend_fontsize_scaling = 0.9

# ----- END USER INPUTS -----

# make output directory
outdir = os.path.join(FIGDIR, "peak_regress_glob")
if not os.path.isdir(outdir):
    os.mkdir(outdir)

nforce = len(FORCELIST)
nens = len(ENSLIST)

# collect variables in regression, do not include forcing variable
varnames_regress = list(set(sum(regress_lists, [])))
varnames_regress = [var for var in varnames_regress if var != FORCE_VAR]
assert all([var in VARNAMES for var in varnames_regress]), "Invalid variable in regression list"

# for plotting forcing levels
force_ticks = [FORCELIST, [f"{force}" for force in FORCELIST]]

# load global time series data
data_dict = load_glob_data(DATADIR, INFILE_BASE, varnames_regress, FORCELIST, ENSLIST)

# re-insert forcing variable
varnames_regress = [FORCE_VAR] + varnames_regress

for regress_idx, regress_list in enumerate(regress_lists):

    nvars = len(regress_list)

    if nvars == 2:
        # univariate model
        fig, ax = plt.subplots(1, 1)
    elif nvars == 3:
        # two-variable model
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(projection='3d')
    else:
        # no plotting for higher dimensions
        raise ValueError(f"Invalid nvars: {nvars}")

    # predictor and target variable names
    xvars = regress_list[:-1]
    yvar = regress_list[-1]

    # collect concatenated predictor and target data for regression
    # "observational" data is separated
    x_deltas_concat, y_deltas_concat, x_deltas_obs, y_deltas_obs = collect_linregress_data(
        data_dict,
        xvars,
        yvar,
        FORCELIST,
        ENSLIST,
        EXPECTED_DIRECTIONS,
        FORCE_VAR,
        FORCE_OBSERVED,
    )

    # plot scatter data
    enscount = 0
    for force_idx, force in enumerate(FORCELIST):

        # select data and plot marker
        if force == FORCE_OBSERVED:
            scatter_marker = "x"
            x_plot = x_deltas_obs.copy()
            y_plot = y_deltas_obs.copy()
        else:
            scatter_marker = "o"
            x_plot = x_deltas_concat[enscount:enscount+nens, :]
            y_plot = y_deltas_concat[enscount:enscount+nens, :]
            enscount += nens

        # plot data
        if nvars == 2:
            ax.scatter(x_plot, y_plot, color=PLOTCOLORS[str(force)], marker=scatter_marker)

        elif nvars == 3:
            ax.scatter(x_plot[:, 0], x_plot[:, 1], y_plot, color=PLOTCOLORS[str(force)], marker=scatter_marker, s=40)

    # compute linear regression
    betas = np.squeeze(calc_ols(x_deltas_concat, y_deltas_concat, intercept=True))

    # plot best fit line/plane
    y_pred = betas[0] + x_deltas_concat @ betas[1:]
    x_min = np.amin(x_deltas_concat, axis=0)
    x_max = np.amax(x_deltas_concat, axis=0)
    if nvars == 2:
        ax.plot(x_deltas_concat, y_pred, 'k')
    elif nvars == 3:
        x_pred_1 = np.linspace(x_min[0], x_max[0])
        x_pred_2 = np.linspace(x_min[1], x_max[1])
        X_1, X_2 = np.meshgrid(x_pred_1, x_pred_2)
        x_1_calc = X_1.ravel(order="F")
        x_2_calc = X_2.ravel(order="F")
        y_pred_plot = betas[0] + x_1_calc * betas[1] + x_2_calc * betas[2]
        Y_pred_plot = np.reshape(y_pred_plot, X_1.shape, order="F")
        ax.plot_surface(X_1, X_2, Y_pred_plot, alpha=0.2)

    # compute fit statistics
    nsamps = x_deltas_concat.shape[0]
    ymean = np.mean(y_deltas_concat)
    residuals = y_pred - np.squeeze(y_deltas_concat)
    ss_res = np.sum(np.square(residuals))
    ss_y = np.sum(np.square(y_deltas_concat - ymean))
    r2 = 1.0 - ss_res / ss_y

    # compute/plot confidence and prediction intervals
    if nvars == 2:
        mse = ss_res / (nsamps - 2)
        xmean = np.mean(x_deltas_concat)
        ss_x = np.sum(np.square(xmean - x_deltas_concat))

        xvals = np.linspace(x_min, x_max, 100)
        yvals = betas[0] + betas[1] * xvals

        s_ci = np.sqrt(mse * (1.0 / nsamps + np.square(xvals - xmean) / ss_x))
        s_pi = np.sqrt(mse * (1.0 + 1.0 / nsamps + np.square(xvals - xmean) / ss_x))

        perc = 1.0 - (1.0 - confidence) / 2.0
        tmult = t.ppf(perc, nsamps - 2)
        conf_int = tmult * s_ci
        pred_int = tmult * s_pi

        ax.plot(xvals, yvals + conf_int, "k--")
        ax.plot(xvals, yvals - conf_int, "k--")
        ax.plot(xvals, yvals + pred_int, "k:")
        ax.plot(xvals, yvals - pred_int, "k:")

    # title formatting
    param_eq = f"{regress_list[-1]} = {betas[0]:#.3g} "
    for var_idx in range(1, len(regress_list)):
        sign = copysign(1, betas[var_idx])
        if sign == 1:
            sign_str = "+"
        elif sign == -1:
            sign_str = "-"
        else:
            raise ValueError("Problem with sign")
        param_eq += f"{sign_str} {abs(betas[var_idx]):#.3g}$\\times${regress_list[var_idx-1]} "
    ax.set_title("R$^2$ = " + f"{r2:#.3g}\n{param_eq}", fontsize=TITLE_FONTSIZE)

    # axis formatting
    ax.set_xlabel(f"{regress_list[0]} impact ({UNITS[regress_list[0]]})", fontsize=AXIS_FONTSIZE, labelpad=10)
    ax.set_ylabel(f"{regress_list[1]} impact ({UNITS[regress_list[1]]})", fontsize=AXIS_FONTSIZE, labelpad=10)
    if nvars == 3:
        ax.set_zlabel(f"{regress_list[2]} impact ({UNITS[regress_list[2]]})", fontsize=AXIS_FONTSIZE, labelpad=10)

    ax.set_xlim(axis_lims[regress_list[0]])
    if regress_list[0] == FORCE_VAR:
        ax.set_xticks(force_ticks[0], force_ticks[1])
    ax.set_ylim(axis_lims[regress_list[1]])
    if regress_list[1] == FORCE_VAR:
        ax.set_yticks(force_ticks[0], force_ticks[1])
    if nvars == 3:
        ax.set_zlim(axis_lims[regress_list[2]])

    ax.tick_params(axis="both", which="major", labelsize=TICKLABELS_FONTSIZE)

    # save plot
    plt.tight_layout()
    outfile = "-".join(regress_list) + f"_scatter.png"
    outfile = os.path.join(outdir, outfile)
    print(f"Saving image to {outfile}")
    plt.savefig(outfile)
    plt.close(fig)