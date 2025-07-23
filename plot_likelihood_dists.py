import os

import numpy as np

from utils import load_avg_data, calc_avg_values, calc_regressions
from utils import calc_likelihood_normal, plot_1d_likelihoods
from constants import DATADIR_BASE, FIGDIR_BASE, INFILE_BASE
from constants import VARNAMES, VARLABELS, ENSLIST, FORCELIST
from constants import FORCE_VAR, FORCE_OBSERVED, FORCE_UNITS
from constants import PLOTCOLORS, LINEWIDTHS
from constants import AXIS_FONTSIZE, TICKLABELS_FONTSIZE, LEGEND_FONTSIZE


# ----- START USER INPUTS -----

# each dict entry represents a single pathway
# pathway list defines graph edges
# first entry in sublist defines source node, second entry is end node
pathdicts = {
    "surf-single": {
        "child_var": "TREFHT",
        "parent_vars": ["SO2"],
        "xlim": [-0.5, 0.2],
        "ylim": [-0.5, 10],
        "legend": False,
    },
    "surf-inter": {
        "child_var": "FSNT",
        "parent_vars": ["SO2"],
        "xlim": [-3.5, 0.5],
        "ylim": [-0.25, 3.5],
        "legend": False,
    },
    "surf-multi": {
        "child_var": "TREFHT",
        "parent_vars": ["SO2", "FSNT"],
        "xlim": [-0.5, 0.2],
        "ylim": [-0.5, 14],
        "legend": False,
    },
}

joint_dicts = {
    "surf-multi-joint": {
        "inter": "surf-inter",
        "final": "surf-multi",
        "log": True,
        "xlim": [-0.5, 0.2],
        "ylim": [1e-12, 1e2],
        "legend": False,
    },
}

region = "glob"
period = "all"

# ----- END USER INPUTS -----

casename = f"{region}-{period}"
datadir = os.path.join(DATADIR_BASE, f"data-{casename}")
figdir = os.path.join(FIGDIR_BASE, f"figs-{casename}")

# make output directory
outdir = os.path.join(figdir, "likelihood_dists")
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# for convenience
nforce = len(FORCELIST)

# temporarily remove forcing from variable list
varnames = []
for keys, vals in pathdicts.items():
    path_vars = [var for var in vals["parent_vars"] if var != FORCE_VAR]
    varnames += path_vars + [vals["child_var"]]
varnames = list(set(varnames))

assert all([var in VARNAMES for var in varnames]), "Invalid variable in varnames"

# load global time series data
data_dict = load_avg_data(datadir, INFILE_BASE, varnames, FORCELIST, ENSLIST)

# re-insert forcing variable
varnames_regress = [FORCE_VAR] + varnames

# collect "observational" data
obs_idx = FORCELIST.index(FORCE_OBSERVED)
obs_dict = {}
for var in varnames:
    # ignore forcing, as this isn't an observational quantity
    if var == FORCE_VAR:
        continue
    da_list = data_dict[var]["da_list"][obs_idx]
    y_obs = calc_avg_values(da_list)

    # observation is ensemble mean of peaks
    obs_dict[var] = np.mean(y_obs)

dists_dict = {}
for path_idx, (path_name, pathdict) in enumerate(pathdicts.items()):

    child_var   = pathdict["child_var"]
    parent_vars = pathdict["parent_vars"]

    # distribution plotting
    compare_range = pathdict["xlim"]
    yval_linspace = np.linspace(compare_range[0], compare_range[1], 1000)

    # determine parent sets
    parents_list = [{
        "child_var": child_var,
        "parent_vars": parent_vars,
    }]

    parents_list = calc_regressions(
        parents_list,
        data_dict,
        FORCELIST,
        ENSLIST,
        FORCE_VAR,
        FORCE_OBSERVED,
    )
    variance = parents_list[0]["variance"]
    betas = parents_list[0]["betas"]

    # compute likelihood distributions
    likelihood_dist_list = []
    for force in FORCELIST:

        # forcing and observational data point
        xvals = []
        for parent_idx, parent_var in enumerate(parent_vars):
            if parent_var == FORCE_VAR:
                xvals.append(force)
            else:
                xval = obs_dict[parent_var]
                xvals.append(xval)
        xvals = np.array(xvals, dtype=np.float64)

        dist = calc_likelihood_normal(
            betas,
            variance,
            xvals,
            yval_linspace,
        )
        likelihood_dist_list.append(dist.copy())

    dists_dict[path_name] = likelihood_dist_list.copy()

    plot_1d_likelihoods(
        yval_linspace,
        likelihood_dist_list,
        obs_dict[child_var],
        FORCELIST,
        FORCE_OBSERVED,
        FORCE_UNITS,
        FORCE_VAR,
        child_var,
        parent_vars,
        VARLABELS,
        PLOTCOLORS,
        LINEWIDTHS,
        path_name,
        outdir,
        xlim=pathdict["xlim"],
        ylim=pathdict["ylim"],
        axis_fontsize=AXIS_FONTSIZE,
        legend_fontsize=LEGEND_FONTSIZE,
        ticklabels_fontsize=TICKLABELS_FONTSIZE,
        legend=pathdict["legend"],
        separate_legend=False,
    )

# NOTE: only handles two steps
for joint_name, joint_dict in joint_dicts.items():

    # collect joint path data
    inter_path = joint_dict["inter"]
    final_path = joint_dict["final"]
    inter_dict = pathdicts[inter_path]
    final_dict = pathdicts[final_path]
    inter_child = inter_dict["child_var"]
    final_child = final_dict["child_var"]
    final_parents = final_dict["parent_vars"]
    inter_dists = dists_dict[inter_path]
    final_dists = dists_dict[final_path]

    # get intermediate observation value
    inter_range = inter_dict["xlim"]
    inter_linspace = np.linspace(inter_range[0], inter_range[1], 1000)
    inter_obs_idx = np.argmin(np.abs(inter_linspace - obs_dict[inter_child]))

    # compute joint distribution
    joint_dists = []
    for dist_idx, dist in enumerate(final_dists):
        inter_dist = inter_dists[dist_idx]
        inter_obs = inter_dist[inter_obs_idx]
        joint_dists.append(final_dists[dist_idx] * inter_obs)

    # plot
    final_range = final_dict["xlim"]
    final_linspace = np.linspace(final_range[0], final_range[1], 1000)
    plot_1d_likelihoods(
        final_linspace,
        joint_dists,
        obs_dict[final_child],
        FORCELIST,
        FORCE_OBSERVED,
        FORCE_UNITS,
        FORCE_VAR,
        final_child,
        [],
        VARLABELS,
        PLOTCOLORS,
        LINEWIDTHS,
        joint_name,
        outdir,
        other_vars=final_parents,
        xlim=joint_dict["xlim"],
        ylim=joint_dict["ylim"],
        axis_fontsize=AXIS_FONTSIZE,
        legend_fontsize=LEGEND_FONTSIZE,
        ticklabels_fontsize=TICKLABELS_FONTSIZE,
        log=joint_dict["log"],
        legend=joint_dict["legend"],
        separate_legend=True,
    )


print("Finished")