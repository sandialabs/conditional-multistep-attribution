import os

import numpy as np
import matplotlib.pyplot as plt

from utils import load_glob_data, calc_peak_values, collect_linregress_data
from utils import calc_ols, calc_likelihood_normal
from constants import DATADIR, INFILE_BASE, FIGDIR
from constants import VARNAMES, ENSLIST, FORCELIST, EXPECTED_DIRECTIONS, PATHNAMES_PLOT
from constants import FORCE_VAR, FORCE_OBSERVED, FORCE_UNITS
from constants import TITLE_FONTSIZE, AXIS_FONTSIZE, TICKLABELS_FONTSIZE, LEGEND_FONTSIZE


# ----- START USER INPUTS -----

# each dict entry represents a single pathway
# pathway list defines graph edges
# first entry in sublist defines source node, second entry is end node
pathdict = {
    "strat-single-f-direct": [
        ["SO2", "T050"],
    ],
    "strat-multi-f-direct": [
        ["SO2", "FLNT"],
        ["SO2", "T050"],
        ["FLNT", "T050"],
    ],

    "surf-single-f-direct": [
        ["SO2", "TREFHT"],
    ],
    "surf-multi-f-direct": [
        ["SO2", "FSDS"],
        ["SO2", "TREFHT"],
        ["FSDS", "TREFHT"],
    ],

    "combo-single-f-direct": [
        ["SO2", "T050"],
        ["SO2", "TREFHT"],
    ],
    "combo-multi-f-direct": [
        ["SO2", "FLNT"],
        ["SO2", "T050"],
        ["FLNT", "T050"],
        ["SO2", "FSDS"],
        ["SO2", "TREFHT"],
        ["FSDS", "TREFHT"],
    ],
}

# each dict entry is a single prior distribution
# each list corresponds to prior prob on entries in FORCELIST
priors = {
    "well-spec": [0.02, 0.02, 0.02, 0.02, 0.3,  0.3,  0.3,  0.02],
    "poor-spec": [0.02, 0.02, 0.3,  0.3,  0.3,  0.02, 0.02, 0.02],
    "agnostic":  [0.93, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
}

# one entry per pathway
plot_legend = [True, False, False, False, False, False]

# ----- END USER INPUTS -----

# make output directory
outdir = os.path.join(FIGDIR, "posteriors_glob")
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# for convenience
nforce = len(FORCELIST)
for prior_name, prior in priors.items():
    priors[prior_name] = np.array(prior, dtype=np.float64)

# temporarily remove forcing from variable list
varnames = []
for keys, vals in pathdict.items():
    path_vars = sum(vals, [])
    path_vars = [var for var in path_vars if var != FORCE_VAR]
    varnames += path_vars
varnames = list(set(varnames))

assert all([var in VARNAMES for var in varnames]), "Invalid variable in varnames"

# load global time series data
data_dict = load_glob_data(DATADIR, INFILE_BASE, varnames, FORCELIST, ENSLIST)

# re-insert forcing variable
varnames_regress = [FORCE_VAR] + varnames

for path_idx, (path_name, pathlists) in enumerate(pathdict.items()):

    # determine parent sets
    parents_dict = {}
    for edge_vars in pathlists:
        if edge_vars[1] not in parents_dict:
            parents_dict[edge_vars[1]] = {"parent_vars": [edge_vars[0]]}
        elif edge_vars[0] not in parents_dict[edge_vars[1]]:
            parents_dict[edge_vars[1]]["parent_vars"] += [edge_vars[0]]
        else:
            raise ValueError(f"Repeated chain parent: {edge_vars[0]} -> {edge_vars[1]}")

    # compute linear models for each parent set
    for child_var, parent_dict in parents_dict.items():

        parent_vars = parent_dict["parent_vars"]
        nparents = len(parent_vars)

        # compute linear regression parameters
        x_deltas_concat, y_deltas_concat, x_deltas_obs, y_deltas_obs = collect_linregress_data(
            data_dict,
            parent_vars,
            child_var,
            FORCELIST,
            ENSLIST,
            EXPECTED_DIRECTIONS,
            FORCE_VAR,
            FORCE_OBSERVED,
        )
        betas = np.squeeze(calc_ols(x_deltas_concat, y_deltas_concat, intercept=True))

        # compute residual sample variance
        y_pred = betas[0] + x_deltas_concat @ betas[1:]
        residuals = y_pred - np.squeeze(y_deltas_concat)
        variance = np.sum(np.square(residuals)) / (residuals.shape[0] - betas.shape[0])

        # store parameters for later
        parent_dict["betas"] = betas.copy()
        parent_dict["variance"] = variance

    # collect "observational" data
    obs_idx = FORCELIST.index(FORCE_OBSERVED)
    obs_dict = {}
    for var in varnames:
        # ignore forcing, as this isn't an observational quantity
        if var == FORCE_VAR:
            continue
        da_list = data_dict[var]["da_list"][obs_idx]
        deltas, _ = calc_peak_values(da_list, EXPECTED_DIRECTIONS[var])

        # observation is ensemble mean of peaks
        obs_dict[var] = np.mean(deltas)

    # compute likelihoods
    likelihoods = np.zeros(nforce, dtype=np.float64)
    for force_idx, force in enumerate(FORCELIST):
        likelihood_force = 1.0
        for child_var, parent_dict in parents_dict.items():

            parent_vars = parent_dict["parent_vars"]
            nparents = len(parent_vars)

            # forcing and observational data point
            xvals = []
            for parent_idx, parent_var in enumerate(parent_vars):
                if parent_var == FORCE_VAR:
                    xvals.append(force)
                else:
                    xvals.append(obs_dict[parent_var])
            xvals = np.array(xvals, dtype=np.float64)
            yval = obs_dict[child_var]

            # likelihood for this step of the pathway
            likelihood_step = calc_likelihood_normal(
                parent_dict["betas"],
                parent_dict["variance"],
                xvals,
                yval,
            )

            # likelihood for this forcing is product of steps in pathway
            likelihood_force *= likelihood_step

        # store likelihood
        likelihoods[force_idx] = likelihood_force

    # compute posterior for each prior distribution
    for prior_name, prior in priors.items():

        # compute evidence and posterior
        evidence = np.sum(likelihoods * prior)
        posterior = likelihoods * prior / evidence

        # plot
        fig, ax = plt.subplots(1, 1)

        barwidth = 0.4
        # prior bars
        offset = -0.4 + barwidth * 1 / 2
        ax.bar(np.arange(nforce) + offset, prior, width=barwidth, color="orange", edgecolor="k")
        # posterior bars
        offset = -0.4 + barwidth * 3 / 2
        ax.bar(np.arange(nforce) + offset, posterior, width=barwidth, color="royalblue", edgecolor="k")

        # plot formatting
        if plot_legend[path_idx]:
            ax.legend(["Prior", "Posterior"], loc="upper center", fontsize=LEGEND_FONTSIZE)
        ax.set_xlabel(f"{FORCE_VAR} forcing ({FORCE_UNITS})", fontsize=AXIS_FONTSIZE)
        ax.set_ylabel("Probability", fontsize=AXIS_FONTSIZE)
        ax.set_ylim([0.0, 1.0])
        ax.set_xticks(np.arange(nforce), [f"{force}" for force in FORCELIST])
        ax.tick_params(axis="both", which="major", labelsize=TICKLABELS_FONTSIZE)
        ax.set_title(PATHNAMES_PLOT[path_name] + f", Post({FORCE_OBSERVED} {FORCE_UNITS}) = {posterior[obs_idx]:.3f}", fontsize=TITLE_FONTSIZE)

        # save figure
        prior_dir = os.path.join(outdir, prior_name)
        if not os.path.isdir(prior_dir):
            os.mkdir(prior_dir)

        plt.tight_layout()
        outfile = f"{path_name}_posteriors.png"
        outfile = os.path.join(prior_dir, outfile)
        print(f"Saving image to {outfile}")
        plt.savefig(outfile)
        plt.close(fig)