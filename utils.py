import os
from copy import deepcopy
from math import sqrt
from math import pi

import cftime
import numpy as np
from scipy.stats import norm
import xarray as xr
import matplotlib.pyplot as plt


def load_avg_data(datadir, datafile_base, varlist, forcelist, enslist, debug=False):

    data_dict = {}

    # compute for each variable
    for var in varlist:

        data_dict[var] = {
            "da_list": [],
            "da_mean_list": [],
        }

        # collect ensemble means for each forcing level
        for force in forcelist:

            # load global time series for all ensemble members
            da_list = []
            for ens in enslist:

                datafile = datafile_base.format(
                    varname=var,
                    force=force,
                    ens=ens,
                )
                datafile = os.path.join(datadir, datafile)

                da = xr.open_dataarray(datafile)
                da_list.append(da.copy(deep=True))

            # store raw data lists
            data_dict[var]["da_list"].append(deepcopy(da_list))

            # compute ensemble mean, store
            da_mean = (xr.concat(da_list, dim="member")).mean(dim="member")
            data_dict[var]["da_mean_list"].append(da_mean.copy(deep=True))

    return data_dict


def dates_to_years(
    dates,
    calendar="noleap",
    has_year_zero=True,
    from_monthly=False,
):
    DAYS_OF_MONTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    first_year = dates[0].year

    days = cftime.date2num(
        dates,
        f"days since {first_year}-01-01",
        calendar=calendar,
        has_year_zero=has_year_zero,
    )

    years = first_year + days / 365

    # offsets to center of previous month
    # monthly data is originally set to first day of following month
    if from_monthly:
        for date_idx, date in enumerate(dates):
            month_idx = date.month - 2  # previous month, zero index
            days_in_month = DAYS_OF_MONTHS[month_idx]
            years[date_idx] -= (days_in_month / 2) / 365

    return years


def calc_avg_values(da_list_in):

    if not isinstance(da_list_in, list):
        da_list = [da_list_in]
    else:
        da_list = da_list_in

    avg_list = []
    for da in da_list:
        vals_glob = da.values
        avg = np.mean(vals_glob)
        avg_list.append(avg)

    return avg_list


def collect_linregress_data(
    data_dict,
    xvars,
    yvar,
    forcelist,
    enslist,
    force_var,
    force_obs,
    normalize=False,
):

    nens = len(enslist)

    x_concat = None
    y_concat = None
    for force_idx, force in enumerate(forcelist):

        # collect impacts for predictor variables
        for x_idx, xvar in enumerate(xvars):

            if xvar == force_var:
                # forcing variable is not in the impact data
                x = force * np.ones(nens, dtype=np.float64)
            else:
                x_da_list = data_dict[xvar]["da_list"][force_idx]
                x = calc_avg_values(x_da_list)
            x = np.expand_dims(x, axis=-1)

            if x_idx == 0:
                x_tot = x.copy()
            else:
                x_tot = np.concatenate((x_tot, x), axis=1)

        # impacts for target variables
        y_da_list = data_dict[yvar]["da_list"][force_idx]
        y = calc_avg_values(y_da_list)
        y = np.expand_dims(y, axis=-1)

        # separate "observed" forcing
        if force == force_obs:
            x_obs = x_tot.copy()
            y_obs = y.copy()
        else:
            # concatenate for later regression
            if x_concat is None:
                x_concat = x_tot.copy()
                y_concat = y.copy()
            else:
                x_concat = np.concatenate((x_concat, x_tot), axis=0)
                y_concat = np.concatenate((y_concat, y), axis=0)

    # feature scaling, if requested
    if normalize:
        # scale x values
        xmin = np.min(x_concat, axis=0, keepdims=True)
        xmax = np.max(x_concat, axis=0, keepdims=True)
        x_concat = calc_minmax_scaling(x_concat, xmin, xmax)
        x_obs    = calc_minmax_scaling(x_obs,    xmin, xmax)
        # scale y values
        ymin = np.min(y_concat, axis=0, keepdims=True)
        ymax = np.max(y_concat, axis=0, keepdims=True)
        y_concat = calc_minmax_scaling(y_concat, ymin, ymax)
        y_obs    = calc_minmax_scaling(y_obs,    ymin, ymax)

    return x_concat, y_concat, x_obs, y_obs


def calc_minmax_scaling(arr, mins, maxs):

    assert arr.ndim == mins.ndim
    assert arr.ndim == maxs.ndim
    assert arr.shape[1:] == mins.shape[1:]
    assert arr.shape[1:] == maxs.shape[1:]
    assert mins.shape[0] == 1
    assert maxs.shape[0] == 1

    arr_out = -1.0 + 2 * (arr - mins) / (maxs - mins)

    return arr_out


def calc_ols(
    Xin,
    Yin,
    intercept=False,
    stats=False,
):
    # compute ordinary least squares regression coefficients

    nsamps = Yin.shape[0]
    assert Xin.shape[0] == nsamps, f"Number of samples do not match: {nsamps} v {Xin.shape[0]}"

    if Yin.ndim == 1:
        Yin = Yin[:, None]
    if Xin.ndim == 1:
        Xin = Xin[:, None]

    # add intercept
    if intercept:
        Xin = np.concatenate((np.ones((nsamps, 1), dtype=Xin.dtype), Xin), axis=1)

    # compute approximate parameters
    A = Xin.T @ Xin
    B = Xin.T @ Yin
    betas = np.linalg.solve(A, B)

    # compute statistics
    if stats:
        # predictions
        Ypred = Xin @ betas
        Ymean = np.mean(Yin)

        # variance
        residuals = Ypred - Yin
        variance = np.sum(np.square(residuals)) / (residuals.shape[0] - betas.shape[0])

        # beta standard errors
        se2_mat = variance * np.linalg.inv(A)
        standard_errs = np.sqrt(np.diagonal(se2_mat))

        # errors
        ss_y = np.sum(np.square(Yin - Ymean))
        rss = np.sum(np.square(residuals))
        mse = rss / (nsamps - 2)
        r2 = 1.0 - rss / ss_y

        return betas, Ypred, variance, standard_errs, rss, ss_y, mse, r2

    else:
        return betas



def calc_likelihood_normal(betas, var, xvals_list, yvals):

    y_pred = betas[0]
    for xidx, xvals in enumerate(xvals_list):
        y_pred += betas[xidx+1] * np.expand_dims(xvals, axis=0)

    exponential = np.exp(-0.5 * (yvals - y_pred)**2 / var)
    likelihood = (1.0 / (np.sqrt(var * 2 * pi))) * exponential

    return likelihood


def calc_regressions(
    parents_list,
    data_dict,
    forcelist,
    enslist,
    force_var,
    force_observed,
    normalize=False,
):

    nens = len(enslist)
    obs_idx = forcelist.index(force_observed)

    for stepdict in parents_list:

        child_var = stepdict["child_var"]
        parent_vars = stepdict["parent_vars"]
        nparents = len(parent_vars)

        # compute linear regression parameters
        x_concat, y_concat, x_obs, _ = collect_linregress_data(
            data_dict,
            parent_vars,
            child_var,
            forcelist,
            enslist,
            force_var,
            force_observed,
            normalize=normalize,
        )
        betas = np.squeeze(calc_ols(x_concat, y_concat, intercept=True))

        # compute residual sample variance
        y_pred = betas[0] + x_concat @ betas[1:]
        residuals = y_pred - np.squeeze(y_concat)
        variance = np.sum(np.square(residuals)) / (residuals.shape[0] - betas.shape[0])

        # store parameters for later
        stepdict["betas"] = betas.copy()
        stepdict["variance"] = variance
        stepdict["std"] = sqrt(variance)

        x_all = np.reshape(x_concat, (nens, -1, nparents), order="F")
        x_all = np.insert(x_all, obs_idx, x_obs, axis=1)

        # compute means
        x_means = np.mean(x_all, axis=0)
        y_means = betas[0]
        for var_idx in range(1, nparents+1):
            y_means += betas[var_idx] * x_means[:, var_idx - 1]
        stepdict["means"] = y_means.copy()

    return parents_list


def plot_1d_likelihoods(
    yvals,
    like_dist_list,
    yval_obs,
    forcelist,
    force_observed,
    force_units,
    force_var,
    dist_var,
    cond_vars,
    varlabels,
    colors,
    linewidths,
    path_name,
    outdir,
    other_vars=[],
    xlim=[None, None],
    ylim=[None, None],
    log=False,
    axis_fontsize=12,
    legend_fontsize=12,
    ticklabels_fontsize=12,
    legend=False,
    separate_legend=False,
):

    fig, ax = plt.subplots(1, 1)

    ndists = len(like_dist_list)
    artist_list = [None for _ in range(ndists)]
    for dist_idx, dist in enumerate(like_dist_list):

        color = colors[str(forcelist[dist_idx])]
        linewidth = 1.5 * linewidths[str(forcelist[dist_idx])]  # slightly thicker
        if forcelist[dist_idx] == force_observed:
            linestyle = "--"
        else:
            linestyle = "-"

        if log:
            artist, = ax.semilogy(yvals, dist, color=color, linewidth=linewidth, linestyle=linestyle)
        else:
            artist, = ax.plot(yvals, dist, color=color, linewidth=linewidth, linestyle=linestyle)

        artist_list[dist_idx] = deepcopy(artist)

    ax.axvline(yval_obs, color="k", linestyle=":", linewidth=2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(varlabels[dist_var], fontsize=axis_fontsize)
    y_title = \
        f"P( {varlabels[dist_var]}" \
        + "".join([f", {varlabels[var]}$_o$" for var in other_vars if (var != force_var)]) \
        + " | " \
        + "".join([f"{varlabels[var]}$_o$, " for var in cond_vars if (var != force_var)]) \
        + f"{varlabels[force_var]} )"
    ax.set_ylabel(y_title, fontsize=axis_fontsize)

    if legend or separate_legend:
        legend_labels = [f"{force} {force_units}" for force in forcelist] + ["Obs."]
        if legend:
            ax.legend(artist_list, legend_labels, loc="upper left", fontsize=legend_fontsize)
        if separate_legend:
            fig_leg, ax_leg = plt.subplots(1, 1)
            ax_leg.legend(artist_list, legend_labels, loc="center")
            ax_leg.axis('off')
            outfile = os.path.join(outdir, f"{path_name}-legend.png")
            plt.savefig(outfile)
            plt.close(fig_leg)

    plt.sca(ax)

    ax.tick_params(axis="both", which="major", labelsize=ticklabels_fontsize)

    plt.tight_layout()

    if log:
        logstr = "-log"
    else:
        logstr = ""
    outfile = f"{path_name}-like-dists{logstr}.png"
    outfile = os.path.join(outdir, outfile)
    print(f"Saving image to {outfile}")
    plt.savefig(outfile)
    plt.close(fig)


def ll(D, mu, std):
    return norm.logpdf(D, loc=mu, scale=std)


def calc_pvals(
    pathlist,
    data_dict,
    forcelist,
    enslist,
    force_var,
    force_observed,
    null_forces,
    mc_evals,
    norm_dist,
    experimental=False,
):

    alt_idx  = forcelist.index(force_observed)

    # determine parent sets
    parents_list = []
    nvars = len(pathlist)
    for var_idx in range(1, nvars):
        parents_list.append({
            "child_var": pathlist[var_idx],
            "parent_vars": pathlist[:var_idx]},
        )

    parents_list = calc_regressions(
        parents_list,
        data_dict,
        forcelist,
        enslist,
        force_var,
        force_observed,
        normalize=False,
    )

    # compute mean predictions
    # null_means = {str(force): {force_var: float(force)} for force in null_forces}
    # alt_means  = {force_var: float(force_observed)}
    # for step_idx, step_dict in enumerate(parents_list):
    #     child_var   = step_dict["child_var"]
    #     parent_vars = step_dict["parent_vars"]
    #     betas = step_dict["betas"]

    #     # compute alt mean
    #     alt_mean = betas[0]
    #     for var_idx, varname in enumerate(parent_vars):
    #         alt_mean += betas[var_idx+1] * alt_means[varname]
    #     alt_means[child_var] = alt_mean

    #     # compute null means
    #     for force in null_forces:
    #         null_mean = betas[0]
    #         for var_idx, varname in enumerate(parent_vars):
    #             null_mean += betas[var_idx+1] * null_means[str(force)][varname]
    #         null_means[str(force)][child_var] = null_mean

    # draw samples
    cond_means = {str(f): {force_var: float(f)} for f in null_forces}
    samples = {str(f): {} for f in null_forces}
    for step_idx, step_dict in enumerate(parents_list):
        child_var   = step_dict["child_var"]
        parent_vars = step_dict["parent_vars"]
        betas = step_dict["betas"]

        for f in null_forces:
            fstr = str(f)
            cond_means[fstr][child_var] = betas[0]
            for var_idx, varname in enumerate(parent_vars):
                cond_means[fstr][child_var] += betas[var_idx+1] * cond_means[fstr][varname]
            samples[fstr][child_var] = norm.rvs(
                size=mc_evals,
                loc=cond_means[fstr][child_var],
                scale=step_dict["std"],
            )

    pvals = []
    for null_idx, force_null in enumerate(null_forces):
        fstr = str(force_null)

        # compute test statistic distribution under null samples
        for step_idx, step_dict in enumerate(parents_list):
            child_var   = step_dict["child_var"]
            parent_vars = step_dict["parent_vars"]
            betas = step_dict["betas"]

            # collect downstream observation
            da_list = data_dict[child_var]["da_list"][alt_idx]
            obs = calc_avg_values(da_list)
            samp_obs = np.array([[np.mean(obs)]], dtype=np.float64)

            # retrieve samples under null
            samps = samples[fstr][child_var]

            # compute means at alternative and null forcings
            mu_alt  = betas[0]
            mu_null = betas[0]
            for var_idx, varname in enumerate(parent_vars):
                if varname == force_var:
                    cond_mean_alt  = force_observed
                    cond_mean_null = force_null
                else:
                    cond_mean_alt = np.mean(calc_avg_values(data_dict[varname]["da_list"][alt_idx]))
                    # cond_mean_alt  = cond_means[fstr][varname]
                    cond_mean_null = cond_means[fstr][varname]
                mu_alt  += betas[var_idx+1] * cond_mean_alt
                mu_null += betas[var_idx+1] * cond_mean_null

            # compute null log likelihood under alternative forcing
            ll_alt     = ll(samps,    mu_alt, step_dict["std"])
            ll_alt_obs = ll(samp_obs, mu_alt, step_dict["std"])

            # compute null log likelihood under null forcing
            ll_null     = ll(samps,    mu_null, step_dict["std"])
            ll_null_obs = ll(samp_obs, mu_null, step_dict["std"])

            if step_idx == 0:
                ll_alt_tot      = ll_alt.copy()
                ll_null_tot     = ll_null.copy()
                ll_alt_obs_tot  = ll_alt_obs.copy()
                ll_null_obs_tot = ll_null_obs.copy()
            else:
                ll_alt_tot      += ll_alt
                ll_null_tot     += ll_null
                ll_alt_obs_tot  += ll_alt_obs
                ll_null_obs_tot += ll_null_obs

        test_vals_null = ll_alt_tot      - ll_null_tot
        test_val_obs   = (ll_alt_obs_tot - ll_null_obs_tot)[0]

        pval = np.mean(test_vals_null > test_val_obs)
        print(f"{force_null}: {pval}")
        pvals.append(pval)

    # breakpoint()

    return np.array(pvals, dtype=np.float64)