import os
from copy import deepcopy
from math import erf, log, sqrt, exp
from math import pi

import cftime
import numpy as np
import xarray as xr


def load_glob_data(datadir, datafile_base, varlist, forcelist, enslist):

    data_dict = {}

    # compute for each variable
    for var in varlist:
        print(f"Variable {var}")

        data_dict[var] = {
            "da_list": [],
            "da_mean_list": [],
        }

        # collect ensemble means for each forcing level
        for force in forcelist:
            print(f"Forcing {force}")

            # load global time series for all ensemble members
            da_list = []
            for ens in enslist:
                print(f"Ensemble {ens}")

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


def threshold_array(arr_in, direction):
    if direction == 1:
        arr_out = (arr_in >= 0.0) * arr_in
    elif direction == -1:
        arr_out = (arr_in <= 0.0) * arr_in
    else:
        raise ValueError(f"Unexpected threshold direction: {direction}")

    return arr_out


def calc_peak_values(da_list_in, expected_direction):

    if not isinstance(da_list_in, list):
        da_list = [da_list_in]
    else:
        da_list = da_list_in

    delta_list = []
    peak_idx_list = []
    for da in da_list:
        vals_glob = da.values

        # compute deviation with respect to first data point
        dev_vals = vals_glob - vals_glob[0]

        # expert-informed peak-finding
        dev_vals_exp = threshold_array(dev_vals, expected_direction)

        # find index of peak, store
        peak_idx = np.nanargmax(np.abs(dev_vals_exp))
        peak_idx_list.append(peak_idx)

        # extract peak deviation, store
        delta = dev_vals[peak_idx]
        delta_list.append(delta)

    return delta_list, peak_idx_list


def calc_ovl_binormal(mu_1, mu_2, s_1, s_2):

    # assumes that s_1 < s_2
    # switch if not the case
    if s_2 < s_1:
        temp = s_1
        s_1 = s_2
        s_2 = temp
        temp = mu_1
        mu_1 = mu_2
        mu_2 = temp

    s_1_sq = s_1**2
    s_2_sq = s_2**2

    # compute x_1 and x_2
    a = mu_1 * s_2_sq - mu_2 * s_1_sq
    b = s_1 * s_2 * sqrt( (mu_1 - mu_2)**2 + (s_1_sq - s_2_sq) * log(s_1_sq / s_2_sq) )
    c = s_2_sq - s_1_sq
    x_1 = (a - b) / c
    x_2 = (a + b) / c

    # compute OVL
    ovl = 1.0 + normal_cdf(x_1, mu_1, s_1) - normal_cdf(x_1, mu_2, s_2) - normal_cdf(x_2, mu_1, s_1) + normal_cdf(x_2, mu_2, s_2)

    return ovl


def normal_cdf(x, mu, s):
    return 0.5 * (1.0 + erf((x - mu) / (s * sqrt(2))))


def collect_linregress_data(
    data_dict,
    xvars,
    yvar,
    forcelist,
    enslist,
    expected_directions,
    force_var,
    force_obs,
):

    nens = len(enslist)

    x_deltas_concat = None
    y_deltas_concat = None
    for force_idx, force in enumerate(forcelist):

        # collect peak impacts for predictor variables
        for x_idx, xvar in enumerate(xvars):

            if xvar == force_var:
                # forcing variable is not in the peak impact data
                x_deltas = force * np.ones(nens, dtype=np.float64)
            else:
                x_da_list = data_dict[xvar]["da_list"][force_idx]
                x_deltas, _ = calc_peak_values(x_da_list, expected_directions[xvar])
            x_deltas = np.expand_dims(x_deltas, axis=-1)

            if x_idx == 0:
                x_deltas_tot = x_deltas.copy()
            else:
                x_deltas_tot = np.concatenate((x_deltas_tot, x_deltas), axis=1)

        # peak impacts for target variables
        y_da_list = data_dict[yvar]["da_list"][force_idx]
        y_deltas, _ = calc_peak_values(y_da_list, expected_directions[yvar])
        y_deltas = np.expand_dims(y_deltas, axis=-1)

        # separate "observed" forcing
        if force == force_obs:
            x_deltas_obs = x_deltas_tot.copy()
            y_deltas_obs = y_deltas.copy()
        else:
            # concatenate for later regression
            if x_deltas_concat is None:
                x_deltas_concat = x_deltas_tot.copy()
                y_deltas_concat = y_deltas.copy()
            else:
                x_deltas_concat = np.concatenate((x_deltas_concat, x_deltas_tot), axis=0)
                y_deltas_concat = np.concatenate((y_deltas_concat, y_deltas), axis=0)

    return x_deltas_concat, y_deltas_concat, x_deltas_obs, y_deltas_obs


def calc_ols(Xin, Yin, intercept=False):
    # compute ordinary least squares regression coefficients

    yobs = Yin.shape[0]
    xobs = Xin.shape[0]
    assert yobs == xobs, f"Number of observations do not match: {yobs} v {xobs}"

    if Yin.ndim == 1:
        Yin = Yin[:, None]
    if Xin.ndim == 1:
        Xin = Xin[:, None]

    # add intercept
    if intercept:
        Xin = np.concatenate((np.ones((xobs, 1), dtype=Xin.dtype), Xin), axis=1)

    # compute approximate parameters
    A = Xin.T @ Xin
    B = Xin.T @ Yin
    beta = np.linalg.solve(A, B)

    return beta


def calc_likelihood_normal(betas, var, xvals, yval):
    assert betas.shape[0] == (xvals.shape[0] + 1)
    y_pred = (betas[0] + xvals @ betas[1:]).item()

    exponential = exp(-0.5 * (yval - y_pred)**2 / var)
    likelihood = (1.0 / (sqrt(var * 2 * pi))) * exponential

    return likelihood