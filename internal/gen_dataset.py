import os

import xarray as xr

from clderapp.pythonDataScripts.data_utils import transform_dataset
from constants import DATADIR_BASE, FORCELIST, ENSLIST, VARNAMES
from constants import SPACETIMES, TIMEBOUNDS, REGIONBOUNDS

# ----- START USER INPUTS -----

basedir = "<...>"
datadir =  "limvar_{mass}Tg/v2.LR.WCYCL20TR.pmcpu.limvar.ens{ens}"

freq = "monthly"
basefile = "{varname}_199106_199805.nc"
postpath = f"post/atm/180x360_aave/ts/{freq}/8yr"

enslist_offset = 6

# ----- END USER INPUTS -----

anom_file = os.path.join(DATADIR_BASE, "../meanstates", "{varname}" + f"_{freq}_meanstate.nc")

nvars = len(VARNAMES)
nforce = len(FORCELIST)

if freq == "monthly":
    from_monthly = True
else:
    from_monthly = False

area = None
area_lat = None

for var_idx, varname in enumerate(VARNAMES):
    print(f"Variable {varname}")
    varfile = basefile.format(varname=varname)

    anom_varfile = anom_file.format(varname=varname)
    da_anom = xr.open_dataset(anom_varfile)

    for region, period in SPACETIMES:
        casename = f"{region}-{period}"
        print(casename)
        outdir = os.path.join(DATADIR_BASE, f"data-{casename}")

        latbounds = REGIONBOUNDS[region]["latbounds"]
        lonbounds = REGIONBOUNDS[region]["lonbounds"]
        timebounds = TIMEBOUNDS[period]["timebounds"]

        da_anom_clip = transform_dataset(
            varname,
            "latlon",
            data_in=da_anom,
            latbounds=latbounds,
            lonbounds=lonbounds,
            timebounds=timebounds,
        )

        # load Pinatubo data
        for mass_idx, mass in enumerate(FORCELIST):
            print(f"Ensemble {mass_idx+1}/{nforce}")

            for ens_idx, ens in enumerate(ENSLIST):
                print(f"Member {ens+enslist_offset} -> {ens_idx+1}")

                datadir_mass = datadir.format(mass=mass, ens=ens+enslist_offset)
                if mass == 0:
                    datadir_mass += ".cf"

                # load data
                datapath = os.path.join(
                    basedir,
                    datadir_mass,
                    postpath,
                    varfile,
                )

                ds_in = xr.open_dataset(datapath)
                if area is None:
                    area = ds_in["area"]

                # clip
                da_in = transform_dataset(
                    varname,
                    "latlon",
                    data_in=ds_in,
                    latbounds=latbounds,
                    lonbounds=lonbounds,
                    timebounds=timebounds,
                )

                # anomalize
                da_in = da_in - da_anom_clip

                # global mean
                da_in_global = transform_dataset(
                    varname,
                    "global",
                    data_in=da_in,
                    area=area,
                )

                if freq == "daily":
                    da_in_global = da_in_global.rolling(
                        time=30, min_periods=int(15 / 2), center=True
                    ).mean("time")

                outfile_glob  = os.path.join(outdir, f"{varname}_{mass}Tg_ens{ens_idx+1}_avg_{freq}.nc")
                da_in_global.to_netcdf(outfile_glob)

