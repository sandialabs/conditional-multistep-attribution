import os

import xarray as xr

from constants import ENSLIST, VARNAMES, DATADIR_BASE

# ----- START USER INPUTS -----

basedir = "<...>"
datadir =  "limvar_{mass}Tg/v2.LR.WCYCL20TR.pmcpu.limvar.ens{ens}"

freq = "monthly"
basefile = "{varname}_199106_199805.nc"
postpath = f"post/atm/180x360_aave/ts/{freq}/8yr"

ens_offset = 6

# ----- END USER INPUTS -----

outdir = os.path.join(DATADIR_BASE, "../meanstates")
if not os.path.isdir(outdir):
    os.mkdir(outdir)


for var_idx, varname in enumerate(VARNAMES):
    print(f"Variable {varname}")
    varfile = basefile.format(varname=varname)

    ds_list = []
    for ens_idx, ens in enumerate(ENSLIST):
        print(f"Ensemble {ens+ens_offset}")

        datadir_mass = datadir.format(mass=0, ens=ens+ens_offset)
        datadir_mass += ".cf"

        infile = os.path.join(
            basedir,
            datadir_mass,
            postpath,
            varfile,
        )
        ds_in = xr.open_dataset(infile)
        ds_list.append(ds_in.copy(deep=True))

    ds_concat = xr.concat(ds_list, dim="member")
    ds_mean = ds_concat.mean(dim="member")

    outfile = os.path.join(outdir, f"{varname}_{freq}_meanstate.nc")
    print(f"Writing to {outfile}")
    ds_mean.to_netcdf(outfile)
