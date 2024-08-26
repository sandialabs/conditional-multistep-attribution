# Conditional multi-step attribution for climate forcings

This directory contains data and Python scripts for reproducing results presented in the paper titled "Conditional multi-step attribution for climate forcings" by Christopher R. Wentland, Michael Weylandt, Laura P. Swiler, Thomas S. Ehrmann, and Diana Bull. The scripts are released under SCR#3080.


## Data

Global average impact (centered about counterfactual ensemble mean) time series data files are included in the `data/` directory. These are derived from the "CLDERA E3SM-SPA Simulation Ensembles" dataset, which are released under SCR#3051. These span a range of climate variables, Mt. Pinatubo stratospheric SO2 injection mass, and ensemble member numbers. The general format of each file follows

```
{variable}_{mass}Tg_ens{member_num}_glob_monthly.nc
```

These data files are formatted as E3SMv2 NetCDF files, and are processed here with the `netcdf4` and `xarray` Python packages.


## Requirements

These scripts have only been tested with Python 3.8.19, and the package versions specified in `requirements.txt`. To guarantee version satisfaction, a fresh `conda` environment is recommended. For example,

```
conda config --add channels conda-forge
conda create --name multistep-attrib python=3.8 --file requirements.txt
conda activate multistep-attrib
```

Alternatively, one may attempt to simply satisfy package version requirements with `pip` via

```
pip3 install -r ./requirements.txt
```

## Running scripts

There are four scripts for generating plots: `plot_time_series.py`, `plot_ovl_contours.py`, `plot_regressions.py`, and `plot_posteriors.py`. Utility functions are held in `utils.py`, and `constants.py` contains global data and plotting parameters.

All scripts are assumed to be executed from this root directory, and generate images in a `figs/` directory. They can be run by, for example,

```
python3 plot_posteriors.py
```

