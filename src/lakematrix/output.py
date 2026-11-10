"""NetCDF output for the lakematrix case-1 runner."""

from pathlib import Path

import numpy as np
import xarray as xr


def save_profiles(path, times, z, Tw1, Tsoil1=None, h1=None, attrs=None):
    """Write profile time series to a NetCDF file.

    ``Tw1`` has shape (ntime, nz); ``Tsoil1`` (ntime, ns, nsoilcols);
    ``times`` are seconds from the start.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"water_temp": (("time", "z"), np.asarray(Tw1, dtype=float))},
        coords={"time": np.asarray(times, dtype=float) / 86400.0,
                "z": np.asarray(z, dtype=float)})
    if Tsoil1 is not None:
        ds["soil_temp"] = (("time", "zsoil", "soilcol"),
                           np.asarray(Tsoil1, dtype=float))
    if h1 is not None:
        ds["h1"] = ("time", np.asarray(h1, dtype=float))
    ds.attrs.update(attrs or {})
    ds.to_netcdf(path)
    return path


def read_profiles(path):
    return xr.open_dataset(path)
