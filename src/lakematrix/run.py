"""Batch driver: several lakes, one state per lake, combined NetCDF output.

Version 1 of the physics is single-lake (`step2.step_any` takes a lake index
into a `StepState`); this driver advances one `StepState` per lake and stacks
the output.  It is the correctness path used by the CLI and the 4-lake
acceptance test; the fully vectorised JAX runner is the M5 performance task.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from .forcing import read_ascii
from .grid import create_grid
from .setup_io import read_case
from .step2 import step_any


def _grid(case):
    return create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)


def run_batch(case_dirs, days, meteo_paths=None, callback=None,
              out_depth=None):
    """Run several case directories for `days`; return stacked profiles.

    Returns ``dict`` with ``time`` (days), ``z`` (m), and arrays
    ``water_temp``/``soil_temp``/``ice_temp``/``h_water``/``h_ice``/``h_snow``/
    ``snowmass``/``layer_case`` shaped ``(lake, time, ...)``.
    """
    case_dirs = [Path(p) for p in case_dirs]
    cases = [read_case(p) for p in case_dirs]
    meteo_paths = meteo_paths or [p / "meteo" / "test.dat" for p in case_dirs]
    rows = [read_ascii(p) for p in meteo_paths]
    grids = [_grid(c) for c in cases]
    from .step import build_step_state
    models = [build_step_state(c, g) for c, g in zip(cases, grids)]
    nsteps = int(round(days * 86400.0 / cases[0].dt))
    every = int(round(86400.0 / cases[0].dt))
    times, rec = [], []

    for n in range(1, nsteps + 1):
        for l, (case, grid, ss) in enumerate(zip(cases, grids, models)):
            step_any(ss, case, grid, rows[l], n, l=0, nstep0=1)
        if n % every == 0 or n == nsteps:
            st = [ss.state for ss in models]
            times.append(n * cases[0].dt / 86400.0)
            rec.append(dict(
                water_temp=np.stack([s.Tw1[0] for s in st]),
                soil_temp=np.stack([s.Tsoil1[0] for s in st]),
                ice_temp=np.stack([s.Ti1[0] for s in st]),
                h_water=np.array([float(s.h1[0]) for s in st]),
                h_ice=np.array([float(s.l1[0]) for s in st]),
                h_snow=np.array([float(s.hs1[0]) for s in st]),
                snowmass=np.array([float(s.snowmass[0]) for s in st]),
                layer_case=np.array([int(s.layer_case[0]) for s in st]),
            ))
            if callback is not None:
                callback(n, models, rec[-1])

    z = np.asarray(out_depth) if out_depth is not None else (
        grids[0].dzeta_int * float(models[0].state.h1[0]))
    def _stack(key):
        a = np.array([r[key] for r in rec])
        return a.swapaxes(0, 1)  # (time, lake, ...) -> (lake, time, ...)

    return dict(
        time=np.array(times), z=z,
        water_temp=_stack("water_temp"),
        soil_temp=_stack("soil_temp"),
        ice_temp=_stack("ice_temp"),
        h_water=_stack("h_water"),
        h_ice=_stack("h_ice"),
        h_snow=_stack("h_snow"),
        snowmass=_stack("snowmass"),
        layer_case=_stack("layer_case"),
    )


def save_netcdf(path, data, names=None):
    """Write a `(lake, time, ...)` NetCDF file from :func:`run_batch`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nt = len(data["time"])
    nz = len(data["z"])
    nzsoil = data["soil_temp"].shape[-2]
    nc = data["soil_temp"].shape[-1]
    mice = data["ice_temp"].shape[-1]
    ds = xr.Dataset({
        "water_temp": (("lake", "time", "z"), data["water_temp"]),
        "soil_temp": (("lake", "time", "zsoil", "soilcol"), data["soil_temp"]),
        "ice_temp": (("lake", "time", "zice"), data["ice_temp"]),
        "h_water": (("lake", "time"), data["h_water"]),
        "h_ice": (("lake", "time"), data["h_ice"]),
        "h_snow": (("lake", "time"), data["h_snow"]),
        "snowmass": (("lake", "time"), data["snowmass"]),
        "layer_case": (("lake", "time"), data["layer_case"]),
    }, coords={"lake": np.arange(data["water_temp"].shape[0]),
               "time": data["time"], "z": data["z"],
               "zsoil": np.arange(nzsoil), "soilcol": np.arange(nc),
               "zice": np.arange(mice)})
    if names is not None:
        ds = ds.assign_coords(lake_name=("lake", [Path(n).name for n in names]))
    ds.to_netcdf(path)
    return path
