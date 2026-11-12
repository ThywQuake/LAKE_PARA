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


def _depth_profile(profile, z_l, depth):
    out = np.full(len(depth), np.nan)
    ok = depth <= z_l[-1]
    if np.any(ok):
        out[ok] = np.interp(depth[ok], z_l, profile)
    return out


def run_batch(case_dirs, days, meteo_paths=None, callback=None,
              out_depth=None, ckpt=None, resume=False):
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
    nsteps = int(round(days * 86400.0 / cases[0].dt))
    every = int(round(86400.0 / cases[0].dt))
    start = 1
    times, rec = [], []
    if ckpt is not None and resume and Path(ckpt).exists():
        import pickle
        with Path(ckpt).open("rb") as fh:
            ck = pickle.load(fh)
        models, start, times, rec = ck["models"], ck["n"] + 1, ck["times"], ck["rec"]
    else:
        models = [build_step_state(c, g) for c, g in zip(cases, grids)]

    def _checkpoint(n):
        if ckpt is None:
            return
        import pickle
        p = Path(ckpt)
        tmp = p.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(dict(models=models, n=n, times=times, rec=rec), fh,
                        protocol=4)
        import os
        os.replace(tmp, p)

    for n in range(start, nsteps + 1):
        for l, (case, grid, ss) in enumerate(zip(cases, grids, models)):
            step_any(ss, case, grid, rows[l], n, l=0, nstep0=1)
        if n % every == 0 or n == nsteps:
            st = [ss.state for ss in models]
            times.append(n * cases[0].dt / 86400.0)
            if out_depth is None:
                wz = [np.stack([s.Tw1[0] for s in st])]
                uz = [np.stack([np.asarray(ss.mom.u1) for ss in models])]
                vz = [np.stack([np.asarray(ss.mom.v1) for ss in models])]
                ez = [np.stack([s.E[0] for s in st])]
                qz = [np.stack([s.eps[0] for s in st])]
            else:
                def prof(get):
                    return np.stack([
                        _depth_profile(get(ss, s),
                                       g.dzeta_int * float(s.h1[0]), out_depth)
                        for ss, s, g in zip(models, st, grids)])
                wz = [prof(lambda ss, s: s.Tw1[0])]
                uz = [prof(lambda ss, s: np.asarray(ss.mom.u1))]
                vz = [prof(lambda ss, s: np.asarray(ss.mom.v1))]
                ez = [prof(lambda ss, s: s.E[0])]
                qz = [prof(lambda ss, s: s.eps[0])]
            rec.append(dict(
                water_temp=wz[0],
                soil_temp=np.stack([s.Tsoil1[0] for s in st]),
                ice_temp=np.stack([s.Ti1[0] for s in st]),
                h_water=np.array([float(s.h1[0]) for s in st]),
                h_ice=np.array([float(s.l1[0]) for s in st]),
                h_snow=np.array([float(s.hs1[0]) for s in st]),
                snowmass=np.array([float(s.snowmass[0]) for s in st]),
                layer_case=np.array([int(s.layer_case[0]) for s in st]),
                u=uz[0], v=vz[0], tke=ez[0], epsilon=qz[0],
                t_surface=np.array([float(ss.Tsurf_prev) for ss in models]),
            ))
            _checkpoint(n)
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
        u=_stack("u"), v=_stack("v"), tke=_stack("tke"),
        epsilon=_stack("epsilon"), t_surface=_stack("t_surface"),
    )


def _run_one(args):
    case, days, meteo = args
    return run_batch([case], days, meteo_paths=[meteo] if meteo else None)


def run_batch_parallel(case_dirs, days, meteo_paths=None, jobs=4):
    """Run one lake per process and stack the results (M5 throughput path)."""
    from concurrent.futures import ProcessPoolExecutor

    case_dirs = [Path(p) for p in case_dirs]
    met = meteo_paths or [None] * len(case_dirs)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        parts = list(pool.map(_run_one, zip(case_dirs, [days] * len(case_dirs), met)))
    out = dict(parts[0])
    # every field except the shared coordinates has a leading lake axis
    for key in out:
        if key not in ("time", "z"):
            out[key] = np.concatenate([p[key] for p in parts], axis=0)
    return out


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
        "u": (("lake", "time", "z"), data["u"]),
        "v": (("lake", "time", "z"), data["v"]),
        "tke": (("lake", "time", "z"), data["tke"]),
        "epsilon": (("lake", "time", "z"), data["epsilon"]),
        "t_surface": (("lake", "time"), data["t_surface"]),
    }, coords={"lake": np.arange(data["water_temp"].shape[0]),
               "time": data["time"], "z": data["z"],
               "zsoil": np.arange(nzsoil), "soilcol": np.arange(nc),
               "zice": np.arange(mice)})
    if names is not None:
        ds = ds.assign_coords(lake_name=("lake", [Path(n).name for n in names]))
    ds.to_netcdf(path)
    return path
