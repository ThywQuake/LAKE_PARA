"""B3a: JAX batched SOIL_COND_HEAT_COEF matches the NumPy reference."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state  # noqa: E402


def test_jax_soil_cond_heat_coef():
    from lakematrix.jaxbatch.soil import soil_cond_heat_coef as jax_s
    from lakematrix.physics.soil import soil_cond_heat_coef as np_s

    case = read_case(ROOT / "ref/v1/summer60d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    ss = build_step_state(case, grid)
    sp = ss.soil_params
    is_ = case.nsoilcols
    L = 4
    wl = np.repeat(ss.soil.wl1[None, ...], L, axis=0)
    wi = np.repeat(ss.soil.wi1[None, ...], L, axis=0)
    out = jax_s(wl, wi, sp.POR, sp.rosdry, sp.dzs, sp.BH, sp.DLMAX,
                sp.FLWMAX, case.ns, is_)
    ref_state = build_step_state(case, grid)
    np_s(ref_state.soil, sp, is_=is_)
    for arr, name in zip(out, ("csoil", "rosoil", "lamsoil", "lammoist",
                               "filtr", "wsoil")):
        np.testing.assert_allclose(np.asarray(arr)[0],
                                   getattr(ref_state.soil, name),
                                   rtol=0, atol=0)
