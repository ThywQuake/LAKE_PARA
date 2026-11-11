"""B2: JAX batched T_SOLVER (surface + T_DIFF) matches the NumPy reference."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.forcing import read_ascii  # noqa: E402
from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state, step_case1_to_solver  # noqa: E402


def test_jax_batched_t_solver():
    from lakematrix.jaxbatch.tsolver import make_solver
    from lakematrix.physics.surface import heatbalance_water as np_hb
    from lakematrix.physics.tsolver import t_solver as np_ts

    case = read_case(ROOT / "ref/v1/summer60d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(ROOT / "ref/v1/summer60d/meteo/test.dat")
    ss = build_step_state(case, grid)
    ctx = step_case1_to_solver(ss, case, grid, rows, 1, 0,
                               init_step=True)["ctx"]
    L = 4
    ctxj = {k: (np.repeat(np.asarray(v)[None, ...], L, axis=0)
                if isinstance(v, np.ndarray) else v)
            for k, v in ctx.__dict__.items()}
    for k in ("h1", "dhw", "dhw0"):
        ctxj[k] = np.full(L, getattr(ctx, k))
    solver = make_solver(1)
    import jax.numpy as jnp
    r = solver(ctxj, jnp.full(L, -999.0), jnp.full(L, 4), jnp.zeros(L, int))
    ref = np_ts(lambda T: np_hb(T, ctx), 1, 4, 0, 0.0)
    np.testing.assert_allclose(np.asarray(r["Tsurf3"]), ref["Tsurf3"],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(r["state"]["Tw2"])[0],
                               ref["state"]["Tw2"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(r["state"]["Tsoil2"])[0],
                               ref["state"]["Tsoil2"], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(np.asarray(r["Tsurf3"]),
                                  np.broadcast_to(np.asarray(r["Tsurf3"])[0],
                                                  (L,)))
