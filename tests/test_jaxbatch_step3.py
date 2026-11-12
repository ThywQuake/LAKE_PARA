"""Self-consistency of the synthetic case-3 JAX step.

There is no stable Fortran case-3 reference (zero water depth segfaults
``BATHYMETRY``); the reference here is the synthetic NumPy sequence in
:mod:`lakematrix.step3`.  Three steps are compared field by field.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.forcing import read_ascii  # noqa: E402
from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.jaxbatch.step2 import from_numpy_state, make_params  # noqa: E402
from lakematrix.jaxbatch.step3 import step_case3_jax  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step3 import build_case3_state, step_case3  # noqa: E402

CASE = ROOT / "ref" / "v1" / "summer1d"
ATOL = 1e-10


def test_case3_step_self_consistency_three_steps():
    case = read_case(CASE)
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(CASE / "meteo" / "test.dat")
    ss = build_case3_state(case, grid)
    state = from_numpy_state(ss, ss.ice)
    params = make_params(case, grid)

    for nstep in (2, 3):
        ref = step_case3(ss, case, grid, rows, nstep)
        state = step_case3_jax(state, params, rows, nstep)
        np.testing.assert_allclose(np.asarray(state["ice"]["Ti1"])[0, 1:],
                                   ref["Ti1"], rtol=0, atol=ATOL)
        np.testing.assert_allclose(np.asarray(state["soil"]["Tsoil1"])[0, 1:, 1:],
                                   ref["Tsoil1"], rtol=0, atol=ATOL)
        np.testing.assert_allclose(float(np.asarray(state["ice"]["l1"])[0]),
                                   ref["l1"], rtol=0, atol=ATOL)
        np.testing.assert_allclose(
            float(np.asarray(state["water"]["Tsurf_prev"])[0]),
            ref["Tsurf3"], rtol=0, atol=ATOL)
        assert np.all(np.isfinite(np.asarray(state["ice"]["Ti1"])))
