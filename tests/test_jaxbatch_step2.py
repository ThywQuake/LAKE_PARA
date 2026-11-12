"""B5 case-2: JAX step vs NumPy step_case2 (synthetic ice states)."""

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.forcing import read_ascii  # noqa: E402
from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.physics.ice import ice_properties, make_ice_state  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state  # noqa: E402
from lakematrix.step2 import step_case2  # noqa: E402


def _case_snow(snow):
    case = read_case(ROOT / "ref/v1/summer1d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(ROOT / "ref/v1/summer1d/meteo/test.dat")
    ss = build_step_state(case, grid)
    st = ss.state
    st.l1[0] = 0.2
    st.h1[0] = 22.46
    st.Ti1[0] = np.linspace(-4.0, 0.0, case.Mice + 1)
    st.Tis1[0] = 0.0
    ss.ice = make_ice_state(
        case.M, case.Mice, case.ns, 100, geom=ss.soil_geom,
        isoilcol=case.nsoilcols,
        nmeltpoint=int(np.atleast_1d(case.nmeltpoint)[0]),
        soilbotflx=float(np.atleast_1d(case.soilbotflx)[0]))
    ice = ss.ice
    ice.l1 = 0.2
    ice.h1 = 22.46
    ice.Ti1[1:] = st.Ti1[0]
    ice.Ti2[1:] = st.Ti1[0]
    ice.Tis1[1:] = 0.0
    ice.Tis2[1:] = 0.0
    ice.Tw1[1:] = st.Tw1[0]
    ice.lamw[1:] = st.lamw[0]
    ice.Tsoil1 = ss.soil.Tsoil1
    ice.Tsoil2 = ss.soil.Tsoil2
    ice.flag_snow = int(snow)
    ice.flag_snow_init = 0
    if snow:
        st.hs1[0] = 0.03
        st.snowmass[0] = 2.0
        ice.hs1 = 0.03
        ice.snowmass = 2.0
        ice.itop = 98
        for i in (98, 99):
            ice.dens[i] = 100.0
            ice.dz[i] = 0.015
            ice.Tsn[i] = -5.0
            ice.T[i] = -5.0
            ice.cs[i] = 2000.0
            ice.lams[i] = 0.1
    ice_properties(ice)
    return case, grid, rows, ss


def _run(snow, nsteps=1):
    from lakematrix.jaxbatch.step2 import (from_numpy_state, make_params,
                                           run_case2_jax)
    case, grid, rows, ss = _case_snow(snow)
    params = make_params(case, grid)
    state = from_numpy_state(ss, ss.ice)
    ref = copy.deepcopy(ss)
    for n in range(2, 2 + nsteps):
        r = step_case2(ref, case, grid, rows, n, 0)
        state = run_case2_jax(params, rows, 1, state=state, nstep0=n)
        np.testing.assert_allclose(np.asarray(state["water"]["Tw1"])[0],
                                   r["Tw1"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(np.asarray(state["soil"]["Tsoil1"])[0, 1:, 1:],
                                   r["Tsoil1"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(np.asarray(state["ice"]["l1"])[0],
                                   r["l1"], rtol=0, atol=1e-10)
        # advance the NumPy reference exactly like run_case2
        st = ref.state
        st.Tw1[0] = r["Tw1"]; st.Sal1[0] = r["Sal1"]
        st.Tsoil1[0] = r["Tsoil1"]; st.Ti1[0] = r["Ti1"]
        st.Tis1[0] = r["Tis1"]; st.h1[0] = r["h1"]; st.l1[0] = r["l1"]
        st.ls1[0] = r["ls1"]; st.hs1[0] = r["hs1"]
        st.snowmass[0] = r["snowmass"]; st.E[0] = r["E1"]
        st.eps[0] = r["eps1"]; st.lamw[0] = r["lamw"]
        ref.Elatent = r["Elatent"]; ref.soil.dhwfsoil = r["dhwfsoil"]


def test_case2_no_snow_one_step():
    _run(False, 1)


def test_case2_snow_one_step():
    _run(True, 1)


def test_case2_no_snow_three_steps():
    _run(False, 3)
