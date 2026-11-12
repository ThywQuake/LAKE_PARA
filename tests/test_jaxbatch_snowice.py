"""Tests for the JAX ice/snow helpers."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_ice_state, build_step_state  # noqa: E402


def _batched(st, L):
    out = {}
    for k, v in vars(st).items():
        a = np.asarray(v)
        out[k] = np.repeat(a[None, ...], L, axis=0) if a.ndim else np.full(L, a)
    return out


def _case():
    case = read_case(ROOT / "ref/v1/summer1d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    ss = build_step_state(case, grid)
    st = build_ice_state(ss, case, grid)
    st.l1 = 0.2
    st.h1 = 22.46
    st.Ti1[1:] = np.linspace(-4.0, 0.0, case.Mice + 1)
    st.Ti2[1:] = st.Ti1[1:]
    st.Tis1[1:] = 0.0
    st.Tis2[1:] = 0.0
    return case, st


def test_ice_properties_match_numpy():
    from lakematrix.jaxbatch.snowice import ice_properties
    from lakematrix.physics.ice import ice_properties as np_ip

    case, st = _case()
    L = 4
    js = ice_properties(_batched(st, L))
    import copy
    ns = copy.deepcopy(st)
    np_ip(ns)
    np.testing.assert_allclose(np.asarray(js["ci_m_roi_v"])[0], np.asarray(ns.ci_m_roi_v),
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(js["lami_v"])[0], np.asarray(ns.lami_v),
                               rtol=0, atol=0)


def test_ice_increments_match_numpy():
    from lakematrix.jaxbatch.snowice import ice_increments
    from lakematrix.physics.ice import ice_increments as np_ii

    case, st = _case()
    L = 4
    # no snow: use the no-snow variant
    from lakematrix.jaxbatch.snowice import ice_increments_nosnow
    from lakematrix.physics.ice import ice_increments_nosnow as np_iin
    js = ice_increments_nosnow(_batched(st, L), 20.0, 0.0, -1.0, 0.0)
    import copy
    ns = copy.deepcopy(st)
    ref = np_iin(ns, 20.0, precip=0.0, tempair=-1.0, Elatent=0.0)
    for k in ("dhi", "dhi0", "dhw", "dhw0", "dhi0", "dhilow", "dhwhigh"):
        np.testing.assert_allclose(np.asarray(js[k])[0], ref[k], rtol=0, atol=1e-12)
