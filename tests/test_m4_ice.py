"""M4 smoke test: ice state + T_DIFF ice branch."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_ice_state, build_step_state  # noqa: E402


def test_ice_tdiff_smoke():
    case = read_case(ROOT / "ref" / "v1" / "summer1d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    ss = build_step_state(case, grid)
    st = build_ice_state(ss, case, grid)
    st.l1 = 0.5
    st.h1 = 22.16
    st.Ti1[1:] = np.linspace(-5.0, 0.0, case.Mice + 1)
    st.Ti2[1:] = st.Ti1[1:]
    from lakematrix.physics.ice import t_diff_ice
    st = t_diff_ice(1, -2.0, 20.0, 0, 1, 1, 0, case.nsoilcols, False, st)
    assert np.all(np.isfinite(st.Ti2))
    assert np.all(np.isfinite(st.Tw2))
    assert st.Ti2[1] > -10.0
