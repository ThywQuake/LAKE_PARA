"""M3 tests: salinity diffusion and the case-1 layer update."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.physics.layers import advance_case1  # noqa: E402
from lakematrix.physics.salinity import s_diff  # noqa: E402


def _profile(M, val=0.0):
    x = np.zeros(M + 2)
    x[1:] = val
    return x


def test_s_diff_zero_salinity_is_zero():
    M = 4
    Sal1 = _profile(M, 0.0)
    area = np.ones(M + 2) * 100.0
    lam = np.ones(M + 1) * 1e-9
    ddz = np.full(M + 1, 0.25)
    dzeta = np.linspace(0, 1, M + 2)
    ddz05 = np.full(M + 1, 0.25)
    out = s_diff(Sal1, 20.0, 0.0, 0.0, 0.0, 0.0, lam, area, area, ddz,
                 10.0, dzeta, ddz05, M)
    np.testing.assert_array_equal(out, np.zeros(M + 2))


def test_advance_case1_thickness():
    from types import SimpleNamespace
    st = SimpleNamespace(h1=np.array([22.66]), ls1=np.array([0.0]))
    d = advance_case1(st, np.array([0.01]))
    assert d["h1"][0] == 22.67
    assert d["l1"][0] == 0.0
    assert d["hs1"][0] == 0.0
