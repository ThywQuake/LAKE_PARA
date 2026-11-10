"""M3 test: SOILFORLAKE against the step-1 probe."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.physics.soil import soilforlake  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state  # noqa: E402

PROBE = ROOT / "scratch" / "m0.4" / "thermo" / "probe-soilf" / "probe" / "1"


def _p1(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x) + 1)
    out[1:] = x
    return out


def _p2(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros((x.shape[0] + 1, x.shape[1] + 1))
    out[1:, 1:] = x
    return out


@pytest.mark.skipif(not (PROBE / "SOILF_in.bin").exists(), reason="SOILFORLAKE probe not generated")
def test_soilforlake_matches_probe():
    from probe import read_probe_sequence

    i = {n: v for n, v in read_probe_sequence(PROBE / "SOILF_in.bin")}
    o = {n: v for n, v in read_probe_sequence(PROBE / "SOILF_out.bin")}
    case = read_case(ROOT / "ref" / "v1" / "full425d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    ss = build_step_state(case, grid)
    s = ss.soil
    for n in ("Tsoil1", "Tsoil3", "wl1", "wi1", "wl4", "wi2", "Sals1"):
        setattr(s, n, _p2(i[n]))
    for n in ("Tsoil2", "csoil", "rosoil", "lamsoil"):
        setattr(s, n, _p1(i[n]))
    s.dhwfsoil = float(i["dhwfsoil"])
    s.h1, s.l1, s.ls1 = float(i["h1"]), float(i["l1"]), float(i["ls1"])
    s.Tw1 = _p1(i["Tw1"])
    s = soilforlake(s, ss.soil_params, 20.0, is_=5)
    for n in ("Tsoil1", "Tsoil3", "wl1", "wi1", "wl4", "wi2"):
        np.testing.assert_allclose(getattr(s, n)[1:, 1:], o[n], rtol=0, atol=1e-12)
    assert s.dhwfsoil == pytest.approx(float(o["dhwfsoil"]), abs=1e-15)
