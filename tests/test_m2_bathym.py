"""M2 test: bathymetry against the Fortran `BATHYMETRY` probe."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.physics.bathym import bathymetry  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.state import init_state  # noqa: E402

CASE = ROOT / "ref" / "v1" / "full425d"
PROBE = ROOT / "scratch" / "m0.4" / "thermo" / "probe-bathym" / "probe" / "1" / "BATHY_902_out.bin"


@pytest.mark.skipif(not (CASE.exists() and PROBE.exists()), reason="bathymetry probe not generated")
def test_bathymetry_matches_fortran():
    from probe import read_probe

    p = read_case(CASE)
    g = create_grid(p.M, p.Mice, p.ns, p.d_surf, p.d_bot, p.soil_depth, 0.02)
    s = init_state(p, g)
    q = read_probe(PROBE)
    b = bathymetry(p, g, s, float(q["preswat"][0]))
    np.testing.assert_allclose(b.z_full[0], q["z_full"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(b.preswat[0], q["preswat"], rtol=0, atol=1e-2)
    np.testing.assert_allclose(b.area_int[0], q["area_int"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(b.area_half[0], q["area_half"][:p.M], rtol=0, atol=1e-6)
    np.testing.assert_allclose(b.vol[0], q["vol"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(b.botar[0], q["botar"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(b.Lx[0], q["Lx"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(b.Ly[0], q["Ly"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(b.rad_int[0], q["rad_int"], rtol=0, atol=1e-9)
