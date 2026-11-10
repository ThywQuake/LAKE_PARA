"""M1 tests: initial state, grid and forcing against the Fortran probe.

The probe files are produced by the M0.4 probe build and live under
`lakebatch/scratch/` (git-ignored).  When they are absent the tests skip; the
procedure to regenerate them is in `docs/reference_cases.md` and `NOTES.md`.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LAKEBATCH = ROOT.parent / "lakebatch"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from lakematrix.constants import C  # noqa: E402
from lakematrix.forcing import at_step, read_ascii  # noqa: E402
from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.setup_io import read_case, read_cases  # noqa: E402
from lakematrix.state import init_state  # noqa: E402
from probe import read_probe  # noqa: E402

CASE = ROOT / "ref" / "v1" / "full425d"
PROBE_BASE = ROOT / "scratch" / "m0.4" / "thermo"
PROBE_INIT = PROBE_BASE / "probe-initvar-v9" / "probe" / "0" / "INITVAR.bin"
PROBE_GRID = PROBE_BASE / "probe-step1-v7" / "probe" / "1" / "LAKE_entry.bin"
PROBE_FORCING = PROBE_BASE / "probe-forcing" / "probe"

needs_case = pytest.mark.skipif(not CASE.exists(), reason="reference case not generated")
needs_probe = pytest.mark.skipif(not PROBE_INIT.exists(), reason="probe data not generated")


@pytest.fixture(scope="module")
def params():
    return read_case(CASE)


@pytest.fixture(scope="module")
def grid(params):
    return create_grid(params.M, params.Mice, params.ns,
                       params.d_surf, params.d_bot, params.soil_depth, 0.02)


@needs_case
@needs_probe
def test_init_state(params, grid):
    s = init_state(params, grid)
    p = read_probe(PROBE_INIT)
    np.testing.assert_allclose(s.Tw1[0], p["Tw1"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(s.Sal1[0], p["Sal1"], rtol=0, atol=1e-12)
    # Tsoil1 uses a default-real (single precision) exp() argument in the
    # Fortran side columns; see NOTES.md.  The deepest column additionally
    # carries the zsoil single-precision rounding.
    np.testing.assert_allclose(s.Tsoil1[0], p["Tsoil1"], rtol=0, atol=1e-6)
    assert float(s.h1[0]) == pytest.approx(float(p["h1"]))
    assert float(s.l1[0]) == 0.0
    assert float(s.snowmass[0]) == 0.0


@needs_case
@needs_probe
def test_grid(params, grid):
    p = read_probe(PROBE_GRID)
    np.testing.assert_allclose(grid.ddz, p["ddz"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(grid.ddz05, p["ddz05"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(grid.dzeta_int, p["dzeta_int"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(grid.zsoil, p["zsoil"], rtol=0, atol=1e-6)
    zsc = p["zsoilcols(1:nsoilcols+1,ix,iy)"]
    from lakematrix.grid import zsoilcols
    got = zsoilcols(float(p["h1"]), params.morphometry[0], grid, params.nsoilcols)
    np.testing.assert_allclose(got, zsc, rtol=0, atol=1e-12)


@needs_case
@pytest.mark.skipif(not PROBE_FORCING.exists(), reason="forcing probe not generated")
def test_forcing():
    rows = read_ascii(LAKEBATCH.parent / "LAKE3.2" / "meteo" / "testlake.dat")
    for step in (1, 2, 3, 180):
        f = at_step(rows, step, 20.0, 1.0)
        p = read_probe(PROBE_FORCING / str(step) / "LAKE_entry.bin")
        assert f[3] == pytest.approx(float(p["shortwave"]), abs=1e-12)
        assert f[2] == pytest.approx(float(p["longwave"]), abs=1e-12)
        # tempair in LAKE is Celsius; the ASCII column is kelvin.
        assert f[0] - C.Kelvin0 == pytest.approx(float(p["tempair"]), abs=1e-12)


@needs_case
def test_batch_stack_matches_single(params):
    batch = read_cases([CASE, ROOT / "ref" / "v1" / "cold425d"])
    g = create_grid(batch.M, batch.Mice, batch.ns, batch.d_surf,
                    batch.d_bot, batch.soil_depth, 0.02)
    sb = init_state(batch, g)
    for l, case in enumerate((CASE, ROOT / "ref" / "v1" / "cold425d")):
        p = read_case(case)
        g1 = create_grid(p.M, p.Mice, p.ns, p.d_surf, p.d_bot, p.soil_depth, 0.02)
        s1 = init_state(p, g1)
        np.testing.assert_array_equal(sb.Tw1[l], s1.Tw1[0])
        np.testing.assert_array_equal(sb.Tsoil1[l], s1.Tsoil1[0])
