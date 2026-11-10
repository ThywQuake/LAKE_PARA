"""M3 test: K_EPSILON against the step-1 Fortran probe."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

PROBE = ROOT / "scratch" / "m0.4" / "thermo" / "probe-ke7" / "probe" / "1"


@pytest.mark.skipif(not (PROBE / "KE_out.bin").exists(), reason="K_EPSILON probe not generated")
def test_kepsilon_matches_fortran():
    from probe import read_probe_sequence
    from replay_kepsilon import replay

    r, _ = replay(PROBE)
    out = {n: v for n, v in read_probe_sequence(PROBE / "KE_out.bin")}
    np.testing.assert_allclose(r.e2[1:], out["E2"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(r.eps2[1:], out["eps2"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(r.k2[1:], out["k2"], rtol=0, atol=1e-15)
    np.testing.assert_allclose(r.kt[1:], out["kt"], rtol=0, atol=1e-10)
    assert r.H_entrainment == pytest.approx(float(out["H_entrainment"]), abs=1e-15)
