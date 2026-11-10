"""M2 test: the ported SfcFlx_momsenlat against the Fortran probe.

The probe records one `SFC_in`/`SFC_out` pair per call; at step 1 the routine
is called 49 times.  `read_probe_sequence` keeps the file order so the pairs
can be matched.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from lakematrix.physics.sfcflx import (SfcFlx_momsenlat, SfcFlx_rhoair,  # noqa: E402
                                       SfcFlx_satwvpres, SfcFlx_spechum)

PROBE = ROOT / "scratch" / "m0.4" / "thermo" / "probe-sfc" / "probe" / "1"
N_IN, N_OUT = 9, 4


@pytest.mark.skipif(not (PROBE / "SFC_in.bin").exists(), reason="SfcFlx probe not generated")
def test_sfcflx_matches_fortran():
    from probe import read_probe_sequence

    ins = read_probe_sequence(PROBE / "SFC_in.bin")
    outs = read_probe_sequence(PROBE / "SFC_out.bin")
    assert len(ins) % N_IN == 0 and len(outs) % N_OUT == 0
    n = len(ins) // N_IN
    assert n == len(outs) // N_OUT and n > 0
    for k in range(n):
        args = [float(ins[k * N_IN + i][1]) for i in range(N_IN)]
        ref = [float(outs[k * N_OUT + i][1]) for i in range(N_OUT)]
        got = SfcFlx_momsenlat(*args)
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-10)


def test_sfcflx_smoke():
    assert SfcFlx_satwvpres(273.15, 0.0) > 0
    assert 0.0 < SfcFlx_spechum(611.0, 100000.0) < 0.01
    assert SfcFlx_rhoair(280.0, 0.005, 100000.0) > 0
