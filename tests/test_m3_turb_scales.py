"""M3 test: TURB_SCALES against the Fortran probe."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from lakematrix.physics.density import set_dens_consts  # noqa: E402
from lakematrix.physics.turb import turb_scales  # noqa: E402

PROBE = ROOT / "scratch" / "m0.4" / "thermo" / "probe-tsc" / "probe" / "1"


@pytest.mark.skipif(not (PROBE / "TSC_in.bin").exists(), reason="TURB_SCALES probe not generated")
def test_turb_scales_matches_fortran():
    from probe import read_probe_sequence

    ins = {n: v for n, v in read_probe_sequence(PROBE / "TSC_in.bin")}
    out = {n: v for n, v in read_probe_sequence(PROBE / "TSC_out.bin")}
    d0, s0, _ = set_dens_consts()
    r = turb_scales(
        ins["row"], ins["ddz"], ins["ddz2"], ins["dzeta_int"],
        ins["dzeta_05int"], float(ins["h1"]), int(ins["M"]), ins["Tw2"],
        ins["Sal2"], ins["u2"], ins["v2"], ins["lamw"], ins["lamsal"],
        ins["k_turb_T_flux"], ins["T_massflux"], float(ins["eflux0_kinem"]),
        float(ins["turb_density_flux"]), float(ins["Buoyancy0"]),
        float(ins["tau"]), float(ins["kor"]), ins["RadWater_integr"],
        ins["area_int"], ins["Lx"], ins["Ly"], float(ins["vol"]),
        ins["z_full"], ins["ddz05"], d0, s0)
    for k, ref in out.items():
        got = np.asarray(r[k])
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12, err_msg=k)
