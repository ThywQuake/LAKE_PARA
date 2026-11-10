"""M3 test: the case-1 `T_DIFF` water+soil solve against the Fortran probe."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "ref"))

from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.physics.tsolver import t_diff_water  # noqa: E402

PROBE = ROOT / "scratch" / "m0.4" / "thermo" / "probe-td2" / "probe" / "1"
M, NS, ISOIL = 40, 10, 5


def _pad1(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape[:-1] + (x.shape[-1] + 1,))
    out[..., 1:] = x
    return out


@pytest.mark.skipif(not (PROBE / "TD_in.bin").exists(), reason="T_DIFF probe not generated")
def test_tdiff_water_matches_fortran():
    from probe import read_probe_sequence

    ins = read_probe_sequence(PROBE / "TD_in.bin")
    outs = read_probe_sequence(PROBE / "TD_out.bin")
    names = [n for n, _ in ins[:19]]
    assert len(ins) % 19 == 0 and len(outs) % 2 == 0
    n = len(ins) // 19
    grid = create_grid(M, 10, NS, 1e-2, 1e-2, 10.0, 0.02)
    for k in range(n):
        a = {names[i]: ins[k * 19 + i][1] for i in range(19)}
        Tsoil1 = np.zeros((NS + 1, 6))
        Tsoil1[1:, 1:] = a["Tsoil1"]
        Tw2, Tsoil2 = t_diff_water(
            float(a["Tsurf"]), float(a["dt"]), _pad1(a["Tw1"]), Tsoil1,
            _pad1(a["lamw"]), _pad1(a["area_int"]), _pad1(a["area_half"]),
            float(a["h1"]), _pad1(a["ddz"]), float(a["dhw"]), float(a["dhw0"]),
            _pad1(a["PEMF"]), _pad1(a["pt_down_f"]), _pad1(a["lamsoil"]),
            _pad1(a["csoil"]), _pad1(a["rosoil"]), _pad1(a["dzs"]),
            a["RadWater_integr"], _pad1(grid.dzeta_int), grid.ddz05,
            M, NS, ISOIL, lsh_water=_pad1(a["lsh_water"]))
        np.testing.assert_allclose(Tw2, outs[k * 2][1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(Tsoil2, outs[k * 2 + 1][1], rtol=0, atol=1e-12)
