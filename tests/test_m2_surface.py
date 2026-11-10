"""M2 tests: RADBALANCE and the water branch of HEATFLUXSURF."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.constants import C  # noqa: E402
from lakematrix.physics.surface import (WATER, heatfluxsurf_water,  # noqa: E402
                                        radbalance)


def test_radbalance_water_skin0():
    T = 20.0
    surfrad = 0.98 * C.sigma * (T + C.Kelvin0) ** 4
    rb, rbs, sb, sr, swup = radbalance(
        T, [500.0, 450.0], [0.0, 0.0], WATER,
        longwave=350.0, shortwave=0.0, netrad=-1.0, cloud=-1.0,
        humair=0.01, pressure=100000.0, tempair=290.0,
        albedo=0.06, albedo_lw=0.02, emissivity=0.98)
    np.testing.assert_allclose(sr, surfrad)
    np.testing.assert_allclose(rb, 50.0 + 350.0 * 0.98 - surfrad)
    np.testing.assert_allclose(rbs, 350.0 * 0.98 - surfrad)
    np.testing.assert_allclose(sb, 50.0)
    np.testing.assert_allclose(swup, 0.0)


def test_heatfluxsurf_water_linear_profile():
    L, M = 2, 4
    Tw1 = np.tile(np.array([10.0, 9.0, 8.0, 7.0, 6.0]), (L, 1))
    Tw2 = Tw1 + 0.1
    lamw = np.full((L, M), 0.5)
    area_int = np.tile(np.array([100.0, 80.0, 60.0, 40.0, 20.0]), (L, 1))
    area_half = np.tile(np.array([90.0, 70.0, 50.0, 30.0]), (L, 1))
    h1 = np.full(L, 10.0)
    ddz = np.full(M, 0.25)
    dhw0 = np.zeros(L)
    eflux, dHdt, conv, bot = heatfluxsurf_water(Tw1, Tw2, lamw, area_int, area_half,
                                                h1, ddz, dhw0, 20.0)
    # Top: -0.5*0.5*(90/100)*(-2.0)/(0.25*10) = 0.18
    np.testing.assert_allclose(eflux, 0.18)
    # dHdt: 0.5*(1/20)*3990*1000*(0.25*10*0.1) = 24937.5
    np.testing.assert_allclose(dHdt, 0.5 / 20.0 * 3990 * 1000 * 0.25 * 10 * 0.1)
    np.testing.assert_allclose(conv, 0.0)
    # Bottom: -0.5*0.5*(30/20)*(-2.0)/(0.25*10) = 0.30
    np.testing.assert_allclose(bot, 0.30)


def test_heatbalance_composition_against_probe():
    """The HB_out probe holds all terms; the Fortran composition must match."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "tools" / "ref"))
    from probe import read_probe_sequence

    base = ROOT / "scratch" / "m0.4" / "thermo" / "probe-surf" / "probe" / "1"
    if not (base / "HB_out.bin").exists():
        import pytest
        pytest.skip("HB probe not generated")
    from lakematrix.physics.surface import heatbalance

    seq = read_probe_sequence(base / "HB_out.bin")
    n = 10
    for c in range(len(seq) // n):
        d = {name: float(seq[c * n + i][1]) for i, (name, _) in enumerate(seq[c * n:c * n + n])}
        HB, eflux0, eflux0_kinem = heatbalance(
            d["dHdt"], d["Radbal"], d["Radbal_surf"], d["eflux"],
            d["convect_flux"], d["hflux"], d["Elatent"])
        np.testing.assert_allclose(HB, d["HEATBALANCE"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(eflux0, d["eflux0"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(eflux0_kinem, d["eflux0_kinem"], rtol=0, atol=1e-15)
