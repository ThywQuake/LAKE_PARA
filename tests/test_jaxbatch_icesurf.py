"""JAX batched ice/snow surface (``jaxbatch.ice_surface``) vs NumPy.

Coverage
========

1. ``SURF_CHAR`` ice/snow tables vs :func:`lakematrix.physics.surface.surf_char`
   (static ``surftyp`` and a mixed per-lake batch).
2. ``HEATFLUXSURF`` ice and snow branches vs
   :func:`lakematrix.physics.surface.heatfluxsurf_ice` /
   ``heatfluxsurf_snow``, batched with *different* ``itop`` per lake.
3. ``RADBALANCE`` ice/snow vs
   :func:`lakematrix.physics.surface.radbalance`.
4. One-step ``HEATBALANCE`` vs the NumPy
   :func:`lakematrix.physics.surface.heatbalance_ice`, for a bare-ice and a
   snow-covered state.  The NumPy kernel is wrapped in a NumPy-backed
   ``t_diff_fn`` (it calls the real ``lakematrix.physics.ice.t_diff_ice``), so
   this test does **not** depend on the batched ice ``T_DIFF`` that B5 will
   provide.
5. Traceability / shapes with a synthetic pure-JAX ``t_diff_fn`` stub: ``jit``,
   ``grad``, mixed ice/snow batch selection and batch consistency (one row of
   an L=4 call vs the same problem in an L=1 call, bit-identical on CPU).

Not covered (deliberate)
========================

Full case-2/3 integration - the batched ice ``T_DIFF`` kernel,
``ice_increments``, ``snow_cond_heat_coef`` and ``T_SOLVER`` - is B5 work and
lives outside this module.  The stub in test 5 stands in for the batched
``T_DIFF`` so that ``heatbalance_ice`` is exercised under ``jit``/``grad``
with per-lake ``snow`` flags.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from lakematrix.constants import C  # noqa: E402
from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.physics import surface as NPS  # noqa: E402
from lakematrix.physics.ice import snow_cond_heat_coef  # noqa: E402
from lakematrix.physics.ice import t_diff_ice as np_t_diff_ice  # noqa: E402
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_ice_state, build_step_state  # noqa: E402

CASE = ROOT / "ref" / "v1" / "summer1d"

#: ``ms`` of ``build_ice_state`` (``make_ice_state(..., ms=100)``); the snow
#: arrays are 1-based with a pad, i.e. ``(ms + 1,)``.
MS = 100

#: Synthetic number of ice layers used by the ``HEATFLUXSURF`` unit test.
NICE = 10


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ref():
    """Case + grid + step state of ``ref/v1/summer1d``."""
    case = read_case(CASE)
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    return case, grid, build_step_state(case, grid)


def make_ice(ref, *, l1=0.5, snow=False, tempair=-5.0):
    """Fresh ``IceState`` with a linear surface-ice profile (case-2 setup)."""
    case, grid, ss = ref
    ice = build_ice_state(ss, case, grid)
    ice.l1 = l1
    ice.h1 = 22.16
    ice.Ti1[1:] = np.linspace(-5.0, 0.0, case.Mice + 1)
    ice.Ti2[1:] = ice.Ti1[1:]
    if snow:
        # Initial snow pack of SNOW_COND_HEAT_COEF (itop = ms-2, three 1 cm
        # layers at tempair plus the interface node ms) and its lams/cs.
        ice.flag_snow = 1
        snow_cond_heat_coef(ice, tempair)
        ice.Tsn[:] = ice.T[:MS + 1]
    return ice


def ice_state_dict(ice):
    """The (unbatched) state dict handed to ``heatbalance_ice``.

    Mirrors the ``SNOWSOIL_MOD``/``ARRAYS`` fields that ``HEATFLUXSURF`` reads;
    ``area_int``/``area_half`` are ``BATHYMICE`` arrays with index 1 =
    ``bathymice(1)`` (only that index is read).
    """
    return dict(
        Ti1=ice.Ti1, Ti2=ice.Ti2, T=ice.T, Tsn=ice.Tsn, cs=ice.cs,
        dens=ice.dens, dz=ice.dz, lams=ice.lams, itop=ice.itop,
        lami_v=ice.lami_v, ci_m_roi_v=ice.ci_m_roi_v, l1=ice.l1,
        dhi0=ice.dhi0, ddzi=ice.ddzi,
        area_int=np.r_[0.0, ice.bathymice[1].area_int],
        area_half=np.r_[0.0, ice.bathymice[1].area_half])


_TDIFF_ARGS = ("surf", "Tsurf", "dt", "snow", "ice", "water", "deepice",
               "isoilcol", "ifflux", "state")


def np_t_diff(surf, Tsurf, dt, snow, ice, water, deepice, isoilcol, ifflux,
              state):
    """NumPy-backed ``t_diff_fn``: the real ``t_diff_ice`` on a real IceState.

    Only used eagerly (no ``jit``): ``t_diff_ice`` mutates the ``IceState`` in
    place, so ``state["_ice"]`` is the fresh state prepared by the caller.  The
    returned dict keeps the *unchanged* ``Ti1``/``T`` (read from the input
    state) next to the *new* ``Ti2``/``Tsn``, exactly like the two separate
    Fortran arrays.
    """
    def one(x):
        return np.asarray(x).reshape(-1)[0].item()

    st = np_t_diff_ice(int(one(surf)), float(Tsurf), float(one(dt)),
                       int(one(snow)), int(one(ice)), int(one(water)),
                       int(one(deepice)), int(one(isoilcol)),
                       bool(one(ifflux)), state["_ice"])
    out = {k: v for k, v in state.items() if k != "_ice"}
    out.update(Ti1=st.Ti1, Ti2=st.Ti2, T=st.T, Tsn=st.Tsn)
    return out


_ATMOS = dict(fetch=1000.0, uwind=3.0, vwind=1.0, u=0.05, v=0.0,
              tempair=15.0, pressure=100000.0, humair=0.008, zref=2.0)


def np_ctx(state, sc, *, snow, nsoilcols, radice=(0.0, 0.0), shortwave=200.0,
           longwave=320.0, **extra):
    """The NumPy ``IceSurfaceContext`` of the one-step problem."""
    return NPS.IceSurfaceContext(
        ice=state["_ice"], snow=snow, water=1, dt=20.0,
        roughness=float(sc["roughness"]), relhums=float(sc["relhums"]),
        aM=float(sc["aM"]), bM=float(sc["bM"]), l1=state["_ice"].l1,
        longwave=longwave, shortwave=shortwave, netrad=-1.0, cloud=-1.0,
        albedo=float(sc["albedo"]), albedo_lw=float(sc["albedo_lw"]),
        emissivity=float(sc["emissivity"]), radwater_integr=np.zeros(41),
        radice_integr=np.asarray(radice, dtype=float), isoilcol=nsoilcols,
        **_ATMOS, **extra)


def jax_ctx(state, sc, *, snow, nsoilcols, radice=(0.0, 0.0),
            shortwave=200.0, longwave=320.0):
    """The JAX ``ctx`` dict of the one-step problem (leading lake axis)."""
    st = {k: jnp.asarray(v) for k, v in state.items() if k != "_ice"}
    st["itop"] = jnp.asarray([state["itop"]], dtype=jnp.int32)
    if "_ice" in state:
        st["_ice"] = state["_ice"]      # the NumPy t_diff target, tests only
    ctx = dict(
        snow=jnp.asarray([snow]), water=jnp.asarray([1]),
        ice=jnp.asarray([1]), deepice=jnp.asarray([0]),
        isoilcol=nsoilcols, dt=20.0, state=st,
        longwave=longwave, shortwave=shortwave, netrad=-1.0, cloud=-1.0,
        albedo=jnp.asarray([sc["albedo"]]),
        albedo_lw=jnp.asarray([sc["albedo_lw"]]),
        emissivity=jnp.asarray([sc["emissivity"]]),
        roughness=jnp.asarray([sc["roughness"]]),
        relhums=jnp.asarray([sc["relhums"]]),
        aM=jnp.asarray([sc["aM"]]), bM=jnp.asarray([sc["bM"]]),
        radice_integr=jnp.asarray([radice], dtype=float))
    for key, val in _ATMOS.items():
        ctx[key] = jnp.asarray([val])
    return ctx


# ---------------------------------------------------------------------------
# 0. SfcFlx ice branch (not available in jaxbatch.sfcflx, which is water-only)
# ---------------------------------------------------------------------------
def test_roughness_ice_matches_numpy():
    """``SfcFlx_roughness`` ice branch vs NumPy (both Re regimes)."""
    from lakematrix.jaxbatch.ice_surface import roughness_ice
    from lakematrix.physics.sfcflx import SfcFlx_roughness

    u_star = np.array([1e-5, 1e-4, 1e-3, 0.02, 0.1, 0.4])
    got = jax.jit(roughness_ice)(1000.0, 3.0, u_star)
    for j, us in enumerate(u_star):
        ref = SfcFlx_roughness(1000.0, 3.0, us, 0.5)   # h_ice > 1e-9: ice
        for k in range(5):
            val = np.asarray(got[k])
            val = val[j] if val.ndim else val
            np.testing.assert_allclose(val, ref[k], rtol=1e-15,
                                       err_msg=f"out{k}[{j}]")


def test_senslatmom_fluxes_ice_matches_numpy():
    """The ice ``SfcFlx`` path (5 term changes) vs NumPy ``senslatmom_fluxes``."""
    from lakematrix.jaxbatch.ice_surface import senslatmom_fluxes_ice

    rng = np.random.default_rng(11)
    L = 24
    Tsurf = rng.uniform(-25.0, 0.5, L)
    uwind = rng.uniform(0.01, 12.0, L)
    vwind = rng.uniform(-6.0, 6.0, L)
    u = rng.uniform(-0.2, 0.2, L)
    v = rng.uniform(-0.2, 0.2, L)
    tempair = rng.uniform(5.0, 30.0, L)
    pressure = rng.uniform(6.0e4, 1.05e5, L)
    humair = rng.uniform(0.0, 0.02, L)
    h_ice = rng.uniform(0.01, 2.0, L)
    fetch = np.full(L, 1000.0)
    zref = np.full(L, 2.0)

    got = jax.jit(senslatmom_fluxes_ice)(
        Tsurf, fetch, uwind, vwind, u, v, tempair, pressure, humair, zref,
        h_ice)
    for j in range(L):
        ref = NPS.senslatmom_fluxes(
            Tsurf[j], fetch[j], uwind[j], vwind[j], u[j], v[j], tempair[j],
            pressure[j], humair[j], zref[j], 1e-5, relhums=0.7, l1=h_ice[j])
        for key in ("hflux", "Elatent", "tau", "cdmw", "xlew", "velfrict"):
            np.testing.assert_allclose(np.asarray(got[key])[j], ref[key],
                                       rtol=1e-9, atol=1e-9,
                                       err_msg=f"{key}[{j}]")
    # The latent heat of sublimation must be present over ice: compare with the
    # water-branch fluxes of jaxbatch.surface (same roughness/vapour curve
    # would be wrong by tens of W/m2, so only check the sign of the difference).
    from lakematrix.jaxbatch.surface import senslatmom_fluxes as water_slm

    wat = water_slm(Tsurf, fetch, uwind, vwind, u, v, tempair, pressure,
                    humair, zref, 1e-5)
    assert np.all(np.asarray(got["Elatent"]) < np.asarray(wat["Elatent"]))


# ---------------------------------------------------------------------------
# 1. SURF_CHAR
# ---------------------------------------------------------------------------
def test_surf_char_ice_matches_numpy():
    from lakematrix.jaxbatch.ice_surface import surf_char_ice

    args = (2016, 8, 1, 4.0, 56.0, 1000.0, 1.0, 0.01)
    for surftyp in (NPS.ICE, NPS.SNOW):
        ref_sc = NPS.surf_char(surftyp, *args)
        got = surf_char_ice(surftyp, *args)
        for key in ("albedo", "albedo_lw", "roughness", "aM", "bM",
                    "emissivity", "relhums", "velfrict_prev"):
            np.testing.assert_allclose(np.asarray(got[key]), ref_sc[key],
                                       rtol=0, atol=0, err_msg=key)
    # The ice/snow tables ignore date, hour, site and wind.
    a = surf_char_ice(NPS.ICE, *args)
    b = surf_char_ice(NPS.ICE, 1999, 1, 1, 0.0, 0.0, 1.0, 40.0, 0.01)
    for key in a:
        np.testing.assert_allclose(np.asarray(a[key]), np.asarray(b[key]),
                                   rtol=0, atol=0, err_msg=key)


def test_surf_char_ice_batched_and_errors():
    from lakematrix.jaxbatch.ice_surface import surf_char_ice

    args = (2016, 8, 1, 4.0, 56.0, 1000.0, 1.0, 0.01)
    surftyp = jnp.asarray([2, 3, 2, 3])
    out = jax.jit(lambda st: surf_char_ice(st, *args))(surftyp)
    for j, st in enumerate([2, 3, 2, 3]):
        ref_sc = NPS.surf_char(st, *args)
        for key in ("albedo", "albedo_lw", "roughness", "emissivity"):
            np.testing.assert_allclose(np.asarray(out[key])[j],
                                       ref_sc[key], rtol=0, atol=0,
                                       err_msg=f"{key}[{j}]")
    # The water (4) and soil (1) branches are not part of this module.
    for bad in (1, 4):
        with pytest.raises(ValueError):
            surf_char_ice(bad, *args)


# ---------------------------------------------------------------------------
# 2. HEATFLUXSURF
# ---------------------------------------------------------------------------
def test_heatfluxsurf_ice_matches_numpy():
    from lakematrix.jaxbatch.ice_surface import heatfluxsurf_ice

    rng = np.random.default_rng(3)
    L, Mice = 4, NICE
    Ti1 = rng.normal(-2.0, 0.5, (L, Mice + 2))
    Ti2 = Ti1 + rng.normal(0.0, 0.1, (L, Mice + 2))
    lami_v = np.full((L, Mice + 1), C.lami)
    lami_v[:, 1] = 2.1
    ci_m_roi_v = np.full((L, Mice + 1), C.ci_m_roi)
    ddzi = np.full((L, Mice + 1), 1.0 / Mice)
    area_int = np.linspace(1.0e5, 2.0e5, L)[:, None] * np.ones((L, Mice + 1))
    area_half = 0.5 * (area_int[:, :-1] + area_int[:, 1:])
    area_half = np.concatenate([area_half, area_half[:, -1:]], axis=1)
    l1 = np.array([0.5, 0.12, 1.3, 0.02])
    dhi0 = np.array([1e-4, -2e-5, 0.0, 3e-3])

    got = jax.jit(heatfluxsurf_ice)(Ti1, Ti2, lami_v, ci_m_roi_v, l1, dhi0,
                                    ddzi, area_int, area_half, 20.0)
    for j in range(L):
        ref_hf = NPS.heatfluxsurf_ice(
            Ti1[j], Ti2[j], lami_v[j], ci_m_roi_v[j], l1[j], dhi0[j],
            ddzi[j], [0.0] + list(area_int[j, 1:]),
            [0.0] + list(area_half[j, 1:]), 20.0)
        np.testing.assert_allclose(np.asarray(got[0])[j], ref_hf[0],
                                   rtol=1e-15)
        np.testing.assert_allclose(np.asarray(got[1])[j], ref_hf[1],
                                   rtol=1e-15)
        assert np.asarray(got[2]) == 0.0 and ref_hf[2] == 0.0


def test_heatfluxsurf_snow_matches_numpy():
    from lakematrix.jaxbatch.ice_surface import heatfluxsurf_snow

    rng = np.random.default_rng(5)
    L, ms = 4, MS
    itop = np.array([ms - 2, ms - 2, ms - 5, ms - 3])
    T = rng.normal(-3.0, 0.5, (L, ms + 1))
    Tsn = T + rng.normal(0.0, 0.2, (L, ms + 1))
    cs = np.full((L, ms + 1), C.csnow)
    dens = rng.uniform(80.0, 300.0, (L, ms + 1))
    dz = rng.uniform(0.005, 0.05, (L, ms + 1))
    lams = rng.uniform(0.05, 0.5, (L, ms + 1))

    got = jax.jit(heatfluxsurf_snow)(Tsn, T, cs, dens, dz, lams, itop, 20.0)
    for j in range(L):
        ref_hf = NPS.heatfluxsurf_snow(Tsn[j], T[j], cs[j], dens[j], dz[j],
                                       lams[j], itop[j], 20.0)
        np.testing.assert_allclose(np.asarray(got[0])[j], ref_hf[0],
                                   rtol=1e-15)
        np.testing.assert_allclose(np.asarray(got[1])[j], ref_hf[1],
                                   rtol=1e-15)
        assert np.asarray(got[2]) == 0.0 and ref_hf[2] == 0.0
    # itop out of the 0-based array bounds fails loudly (static itop only).
    with pytest.raises(ValueError):
        heatfluxsurf_snow(Tsn, T, cs, dens, dz, lams, ms, 20.0)
    with pytest.raises(ValueError):
        heatfluxsurf_snow(Tsn, T, cs, dens, dz, lams, 0, 20.0)


# ---------------------------------------------------------------------------
# 3. RADBALANCE
# ---------------------------------------------------------------------------
def test_radbalance_ice_matches_numpy():
    from lakematrix.jaxbatch.ice_surface import radbalance_ice

    L = 3
    Tsurf = np.array([-3.0, -0.2, -12.0])
    radice = np.array([[10.0, 3.0], [60.0, 20.0], [0.0, 0.0]])
    kw = dict(longwave=320.0, shortwave=200.0, albedo=0.4, albedo_lw=0.03,
              emissivity=0.97)
    for surftyp in (NPS.ICE, NPS.SNOW):
        got = jax.jit(lambda t, r: radbalance_ice(t, r, surftyp, **kw))(
            Tsurf, radice)
        for j in range(L):
            ref_rb = NPS.radbalance(Tsurf[j], np.zeros(2), radice[j], surftyp,
                                    netrad=-1.0, cloud=-1.0, humair=0.008,
                                    pressure=100000.0, tempair=15.0, **kw)
            np.testing.assert_allclose(np.asarray(got[0])[j], ref_rb[0],
                                       rtol=1e-14, atol=1e-13)
            assert got[1] is None and ref_rb[1] is None
            for k in (2, 3, 4):
                # shortbal is per-lake, surfrad/swup stay scalar here
                # because albedo/emissivity/``shortwave`` are scalars.
                val = np.asarray(got[k])
                val = val[j] if val.ndim else val
                np.testing.assert_allclose(val, ref_rb[k], rtol=1e-14,
                                           atol=1e-13)
    # Mixed per-lake surftyp: ice shortwave for 2, zero shortbal for 3.
    st = jnp.asarray([2, 3, 2])
    rb, rbs, sb, sr, swup = radbalance_ice(-1.0, radice, st, **kw)
    np.testing.assert_allclose(np.asarray(sb), [7.0, 0.0, 0.0], rtol=0, atol=0)
    assert rbs is None and np.asarray(rb).shape == (3,)
    # radice_integr=None is the snow-only shortcut (no ice shortwave).
    rb_none, _, sb_none, _, _ = radbalance_ice(-1.0, None, 3, **kw)
    np.testing.assert_allclose(np.asarray(sb_none), 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(rb_none),
                               320.0 * (1.0 - 0.03)
                               - 0.97 * C.sigma * (-1.0 + C.Kelvin0) ** 4,
                               rtol=1e-14, atol=1e-13)


# ---------------------------------------------------------------------------
# 4. one-step HEATBALANCE vs the NumPy reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("snow", [0, 1])
def test_heatbalance_ice_one_step_matches_numpy(ref, snow):
    """One ``T_DIFF`` step + the surface balance, JAX vs NumPy."""
    from lakematrix.jaxbatch.ice_surface import heatbalance_ice

    case = ref[0]
    surftyp = NPS.SNOW if snow else NPS.ICE
    sc = NPS.surf_char(surftyp, 2016, 8, 1, 4.0, 56.0, 1000.0, 3.0, 0.01)
    radice = (12.0, 4.0)
    Tsurf = -2.5

    # --- NumPy reference (mutates its own IceState) -----------------------
    state_ref = dict(_ice=make_ice(ref, snow=bool(snow)))
    ctx_np = np_ctx(state_ref, sc, snow=snow, nsoilcols=case.nsoilcols,
                    radice=radice)
    HB_ref, out_ref = NPS.heatbalance_ice(Tsurf, ctx_np)

    # --- JAX kernel driven by the NumPy-backed t_diff_fn ------------------
    state = dict(_ice=make_ice(ref, snow=bool(snow)))
    base = ice_state_dict(state["_ice"])
    ctx = jax_ctx(dict(base, _ice=state["_ice"]), sc, snow=snow,
                  nsoilcols=case.nsoilcols, radice=radice)
    HB, out = heatbalance_ice(Tsurf, ctx, np_t_diff)

    np.testing.assert_allclose(np.asarray(HB), HB_ref, rtol=1e-13,
                               atol=1e-11, err_msg="HB")
    for key in ("eflux", "dHdt", "convect_flux", "hflux", "Elatent",
                "Radbal", "eflux0", "eflux0_kinem", "cdmw"):
        np.testing.assert_allclose(np.asarray(out[key])[0], out_ref[key],
                                   rtol=1e-13, atol=1e-11, err_msg=key)
    assert out["Radbal_surf"] is None

    # The updated temperatures travel in out["ice"]; the old ones are the
    # input state values (SNOWSOIL keeps T and Tsn apart).
    ic, ic_ref = out["ice"], out_ref["ice"]
    np.testing.assert_allclose(np.asarray(ic["Ti2"]), ic_ref.Ti2, rtol=0,
                               atol=0)
    np.testing.assert_allclose(np.asarray(ic["Ti1"]), ic_ref.Ti1, rtol=0,
                               atol=0)
    np.testing.assert_allclose(np.asarray(ic["Tsn"]), ic_ref.Tsn, rtol=0,
                               atol=0)
    np.testing.assert_allclose(np.asarray(ic["T"]), ic_ref.T, rtol=0, atol=0)
    assert np.asarray(ic["Tsn"]).shape == (MS + 1,)


def test_heatbalance_ice_missing_key_message(ref):
    """A missing ctx/state entry names the place that was searched."""
    from lakematrix.jaxbatch.ice_surface import heatbalance_ice

    case = ref[0]
    sc = NPS.surf_char(NPS.ICE, 2016, 8, 1, 4.0, 56.0, 1000.0, 3.0, 0.01)
    ice = make_ice(ref)
    ctx = jax_ctx(dict(ice_state_dict(ice), _ice=ice), sc, snow=0,
                  nsoilcols=case.nsoilcols)
    del ctx["longwave"]
    with pytest.raises(KeyError, match="longwave"):
        heatbalance_ice(-2.5, ctx, np_t_diff)


# ---------------------------------------------------------------------------
# 5. traceability / shapes with a synthetic JAX t_diff stub
# ---------------------------------------------------------------------------
def _stub_t_diff(surf, Tsurf, dt, snow, ice, water, deepice, isoilcol,
                 ifflux, state):
    """Synthetic, pure ``T_DIFF`` stand-in (B5 will supply the real kernel).

    It relaxes each temperature towards ``Tsurf`` and leaves everything else
    untouched, so the balance depends on ``Tsurf`` and stays differentiable:
    the point is to prove that ``heatbalance_ice`` traces, jits and vmaps.
    """
    out = dict(state)
    Tsurf = jnp.asarray(Tsurf)[..., None]      # (L,) -> (L, 1) column
    out["Ti2"] = state["Ti1"] + 0.1 * (Tsurf - state["Ti1"])
    # SNOWSOIL keeps Tsn (ms) and T (ML) as separate arrays; the stub keeps the
    # Tsn shape and still depends on both the old Tsn and T.
    ms1 = jnp.shape(state["Tsn"])[-1]
    T_old = state["T"][..., :ms1]
    out["Tsn"] = 0.5 * (state["Tsn"] + T_old) + 0.1 * (Tsurf - T_old)
    return out


def _repeat(x, L):
    x = np.asarray(x, dtype=float)
    return jnp.asarray(np.repeat(x[None, ...], L, axis=0))


def _batch_problem(ref, L=4, snow=(0, 1, 0, 1)):
    """An ``(L, ...)`` ice/snow problem: one snow-bearing IceState, L copies."""
    ice = make_ice(ref, snow=True)          # snow pack present in the state
    base = ice_state_dict(ice)
    st = {k: _repeat(v, L) for k, v in base.items()}
    st["itop"] = jnp.repeat(jnp.asarray([base["itop"]], dtype=jnp.int32), L,
                            axis=0)
    sc_ice = NPS.surf_char(NPS.ICE, 2016, 8, 1, 4.0, 56.0, 1000.0, 3.0, 0.01)
    sc_snow = NPS.surf_char(NPS.SNOW, 2016, 8, 1, 4.0, 56.0, 1000.0, 3.0, 0.01)
    ctx = dict(
        snow=jnp.asarray(snow, dtype=jnp.int32),
        water=jnp.ones(L, dtype=jnp.int32),
        ice=jnp.ones(L, dtype=jnp.int32),
        deepice=jnp.zeros(L, dtype=jnp.int32),
        isoilcol=ref[0].nsoilcols, dt=20.0, state=st,
        longwave=320.0, shortwave=200.0, netrad=-1.0, cloud=-1.0,
        roughness=jnp.asarray([sc_snow["roughness"] if s else
                               sc_ice["roughness"] for s in snow]),
        relhums=jnp.full((L,), 0.7), aM=jnp.full((L,), C.aMagi),
        bM=jnp.full((L,), C.bMagi),
        albedo=jnp.asarray([sc_snow["albedo"] if s else sc_ice["albedo"]
                            for s in snow]),
        albedo_lw=jnp.asarray([sc_snow["albedo_lw"] if s
                               else sc_ice["albedo_lw"] for s in snow]),
        emissivity=jnp.asarray([sc_snow["emissivity"] if s
                                else sc_ice["emissivity"] for s in snow]),
        radice_integr=jnp.asarray(np.tile([[12.0, 4.0]], (L, 1))))
    for key, val in _ATMOS.items():
        ctx[key] = jnp.full((L,), val)
    return ctx


def test_heatbalance_ice_traces_and_shapes(ref):
    """``jit`` + ``grad`` + mixed ice/snow selection on a synthetic batch."""
    from lakematrix.jaxbatch.ice_surface import heatbalance_ice

    ctx = _batch_problem(ref, L=4, snow=(0, 1, 0, 1))
    Tsurf = jnp.asarray([-2.5, -1.0, -8.0, -0.5])

    HB, out = jax.jit(heatbalance_ice, static_argnames=("t_diff_fn",))(
        Tsurf, ctx, _stub_t_diff)
    assert HB.shape == (4,)
    for key in ("eflux", "dHdt", "convect_flux", "hflux", "Elatent",
                "Radbal", "eflux0", "eflux0_kinem", "cdmw"):
        assert out[key].shape == (4,), key
        assert np.all(np.isfinite(np.asarray(out[key]))), key
    assert out["Radbal_surf"] is None
    for key, val in out["ice"].items():
        assert np.all(np.isfinite(np.asarray(val))), key
    assert np.asarray(out["ice"]["Tsn"]).shape == (4, MS + 1)
    assert np.asarray(out["ice"]["Ti2"]).shape == (4, ref[0].Mice + 2)

    # Per-lake branch selection: the stub does not read the `snow` flag, so
    # forcing snow to all-0 / all-1 must reproduce the matching entries of the
    # mixed batch exactly.
    HB_ice, _ = heatbalance_ice(Tsurf, dict(ctx, snow=jnp.zeros_like(
        ctx["snow"])), _stub_t_diff)
    HB_snow, _ = heatbalance_ice(Tsurf, dict(ctx, snow=jnp.ones_like(
        ctx["snow"])), _stub_t_diff)
    np.testing.assert_allclose(np.asarray(HB)[[0, 2]],
                               np.asarray(HB_ice)[[0, 2]], rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(HB)[[1, 3]],
                               np.asarray(HB_snow)[[1, 3]], rtol=0, atol=0)

    # vmappable: mapping the same kernel over the per-lake ctx pytree gives the
    # batched answer (a few ulps, because the CPU codegen differs with shape).
    L = 4
    in_axes = jax.tree_util.tree_map(
        lambda v: 0 if (np.ndim(v) and np.shape(v)[0] == L) else None, ctx)
    HBv, _ = jax.vmap(lambda c, t: heatbalance_ice(t, c, _stub_t_diff),
                      in_axes=(in_axes, 0))(ctx, Tsurf)
    np.testing.assert_allclose(np.asarray(HBv), np.asarray(HB), rtol=1e-12,
                               atol=1e-10)


def test_heatbalance_ice_batch_consistency(ref):
    """One row of an L=4 batch equals the same problem in an L=1 call."""
    from lakematrix.jaxbatch.ice_surface import heatbalance_ice

    L = 4
    ctx4 = _batch_problem(ref, L=L)
    Tsurf4 = jnp.asarray([-2.5, -1.0, -8.0, -0.5])
    HB4, out4 = heatbalance_ice(Tsurf4, ctx4, _stub_t_diff)

    def one(v):
        v = jnp.asarray(v)
        return v[j:j + 1] if v.ndim >= 1 and v.shape[0] == L else v

    j = 2
    ctx1 = {k: (v if k == "state" else one(v)) for k, v in ctx4.items()}
    ctx1["state"] = {k: one(v) for k, v in ctx4["state"].items()}
    HB1, out1 = heatbalance_ice(Tsurf4[j:j + 1], ctx1, _stub_t_diff)
    np.testing.assert_allclose(np.asarray(HB1)[0], np.asarray(HB4)[j],
                               rtol=0, atol=0)
    for key in ("eflux", "dHdt", "hflux", "Elatent", "Radbal", "eflux0"):
        np.testing.assert_allclose(np.asarray(out1[key])[0],
                                   np.asarray(out4[key])[j], rtol=0, atol=0)
