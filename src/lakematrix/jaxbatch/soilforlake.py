"""Batched JAX ``SOILFORLAKE`` for the case-1 open-water deepest soil column.

Fortran: ``SOILFORLAKE(nn,dt,a,b,c,d,is)`` (``soil_mod.f90:215-709``);
reference NumPy implementation: :func:`lakematrix.physics.soil.soilforlake`.
This is the M5/B3 port of the **case-1 open-water, deepest-column** path:

    h1 > 0,  l1 = 0,  ls1 = 0,  hs1 = 0,  is = nsoilcols

i.e. no ice/snow at the surface and the soil column in contact with the water
body.  The full sequence of the reference is implemented:

1. implicit moisture diffusion (``PROGONKA`` on ``wl1`` -> ``wl2``) with the
   saturated (Dirichlet) top boundary ``wl1(1) = WL_MAX``;
2. gravitational infiltration (``filtr``) -> ``wl3``;
3. redistribution of water above ``WL_MAX`` (and of negative deficits) over the
   column -> ``wl3``, accumulating ``dhwfsoil3``;
4. the iterative phase change (melting/freezing with the under-relaxation
   weights ``alp``) -> ``Tsoil3``/``wl4``/``wi2``;
5. negative ice returned to liquid water;
6. porosity (pore-volume) adjustment of ``wl4`` -> ``dhwfsoil4``;
7. ``dhwfsoil = dhwfsoil1 + dhwfsoil2 + dhwfsoil3 + dhwfsoil4``;
8. the prognostic updates ``Tsoil1 = Tsoil3``, ``wl1 = wl4``, ``wi1 = wi2``
   and the diagnostic air content ``wa`` (and ``methgenmh``).

Interface
=========

All arrays carry a leading lake axis ``L`` and are **1-based along the level
axis** exactly like the NumPy reference (index 0 is the unused Fortran pad).
``state`` values are read and returned in a plain ``dict``:

=============================  ==================  ===========================
key                            shape               meaning
=============================  ==================  ===========================
``Tsoil1``, ``wl1``, ``wi1``   ``(L, ns+1, nc+1)`` prognostic soil fields
``Sals1``                      ``(L, ns+1, nc+1)`` soil salinity, kg/kg
``Tsoil3``, ``wl4``, ``wi2``   ``(L, ns+1, nc+1)`` phase-change outputs
``csoil``, ``rosoil``,         ``(L, ns+1)``       ``SOIL_COND_HEAT_COEF``
``lammoist``, ``filtr``                            coefficients of column
                                                   ``is_`` (see note below)
``h1``                         ``(L,)``            water depth, m
``wa``, ``methgenmh``          ``(L, ns+1)``       optional diagnostics
``dhwfsoil``                   ``(L,)``            optional diagnostic
=============================  ==================  ===========================

``params`` (static Python ints are used for the unrolled loops, so they must be
real Python ``int``/arrays, not tracers):

=============================  ==================  ===========================
key                            shape               meaning
=============================  ==================  ===========================
``ns``, ``nsoilcols``          ``int``             levels / soil columns
``nmeltpoint``                 ``int``             ``MELTPNT`` selector (0/1)
``POR``, ``rosdry``,           ``(ns+1,)``         soil geometry/type arrays
``dzs``, ``dzss``, ``zsoil``                       (1-based, index 0 pad)
``WLM0``, ``WLM7``             ``(ns+1,)``         ``UNFRWAT`` coefficients
``pressure``                   scalar or ``(L,)``  atmospheric pressure, Pa
``soilbotflx``                 scalar              unused here (T_DIFF only)
=============================  ==================  ===========================

``Tsoil2`` is the **diffused** soil temperature of the same column
(``(L, ns+1)``, produced by ``T_DIFF`` with ``isoilcol = nsoilcols``); ``dt``
is the time step in seconds.  ``is_`` defaults to ``params["nsoilcols"]`` (the
deepest column, which is the only supported configuration).

Returns a new dict: a copy of ``state`` with ``Tsoil1``/``wl1``/``wi1``
updated in column ``is_``, plus ``Tsoil2`` (the iteration work value, as the
in-place reference leaves it), ``Tsoil3``, ``wl4``, ``wi2``, ``wa``,
``methgenmh`` and ``dhwfsoil``.  The input dict and its arrays are not mutated.

Notes / deviations
==================

* ``tricemethhydr > 0`` (methane hydrates) and ``nmeltpoint == 2`` (Jackett et
  al. 2004) raise ``NotImplementedError``; the reference has the same gap for
  ``nmeltpoint == 2`` and the methane constants are not part of this port.
* ``soilbotflx`` is accepted for interface parity but not used: it belongs to
  ``T_DIFF`` (heat diffusion), not to ``SOILFORLAKE``.
* Only the case-1 branch selection is implemented (no ice/snow/no-water
  branches and no ``dhwfsoil`` ``Wflow``/``Wfhigh`` diagnostics, which are dead
  stores in the source for this branch).
* Scalar ``if`` branches become ``jnp.where``; the ``ns``-sized loops are
  unrolled static Python loops; the phase-change ``while`` loop (at most 21
  passes: 1 initial + 20 under-relaxations + final check, ``iter3 < 20``) is a
  ``lax.fori_loop`` with a per-lake ``done`` mask so a lake that converged
  stops exactly as the Fortran ``exit`` does.
* Float32 Fortran literals are reused from the NumPy module
  (``S._S001``, ``S._MELTPNT_DTDC``); the under-relaxation weights are computed
  on the host with ``math.cos`` so they are bit-identical to the reference.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ..constants import C
from ..physics import soil as S
from ..physics.tridiag import progonka_jax
from .soil import wl_max

__all__ = [
    "make_soilforlake",
    "meltingpoint",
    "soilforlake",
    "unfrwat",
    "wl_max",
]

_S001 = S._S001


def _iteration_weights():
    """``alp(1:8)`` (soil_mod.f90:299-303), evaluated exactly as the NumPy port."""
    ma, mi = 10.5, 4.5
    return tuple(
        2.0 * (mi + ma - (ma - mi)
               * math.cos((2.0 * S._PSIIT[i] - 1.0) / 16.0 * C.pi)) ** (-1.0)
        for i in range(1, 9))


_ALP = _iteration_weights()


def unfrwat(t, wlm0, wlm7):
    """``UNFRWAT`` (phys_func.f90:1346) for already-sliced per-level arrays."""
    return (wlm0 - wlm7) * jnp.exp(t / 3.0) + wlm7


def meltingpoint(s, nmeltpoint):
    """``MELTINGPOINT``/``MELTPNT`` (phys_func.f90:778,1548) for ``mh = 0``."""
    if nmeltpoint == 0:
        return jnp.zeros_like(s)
    if nmeltpoint == 1:
        return 0.0 - s * S._MELTPNT_DTDC
    raise NotImplementedError(
        "MELTPNT: nmeltpoint=2 (Jackett et al. 2004, FP_THETA) is not ported")


def _mfdiv(x, y):
    """Division with Fortran/IEEE semantics (``inf``/``NaN`` instead of raising)."""
    return x / y


def _get(state, key, like):
    """``state[key]`` or a zero template ``like`` for optional work arrays."""
    return jnp.asarray(state[key]) if key in state else jnp.zeros_like(like)


def _excess_water(wl, cap, den):
    """Fortran excess-water redistribution (soil_mod.f90:439-487 / 566-...).

    ``wl``, ``cap``, ``den`` are 1-based ``(L, ns+1)`` arrays; ``den(i) =
    rosdry(i)*(1-por(i))*dzss(i)`` (kg/m2 per unit of ``wl``).  Layers whose
    content exceeds ``cap`` push the surplus into other layers (increasing
    index order, stopping when the surplus is exhausted), negative deficits
    take water from the layer above (decreasing index) or else from below.

    Returns ``(wl, dhw)`` with ``dhw`` the water lost from the column in
    kg/m2 (``surplus*den/row0``), i.e. ``dhwfsoil3``/``dhwfsoil4``.
    """
    L = wl.shape[0]
    ns = wl.shape[1] - 1
    dhw = jnp.zeros((L,))

    for i in range(1, ns + 1):
        over = wl[:, i] > cap[:, i]
        surplus = jnp.where(over, wl[:, i] - cap[:, i], 0.0)
        den_i = den[:, i]
        active = over
        for j in range(1, ns + 1):
            room = cap[:, j] - wl[:, j]
            need = surplus * den_i / den[:, j]
            take_all = room > need
            apply = active & (room > 0.0)
            wl = wl.at[:, j].set(jnp.where(
                apply, jnp.where(take_all, wl[:, j] + need, cap[:, j]),
                wl[:, j]))
            surplus = jnp.where(
                apply,
                jnp.where(take_all, 0.0, surplus - room * den[:, j] / den_i),
                surplus)
            active = active & ~(apply & take_all)
        dhw = dhw + surplus * den_i / C.row0
        wl = wl.at[:, i].set(jnp.where(over, cap[:, i], wl[:, i]))

        # Negative deficit: label 1 (wl(i) = 0) is reached in all cases.
        deficit = wl[:, i] < 0.0
        moved = jnp.zeros((L,), dtype=bool)
        for j in range(i - 1, 0, -1):
            need = -wl[:, i] * den_i / den[:, j]
            ok = deficit & ~moved & (wl[:, j] > need)
            wl = wl.at[:, j].set(
                jnp.where(ok, wl[:, j] + wl[:, i] * den_i / den[:, j],
                          wl[:, j]))
            moved = moved | ok
        if i != ns:
            for j in range(i + 1, ns + 1):
                need = -wl[:, i] * den_i / den[:, j]
                ok = deficit & ~moved & (wl[:, j] > need)
                wl = wl.at[:, j].set(
                    jnp.where(ok, wl[:, j] + wl[:, i] * den_i / den[:, j],
                              wl[:, j]))
                moved = moved | ok
        wl = wl.at[:, i].set(jnp.where(deficit, 0.0, wl[:, i]))
    return wl, dhw


def _phase_candidates(wl3, wi1, Tsoil2, Sals1, por, rosdry, dzss, wlm0, wlm7,
                      pressoil, csoil, rosoil, nmeltpoint):
    """One pass of the phase-change block (soil_mod.f90:490-570).

    ``wl3``/``wi1``/``Tsoil2``/``Sals1``/``pressoil`` are 1-based ``(L, ns+1)``;
    returns the candidate ``(wl4, wi2, Tsoil3)``.  With ``tricemethhydr = 0``
    the reference's ``mhsep`` is exactly 1 and is omitted.
    """
    w = wl3[:, 1:]
    wi = wi1[:, 1:]
    T2 = Tsoil2[:, 1:]
    sal = Sals1[:, 1:]
    porc = por[None, 1:]
    rosc = rosdry[None, 1:]
    dzc = dzss[None, 1:]
    cso = csoil[:, 1:]
    ros = rosoil[:, 1:]
    dz = dzss[None, 1:]
    wlm0c = wlm0[None, 1:]
    wlm7c = wlm7[None, 1:]
    hicemelt = C.Lwi

    work = meltingpoint(_mfdiv(sal, w), nmeltpoint)
    unfr_t2 = unfrwat(T2, wlm0c, wlm7c)
    # Energy per unit of ice/liquid content, J/m2 (Fortran divisor of the
    # partial branches: rosdry*dzss*(1-por)*hicemelt, without the content).
    freeze_den = rosc * dz * (1.0 - porc) * hicemelt
    # Energy available for melting / freezing, J/m2.
    pot_melt = (T2 - (work + _S001)) * cso * ros * dz
    ice_energy = wi * freeze_den
    pot_freeze = -(T2 - work + _S001) * cso * ros * dz
    wata = (w - unfr_t2) * freeze_den

    melting = (wi > 0.0) & (T2 > work + _S001)
    freezing = (T2 < work - _S001) & (w >= 0.0)
    melt_full = melting & (pot_melt >= ice_energy)

    # Melting of soil ice.
    wl4_melt = jnp.where(melt_full, w + _mfdiv(wi, 1.0),
                         w + pot_melt / freeze_den / 1.0)
    wi2_melt = jnp.where(melt_full, 0.0, wi - pot_melt / freeze_den)
    T3_melt = jnp.where(
        melt_full,
        work + _S001 + (pot_melt - ice_energy) / (cso * ros * dz),
        work + _S001)

    # Freezing of soil water; the first branch (wata < 0) is Fortran's cycle c2.
    # The remaining freezing case (wata >= 0 and pot_freeze < wata) is the
    # implicit ``else`` of the two ``jnp.where`` chains below.
    fz_neg = freezing & (wata < 0.0)
    fz_avail = freezing & (wata >= 0.0) & (pot_freeze >= wata)
    fz_keep = fz_neg | fz_avail
    wl4_fz = jnp.where(fz_keep, unfr_t2, w - pot_freeze / freeze_den)
    wi2_fz = jnp.where(fz_keep, wi + (w - unfr_t2),
                       wi + pot_freeze / freeze_den)
    T3_fz = jnp.where(
        fz_neg, T2 + wata / (cso * ros * dz),
        jnp.where(fz_avail,
                  work - _S001 - (pot_freeze - wata) / (cso * ros * dz),
                  work - _S001))

    wl4 = jnp.where(melting, wl4_melt,
                    jnp.where(freezing, wl4_fz, w))
    wi2 = jnp.where(melting, wi2_melt,
                    jnp.where(freezing, wi2_fz, wi))
    T3 = jnp.where(melting, T3_melt,
                   jnp.where(freezing, T3_fz, T2))
    L = wl3.shape[0]
    ns = wl3.shape[1]
    z = jnp.zeros((L, 1))
    return (jnp.concatenate((z, wl4), axis=-1),
            jnp.concatenate((z, wi2), axis=-1),
            jnp.concatenate((z, T3), axis=-1))


def _phase_max1(wl4, Tsoil3, Sals1, wlm0, wlm7, pressoil, nmeltpoint):
    """``max1`` of the phase-change consistency check (soil_mod.f90:539-548)."""
    w = wl4[:, 1:]
    T = Tsoil3[:, 1:]
    mp = meltingpoint(_mfdiv(Sals1[:, 1:], w), nmeltpoint)
    unfr = unfrwat(T, wlm0[None, 1:], wlm7[None, 1:])
    diff = jnp.abs(w - unfr)
    cond = (T < mp - _S001) & (diff > 0.0)
    return jnp.max(jnp.where(cond, diff, 0.0), axis=1)


def _phase_iterate(wl3, wi1, Tsoil2, Sals1, por, rosdry, dzss, wlm0, wlm7,
                   pressoil, csoil, rosoil, nmeltpoint):
    """The ``while`` loop with under-relaxation (soil_mod.f90:533-566)."""
    L = wl3.shape[0]
    wl4 = jnp.zeros_like(wl3)
    wi2 = jnp.zeros_like(wl3)
    T3 = jnp.zeros_like(wl3)
    done = jnp.zeros((L,), dtype=bool)
    weights = jnp.asarray((0.0,) + _ALP)
    # Level 0 is the Fortran pad; the reference only relaxes 1..ns.
    level = (jnp.arange(wl3.shape[1]) > 0)[None, :]

    def body(k, carry):
        wl3, wi1, Tsoil2, wl4, wi2, T3, done = carry
        c_wl4, c_wi2, c_T3 = _phase_candidates(
            wl3, wi1, Tsoil2, Sals1, por, rosdry, dzss, wlm0, wlm7,
            pressoil, csoil, rosoil, nmeltpoint)
        dmask = done[:, None]
        wl4 = jnp.where(dmask, wl4, c_wl4)
        wi2 = jnp.where(dmask, wi2, c_wi2)
        T3 = jnp.where(dmask, T3, c_T3)
        max1 = jnp.where(done, 0.0,
                         _phase_max1(wl4, T3, Sals1, wlm0, wlm7, pressoil,
                                     nmeltpoint))
        relax = (~done) & (max1 > _S001) & (k < 20)
        w = weights[k % 8 + 1]
        rmask = relax[:, None] & level
        wl3 = jnp.where(rmask, wl3 + w * (wl4 - wl3), wl3)
        wi1 = jnp.where(rmask, wi1 + w * (wi2 - wi1), wi1)
        Tsoil2 = jnp.where(rmask, Tsoil2 + w * (T3 - Tsoil2), Tsoil2)
        return (wl3, wi1, Tsoil2, wl4, wi2, T3, done | ~relax)

    wl3, wi1, Tsoil2, wl4, wi2, T3, _ = jax.lax.fori_loop(
        0, 21, body, (wl3, wi1, Tsoil2, wl4, wi2, T3, done))
    return wl3, wi1, Tsoil2, wl4, wi2, T3


def _soilforlake_case1(state, params, Tsoil2, dt, is_, ns, nsoilcols,
                       nmeltpoint, tricemethhydr):
    if tricemethhydr > 0.0:
        raise NotImplementedError(
            "tricemethhydr > 0 (methane hydrate dissociation) is not ported")
    if nmeltpoint not in (0, 1):
        raise NotImplementedError(
            "MELTPNT: nmeltpoint=2 (Jackett et al. 2004, FP_THETA) is not set up")

    por = jnp.asarray(params["POR"])
    rosdry = jnp.asarray(params["rosdry"])
    dzs = jnp.asarray(params["dzs"])
    dzss = jnp.asarray(params["dzss"])
    zsoil = jnp.asarray(params["zsoil"])
    wlm0 = jnp.asarray(params["WLM0"])
    wlm7 = jnp.asarray(params["WLM7"])
    pressure = jnp.atleast_1d(jnp.asarray(params["pressure"]))[:, None]
    # params["soilbotflx"] is unused: it belongs to T_DIFF (heat diffusion).
    h1_ = jnp.atleast_1d(jnp.asarray(state["h1"]))[:, None]
    Lh = h1_.shape[0]
    def _lane_scalar(key):
        a = jnp.atleast_1d(jnp.asarray(state.get(key, 0.0)))
        return jnp.broadcast_to(a.reshape(-1)[:1] if a.size == 1 else a,
                                (Lh,))
    l1_ = _lane_scalar("l1")
    ls1_ = _lane_scalar("ls1")
    hs1_ = _lane_scalar("hs1")
    h1f, l1f, ls1f, hs1f = h1_[:, 0], l1_, ls1_, hs1_
    open_water = (h1f != 0.0) & (ls1f == 0.0)
    no_water = (((l1f != 0.0) & (h1f == 0.0)) | (ls1f != 0.0)
                | ((hs1f != 0.0) & (l1f == 0.0) & (h1f == 0.0)))

    Tsoil2 = jnp.asarray(Tsoil2)
    Tsoil1 = jnp.asarray(state["Tsoil1"])
    wl1 = jnp.asarray(state["wl1"])
    wi1 = jnp.asarray(state["wi1"])
    Sals1 = jnp.asarray(state["Sals1"])
    L = Tsoil1.shape[0]

    z2 = jnp.zeros((L, ns + 1))
    Tsoil3 = _get(state, "Tsoil3", Tsoil1)
    wl4 = _get(state, "wl4", Tsoil1)
    wi2 = _get(state, "wi2", Tsoil1)
    csoil = _get(state, "csoil", z2)
    rosoil = _get(state, "rosoil", z2)
    lammoist = _get(state, "lammoist", z2)
    filtr = _get(state, "filtr", z2)

    wl1_is = wl1[:, :, is_]
    wi1_is = wi1[:, :, is_]
    Sals_is = Sals1[:, :, is_]

    # WL_MAX with tricemethhydr = 0 (phys_func.f90:1364).
    wlmax1 = wl_max(por[None, :], rosdry[None, :], wi1_is)

    # HYPSOGRAPHIC soil pressure at the level mid-points (row0*g*(h1+zsoil)).
    pressoil = pressure + C.row0 * C.g * (h1_ + zsoil[None, :])

    # --- Step 1: implicit moisture diffusion (soil_mod.f90:341-368) --------
    a = z2
    b = z2
    c = z2
    d = z2
    # Open water (h1 /= 0 and ls1 == 0): saturated top layer; no water over
    # ice/snow: zero-flux top boundary (soil_mod.f90:313-320).
    coef_top = dt * (lammoist[:, 1] + lammoist[:, 2]) / (dzs[1] * dzs[1])
    c = c.at[:, 1].set(jnp.where(no_water, -1.0 - coef_top, 1.0))
    b = b.at[:, 1].set(jnp.where(no_water, -coef_top, 0.0))
    d = d.at[:, 1].set(jnp.where(no_water, -wl1_is[:, 1], wlmax1[:, 1]))
    lammoist1_ns = 0.5 * (lammoist[:, ns - 1] + lammoist[:, ns])
    coef_ns = 2.0 * dt * lammoist1_ns / (dzs[ns - 1] * dzs[ns - 1])
    c = c.at[:, ns].set(-1.0 - coef_ns)
    a = a.at[:, ns].set(-coef_ns)
    d = d.at[:, ns].set(-wl1_is[:, ns])
    for i in range(2, ns):
        lammoist2 = 0.5 * (lammoist[:, i] + lammoist[:, i + 1])
        lammoist1 = 0.5 * (lammoist[:, i - 1] + lammoist[:, i])
        dzmean = 0.5 * (dzs[i - 1] + dzs[i])
        a = a.at[:, i].set(-dt / dzmean * lammoist1 / dzs[i - 1])
        b = b.at[:, i].set(-dt / dzmean * lammoist2 / dzs[i])
        c = c.at[:, i].set(
            -dt / dzmean * (lammoist1 / dzs[i - 1] + lammoist2 / dzs[i])
            - 1.0)
        d = d.at[:, i].set(-wl1_is[:, i])
    sol = progonka_jax(a[:, 1:], b[:, 1:], c[:, 1:], d[:, 1:])
    wl2 = jnp.concatenate((jnp.zeros((L, 1)), sol), axis=-1)

    lammoist1_1 = 0.5 * (lammoist[:, 1] + lammoist[:, 2])
    dhwfsoil1 = jnp.where(open_water, (
        -(wl2[:, 1] * (dzs[1] / (2.0 * dt) + lammoist1_1 / dzs[1])
          - wl2[:, 2] * lammoist1_1 / dzs[1]
          - wl1_is[:, 1] * dzs[1] / (2.0 * dt))
        * rosdry[1] * (1.0 - por[1]) / C.row0 * dt), 0.0)

    # --- Step 2: gravitational infiltration (soil_mod.f90:375-434) ---------
    # The pad (level 0) is never written by the reference; keep the incoming
    # value there while all physical levels 1..ns are overwritten below.
    wl3 = _get(state, "wl3", z2)
    for i in range(2, ns):
        wl3 = wl3.at[:, i].set(
            wl2[:, i] + dt * (filtr[:, i - 1] - filtr[:, i]) / dzss[i])
    # Open water: the top layer is kept at WL_MAX.  With ice/snow and no
    # water it gains only the gravitational flux from below (soil_mod.f90:402).
    wl3 = wl3.at[:, 1].set(jnp.where(
        no_water, wl2[:, 1] + dt * (-filtr[:, 1]) / dzss[1], wlmax1[:, 1]))
    dhwfsoil2 = jnp.where(open_water,
                          -filtr[:, 1] * rosdry[1] * (1.0 - por[1])
                          / C.row0 * dt, 0.0)
    wl3 = wl3.at[:, ns].set(
        wl2[:, ns] + dt * (-0.0 + filtr[:, ns - 1]) / dzss[ns])

    # Redistribute water that exceeds WL_MAX (soil_mod.f90:439-487).
    den = jnp.broadcast_to(rosdry * (1.0 - por) * dzss, (L, ns + 1))
    wl3, dhwfsoil3 = _excess_water(wl3, wlmax1, den)

    # --- Step 3: phase processes (soil_mod.f90:490-570) --------------------
    wl3, wi1_is, Tsoil2, wl4_is, wi2_is, Tsoil3_is = _phase_iterate(
        wl3, wi1_is, Tsoil2, Sals_is, por, rosdry, dzss, wlm0, wlm7,
        pressoil, csoil, rosoil, nmeltpoint)

    # The reference only writes levels 1..ns, so the pad (level 0) of the
    # phase-change outputs keeps the incoming state's value.
    def keep_pad(new, old):
        return jnp.concatenate((old[:, :1], new[:, 1:]), axis=-1)

    wl4_is = keep_pad(wl4_is, wl4[:, :, is_])
    wi2_is = keep_pad(wi2_is, wi2[:, :, is_])
    Tsoil3_is = keep_pad(Tsoil3_is, Tsoil3[:, :, is_])

    # Negative ice content: return it to liquid water (mhsep = 1).
    neg = wi2_is < 0.0
    wl4_is = jnp.where(neg, wl4_is + wi2_is, wl4_is)
    wi2_is = jnp.where(neg, 0.0, wi2_is)

    # methgenmh (soil_mod.f90:575-579) is identically zero for mh = 0; like the
    # reference only levels 1..ns are written (0 keeps the incoming value).
    methgenmh = _get(state, "methgenmh", z2).at[:, 1:].set(0.0)

    # Water above the pore volume: push it to other layers / out of the column.
    cap = (por[None, :] * C.row0 / (rosdry[None, :] * (1.0 - por[None, :]))
           - wi2_is * C.row0_d_roi)
    wl4_is, dhwfsoil4 = _excess_water(wl4_is, cap, den)

    dhwfsoil = dhwfsoil1 + dhwfsoil2 + dhwfsoil3 + dhwfsoil4

    # --- Prognostic updates (soil_mod.f90:691-703) -------------------------
    Tsoil1 = Tsoil1.at[:, :, is_].set(Tsoil3_is)
    wl1 = wl1.at[:, :, is_].set(wl4_is)
    wi1 = wi1.at[:, :, is_].set(wi2_is)
    Tsoil3 = Tsoil3.at[:, :, is_].set(Tsoil3_is)
    wl4 = wl4.at[:, :, is_].set(wl4_is)
    wi2 = wi2.at[:, :, is_].set(wi2_is)

    wa_new = (por[None, :] * C.roa0 / ((1.0 - por[None, :]) * rosdry[None, :])
              - C.roa0_d_row0 * wl1[:, :, is_]
              - C.roa0_d_roi * wi1[:, :, is_])
    wa = _get(state, "wa", z2)
    wa = wa.at[:, 1:].set(jnp.maximum(wa_new[:, 1:], 0.0))

    out = dict(state)
    out.update({
        "Tsoil1": Tsoil1,
        "Tsoil2": Tsoil2,
        "Tsoil3": Tsoil3,
        "wl1": wl1,
        "wl4": wl4,
        "wi1": wi1,
        "wi2": wi2,
        "wa": wa,
        "methgenmh": methgenmh,
        "dhwfsoil": dhwfsoil,
        # diagnostics of the moisture splitting
        "wl2": wl2,
        "wl3": wl3,
    })
    return out


def soilforlake(state, params, Tsoil2, dt, is_=None):
    """Case-1 open-water ``SOILFORLAKE``; see the module docstring for the API.

    ``state`` and ``params`` are dicts, ``Tsoil2`` is the ``(L, ns+1)`` diffused
    temperature of column ``is_`` (default ``params["nsoilcols"]``) and ``dt``
    is the time step in seconds.  Returns the updated state dict.
    """
    ns = int(params["ns"])
    nsoilcols = int(params["nsoilcols"])
    nmeltpoint = int(params.get("nmeltpoint", 1))
    tricemethhydr = float(params.get("tricemethhydr", 0.0))
    is_ = nsoilcols if is_ is None else int(is_)
    return _soilforlake_case1(state, params, Tsoil2, dt, is_, ns, nsoilcols,
                              nmeltpoint, tricemethhydr)


def make_soilforlake(ns, nsoilcols, nmeltpoint=1, tricemethhydr=0.0, is_=None):
    """Return a ``jax.jit``-compiled kernel closed over the static sizes.

    The returned callable has signature ``kernel(state, params, Tsoil2, dt)``
    and the same semantics as :func:`soilforlake`.
    """
    is_ = nsoilcols if is_ is None else int(is_)

    def kernel(state, params, Tsoil2, dt):
        return _soilforlake_case1(state, params, Tsoil2, dt, is_, int(ns),
                                  int(nsoilcols), int(nmeltpoint),
                                  float(tricemethhydr))

    return jax.jit(kernel)
