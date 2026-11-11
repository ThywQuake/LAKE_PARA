"""Traceable (``jax.jit``/``lax.scan``-safe) variant of the ``SOILCOLSTEMP`` pass.

Why this module exists
======================

The batched open-water soil pass lives in :mod:`lakematrix.jaxbatch.soilcols`
(``soilcolstemp_water``) and is validated there against the NumPy reference.
It cannot be used inside a traced step, for two independent reasons:

1. its auxiliary routine
   :func:`lakematrix.jaxbatch.soil.soil_cond_heat_coef` evaluates

   .. code-block:: python

       cr = float(jnp.float32(0.2)) * 4180.0

   *inside* the function body.  ``float()`` of a concrete JAX scalar is fine
   eagerly, but under ``jax.jit`` the value is an abstract tracer and raises
   ``ConcretizationTypeError`` (see the "Not ported / gaps" section of the
   ``soilcols`` docstring);
2. its column loop is a Python ``for`` loop that unrolls
   ``nsoilcols - 1`` copies of ``SOILFORLAKE``.  That is correct but makes the
   compiled program (and the compile time) scale with the number of soil
   columns -- 5 copies of ~6 s each for the reference setup.

This module adds **no new physics**: it is

* a verbatim copy of ``jaxbatch.soil.soil_cond_heat_coef`` with the
  single-precision 0.2 literal hoisted to a module constant (``_CR``), and
* the column loop of ``jaxbatch.soilcols._soilcolstemp_water`` rewritten as a
  :func:`jax.lax.scan` over the side columns, calling the *shared*
  ``tsurfsoilcol_water``, ``t_diff_soil`` and ``SOILFORLAKE`` kernels.

``SOILFORLAKE`` is invoked as
``jaxbatch.soilforlake._soilforlake_case1`` because the public wrapper
concretises ``is_`` with ``int()``; the private entry point takes ``is_`` as an
ordinary argument and indexes with it, so a traced column index is fine.

``tests/test_jaxbatch_step.py`` asserts that :func:`soilcolstemp_water`
reproduces :func:`lakematrix.jaxbatch.soilcols.soilcolstemp_water` to machine
precision, so the copy cannot drift away from the reference implementation
silently.

``_CR`` is the value Fortran's default-real literal ``0.2`` has once rounded to
``real(8)`` (``float(np.float32(0.2)) = 0.20000000298023224``); the arithmetic
is identical to the shared module, only evaluated at import time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import C
from ..physics import soil as S
from .soil import soil_cond_johansen, wl_max
from .soilcols import t_diff_soil, tsurfsoilcol_water
from .soilforlake import _soilforlake_case1

__all__ = ["soil_cond_heat_coef", "soilcolstemp_water"]

#: ``cr = 0.2_ireals*4180.`` of ``SOIL_COND_HEAT_COEF`` (soil_mod.f90:63):
#: the literal ``0.2`` is default real, so it is rounded to float32 first.
_CR = float(jnp.float32(0.2)) * 4180.0


def soil_cond_heat_coef(wl1, wi1, POR, rosdry, dzs, BH, DLMAX, FLWMAX,
                        ns, is_):
    """Traceable ``SOIL_COND_HEAT_COEF`` for column ``is_``.

    Copy of :func:`lakematrix.jaxbatch.soil.soil_cond_heat_coef` with ``_CR``
    hoisted to module level (same arithmetic); ``is_`` may be a traced scalar.
    """
    L = wl1.shape[0]
    cr = _CR
    cimh = C.ci
    wl = wl1[:, :, is_]
    wi = wi1[:, :, is_]
    i = jnp.arange(1, ns + 1)
    wl_i = wl[:, i]
    wi_i = wi[:, i]
    csoil = jnp.zeros((L, ns + 1))
    rosoil = jnp.zeros((L, ns + 1))
    lamsoil = jnp.zeros((L, ns + 1))
    lammoist = jnp.zeros((L, ns + 1))
    filtr = jnp.zeros((L, ns + 1))
    wsoil = jnp.zeros((L, ns + 1))
    csoil = csoil.at[:, i].set(cr + wl_i * C.cw + wi_i * cimh)
    rosoil = rosoil.at[:, i].set(rosdry[None, i] * (1.0 - POR[None, i])
                                 * (1.0 + wi_i + wl_i))
    lamsoil = lamsoil.at[:, i].set(
        soil_cond_johansen(wl_i, wi_i, rosdry[None, i], POR[None, i]))
    wlmax = wl_max(POR[None, i], rosdry[None, i], wi_i)
    arg = jnp.maximum(wl_i / wlmax, S._ARG_MIN)
    lammoist = lammoist.at[:, i].set(
        DLMAX[None, i] * arg ** (BH[None, i] + 2.0))
    # inter-layer percolation / vapour diffusivity
    j = jnp.arange(1, ns)
    wlmax_j = wl_max(0.5 * (POR[None, j] + POR[None, j + 1]),
                     0.5 * (rosdry[None, j] + rosdry[None, j + 1]),
                     0.5 * (wi[:, j] + wi[:, j + 1]))
    wlmax_i = wl_max(POR[None, j + 1], rosdry[None, j + 1], wi[:, j + 1])
    fil = jnp.where(
        wlmax_i < S._S001, 0.0,
        0.5 * (FLWMAX[None, j + 1] + FLWMAX[None, j])
        * (0.5 * (wl[:, j + 1] + wl[:, j]) / wlmax_j)
        ** (BH[None, j + 1] + BH[None, j] + 3.0))
    filtr = filtr.at[:, j].set(fil)
    lam1 = 0.5 * (lammoist[:, j] + lammoist[:, j + 1])
    wsoil = wsoil.at[:, j].set(
        (-lam1 * (wl[:, j + 1] - wl[:, j]) / dzs[None, j] + filtr[:, j])
        * 0.5 * (rosdry[None, j] + rosdry[None, j + 1])
        * (1.0 - 0.5 * (POR[None, j] + POR[None, j + 1])) / C.row0)
    return csoil, rosoil, lamsoil, lammoist, filtr, wsoil


def soilcolstemp_water(state, params, geom, dt):
    """Traceable case-1 ``SOILCOLSTEMP`` heat pass over the side columns.

    Same interface and same result as
    :func:`lakematrix.jaxbatch.soilcols.soilcolstemp_water`, with the column
    loop replaced by a ``lax.scan`` and a traceable ``SOIL_COND_HEAT_COEF``.
    """
    Tsoil1 = jnp.asarray(state["Tsoil1"])
    ns = Tsoil1.shape[-2] - 1
    ncols = Tsoil1.shape[-1] - 1
    M = jnp.asarray(geom["ddz"]).shape[-1] - 1
    return _soilcolstemp_water(state, params, geom, dt, ns, ncols, M)


def _soilcolstemp_water(state, params, geom, dt, ns, ncols, M):
    ns, ncols, M = int(ns), int(ncols), int(M)
    conj = int(params.get("soilcolconjtype", 2))
    if conj != 2:
        raise NotImplementedError(
            f"SOILCOLSTEMP: soilcolconjtype={conj} is not ported")
    nmeltpoint = int(params.get("nmeltpoint", 1))
    tricemethhydr = float(params.get("tricemethhydr", 0.0))
    if nmeltpoint not in (0, 1):
        raise NotImplementedError(
            "MELTPNT: nmeltpoint=2 (Jackett et al. 2004) is not ported")
    if tricemethhydr > 0.0:
        raise NotImplementedError("tricemethhydr > 0 is not ported")
    p = dict(params)
    p.update(ns=ns, nsoilcols=ncols, nmeltpoint=nmeltpoint,
             tricemethhydr=tricemethhydr)

    Tsoil1 = jnp.asarray(state["Tsoil1"])
    Tw1 = jnp.asarray(state["Tw1"])
    h1 = jnp.atleast_1d(jnp.asarray(state["h1"]))
    L = Tsoil1.shape[0]

    POR = jnp.asarray(params["POR"])
    rosdry = jnp.asarray(params["rosdry"])
    dzs = jnp.asarray(params["dzs"])
    BH = jnp.asarray(params["BH"])
    DLMAX = jnp.asarray(params["DLMAX"])
    FLWMAX = jnp.asarray(params["FLWMAX"])
    soilbotflx = params["soilbotflx"]

    # Tsoilsurf for the side columns, computed once before the loop.
    Tsoilsurf = tsurfsoilcol_water(
        Tw1, h1, geom["zsoilcols"], geom["ddz"], geom["w_lx"], geom["w_ly"],
        geom["w_lx_half"], geom["w_ly_half"], geom["s_lx"], geom["s_ly"],
        M, ncols)

    if "soilflux" in state:
        soilflux = jnp.asarray(state["soilflux"])
    else:
        soilflux = jnp.zeros((L, ncols + 1))

    out = dict(state)
    out["soilflux"] = soilflux
    if "dhwfsoil" not in out:          # SOILFORLAKE writes it; keep the
        out["dhwfsoil"] = jnp.zeros((L,))   # scan structure stable
    dt_inv = 1.0 / dt

    def column(carry, is_):
        """One ``SOILCOLSTEMP`` iteration; ``is_`` is a traced column index."""
        csoil, rosoil, lamsoil, lammoist, filtr, wsoil = soil_cond_heat_coef(
            carry["wl1"], carry["wi1"], POR, rosdry, dzs, BH, DLMAX, FLWMAX,
            ns, is_)

        Tsoil2 = t_diff_soil(carry["Tsoil1"][:, :, is_], lamsoil, csoil,
                             rosoil, dzs, Tsoilsurf[:, is_], dt, soilbotflx,
                             ns)

        # Heat flux into the soil, W/m2 (soil_mod.f90:929-932).
        flux_is = (csoil[:, 1] * rosoil[:, 1] * dt_inv * 0.5 * dzs[1]
                   * (Tsoil2[:, 1] - carry["Tsoil1"][:, 1, is_])
                   - 0.25 * (lamsoil[:, 1] + lamsoil[:, 2])
                   * (Tsoil2[:, 2] + carry["Tsoil1"][:, 2, is_]
                      - Tsoil2[:, 1] - carry["Tsoil1"][:, 1, is_]) / dzs[1])
        soilflux = carry["soilflux"].at[:, is_].set(flux_is)

        # SOILFORLAKE reads the module-wide csoil/rosoil/lammoist/filtr of this
        # column and Tsoil2 just computed; hand them over in the state dict.
        work = dict(carry)
        work.update(csoil=csoil, rosoil=rosoil, lamsoil=lamsoil,
                    lammoist=lammoist, filtr=filtr, wsoil=wsoil,
                    Tsoil2=Tsoil2, soilflux=soilflux)
        new = _soilforlake_case1(work, p, Tsoil2, dt, is_, ns, ncols,
                                 nmeltpoint, tricemethhydr)
        new["soilflux"] = soilflux
        return new, None

    # side columns 1 .. ncols-1 (the deepest column is handled by the caller,
    # after T_SOLVER), exactly like SOILCOLSTEMP.
    out, _ = jax.lax.scan(column, out, jnp.arange(1, ncols))
    out["Tsoilsurf"] = Tsoilsurf
    return out
