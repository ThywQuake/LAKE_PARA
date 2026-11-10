"""Ice and snow thermodynamics: ``T_DIFF`` ice/snow branches, ``DIFF_COEF``
ice / ice-salinity / snow / deep-ice cases, and the snow physics.

Fortran source
==============

===============  ===========================================================
Fortran          Python
===============  ===========================================================
``T_DIFF``       :func:`t_diff_ice`        (``t_solver_mod.f90:219-656``)
``DIFF_COEF``    :func:`diff_coef_ice`, :func:`diff_coef_ice_sal`,
                 :func:`diff_coef_snow`, :func:`diff_coef_deepice`
                 (``t_solver_mod.f90:662-937``; the soil and water cases stay
                 in :mod:`lakematrix.physics.tsolver`)
``SNOWSOIL_MOD`` :class:`IceState`          (``lake_modules.f90:2316-2351``:
                 ``T``, ``Tsn``, ``cs``, ``dens``, ``dz``, ``itop``, ``lams``,
                 ``wl``, ``z``)
``SNOW_COND_HEAT_COEF``  :func:`snow_cond_heat_coef` (``snowtemp.f90:100-179``)
``SNOW_CALC``    :func:`snow_calc`          (``snowcalc.f90:5-497``; the
                 FLake ``SNOWSOIL`` routine's densification/melt)
``SR_botsnow``   :func:`sr_botsnow`         (``lake.f90:1339-1344``,
                 ``1921-1926``)
case 2 ice       :func:`ice_increments`     (``lake.f90:1334-1445``)
``UPDATE_ICE_SALINITY`` :func:`update_ice_salinity`
                 (``salinity_mod.f90:306-395``)
``MELTPNT``      :func:`lakematrix.physics.soil.meltpnt`
``EXTINCT_SNOW`` :func:`extinct_snow`       (``phys_func.f90:1324-1343``)
===============  ===========================================================

Note on the Fortran layout
==========================

The LAKE 3.2 checkout this project ports (``lakebatch/core/source/model``)
has **no** ``snowsoil_mod.f90``: ``SNOWSOIL_MOD`` (``lake_modules.f90:2316``)
only declares the module storage (``lams``, ``q``, ``T``, ``Tsn``, ``cs``,
``dens``, ``dz``, ``wl``, ``z``, ``itop``) and its allocator.  The physics
that the FLake ``SNOWSOIL`` routine performs lives in ``SNOW_CALC`` /
``addPrecipToSnow`` (``snowcalc.f90``) and, for the ice thickness increments
``dhi``/``dhi0``/``dls``/``snmeltice``, inline in ``lake.f90`` case 2/3.
Those are the routines transcribed here; see "Not transcribed" below.

Indexing convention
===================

Like :mod:`lakematrix.physics.soil` and :mod:`lakematrix.physics.tsolver`,
every array whose Fortran lower bound is 1 is stored **1-based**: index 0 is
an unused pad and ``a[i]`` is Fortran ``a(i)``.  This keeps the transcription
auditable (``Ti1[j]``, ``cs[j]``, ``lams[j]`` read like the source).

Arrays whose Fortran lower bound is 0 keep it: ``ddzi05[0:Mice]``,
``ddz05[0:M]``, ``rad_integr_ice[0:Mice]``, ``rad_integr_water[0:M]``,
``rad_integr_deepice[0:Mice]``.  ``dzeta_int``/``dzetai_int`` are rebuilt
from ``ddz``/``ddzi`` with the ``DZETA``/``DZETAI`` recurrence and padded, so
``dzeta_int[j]`` is Fortran ``dzeta_int(j)`` (this is *not* the same offset as
``lakematrix.grid.Grid.dzeta_int``, whose index ``j`` is Fortran ``j+1``).

Floating point
==============

``ireals`` is ``real(8)``, but most literals in the source are default real
(``real(4)``).  A default-real literal assigned to a ``real(8)`` variable (or
used in a mixed expression) is first rounded to ``float32``, so ``0.419``
becomes ``float(np.float32(0.419))``.  ``0.25``/``0.5``/``2.`` are exact and
need no rounding.  Literals written with a ``d`` exponent are double.  The
helper ``_s()`` marks every single-precision literal; ``_sdiv()`` marks a
quotient of two default-real values evaluated in single precision.

**Multiplication order is preserved literally.**  The Fortran expressions are
not algebraically rearranged (``0.5*Ti1*ci*ddzi*l1*dt_inv`` in one branch is
``Ti1*0.5*dt_inv*ci*ddzi*l1`` in another, and the two round differently), so
the transcribed expression keeps the same operand order.  Do not "simplify".

Typical call order (``lake.f90`` case 2.1)
==========================================

1. :func:`ice_properties` (``ci_m_roi_v``/``lami_v``) and
   :func:`lakematrix.physics.soil.soil_cond_heat_coef` for the soil columns.
2. :func:`sr_botsnow` and the ``RadIce``/``RadWater``/``RadDeepIce`` updates
   (already in :mod:`lakematrix.physics.radiation`).
3. :func:`snow_cond_heat_coef` (``cs``/``lams`` needed by ``T_DIFF`` and by
   the snow-ice interface heat capacity ``y``).
4. :func:`ice_increments` (or :func:`ice_increments_nosnow`) for
   ``dhi``/``dhi0``/``dls``/``dls0``/``snmeltice``/``dhiimp``/``dhw``.
5. :func:`lakematrix.physics.tsolver.t_solver` with
   ``fn = lambda Ts: (heat_balance, t_diff_ice(1, Ts, ...))``; the final
   ``t_diff_ice(0, Tsurf3, ...)`` of ``T_SOLVER`` updates ``Tw2``/``Tis2``.
6. ``SOILFORLAKE``/``S_DIFF`` and then :func:`snow_calc` (``SNOWTEMP``
   calls ``addPrecipToSnow`` first - see below).

Not transcribed (TODO, deliberately left rather than invented)
==============================================================

* ``addPrecipToSnow`` (``snowcalc.f90:500-706``) - precipitation
  accumulation and the layer split at the top of the pack.  :func:`snow_calc`
  therefore takes its three outputs ``pheat``, ``pmelt`` and ``CCT`` as
  arguments.  ``pmelt``/``CCT`` are not read by ``SNOW_CALC`` (``pheat`` is,
  as ``Erain``), so a caller that has no precipitation can pass zeros.
* ``T_SOLVER``'s scan/Chorde loop is already in
  :func:`lakematrix.physics.tsolver.t_solver`; it is surface-type agnostic.
* ``nmeltpoint == 2`` (Jackett et al. 2004) - raises ``NotImplementedError``
  in :func:`lakematrix.physics.soil.meltpnt`.
* ``saltice == 1`` (saline ice) property update ``ci_m_roi_v``/``lami_v``
  (``lake.f90:1316-1329``) is not part of ``T_DIFF``; :func:`ice_properties`
  is provided for convenience but ``T_DIFF`` only reads the arrays.
* ``SOILFORLAKE``/``SOILCOLSTEMP`` after ``T_SOLVER`` stay in
  :mod:`lakematrix.physics.soil`.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from ..constants import C
from .soil import Bathym, meltpnt
from .tsolver import diff_coef_soil, diff_coef_water


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _s(x):
    """Value of a default-real (``real(4)``) literal widened to ``real(8)``."""
    return float(np.float32(x))


def _sdiv(a, b):
    """A quotient of two default-real operands, evaluated in ``float32``."""
    return float(np.float32(a) / np.float32(b))


def _area_int(bathym):
    """``area_int`` of a 1-based :class:`Bathym` list as a 1-based array."""
    out = np.zeros(len(bathym))
    for j in range(1, len(bathym)):
        out[j] = bathym[j].area_int
    return out


def _area_half(bathym):
    """``area_half`` of a 1-based :class:`Bathym` list as a 1-based array."""
    out = np.zeros(len(bathym))
    for j in range(1, len(bathym)):
        out[j] = bathym[j].area_half
    return out


# ``DIFF_COEF`` substrate indicators (lake_modules.f90:729-733).
SOIL_INDIC = 1
ICE_INDIC = 2
SNOW_INDIC = 3
WATER_INDIC = 4
DEEPICE_INDIC = 5
ICE_SAL_INDIC = 12

#: ``soilswitch%par`` values used below (``DRIVING_PARAMS``).
SOILSWITCH_ON = 1
SOILSWITCH_OFF = 0


def extinct_snow(snow_density):
    """``EXTINCT_SNOW`` (``phys_func.f90:1324-1343``).

    ``exp(-(0.13d0*dens + 3.4d0))``; the two constants are double.
    ``extinct_snow_max`` in the source is computed but never used.
    """
    return np.exp(-(0.13 * snow_density + 3.4))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class IceState:
    """``SNOWSOIL_MOD`` storage plus the ice/snow inputs of ``T_DIFF``.

    All arrays are 1-based (index 0 unused) unless the field comment says
    otherwise.  The object is *the* module storage of the Fortran: several
    branches leave entries unwritten and the next call inherits them, so do
    not recreate it between steps (same rule as
    :class:`lakematrix.physics.soil.SoilState`).

    Grid (``ARRAYS_GRID`` / :class:`lakematrix.grid.Grid`)
        ``M``, ``Mice``, ``ns``, ``ms``, ``ml``, ``nveclen``;
        ``ddz`` (1-based, ``M``), ``ddz05`` (0-based, ``0:M``),
        ``ddzi`` (1-based, ``Mice``), ``ddzi05`` (0-based, ``0:Mice``),
        ``dzeta_int`` (1-based), ``dzetai_int`` (1-based).

    Bathymetry (``BATHYMETRY``, rebuilt every step)
        ``bathymice``, ``bathymwater``, ``bathymdice`` - 1-based lists of
        :class:`lakematrix.physics.bathym.Bathym`, so ``bathymice[j].area_int``
        is Fortran ``bathymice(j)%area_int``.

    Snow (``SNOWSOIL_MOD``; ``T``/``z``/``wl`` are allocated ``1:ML`` in the
        source while ``Tsn``/``cs``/``dens``/``dz``/``lams`` are ``1:ms``)
        ``T``, ``Tsn``, ``cs``, ``dens``, ``dz``, ``lams``, ``wl``,
        ``extinct``, ``z``, ``itop``, ``hs1``, ``snowmass``,
        ``snowmass_init``, ``snmelt``, ``swe``, ``flag_snow_init``.

    Ice (``ARRAYS_WATERSTATE``)
        ``Ti1``, ``Ti2`` (``Mice+1``), ``ci_m_roi_v``, ``lami_v``,
        ``salice``, ``porice``, ``dhi``, ``dhi0``.

    Deep ice
        ``Tis1``, ``Tis2`` (``Mice+1``), ``dls``, ``dls0``.

    Water (read by the ice/snow branches)
        ``Tw1``, ``Tw2``, ``lamw``, ``dhw``, ``dhw0``, ``PEMF``,
        ``pt_down_f``, ``Sal2``, ``preswat``.

    Soil (read by the ice-soil / snow-soil interfaces and the soil tail)
        ``Tsoil1`` ``(ns, nsoilcols)``, ``Tsoil2`` (``ns``), ``lamsoil``,
        ``csoil``, ``rosoil``, ``dzs``, ``isoilcol``.

    Radiation (``integr`` keeps its 0-based bound)
        ``rad_integr_ice``, ``rad_integr_water``, ``rad_integr_deepice``.

    Lateral soil heat source (``lsh_type``)
        ``lsh_ice`` (``lsh%ice``), ``lsh_dice`` (``lsh%dice``).

    Switches (``DRIVING_PARAMS``)
        ``soilswitch``, ``nmeltpoint``, ``soilbotflx``.
    """

    # --- grid -------------------------------------------------------------
    M: int
    Mice: int
    ns: int
    ms: int
    ml: int
    nveclen: int
    ddz: np.ndarray
    ddz05: np.ndarray
    ddzi: np.ndarray
    ddzi05: np.ndarray
    dzeta_int: np.ndarray
    dzetai_int: np.ndarray
    # --- bathymetry -------------------------------------------------------
    bathymice: List[Bathym]
    bathymwater: List[Bathym]
    bathymdice: List[Bathym]
    # --- snow (SNOWSOIL_MOD) ---------------------------------------------
    T: np.ndarray
    Tsn: np.ndarray
    cs: np.ndarray
    dens: np.ndarray
    dz: np.ndarray
    lams: np.ndarray
    wl: np.ndarray
    extinct: np.ndarray
    z: np.ndarray
    # --- ice --------------------------------------------------------------
    Ti1: np.ndarray
    Ti2: np.ndarray
    ci_m_roi_v: np.ndarray
    lami_v: np.ndarray
    salice: np.ndarray
    porice: np.ndarray
    # --- deep ice ---------------------------------------------------------
    Tis1: np.ndarray
    Tis2: np.ndarray
    # --- water ------------------------------------------------------------
    Tw1: np.ndarray
    Tw2: np.ndarray
    lamw: np.ndarray
    PEMF: np.ndarray
    pt_down_f: np.ndarray
    Sal2: np.ndarray
    preswat: np.ndarray
    # --- soil -------------------------------------------------------------
    Tsoil1: np.ndarray
    Tsoil2: np.ndarray
    lamsoil: np.ndarray
    csoil: np.ndarray
    rosoil: np.ndarray
    dzs: np.ndarray
    # --- radiation / heat sources ----------------------------------------
    rad_integr_ice: np.ndarray
    rad_integr_water: np.ndarray
    rad_integr_deepice: np.ndarray
    lsh_ice: np.ndarray
    lsh_dice: np.ndarray
    lsh_water: np.ndarray
    # --- snow scalars (SNOWSOIL_MOD) --------------------------------------
    itop: int = 1
    #: ``flag_snow``: 0 = no snow cover, 1 = snow cover present (``ARRAYS``).
    flag_snow: int = 0
    hs1: float = 0.0
    snowmass: float = 0.0
    snowmass_init: float = 0.0
    snmelt: float = 0.0
    swe: float = 0.0
    flag_snow_init: int = 1
    # --- layer thicknesses and increments ---------------------------------
    #: ``h1``/``l1``/``ls1`` layer thicknesses (read by several coefficients).
    h1: float = 0.0
    l1: float = 0.0
    ls1: float = 0.0
    #: ``dhi``/``dhi0`` surface-ice and ``dls``/``dls0`` deep-ice increments.
    dhi: float = 0.0
    dhi0: float = 0.0
    dls: float = 0.0
    dls0: float = 0.0
    dhw: float = 0.0
    dhw0: float = 0.0
    #: ``lake.f90`` case-2 scalars read by the snow-ice interface of
    #: ``T_DIFF``: frozen snowmelt and ice impoundment (see
    #: :func:`ice_increments`).  ``cuette`` selects the Couette b.c. of the
    #: water-only branch (``cuette%par``).
    snmeltice: float = 0.0
    dhiimp: float = 0.0
    cuette: int = 0
    #: ``lake.f90:1368-1445`` thickness increments of the current step.
    dhwlow: float = 0.0
    dhilow: float = 0.0
    dhwhigh: float = 0.0
    dhihigh: float = 0.0
    dhwsnmelt: float = 0.0
    dhwls: float = 0.0
    #: case-2.2 surface increments: ``dhis`` (heat-loss ice growth) and
    #: ``dhip`` (precipitation ice).
    dhis: float = 0.0
    dhip: float = 0.0
    isoilcol: int = 1
    #: ``saltice%par``: 0 = pure ice, 1 = ice with brine-filled pores.
    saltice: int = 0
    #: External water increments of the current step (``dhwfsoil`` soil heat
    #: flux, ``dhwtrib`` tributaries), added in :func:`ice_increments`.
    dhwfsoil: float = 0.0
    dhwtrib: float = 0.0
    # --- switches ---------------------------------------------------------
    soilswitch: int = SOILSWITCH_ON
    nmeltpoint: int = 1
    soilbotflx: float = 0.0
    SR_botsnow: float = 0.0


def make_ice_state(M, Mice, ns, ms, *, ml=131, nveclen=None,
                   geom=None, isoilcol=None, soilswitch=SOILSWITCH_ON,
                   nmeltpoint=1, soilbotflx=0.0) -> IceState:
    """Allocate a zeroed :class:`IceState` with the Fortran array bounds.

    ``geom`` (a :class:`lakematrix.physics.soil.SoilGeometry`) supplies
    ``ddz``/``ddzi``/``ddz05``/``ddzi05`` and the three bathymetry lists when
    given; otherwise zero/identity defaults are used.  ``nveclen`` defaults to
    ``NUMERIC_PARAMS_SET``'s ``max(ms+Mice+ns, M+ns)``.
    """
    if nveclen is None:
        nveclen = max(ms + Mice + ns, M + ns)
    z = lambda n: np.zeros(n, dtype=float)
    one = lambda n: np.ones(n, dtype=float)
    nn = max(ms, ml)

    ddz = z(M + 1)
    ddz05 = z(M + 1)
    ddzi = z(Mice + 1)
    ddzi05 = z(Mice + 1)
    dzeta_int = z(M + 2)
    dzetai_int = z(Mice + 2)
    bwater = [Bathym() for _ in range(M + 2)]
    bice = [Bathym() for _ in range(Mice + 2)]
    bdice = [Bathym() for _ in range(Mice + 2)]
    if geom is not None:
        ddz[:] = geom.ddz
        ddz05[:] = geom.ddz05
        ddzi[:] = geom.ddzi
        ddzi05[:] = geom.ddzi05
        bwater = list(geom.bathymwater)
        bice = list(geom.bathymice)
        bdice = list(geom.bathymdice)

    # DZETA/DZETAI (dzeta_mod.f90): dzeta_int(1) = 0,
    # dzeta_int(i) = dzeta_int(i-1) + ddz(i-1); 1-based with a pad at 0.
    for i in range(2, M + 2):
        dzeta_int[i] = dzeta_int[i - 1] + ddz[i - 1]
    for i in range(2, Mice + 2):
        dzetai_int[i] = dzetai_int[i - 1] + ddzi[i - 1]

    return IceState(
        M=M, Mice=Mice, ns=ns, ms=ms, ml=nn, nveclen=nveclen,
        ddz=ddz, ddz05=ddz05, ddzi=ddzi, ddzi05=ddzi05,
        dzeta_int=dzeta_int, dzetai_int=dzetai_int,
        bathymice=bice, bathymwater=bwater, bathymdice=bdice,
        T=z(nn + 1), Tsn=z(ms + 1), cs=z(ms + 1), dens=z(ms + 1),
        dz=z(ms + 1), lams=z(ms + 1), wl=z(nn + 1), extinct=z(nn + 1),
        z=z(nn + 1),
        Ti1=z(Mice + 2), Ti2=z(Mice + 2),
        ci_m_roi_v=one(Mice + 2) * C.ci_m_roi, lami_v=one(Mice + 1) * C.lami,
        salice=z(Mice + 2), porice=z(Mice + 2),
        Tis1=z(Mice + 2), Tis2=z(Mice + 2),
        Tw1=z(M + 2), Tw2=z(M + 2), lamw=z(M + 1),
        PEMF=z(M + 2), pt_down_f=z(M + 2), Sal2=z(M + 2), preswat=z(M + 2),
        Tsoil1=z((ns + 1) * (2 if isoilcol is None else isoilcol + 1)).reshape(
            ns + 1, 2 if isoilcol is None else isoilcol + 1),
        Tsoil2=z(ns + 1),
        lamsoil=z(ns + 1), csoil=z(ns + 1), rosoil=z(ns + 1), dzs=z(ns + 1),
        isoilcol=(1 if isoilcol is None else isoilcol),
        rad_integr_ice=z(Mice + 1), rad_integr_water=z(M + 1),
        rad_integr_deepice=z(Mice + 1),
        lsh_ice=z(Mice + 2), lsh_dice=z(Mice + 2), lsh_water=z(M + 2),
        soilswitch=soilswitch, nmeltpoint=nmeltpoint,
        soilbotflx=float(soilbotflx),
    )


# ---------------------------------------------------------------------------
# DIFF_COEF, ice case
# ---------------------------------------------------------------------------

def diff_coef_ice(a, b, c, d, n0, n1, m0, m1, dt, st: IceState):
    """``DIFF_COEF`` case ``ice_indic`` (``t_solver_mod.f90:737-753``).

    ``n0``/``n1`` are the ice node range (``2..Mice`` in every call site),
    ``m0``/``m1`` the target slots in the ``PROGONKA`` arrays.  ``j`` is the
    ice node index, ``j-1``/``j+1`` its neighbours; ``ddzi05[j-1]`` is
    Fortran ``ddzi05(j-1)`` (0-based array).
    """
    dt_inv = 1.0 / dt
    dt_inv05 = 0.5 * dt_inv
    l1 = st.l1
    for i in range(m0, m1 + 1):
        j = i - m0 + n0
        a[i] = 0.5 * (
            -st.lami_v[j - 1] * st.bathymice[j - 1].area_half
            / (st.ddzi[j - 1] * l1 * st.bathymice[j].area_int)
            + st.ci_m_roi_v[j] * st.dhi * st.dzetai_int[j] * dt_inv05
            - st.ci_m_roi_v[j] * st.dhi0 * dt_inv05)
        b[i] = 0.5 * (
            -st.lami_v[j] * st.bathymice[j].area_half
            / (st.ddzi[j] * l1 * st.bathymice[j].area_int)
            - st.ci_m_roi_v[j] * st.dhi * st.dzetai_int[j] * dt_inv05
            + st.ci_m_roi_v[j] * st.dhi0 * dt_inv05)
        c[i] = (a[i] + b[i]
                - dt_inv * st.ddzi05[j - 1] * l1 * st.ci_m_roi_v[j])
        d[i] = (-st.Ti1[j] * dt_inv * st.ddzi05[j - 1] * l1 * st.ci_m_roi_v[j]
                + st.rad_integr_ice[j] - st.rad_integr_ice[j - 1]
                + a[i] * st.Ti1[j - 1] + b[i] * st.Ti1[j + 1]
                - (a[i] + b[i]) * st.Ti1[j]
                - st.lsh_ice[j] * st.ddzi05[j - 1] * l1)


def diff_coef_ice_sal(a, b, c, d, n0, n1, m0, m1, dt, st: IceState):
    """``DIFF_COEF`` case ``ice_sal_indic`` (``t_solver_mod.f90:754-764``).

    Bulk ice salinity, no diffusion: the equation keeps the metric (moving
    ice) terms and the brine drainage ``asalice*salice``.  Note it uses
    ``dzetai_int[j]`` and ``dhi``/``dhi0`` exactly like the temperature case.
    """
    dt_inv = 1.0 / dt
    dt_inv05 = 0.5 * dt_inv
    l1 = st.l1
    for i in range(m0, m1 + 1):
        j = i - m0 + n0
        a[i] = 0.5 * (st.dhi * st.dzetai_int[j] * dt_inv05
                      - st.dhi0 * dt_inv05)
        b[i] = 0.5 * (-st.dhi * st.dzetai_int[j] * dt_inv05
                      + st.dhi0 * dt_inv05)
        c[i] = a[i] + b[i] - dt_inv * st.ddzi05[j - 1] * l1
        d[i] = (-st.salice[j] * dt_inv * st.ddzi05[j - 1] * l1
                + a[i] * st.salice[j - 1] + b[i] * st.salice[j + 1]
                - (a[i] + b[i]) * st.salice[j]
                - C.asalice * st.salice[j] * st.ddzi05[j - 1] * l1)


# ---------------------------------------------------------------------------
# DIFF_COEF, snow case
# ---------------------------------------------------------------------------

def diff_coef_snow(a, b, c, d, n0, n1, m0, m1, dt, st: IceState):
    """``DIFF_COEF`` case ``snow_indic`` (``t_solver_mod.f90:765-777``).

    ``j`` indexes the snow layers (``1..ms-1``); the node ``ms`` is the
    snow-ice (or snow-soil) interface and is set by :func:`t_diff_ice`.
    """
    dt_inv = 1.0 / dt
    for i in range(m0, m1 + 1):
        j = i - m0 + n0
        lam2 = 0.5 * (st.lams[j] + st.lams[j + 1])
        lam1 = 0.5 * (st.lams[j - 1] + st.lams[j])
        dzmean = 0.5 * (st.dz[j - 1] + st.dz[j])
        a[i] = -0.5 * lam1 / st.dz[j - 1]
        b[i] = -0.5 * lam2 / st.dz[j]
        c[i] = a[i] + b[i] - dt_inv * (st.cs[j] * st.dens[j] * dzmean)
        d[i] = (-st.T[j] * dt_inv * (st.cs[j] * st.dens[j] * dzmean)
                + a[i] * st.T[j - 1]
                - (a[i] + b[i]) * st.T[j]
                + b[i] * st.T[j + 1])


# ---------------------------------------------------------------------------
# DIFF_COEF, deep ice case
# ---------------------------------------------------------------------------

def diff_coef_deepice(a, b, c, d, n0, n1, m0, m1, dt, st: IceState):
    """``DIFF_COEF`` case ``deepice_indic`` (``t_solver_mod.f90:805-824``).

    Deep (bottom) ice uses the *pure ice* constants ``lami``/``ci_m_roi``, not
    the porosity-corrected ``lami_v``/``ci_m_roi_v`` of the surface ice, and
    the deep-ice thickness ``ls1``/``dls``/``dls0``.
    """
    dt_inv = 1.0 / dt
    dt_inv05 = 0.5 * dt_inv
    lami = C.lami
    ci_m_roi = C.ci_m_roi
    ls1 = st.ls1
    for i in range(m0, m1 + 1):
        j = i - m0 + n0
        a[i] = 0.5 * (
            -lami * st.bathymdice[j - 1].area_half
            / (st.ddzi[j - 1] * ls1 * st.bathymdice[j].area_int)
            + ci_m_roi * st.dls * st.dzetai_int[j] * dt_inv05
            - ci_m_roi * st.dls0 * dt_inv05)
        b[i] = 0.5 * (
            -lami * st.bathymdice[j].area_half
            / (st.ddzi[j] * ls1 * st.bathymdice[j].area_int)
            - ci_m_roi * st.dls * st.dzetai_int[j] * dt_inv05
            + ci_m_roi * st.dls0 * dt_inv05)
        c[i] = (a[i] + b[i]
                - dt_inv * st.ddzi05[j - 1] * ls1 * ci_m_roi)
        d[i] = (-st.Tis1[j] * dt_inv * st.ddzi05[j - 1] * ls1 * ci_m_roi
                + st.rad_integr_deepice[j] - st.rad_integr_deepice[j - 1]
                + a[i] * st.Tis1[j - 1] + b[i] * st.Tis1[j + 1]
                - (a[i] + b[i]) * st.Tis1[j]
                - st.lsh_dice[j] * st.ddzi05[j - 1] * ls1)


# ---------------------------------------------------------------------------
# PROGONKA
# ---------------------------------------------------------------------------

def _progonka(a, b, c, d, k, n):
    """``PROGONKA(a,b,c,d,Temp,k,n)`` on the 1-based arrays (numerics_mod.f90).

    ``k``/``n`` are Fortran indices; the returned vector has ``a``'s shape with
    only ``k..n`` filled (the Fortran solver overwrites ``Temp`` in place).
    """
    from .tridiag import progonka
    y = progonka(a[k:n + 1], b[k:n + 1], c[k:n + 1], d[k:n + 1])
    out = np.zeros_like(a)
    out[k:n + 1] = y
    return out


# ---------------------------------------------------------------------------
# T_DIFF
# ---------------------------------------------------------------------------

def t_diff_ice(surf, Tsurf, dt, snow, ice, water, deepice, isoilcol, ifflux,
               st: IceState) -> IceState:
    """``T_DIFF`` (``t_solver_mod.f90:219-656``) for the ice/snow branches.

    ``surf == 1`` solves the whole profile with ``Tsurf`` as the Dirichlet
    surface temperature (used by the ``T_SOLVER`` scan/Chorde iteration);
    ``surf == 0`` re-solves only the water / deep-ice column with the melting
    point at the ice-water interface (``T_DIFF`` returns immediately when
    ``water == 0``, ``t_solver_mod.f90:514``).

    ``snow``/``ice``/``water``/``deepice`` are the ``0/1`` flags of
    ``ARRAYS``; ``isoilcol`` is the soil column used by the ice-soil and
    snow-soil interfaces (``nsoilcols`` at every ``T_SOLVER`` call site);
    ``ifflux`` makes ``Tsurf`` a downward heat flux in the soil-only branch.

    The results are written into ``st`` (``Tsn``, ``T`` is not modified,
    ``Ti2``, ``Tw2``, ``Tis2``, ``Tsoil2``) and ``st`` is returned.
    ``T`` (the *old* snow temperature) is only read here; it is refreshed from
    ``Tsn`` by ``SNOWTEMP`` before ``SNOW_CALC`` runs.

    ``cuette%par`` values other than ``0``/``1``/``11`` leave the water-only
    system unsolved in the source (``Temp`` is never computed, so ``Tw2`` would
    be undefined); a ``NotImplementedError`` is raised instead of returning
    garbage.

    Deviations from a "clean" implementation are the source's own and are
    kept: ``d(Mice+1)`` in the "water + deep ice, no soil" branch does **not**
    scale ``RadDeepIce%integr(Mice)`` by ``xx`` (``t_solver_mod.f90:639-643``)
    while every other deep-ice interface does, and the snow-ice no-soil branch
    mixes ``ci_m_roi`` (pure ice) into a ``ci_m_roi_v`` expression
    (``t_solver_mod.f90:327``).
    """
    M, Mice, ms, ns = st.M, st.Mice, st.ms, st.ns
    nn = st.nveclen
    a = np.zeros(nn + 1)
    b = np.zeros(nn + 1)
    c = np.zeros(nn + 1)
    d = np.zeros(nn + 1)

    dt_inv = 1.0 / dt
    h1, l1, ls1 = st.h1, st.l1, st.ls1
    cw = C.cw_m_row0
    Lwi, row0 = C.Lwi, C.row0
    Tsoil1 = st.Tsoil1
    ic = isoilcol

    # ======================================================================
    # surf == 1: Dirichlet surface temperature
    # ======================================================================
    if surf == 1:

        # ---------------------- snow exists -------------------------------
        if snow == 1:
            itop = st.itop
            c[itop] = 1.0
            b[itop] = 0.0
            d[itop] = Tsurf
            if itop <= ms - 2:
                diff_coef_snow(a, b, c, d, itop + 1, ms - 1, itop + 1, ms - 1,
                               dt, st)

            if ice == 1:
                # ------------------ SNOW-ICE INTERFACE (t_solver 280-288) --
                a[ms] = -0.25 * (st.lams[ms - 1] + st.lams[ms]) / st.dz[ms - 1]
                b[ms] = 0.5 * (-st.lami_v[1] / (st.ddzi[1] * l1)
                               + 0.5 * st.ci_m_roi_v[1] * st.dhi0 * dt_inv)
                c[ms] = (a[ms] + b[ms]
                         - 0.5 * dt_inv * (st.cs[ms] * st.dens[ms] * st.dz[ms]
                                           + st.ci_m_roi_v[1] * st.ddzi[1] * l1))
                d[ms] = (-st.Ti1[1] * 0.5 * dt_inv
                         * (st.cs[ms] * st.dens[ms] * st.dz[ms]
                            + st.ci_m_roi_v[1] * st.ddzi[1] * l1)
                         - st.SR_botsnow + st.rad_integr_ice[1]
                         + a[ms] * st.T[ms - 1]
                         - (a[ms] + b[ms]) * st.Ti1[1]
                         + b[ms] * st.Ti1[2]
                         - (st.snmeltice + st.dhiimp) * Lwi * row0 * dt_inv)
                # ----------------------------------------------------------
                diff_coef_ice(a, b, c, d, 2, Mice, ms + 1, ms + Mice - 1,
                              dt, st)

                if water == 1:
                    # Snow, ice and water
                    a[ms + Mice] = 0.0
                    c[ms + Mice] = 1.0
                    d[ms + Mice] = meltpnt(st.Sal2[1], st.preswat[1],
                                           st.nmeltpoint)
                    Temp = _progonka(a, b, c, d, itop, ms + Mice)
                    st.Tsn[itop:ms + 1] = Temp[itop:ms + 1]
                    st.Ti2[1:Mice + 2] = Temp[ms:ms + Mice + 1]
                else:
                    if st.soilswitch == SOILSWITCH_ON:
                        # ------------- ICE-SOIL INTERFACE (303-313) -------
                        xx = (st.bathymice[Mice + 1].area_int
                              / st.bathymice[Mice].area_half)
                        a[ms + Mice] = 0.5 * (
                            -st.lami_v[Mice] / (st.ddzi[Mice] * l1)
                            + xx * 0.5 * st.ci_m_roi_v[Mice + 1]
                            * (st.dhi - st.dhi0) * dt_inv)
                        b[ms + Mice] = (-0.25
                                        * (st.lamsoil[1] + st.lamsoil[2])
                                        / st.dzs[1])
                        cc = 0.5 * dt_inv * (
                            st.csoil[1] * st.rosoil[1] * st.dzs[1]
                            + xx * st.ci_m_roi_v[Mice + 1]
                            * st.ddzi[Mice] * l1)
                        c[ms + Mice] = a[ms + Mice] + b[ms + Mice] - cc
                        d[ms + Mice] = (
                            -st.Ti1[Mice + 1] * cc
                            - st.rad_integr_ice[Mice]
                            + a[ms + Mice] * st.Ti1[Mice]
                            - (a[ms + Mice] + b[ms + Mice]) * st.Ti1[Mice + 1]
                            + b[ms + Mice] * Tsoil1[2, ic])
                        # --------------------------------------------------
                        diff_coef_soil(a, b, c, d, 2, ns - 1,
                                       ms + Mice + 1, ms + Mice + ns - 2,
                                       dt, st.lamsoil, st.csoil, st.rosoil,
                                       st.dzs, Tsoil1, ic)
                        c[ms + Mice + ns - 1] = 1.0
                        a[ms + Mice + ns - 1] = 1.0
                        d[ms + Mice + ns - 1] = (
                            -st.soilbotflx * st.dzs[ns - 1]
                            / (st.lamsoil[ns - 1] + st.lamsoil[ns]))
                        Temp = _progonka(a, b, c, d, itop,
                                         ms + Mice + ns - 1)
                        st.Tsoil2[1:ns + 1] = Temp[ms + Mice:
                                                    ms + Mice + ns]
                    elif st.soilswitch == SOILSWITCH_OFF:
                        # Snow, ice, no water, no soil (323-336)
                        xx = (st.bathymice[Mice].area_half
                              / st.bathymice[Mice + 1].area_int)
                        c[ms + Mice] = (
                            -xx * 0.5 * st.lami_v[Mice] / (st.ddzi[Mice] * l1)
                            - 0.5 * st.ci_m_roi_v[Mice + 1] * st.ddzi[Mice]
                            * l1 * dt_inv
                            + 0.25 * C.ci_m_roi * (st.dhi - st.dhi0) * dt_inv)
                        a[ms + Mice] = 0.5 * (
                            -xx * st.lami_v[Mice] / (st.ddzi[Mice] * l1)
                            + 0.5 * st.ci_m_roi_v[Mice + 1]
                            * (st.dhi - st.dhi0) * dt_inv)
                        d[ms + Mice] = (
                            -0.5 * st.Ti1[Mice + 1] * st.ci_m_roi_v[Mice + 1]
                            * st.ddzi[Mice] * l1 * dt_inv
                            - xx * st.rad_integr_ice[Mice]
                            # soilbotflx is the top soil boundary flux here
                            + st.soilbotflx
                            + (xx * 0.5 * st.lami_v[Mice]
                               / (st.ddzi[Mice] * l1)
                               - 0.25 * st.ci_m_roi_v[Mice + 1]
                               * (st.dhi - st.dhi0) * dt_inv)
                            * st.Ti1[Mice + 1]
                            + a[ms + Mice] * st.Ti1[Mice])
                        Temp = _progonka(a, b, c, d, itop, ms + Mice)
                        st.Tsoil2[:] = Tsoil1[:, ic]
                    st.Tsn[itop:ms + 1] = Temp[itop:ms + 1]
                    st.Ti2[1:Mice + 2] = Temp[ms:ms + Mice + 1]

            else:
                # ------------------- no ice, snow exists ------------------
                if st.soilswitch == SOILSWITCH_ON:
                    # ------------- SNOW-SOIL INTERFACE (347-354) ----------
                    a[ms] = (-0.25 * (st.lams[ms - 1] + st.lams[ms])
                             / st.dz[ms - 1])
                    b[ms] = (-0.25 * (st.lamsoil[1] + st.lamsoil[2])
                             / st.dzs[1])
                    c[ms] = (a[ms] + b[ms]
                             - 0.5 * dt_inv
                             * (st.cs[ms] * st.dens[ms] * st.dz[ms]
                                + st.csoil[1] * st.rosoil[1] * st.dzs[1]))
                    d[ms] = (-Tsoil1[1, ic] * 0.5 * dt_inv
                             * (st.cs[ms] * st.dens[ms] * st.dz[ms]
                                + st.csoil[1] * st.rosoil[1] * st.dzs[1])
                             - st.SR_botsnow
                             + a[ms] * st.T[ms - 1]
                             - (a[ms] + b[ms]) * Tsoil1[1, ic]
                             + b[ms] * Tsoil1[2, ic])
                    # ------------------------------------------------------
                    diff_coef_soil(a, b, c, d, 2, ns - 1, ms + 1, ms + ns - 2,
                                   dt, st.lamsoil, st.csoil, st.rosoil,
                                   st.dzs, Tsoil1, ic)
                    c[ms + ns - 1] = 1.0
                    a[ms + ns - 1] = 1.0
                    d[ms + ns - 1] = (-st.soilbotflx * st.dzs[ns - 1]
                                      / (st.lamsoil[ns - 1] + st.lamsoil[ns]))
                    Temp = _progonka(a, b, c, d, itop, ms + ns - 1)
                    st.Tsoil2[1:ns + 1] = Temp[ms:ms + ns]
                elif st.soilswitch == SOILSWITCH_OFF:
                    # Snow only, no soil (comment in the source says "Only
                    # ice exists", t_solver_mod.f90:364)
                    a[ms] = (-0.25 * (st.lams[ms - 1] + st.lams[ms])
                             / st.dz[ms - 1])
                    c[ms] = (a[ms]
                             - 0.5 * st.cs[ms] * st.dens[ms] * st.dz[ms]
                             * dt_inv)
                    d[ms] = (-st.T[ms] * 0.5 * st.cs[ms] * st.dens[ms]
                             * st.dz[ms] * dt_inv
                             - st.SR_botsnow
                             # soilbotflx is the top soil boundary flux here
                             + st.soilbotflx
                             + a[ms] * (st.T[ms - 1] - st.T[ms]))
                    Temp = _progonka(a, b, c, d, itop, ms)
                    st.Tsoil2[:] = Tsoil1[:, ic]
                st.Tsn[itop:ms + 1] = Temp[itop:ms + 1]

        # ---------------------- no snow, ice exists -----------------------
        elif ice == 1:
            b[1] = 0.0
            c[1] = 1.0
            d[1] = Tsurf
            diff_coef_ice(a, b, c, d, 2, Mice, 2, Mice, dt, st)

            if water == 1:
                # Ice and water (382-389)
                a[Mice + 1] = 0.0
                c[Mice + 1] = 1.0
                d[Mice + 1] = meltpnt(st.Sal2[1], st.preswat[1],
                                      st.nmeltpoint)
                Temp = _progonka(a, b, c, d, 1, Mice + 1)
                st.Ti2[1:Mice + 2] = Temp[1:Mice + 2]
            else:
                if st.soilswitch == SOILSWITCH_ON:
                    # ------------- ICE-SOIL INTERFACE (392-401) -----------
                    xx = (st.bathymice[Mice + 1].area_int
                          / st.bathymice[Mice].area_half)
                    a[Mice + 1] = 0.5 * (
                        -st.lami_v[Mice] / (st.ddzi[Mice] * l1)
                        + xx * 0.5 * st.ci_m_roi_v[Mice + 1]
                        * (st.dhi - st.dhi0) * dt_inv)
                    b[Mice + 1] = (-0.25 * (st.lamsoil[1] + st.lamsoil[2])
                                   / st.dzs[1])
                    cc = 0.5 * dt_inv * (
                        st.csoil[1] * st.rosoil[1] * st.dzs[1]
                        + xx * st.ci_m_roi_v[Mice + 1] * st.ddzi[Mice] * l1)
                    c[Mice + 1] = a[Mice + 1] + b[Mice + 1] - cc
                    d[Mice + 1] = (
                        -st.Ti1[Mice + 1] * cc
                        - st.rad_integr_ice[Mice]
                        + a[Mice + 1] * st.Ti1[Mice]
                        - (a[Mice + 1] + b[Mice + 1]) * st.Ti1[Mice + 1]
                        + b[Mice + 1] * Tsoil1[2, ic])
                    # ------------------------------------------------------
                    diff_coef_soil(a, b, c, d, 2, ns - 1, Mice + 2,
                                   Mice + ns - 1, dt, st.lamsoil, st.csoil,
                                   st.rosoil, st.dzs, Tsoil1, ic)
                    c[Mice + ns] = 1.0
                    a[Mice + ns] = 1.0
                    d[Mice + ns] = (-st.soilbotflx * st.dzs[ns - 1]
                                    / (st.lamsoil[ns - 1] + st.lamsoil[ns]))
                    Temp = _progonka(a, b, c, d, 1, Mice + ns)
                    st.Tsoil2[1:ns + 1] = Temp[Mice + 1:Mice + ns + 1]
                elif st.soilswitch == SOILSWITCH_OFF:
                    # Ice only (411-424)
                    xx = (st.bathymice[Mice].area_half
                          / st.bathymice[Mice + 1].area_int)
                    c[Mice + 1] = (
                        -xx * 0.5 * st.lami_v[Mice] / (st.ddzi[Mice] * l1)
                        + 0.5 * dt_inv * st.ci_m_roi_v[Mice + 1]
                        * (-st.ddzi[Mice] * l1
                           + 0.5 * (st.dhi - st.dhi0)))
                    a[Mice + 1] = 0.5 * (
                        -xx * st.lami_v[Mice] / (st.ddzi[Mice] * l1)
                        + 0.5 * dt_inv * st.ci_m_roi_v[Mice + 1]
                        * (st.dhi - st.dhi0))
                    d[Mice + 1] = (
                        -st.Ti1[Mice + 1] * 0.5 * dt_inv
                        * st.ci_m_roi_v[Mice + 1] * st.ddzi[Mice] * l1
                        - xx * st.rad_integr_ice[Mice]
                        # soilbotflx is the top soil boundary flux here
                        + st.soilbotflx
                        + (xx * 0.5 * st.lami_v[Mice] / (st.ddzi[Mice] * l1)
                           - 0.25 * dt_inv * st.ci_m_roi_v[Mice + 1]
                           * (st.dhi - st.dhi0)) * st.Ti1[Mice + 1]
                        + a[Mice + 1] * st.Ti1[Mice])
                    Temp = _progonka(a, b, c, d, 1, Mice + 1)
                    st.Tsoil2[:] = Tsoil1[:, ic]
                st.Ti2[1:Mice + 2] = Temp[1:Mice + 2]

        # ---------------------- no snow, no ice, water --------------------
        elif water == 1:
            b[1] = 0.0
            c[1] = 1.0
            d[1] = Tsurf
            diff_coef_water(a, b, c, d, 2, M, 2, M, dt, st.Tw1, st.lamw,
                            _area_int(st.bathymwater), _area_half(st.bathymwater),
                            st.ddz, h1, st.dhw, st.dhw0, st.PEMF,
                            st.pt_down_f, st.rad_integr_water, st.lsh_water,
                            st.dzeta_int, st.ddz05)

            if deepice == 1:
                # Water and deep ice (435-442)
                a[M + 1] = 0.0
                c[M + 1] = 1.0
                d[M + 1] = meltpnt(st.Sal2[M + 1], st.preswat[M + 1],
                                   st.nmeltpoint)
                Temp = _progonka(a, b, c, d, 1, M + 1)
                st.Tw2[1:M + 2] = Temp[1:M + 2]
            else:
                if st.soilswitch == SOILSWITCH_ON:
                    # --------------- WATER-SOIL INTERFACE (446-455) --------
                    xx = (st.bathymwater[M + 1].area_int
                          / st.bathymwater[M].area_half)
                    a[M + 1] = (0.5 * (
                        -st.lamw[M] / (st.ddz[M] * h1)
                        + xx * 0.5 * dt_inv * cw * (st.dhw - st.dhw0))
                        + xx * 0.5 * cw * st.PEMF[M])
                    b[M + 1] = (-0.25 * (st.lamsoil[1] + st.lamsoil[2])
                                / st.dzs[1])
                    cc = 0.5 * dt_inv * (
                        st.csoil[1] * st.rosoil[1] * st.dzs[1]
                        + xx * cw * st.ddz[M] * h1)
                    c[M + 1] = (a[M + 1] + b[M + 1] - cc
                                - xx * cw * st.PEMF[M])
                    d[M + 1] = (-st.Tw1[M + 1] * cc
                                - st.rad_integr_water[M]
                                - xx * cw * st.PEMF[M] * st.pt_down_f[M]
                                + a[M + 1] * st.Tw1[M]
                                - (a[M + 1] + b[M + 1]) * st.Tw1[M + 1]
                                + b[M + 1] * Tsoil1[2, ic])
                    # ------------------------------------------------------
                    diff_coef_soil(a, b, c, d, 2, ns - 1, M + 2, M + ns - 1,
                                   dt, st.lamsoil, st.csoil, st.rosoil,
                                   st.dzs, Tsoil1, ic)
                    c[M + ns] = 1.0
                    a[M + ns] = 1.0
                    d[M + ns] = (-st.soilbotflx * st.dzs[ns - 1]
                                 / (st.lamsoil[ns - 1] + st.lamsoil[ns]))
                    Temp = _progonka(a, b, c, d, 1, M + ns)
                    st.Tsoil2[1:ns + 1] = Temp[M + 1:M + ns + 1]
                elif st.soilswitch == SOILSWITCH_OFF:
                    if st.cuette == 0:
                        # Water only (466-477)
                        xx = (st.bathymwater[M].area_half
                              / st.bathymwater[M + 1].area_int)
                        a[M + 1] = (0.5 * (
                            -xx * st.lamw[M] / (st.ddz[M] * h1)
                            + 0.5 * dt_inv * cw * (st.dhw - st.dhw0))
                            + 0.5 * cw * st.PEMF[M])
                        c[M + 1] = (a[M + 1]
                                    - 0.5 * dt_inv * cw * st.ddz[M] * h1
                                    - cw * st.PEMF[M])
                        d[M + 1] = (
                            -st.Tw1[M + 1] * 0.5 * dt_inv * cw * st.ddz[M] * h1
                            - xx * st.rad_integr_water[M]
                            - cw * st.PEMF[M] * st.pt_down_f[M]
                            # soilbotflx is the top soil boundary flux here
                            + st.soilbotflx
                            + 0.5 * (-xx * st.lamw[M] / (st.ddz[M] * h1)
                                     + 0.5 * dt_inv * cw
                                     * (st.dhw - st.dhw0))
                            * (st.Tw1[M] - st.Tw1[M + 1]))
                        Temp = _progonka(a, b, c, d, 1, M + 1)
                        st.Tsoil2[:] = Tsoil1[:, ic]
                    elif st.cuette in (1, 11):
                        # Couette flow b.c. (478-485)
                        a[M + 1] = 0.0
                        c[M + 1] = 1.0
                        d[M + 1] = st.Tw1[M + 1]
                        Temp = _progonka(a, b, c, d, 1, M + 1)
                        st.Tsoil2[:] = Tsoil1[:, ic]
                    else:
                        raise NotImplementedError(
                            "T_DIFF: cuette%par = %r is not covered by "
                            "t_solver_mod.f90:466-485" % (st.cuette,))
                st.Tw2[1:M + 2] = Temp[1:M + 2]

        # ---------------------- soil only ---------------------------------
        else:
            if ifflux:
                # Tsurf is a downward heat flux (492-497)
                cc = st.rosoil[1] * st.csoil[1] * 0.5 * st.dzs[1] * dt_inv
                xx = 0.25 * (st.lamsoil[1] + st.lamsoil[2]) / st.dzs[1]
                b[1] = xx
                c[1] = cc + xx
                d[1] = (Tsurf + xx * Tsoil1[2, ic]
                        + (cc - xx) * Tsoil1[1, ic])
            else:
                b[1] = 0.0
                c[1] = 1.0
                d[1] = Tsurf
            diff_coef_soil(a, b, c, d, 2, ns - 1, 2, ns - 1, dt,
                           st.lamsoil, st.csoil, st.rosoil, st.dzs, Tsoil1, ic)
            c[ns] = 0.5
            a[ns] = 0.5
            d[ns] = (-st.soilbotflx * st.dzs[ns - 1]
                     / (st.lamsoil[ns - 1] + st.lamsoil[ns])
                     + a[ns] * Tsoil1[ns - 1, ic]
                     - c[ns] * Tsoil1[ns, ic])
            Temp = _progonka(a, b, c, d, 1, ns)
            st.Tsoil2[1:ns + 1] = Temp[1:ns + 1]

    # ======================================================================
    # surf == 0: melting point at the ice-water interface
    # ======================================================================
    else:
        if water == 0:
            return st

        if ice == 1:
            # Ice and water exist (517-529)
            b[1] = 0.0
            c[1] = 1.0
            d[1] = meltpnt(st.Sal2[1], st.preswat[1], st.nmeltpoint)
            diff_coef_water(a, b, c, d, 2, M, 2, M, dt, st.Tw1, st.lamw,
                            _area_int(st.bathymwater), _area_half(st.bathymwater),
                            st.ddz, h1, st.dhw, st.dhw0, st.PEMF,
                            st.pt_down_f, st.rad_integr_water, st.lsh_water,
                            st.dzeta_int, st.ddz05)

            if deepice == 1:
                a[M + 1] = 0.0
                c[M + 1] = 1.0
                d[M + 1] = meltpnt(st.Sal2[M + 1], st.preswat[M + 1],
                                   st.nmeltpoint)
                Temp = _progonka(a, b, c, d, 1, M + 1)
                st.Tw2[1:M + 2] = Temp[1:M + 2]

                # Deep ice column (530-533)
                b[1] = 0.0
                c[1] = 1.0
                d[1] = meltpnt(st.Sal2[M + 1], st.preswat[M + 1],
                               st.nmeltpoint)
                diff_coef_deepice(a, b, c, d, 2, Mice, 2, Mice, dt, st)

                if st.soilswitch == SOILSWITCH_ON:
                    # ------------ DEEPICE-SOIL INTERFACE (536-544) --------
                    xx = (st.bathymdice[Mice + 1].area_int
                          / st.bathymdice[Mice].area_half)
                    a[Mice + 1] = 0.5 * (
                        -C.lami / (st.ddzi[Mice] * ls1)
                        + xx * 0.5 * dt_inv * C.ci_m_roi
                        * (st.dls - st.dls0))
                    b[Mice + 1] = (-0.25 * (st.lamsoil[1] + st.lamsoil[2])
                                   / st.dzs[1])
                    cc = 0.5 * dt_inv * (
                        st.csoil[1] * st.rosoil[1] * st.dzs[1]
                        + xx * C.ci_m_roi * st.ddzi[Mice] * ls1)
                    c[Mice + 1] = a[Mice + 1] + b[Mice + 1] - cc
                    d[Mice + 1] = (
                        -st.Tis1[Mice + 1] * cc
                        - st.rad_integr_deepice[Mice]
                        + a[Mice + 1] * st.Tis1[Mice]
                        - (a[Mice + 1] + b[Mice + 1]) * st.Tis1[Mice + 1]
                        + b[Mice + 1] * Tsoil1[2, ic])
                    # ------------------------------------------------------
                    diff_coef_soil(a, b, c, d, 2, ns - 1, Mice + 2,
                                   Mice + ns - 1, dt, st.lamsoil, st.csoil,
                                   st.rosoil, st.dzs, Tsoil1, ic)
                    c[Mice + ns] = 1.0
                    a[Mice + ns] = 1.0
                    d[Mice + ns] = (-st.soilbotflx * st.dzs[ns - 1]
                                    / (st.lamsoil[ns - 1] + st.lamsoil[ns]))
                    Temp = _progonka(a, b, c, d, 1, Mice + ns)
                    st.Tsoil2[1:ns + 1] = Temp[Mice + 1:Mice + ns + 1]
                elif st.soilswitch == SOILSWITCH_OFF:
                    # Ice, water, deep ice, no soil (555-565)
                    xx = (st.bathymdice[Mice].area_half
                          / st.bathymdice[Mice + 1].area_int)
                    c[Mice + 1] = (
                        -xx * 0.5 * C.lami / (st.ddzi[Mice] * ls1)
                        + 0.5 * dt_inv * C.ci_m_roi
                        * (-st.ddzi[Mice] * ls1
                           + 0.5 * (st.dls - st.dls0)))
                    a[Mice + 1] = (
                        -xx * 0.5 * C.lami / (st.ddzi[Mice] * ls1)
                        + 0.25 * dt_inv * C.ci_m_roi * (st.dls - st.dls0))
                    d[Mice + 1] = (
                        -st.Tis1[Mice + 1] * 0.5 * dt_inv * C.ci_m_roi
                        * st.ddzi[Mice] * ls1
                        - xx * st.rad_integr_deepice[Mice]
                        # soilbotflx is the top soil boundary flux here
                        + st.soilbotflx
                        + (xx * 0.5 * C.lami / (st.ddzi[Mice] * ls1)
                           - 0.25 * dt_inv * C.ci_m_roi
                           * (st.dls - st.dls0))
                        * (st.Tis1[Mice + 1] - st.Tis1[Mice]))
                    Temp = _progonka(a, b, c, d, 1, Mice + 1)
                    st.Tsoil2[:] = Tsoil1[:, ic]
                st.Tis2[1:Mice + 2] = Temp[1:Mice + 2]
            else:
                # Ice and water, no deep ice (570-608)
                if st.soilswitch == SOILSWITCH_ON:
                    # --------------- WATER-SOIL INTERFACE (573-582) --------
                    xx = (st.bathymwater[M + 1].area_int
                          / st.bathymwater[M].area_half)
                    a[M + 1] = (0.5 * (
                        -st.lamw[M] / (st.ddz[M] * h1)
                        + xx * 0.5 * dt_inv * cw * (st.dhw - st.dhw0))
                        + xx * 0.5 * cw * st.PEMF[M])
                    b[M + 1] = (-0.25 * (st.lamsoil[1] + st.lamsoil[2])
                                / st.dzs[1])
                    cc = 0.5 * dt_inv * (
                        st.csoil[1] * st.rosoil[1] * st.dzs[1]
                        + xx * cw * st.ddz[M] * h1)
                    c[M + 1] = (a[M + 1] + b[M + 1] - cc
                                - xx * cw * st.PEMF[M])
                    d[M + 1] = (-st.Tw1[M + 1] * cc
                                - st.rad_integr_water[M]
                                - xx * cw * st.PEMF[M] * st.pt_down_f[M]
                                + a[M + 1] * st.Tw1[M]
                                - (a[M + 1] + b[M + 1]) * st.Tw1[M + 1]
                                + b[M + 1] * Tsoil1[2, ic])
                    # ------------------------------------------------------
                    diff_coef_soil(a, b, c, d, 2, ns - 1, M + 2, M + ns - 1,
                                   dt, st.lamsoil, st.csoil, st.rosoil,
                                   st.dzs, Tsoil1, ic)
                    c[M + ns] = 1.0
                    a[M + ns] = 1.0
                    d[M + ns] = (-st.soilbotflx * st.dzs[ns - 1]
                                 / (st.lamsoil[ns - 1] + st.lamsoil[ns]))
                    Temp = _progonka(a, b, c, d, 1, M + ns)
                    st.Tsoil2[1:ns + 1] = Temp[M + 1:M + ns + 1]
                elif st.soilswitch == SOILSWITCH_OFF:
                    # Ice and water, no soil, no deep ice (592-603)
                    xx = (st.bathymwater[M].area_half
                          / st.bathymwater[M + 1].area_int)
                    a[M + 1] = (0.5 * (
                        -xx * st.lamw[M] / (st.ddz[M] * h1)
                        + 0.5 * dt_inv * cw * (st.dhw - st.dhw0))
                        + 0.5 * cw * st.PEMF[M])
                    c[M + 1] = (a[M + 1]
                                - 0.5 * dt_inv * cw * st.ddz[M] * h1
                                - cw * st.PEMF[M])
                    d[M + 1] = (-st.Tw1[M + 1] * 0.5 * dt_inv * cw
                                * st.ddz[M] * h1
                                - xx * st.rad_integr_water[M]
                                - cw * st.PEMF[M] * st.pt_down_f[M]
                                # soilbotflx is the top soil boundary flux
                                + st.soilbotflx
                                + 0.5 * (-xx * st.lamw[M] / (st.ddz[M] * h1)
                                         + 0.5 * dt_inv * cw
                                         * (st.dhw - st.dhw0))
                                * (st.Tw1[M] - st.Tw1[M + 1]))
                    Temp = _progonka(a, b, c, d, 1, M + 1)
                    st.Tsoil2[:] = Tsoil1[:, ic]
                st.Tw2[1:M + 2] = Temp[1:M + 2]

        elif deepice == 1:
            # Water and deep ice, no top ice (609-649)
            b[1] = 0.0
            c[1] = 1.0
            d[1] = meltpnt(st.Sal2[M + 1], st.preswat[M + 1], st.nmeltpoint)
            diff_coef_deepice(a, b, c, d, 2, Mice, 2, Mice, dt, st)

            if st.soilswitch == SOILSWITCH_ON:
                # -------------- DEEPICE-SOIL INTERFACE (616-624) ----------
                xx = (st.bathymdice[Mice + 1].area_int
                      / st.bathymdice[Mice].area_half)
                a[Mice + 1] = 0.5 * (
                    -C.lami / (st.ddzi[Mice] * ls1)
                    + xx * 0.5 * dt_inv * C.ci_m_roi * (st.dls - st.dls0))
                b[Mice + 1] = (-0.25 * (st.lamsoil[1] + st.lamsoil[2])
                               / st.dzs[1])
                cc = 0.5 * dt_inv * (
                    st.csoil[1] * st.rosoil[1] * st.dzs[1]
                    + xx * C.ci_m_roi * st.ddzi[Mice] * ls1)
                c[Mice + 1] = a[Mice + 1] + b[Mice + 1] - cc
                d[Mice + 1] = (
                    -st.Tis1[Mice + 1] * cc
                    - st.rad_integr_deepice[Mice]
                    + a[Mice + 1] * st.Tis1[Mice]
                    - (a[Mice + 1] + b[Mice + 1]) * st.Tis1[Mice + 1]
                    + b[Mice + 1] * Tsoil1[2, ic])
                # ----------------------------------------------------------
                diff_coef_soil(a, b, c, d, 2, ns - 1, Mice + 2,
                               Mice + ns - 1, dt, st.lamsoil, st.csoil,
                               st.rosoil, st.dzs, Tsoil1, ic)
                c[Mice + ns] = 1.0
                a[Mice + ns] = 1.0
                d[Mice + ns] = (-st.soilbotflx * st.dzs[ns - 1]
                                / (st.lamsoil[ns - 1] + st.lamsoil[ns]))
                Temp = _progonka(a, b, c, d, 1, Mice + ns)
                st.Tsoil2[1:ns + 1] = Temp[Mice + 1:Mice + ns + 1]
            elif st.soilswitch == SOILSWITCH_OFF:
                # Water and deep ice, no soil (635-645).
                # NOTE: the source does not scale RadDeepIce%integr(Mice) by
                # xx here, unlike every other deep-ice interface.
                xx = (st.bathymdice[Mice].area_half
                      / st.bathymdice[Mice + 1].area_int)
                c[Mice + 1] = (
                    -xx * 0.5 * C.lami / (st.ddzi[Mice] * ls1)
                    + 0.5 * dt_inv * C.ci_m_roi
                    * (-st.ddzi[Mice] * ls1 + 0.5 * (st.dls - st.dls0)))
                a[Mice + 1] = (
                    -xx * 0.5 * C.lami / (st.ddzi[Mice] * ls1)
                    + 0.25 * dt_inv * C.ci_m_roi * (st.dls - st.dls0))
                d[Mice + 1] = (
                    -st.Tis1[Mice + 1] * C.ci_m_roi * 0.5 * dt_inv
                    * st.ddzi[Mice] * ls1
                    - st.rad_integr_deepice[Mice]
                    # soilbotflx is the top soil boundary flux here
                    + st.soilbotflx
                    + (-xx * 0.5 * C.lami / (st.ddzi[Mice] * ls1)
                       + 0.25 * dt_inv * C.ci_m_roi * (st.dls - st.dls0))
                    * (st.Tis1[Mice] - st.Tis1[Mice + 1]))
                Temp = _progonka(a, b, c, d, 1, Mice + 1)
                st.Tsoil2[:] = Tsoil1[:, ic]
            st.Tis2[1:Mice + 2] = Temp[1:Mice + 2]

    return st


# ---------------------------------------------------------------------------
# SNOW_COND_HEAT_COEF
# ---------------------------------------------------------------------------

def snow_cond_heat_coef(st: IceState, tempair):
    """``SNOW_COND_HEAT_COEF`` (``snowtemp.f90:100-179``), in place.

    Two jobs: on the first call (``flag_snow_init == 1``) it builds the initial
    snow pack (``itop = ms-2``, three 1 cm layers at ``tempair`` plus the
    interface node ``ms``), and on every call it refreshes ``lams`` (snow heat
    conductance) and ``cs`` (snow heat capacity) for layers ``itop..ms-1``.

    ``dzsn`` (``snowtemp.f90:165-167``) is computed in the source but never
    used; it is not reproduced here.  ``totalmelts``/``totalprecips``/
    ``totalevaps`` are diagnostic counters of ``ARRAYS`` and are not part of
    this state (TODO: track them in the integration layer if the output needs
    them).
    """
    ms = st.ms
    if st.flag_snow_init == 1:
        st.itop = ms - 2
        st.hs1 = _s(0.02)
        rofresh = _s(67.9) + _s(51.3) * np.exp(tempair / _s(2.6))
        for i in range(st.itop, ms):
            st.wl[i] = 0.0
            st.dens[i] = rofresh
            st.dz[i] = _s(0.01)
            st.T[i] = tempair
        st.wl[ms] = 0.0
        st.dens[ms] = rofresh
        st.T[ms] = st.Ti1[1]
        st.dz[ms] = 0.5 * st.dz[ms - 1]
        for i in range(1, st.itop):
            st.T[i] = 0.0
            st.wl[i] = 0.0
            st.dens[i] = 0.0
            st.dz[i] = 0.0

    row0, ci, cw = C.row0, C.ci, C.cw
    for i in range(st.itop, ms):
        # 0.419*(6.*(dens(i)/row0)**4 + 1.9*(dens(i)/row0) + 0.05)
        # The literals are default real.  NOTE: gfortran may expand the integer
        # power into a chain of multiplications, which can differ from this
        # ``**4`` in the last bit; check against the reference build.
        d = st.dens[i] / row0
        st.lams[i] = _s(0.419) * (6.0 * d ** 4 + _s(1.9) * d + _s(0.05))
        xx = st.wl[i] * row0 / (st.dens[i] * st.dz[i])
        st.cs[i] = ci * (1.0 - xx) + cw * xx


# ---------------------------------------------------------------------------
# SR_botsnow
# ---------------------------------------------------------------------------

def sr_botsnow(st: IceState, shortwave):
    """Shortwave radiation reaching the bottom of the snow pack.

    ``lake.f90:1339-1344`` (case 2) and ``lake.f90:1921-1926`` (case 3):
    ``SR_botsnow = shortwave*(1-albedoofsnow)``, attenuated by every snow layer
    ``itop..ms-1`` as ``EXTINCT_SNOW(dens(i))**dz(i)``.  The loop is skipped
    while ``flag_snow_init == 1``.  ``albedoofsnow`` is the float32 constant of
    :mod:`lakematrix.constants`, so ``1-albedoofsnow`` is ``0.149999976...``.
    """
    SR = shortwave * (1.0 - C.albedoofsnow)
    if st.flag_snow_init != 1:
        for i in range(st.itop, st.ms):
            SR = SR * extinct_snow(st.dens[i]) ** st.dz[i]
    st.SR_botsnow = SR
    return SR


# ---------------------------------------------------------------------------
# SNOW_CALC
# ---------------------------------------------------------------------------

def snow_calc(st: IceState, dt, *, ts, t2, snow_sc, eflux, Erad, Elatent,
              precip, pheat):
    """``SNOW_CALC`` (``snowcalc.f90:5-497``): snow melt, percolation,
    densification and layer bookkeeping.

    Parameters
    ----------
    ts : float
        Snow surface temperature, C (``T(itop)`` at the previous step).
    t2 : float
        Air temperature at 2 m, C.
    snow_sc : float
        Water equivalent of the pack *before* the step, m
        (``snow_sc``/``snowmass`` in ``SNOWTEMP``).
    eflux, Erad : float
        Surface heat flux and absorbed shortwave radiation reaching the snow
        surface, W/m2 (``eflux`` and ``shortwave*(1-albedoofsnow)``).
    Elatent : float
        Latent heat flux, W/m2; used only to compute evaporation ``evap``.
    precip : float
        Precipitation rate, m/s.  The routine multiplies it by ``dt`` on entry
        and divides it back on exit, exactly like the Fortran module variable;
        the restored value is returned.
    pheat : float
        Heat content of the rain, W/m2 - ``addPrecipToSnow`` output (see the
        module docstring; use 0 when precipitation is not modelled).

    Returns ``dict(snow_sc, swe, hsnow, snmelt, phase, precip)``: the new
    water equivalent (also stored in ``st.swe``), the snow depth, the melt
    rate (m/s of water, stored in ``st.snmelt``), the (unused) ``Phase``
    accumulator of the source and the restored precipitation rate.

    ``pmelt``/``CCT`` of the Fortran signature are not read by ``SNOW_CALC``
    and are omitted; ``addPrecipToSnow`` is not ported (see module docstring).

    The ``GOTO``-driven layer compaction (labels 222/111/123/333) is
    reproduced with restartable ``while`` loops: the label-222 sweep restarts
    from ``ms-1`` after every relabelled layer, and the label-333 merge sweep
    restarts after every merge, exactly like ``GOTO 222``/``GOTO 333``.
    """
    ms = st.ms
    itop = st.itop
    row0, roi = C.row0, C.roi
    Lwi, Liv = C.Lwi, C.Liv
    whc, hcond = C.whc, C.hcond
    g, csnow = C.g, C.csnow
    rosn_dry_ref = C.rosn_dry_ref
    S1E4 = _s(1.0e-4)
    S1E6 = _s(1.0e-6)

    # t(i) before the update is needed by the Phase accumulator (line 366).
    ti = np.zeros_like(st.T)
    for i in range(itop, ms):
        ti[i] = st.T[i]

    st.snmelt = 0.0
    swe = 0.0
    phase = 0.0

    densny = _s(67.9) + _s(51.3) * np.exp(t2 / _s(2.6))
    Elatent1 = Elatent
    if ts > _s(1.0e-5):
        Elatent1 = 0.0
    elif snow_sc < _s(1.0e-6) and Elatent > 0.0:
        Elatent1 = 0.0

    st.dz[ms - 1] = st.dz[ms - 1] + st.dz[ms]
    evap = Elatent1 / Liv
    hsnow = 0.0

    for i in range(itop, ms):
        st.extinct[i] = extinct_snow(st.dens[i])

    precip = precip * dt
    # Fortran leaves countrad undefined when extinct(itop) == 0; that needs
    # exp(-(...)) == 0, i.e. an infinite density, so it is unreachable.  Start
    # from 0 to keep the loop deterministic.
    countrad = 0.0
    Eout = 0.0

    if itop != ms:
        # ------------------- first (top) snow layer -----------------------
        i = itop
        dztop = st.dz[i]
        Erain = pheat
        Eflux1 = eflux
        Erad1 = Erad
        if st.extinct[itop] == 0.0:
            Eflux1 = Eflux1 + Erad1 + Erain
        else:
            if itop == ms - 1:
                Eflux1 = Eflux1 + Erain + Erad1
            else:
                Eflux1 = Eflux1 + Erain + Erad1 * (1 - st.extinct[i] ** st.dz[i])
            countrad = Erad1 * (1 - st.extinct[i] ** st.dz[i])

        if Eflux1 > 0.0:
            if itop == ms - 1:
                if ts >= _s(-1.0e-4):
                    if Eflux1 / Lwi * dt <= st.dens[i] * st.dz[i] - st.wl[i] * row0:
                        smelt = Eflux1 / Lwi / row0
                        Eout = 0.0
                    else:
                        smelt = (st.dz[i] * st.dens[i] / row0 - st.wl[i]) / dt
                        Eout = Eflux1 - smelt * row0 * Lwi
                else:
                    smelt = 0.0
                    Eout = 0.0
            else:
                smelt = 0.0
                Eout = Eflux1
        else:
            smelt = 0.0
            Eout = 0.0

        # percolation
        smelt = smelt + st.wl[i] / dt
        st.wl[i] = 0.0
        qbase = smelt

        # top layer thickness and density
        dmass = st.dens[i] * dztop - (evap + smelt * row0) * dt
        densev = st.dens[i] if evap > 0.0 else roi
        dztop = dztop - (smelt * row0 / st.dens[i] + evap / densev) * dt
        if dztop == 0.0:
            dztop = S1E6
        st.dens[i] = max(dmass / dztop, 0.0)
        if st.dens[i] > roi:
            dztop = dztop * st.dens[i] / roi
            st.dens[i] = roi
            dmass = st.dens[i] * dztop
        st.dz[i] = dztop
        if itop == ms - 1:
            st.snmelt = qbase

    counter = st.dz[itop]

    if itop < ms - 1:
        # ------------------------- internal layers ------------------------
        for i in range(itop + 1, ms):
            rf = 0.0
            smelti = 0.0
            Ein = Eout
            st.T[i] = (st.T[i] * st.dz[i] * st.dens[i]
                       / (qbase * row0 * dt + st.dz[i] * st.dens[i]))
            dzi = st.dz[i]
            st.wl[i] = st.wl[i] + qbase * dt
            if st.wl[i] < 0.0:
                st.wl[i] = 0.0
            w0 = st.wl[i]
            if st.extinct[i] == 0.0:
                Ei = Ein
            else:
                if i == ms - 1:
                    Ei = Ein + (Erad1 - countrad)
                else:
                    Ei = Ein + Erad1 * (st.extinct[i] ** counter
                                        - st.extinct[i] ** (counter + st.dz[i]))
                    # EInRad is assigned in the source but never used.
                countrad = countrad + Erad1 * (
                    st.extinct[i] ** counter
                    - st.extinct[i] ** (counter + st.dz[i]))
                counter = counter + st.dz[i]

            st.dz[i] = st.dz[i] + qbase * dt
            ziz = dzi / st.dz[i]
            st.dens[i] = ziz * st.dens[i] + (1.0 - ziz) * row0

            if Ei > 0.0:
                if st.T[i] >= S1E4:
                    Ei = Ei + csnow * st.dens[i] * st.T[i] * st.dz[i] / dt
                    st.T[i] = 0.0
                    if Ei / Lwi * dt <= (st.dz[i] * st.dens[i]
                                         - st.wl[i] * row0):
                        smelti = Ei / Lwi / row0
                        Eout = 0.0
                    else:
                        smelti = (st.dz[i] * st.dens[i] / row0 - st.wl[i]) / dt
                        Eout = Ei - smelti * row0 * Lwi
                        dzi = 0.0
                    st.wl[i] = st.wl[i] + smelti * dt
                else:
                    # T < 0
                    st.T[i] = st.T[i] + Ei * dt / (csnow * st.dens[i] * st.dz[i])
                    if st.T[i] < _s(-1.0e-4):
                        if st.wl[i] > (-st.T[i] * csnow * st.dz[i]
                                       * st.dens[i] / Lwi / row0):
                            rf = (-st.T[i] * csnow * st.dz[i] * st.dens[i]
                                  / Lwi / dt / row0)
                            st.T[i] = 0.0
                            st.wl[i] = st.wl[i] - rf * dt
                        else:
                            rf = st.wl[i] / dt
                            st.T[i] = (st.T[i] + rf * row0 * dt * Lwi
                                       / (csnow * st.dz[i] * st.dens[i]))
                            st.wl[i] = 0.0
                        Eout = 0.0
                    if st.T[i] >= S1E4:
                        if (csnow * st.dens[i] * st.T[i] * st.dz[i] / Lwi
                                <= st.dz[i] * st.dens[i]):
                            smelti = (csnow * st.dens[i] * st.T[i]
                                      * st.dz[i] / dt / Lwi / row0)
                            Eout = 0.0
                        else:
                            smelti = st.dz[i] * st.dens[i] / dt / row0
                            Eout = (csnow * st.dens[i] * st.T[i] * st.dz[i] / dt
                                    - smelti * Lwi * row0)
                            dzi = 0.0
                        st.T[i] = 0.0
                        st.wl[i] = st.wl[i] + smelti * dt
            else:
                # E == 0 (E < 0 is an impossible case per the source)
                if st.T[i] >= S1E4:
                    if (csnow * st.dens[i] * st.T[i] * st.dz[i] / Lwi
                            <= st.dz[i] * st.dens[i]):
                        smelti = (csnow * st.dens[i] * st.T[i]
                                  * st.dz[i] / dt / Lwi / row0)
                        st.T[i] = 0.0
                        Eout = 0.0
                        st.wl[i] = st.wl[i] + smelti * dt
                    else:
                        smelti = st.dz[i] * st.dens[i] / dt / row0
                        Eout = (csnow * st.dens[i] * st.T[i] * st.dz[i] / dt
                                - smelti * Lwi * row0)
                        st.T[i] = 0.0
                        dzi = 0.0
                        st.wl[i] = st.wl[i] + smelti * dt
                else:
                    # T < 0
                    if st.wl[i] > (-csnow * st.dens[i] * st.T[i]
                                   * st.dz[i] / Lwi / row0):
                        rf = (-csnow * st.dens[i] * st.T[i] * st.dz[i]
                              / dt / Lwi / row0)
                        st.T[i] = 0.0
                        st.wl[i] = st.wl[i] - rf * dt
                    else:
                        rf = st.wl[i] / dt
                        st.wl[i] = 0.0
                        st.T[i] = (st.T[i] + rf * row0 * dt * Lwi
                                   / (csnow * st.dz[i] * st.dens[i]))
                    Eout = 0.0
                    if st.T[i] >= S1E4:
                        if (csnow * st.dens[i] * st.T[i] * st.dz[i] / Lwi
                                <= st.dz[i] * st.dens[i]):
                            smelti = (csnow * st.dens[i] * st.T[i]
                                      * st.dz[i] / dt / Lwi / row0)
                            Eout = 0.0
                        else:
                            smelti = st.dz[i] * st.dens[i] / dt / row0
                            Eout = (csnow * st.dens[i] * st.T[i]
                                    * st.dz[i] / dt - smelti * Lwi * row0)
                            dzi = 0.0
                        st.T[i] = 0.0
                        st.wl[i] = st.wl[i] + smelti * dt

            # ------------------------ internal q --------------------------
            rho_dry = ((st.dens[i] - w0 / st.dz[i] * row0)
                       / (1 - w0 / st.dz[i]))
            if rho_dry < 0.0:
                # The source prints here; only the correction is reproduced.
                # > @todo{add heat release}
                if st.dens[i] - rosn_dry_ref > 0.0:
                    st.wl[i] = (st.dz[i] * (st.dens[i] - rosn_dry_ref)
                                / (row0 - rosn_dry_ref))
                    rho_dry = rosn_dry_ref
                else:
                    st.wl[i] = 0.0
                    rho_dry = st.dens[i]

            porosity = 1 - rho_dry / roi - st.wl[i] / st.dz[i]
            porosity = max(porosity, whc + 0.1)
            p1 = porosity - whc
            fukt = st.wl[i] / st.dz[i]
            if fukt > whc:
                fukt = (fukt - whc) / p1
                q0 = hcond * fukt ** 3.0
                qbase = min(q0, st.wl[i] / dt)
                st.wl[i] = st.wl[i] - qbase * dt
            else:
                qbase = 0.0
                q0 = 0.0
            if dzi <= S1E6:
                qbase = qbase + st.wl[i] / dt
                st.dz[i] = dzi
            else:
                dzmasi = st.dz[i] * st.dens[i] - qbase * dt * row0
                dzi = st.dz[i] + row0 * (
                    -qbase / row0 - rf / row0 + rf / roi
                    - smelti / rho_dry + smelti / row0) * dt
                st.dens[i] = max(dzmasi / dzi, 0.0)
                st.dz[i] = dzi
                if st.dens[i] > roi:
                    st.dz[i] = st.dz[i] * st.dens[i] / roi
                    st.dens[i] = roi
            phase = phase + (st.T[i] - ti[i]) * st.dens[i] * st.dz[i] / snow_sc / dt

            if i == ms - 1:
                st.snmelt = qbase

    # --------------------- layer removal / relabelling --------------------
    goto123 = False
    if itop == ms - 1 and st.dz[itop] < S1E6:
        st.dz[itop] = 0.0
        st.dens[itop] = 0.0
        goto123 = True
    if not goto123:
        if itop < ms and st.dz[itop] <= S1E6:
            st.dz[itop] = 0.0
            st.dens[itop] = 0.0
            itop = itop + 1

        # label 222: drop empty layers and shift the ones above down
        while True:
            restart = False
            for i in range(ms - 1, itop - 1, -1):
                if st.dz[i] <= S1E6:
                    k = itop - 1  # value of k after a completed DO loop
                    for kk in range(i - 1, itop - 1, -1):
                        if st.dz[kk] > S1E6:
                            k = kk
                            break
                    for j in range(i, itop - 1, -1):
                        st.dz[j] = st.dz[k + j - i]
                        st.dens[j] = st.dens[k + j - i]
                        st.T[j] = st.T[k + j - i]
                        st.wl[j] = st.wl[k + j - i]
                    itop = itop - k + i
                    restart = True
                    break
            if not restart:
                break

        # label 333: merge layers thinner than dzmin into the one below
        while True:
            merged = False
            for i in range(itop, ms):
                if st.dz[i] < C.dzmin and itop < ms - 1:
                    if i < ms - 1:
                        st.dens[i + 1] = (st.dens[i + 1] * st.dz[i + 1]
                                          / (st.dz[i] + st.dz[i + 1])
                                          + st.dens[i] * st.dz[i]
                                          / (st.dz[i] + st.dz[i + 1]))
                        st.T[i + 1] = (st.T[i + 1] * st.dz[i + 1]
                                       / (st.dz[i] + st.dz[i + 1])
                                       + st.T[i] * st.dz[i]
                                       / (st.dz[i] + st.dz[i + 1]))
                        st.wl[i + 1] = st.wl[i + 1] + st.wl[i]
                        st.dz[i + 1] = st.dz[i + 1] + st.dz[i]
                        for j in range(i, itop, -1):
                            st.dens[j] = st.dens[j - 1]
                            st.T[j] = st.T[j - 1]
                            st.wl[j] = st.wl[j - 1]
                            st.dz[j] = st.dz[j - 1]
                        itop = itop + 1
                        merged = True
                        break
            if not merged:
                break

    st.itop = itop

    # ------------------------- swe / snow depth ---------------------------
    swe = 0.0
    hsnow = 0.0
    for j in range(ms - 1, itop - 1, -1):
        swe = swe + st.dz[j] * st.dens[j]
        hsnow = hsnow + st.dz[j]

    # ------------- densification due to gravity and metamorphism ----------
    a2 = _s(0.00000066)
    sigma = 75.0
    for j in range(itop + 1, ms):
        if st.dens[j] < 3 * densny:
            dens_old = st.dens[j]
            eta = (a2 * np.exp(_s(19.3) * st.dens[j] / roi)
                   * np.exp(_sdiv(67300.0, 8.314) / (st.T[j] + _s(273.15))))
            P = 0.0
            for i in range(j, itop - 1, -1):
                P = P + st.dens[i] * g * st.dz[i]
            st.dens[j] = st.dens[j] + _s(0.001) * (
                dt * st.dens[j] * 1000.0 * (sigma + P) / eta)
            st.dens[j] = min(st.dens[j], roi)
            st.dz[j] = st.dz[j] * dens_old / st.dens[j]

    swe = 0.0
    hsnow = 0.0
    for j in range(ms - 1, itop - 1, -1):
        swe = swe + st.dz[j] * st.dens[j]
        hsnow = hsnow + st.dz[j]

    snow_sc_out = swe

    # --------------------- dz(ms) / z bookkeeping -------------------------
    if itop != ms - 1:
        st.dz[ms] = st.dz[ms - 1] / 3.0
        st.dz[ms - 1] = st.dz[ms - 1] - st.dz[ms]
    else:
        st.dz[ms] = st.dz[ms - 1] / 2.0
        st.dz[ms - 1] = st.dz[ms - 1] - st.dz[ms]
    if itop == ms - 1:
        st.z[ms - 1] = st.z[ms] - st.dz[ms - 1] - st.dz[ms]
    else:
        st.z[ms - 1] = st.z[ms] - st.dz[ms - 1] / 2.0 - st.dz[ms]
    for j in range(ms - 2, itop, -1):
        st.z[j] = st.z[j + 1] - st.dz[j] / 2.0 - st.dz[j + 1] / 2.0
    if itop < ms - 1:
        st.z[itop] = st.z[itop + 1] - st.dz[itop + 1] / 2.0 - st.dz[itop]

    for i in range(1, itop):
        st.dz[i] = 0.0

    st.swe = snow_sc_out
    st.hs1 = hsnow
    precip = precip / dt

    return dict(snow_sc=snow_sc_out, swe=snow_sc_out, hsnow=hsnow,
                snmelt=st.snmelt, phase=phase, precip=precip)


# ---------------------------------------------------------------------------
# Ice properties (ci_m_roi_v / lami_v)
# ---------------------------------------------------------------------------

def ice_properties(st: IceState, saltice=None):
    """Volumetric heat capacity and conductivity of the surface ice.

    ``lake.f90:1315-1329`` (case 2) and ``lake.f90:1903-1915`` (case 3).
    ``saltice == 0``: pure ice, ``ci_m_roi_v = ci_m_roi`` and
    ``lami_v = lami``.  ``saltice == 1``: pores filled with water,
    ``ci_m_roi_v(i) = ci*roi*(1-porice(i)) + cw*row0*porice(i)`` for
    ``i = 1..Mice+1`` and
    ``lami_v(i) = lami*(1-x) + lamw0*x`` with ``x = 0.5*(porice(i)+porice(i+1))``
    for ``i = 1..Mice``.

    ``lami_v(Mice+1)`` is **not** written by the source; the existing value
    persists (it is never read either - the ice-soil interface uses
    ``lami_v(Mice)``).
    """
    if saltice is None:
        saltice = st.saltice
    Mice = st.Mice
    if saltice == 0:
        for i in range(1, Mice + 2):
            st.ci_m_roi_v[i] = C.ci_m_roi
        for i in range(1, Mice + 1):
            st.lami_v[i] = C.lami
    elif saltice == 1:
        for i in range(1, Mice + 2):
            st.ci_m_roi_v[i] = (C.ci * C.roi * (1.0 - st.porice[i])
                                + C.cw * C.row0 * st.porice[i])
        for i in range(1, Mice + 1):
            x = 0.5 * (st.porice[i] + st.porice[i + 1])
            st.lami_v[i] = C.lami * (1.0 - x) + C.lamw0 * x
    else:
        raise NotImplementedError("saltice%%par = %r" % (saltice,))


# ---------------------------------------------------------------------------
# Case 2 ice thickness increments (water + ice + snow)
# ---------------------------------------------------------------------------

def ice_increments(st: IceState, dt):
    """``lake.f90:1367-1445``: case 2.1 (water + ice + snow) increments.

    Computes the ice growth/melt increments of one step:

    ``dhwlow``/``dhilow``
        freezing/melting at the ice bottom, limited by the water and ice
        thicknesses (``lake.f90:1374-1375``);
    ``dhwhigh``/``dhihigh``
        melting at the ice top (fresh-water melting point) plus the snowmelt
        frozen onto the ice top (``snmeltice``) and the ice impounded from
        above (``dhiimp``);
    ``dhwsnmelt``
        snowmelt water reaching the water column;
    ``dhwls``/``dls``
        growth/melt of the deep (bottom) ice layer;
    ``dhi``/``dhi0``
        total and "top" ice increments (``dhi0`` is the part that also enters
        the moving-grid terms of ``DIFF_COEF``).

    ``T_DIFF``'s snow-ice interface reads ``snmeltice`` and ``dhiimp`` from the
    state, so call this **before** ``T_SOLVER``, and call
    :func:`snow_cond_heat_coef` before it (the source calls
    ``SNOW_COND_HEAT_COEF`` at ``lake.f90:1399`` "needed for cs update").

    ``Ti1[1]`` is clipped to ``x + T_phase_threshold`` (``x`` = meltpnt(0)) when
    the ice top is above the melting point.  ``dhwfsoil``/``dhwtrib`` are
    external increments produced by the soil/tributary modules and are added
    from the state unchanged.

    Returns the dict of increments (also written into ``st``).
    """
    Mice, M, ms = st.Mice, st.M, st.ms
    Ti1, Tw1 = st.Ti1, st.Tw1
    l1, h1 = st.l1, st.h1
    row0_m_Lwi = C.row0_m_Lwi
    roi_d_row0 = C.roi_d_row0
    cw = C.cw_m_row0
    x = meltpnt(0.0, 0.0, st.nmeltpoint)

    # --- ice bottom / water top fluxes -----------------------------------
    flux1 = (-st.lami_v[Mice] * (Ti1[Mice + 1] - Ti1[Mice])
             / (st.ddzi[Mice] * l1)
             + st.rad_integr_ice[Mice]
             + st.ci_m_roi_v[Mice + 1] * (st.dhi - st.dhi0)
             * (Ti1[Mice + 1] - Ti1[Mice]) / (2.0 * dt))
    flux2 = (-st.lamw[1] * (Tw1[2] - Tw1[1]) / (st.ddz[1] * h1)
             + st.rad_integr_water[1]
             + cw * st.dhw0 * (Tw1[2] - Tw1[1]) / (2.0 * dt)
             + cw * st.PEMF[1] * (st.pt_down_f[1] - 0.5 * (Tw1[2] + Tw1[1])))
    # Limiting the freezing/melting rate by the thicknesses of water and ice
    dhwlow = min(max(dt * (flux1 - flux2) / (row0_m_Lwi),
                     -st.ddz[1] * h1),
                 st.ddzi[Mice] * l1 * C.roi_d_row0)
    dhilow = -dhwlow * C.roi_d_row0
    if st.saltice == 1:
        dhwlow = dhwlow * (1.0 + st.porice[Mice + 1] * roi_d_row0
                           / (1.0 - st.porice[Mice + 1]))
        dhilow = dhilow * (1.0 / (1.0 - st.porice[Mice + 1]))

    # --- melting at the ice top (fresh-water melting point) --------------
    if Ti1[1] > x + C.T_phase_threshold:
        dhwhigh = ((Ti1[1] - x - C.T_phase_threshold)
                   * st.ci_m_roi_v[1] * l1 * st.ddzi[1] * 0.5 / row0_m_Lwi)
        dhihigh = -dhwhigh * C.roi_d_row0
        Ti1[1] = x + C.T_phase_threshold
        if st.saltice == 1:
            dhwhigh = dhwhigh * (1.0 + st.porice[1] * roi_d_row0
                                 / (1.0 - st.porice[1]))
            dhihigh = dhihigh * (1.0 / (1.0 - st.porice[1]))
    else:
        dhwhigh = 0.0
        dhihigh = 0.0

    # --- freezing of snowmelt at the snow-ice interface -------------------
    # y: heat capacity of the snow-ice interface.  NOTE: SNOW_COND_HEAT_COEF
    # refreshes cs only for itop..ms-1, so cs(ms) is whatever the last
    # initialisation left there - a quirk of the source, kept as is.
    y = 0.5 * (st.cs[ms] * st.dens[ms] * st.dz[ms]
               + st.ci_m_roi_v[1] * st.ddzi[1] * l1)
    dTmax = max((1.0 - C.snmeltwat) * st.snmelt * dt * C.row0 * C.Lwi / y, 0.0)
    dTisnmelt = min(dTmax, x + C.T_phase_threshold - Ti1[1])
    dTisnmelt = max(dTisnmelt, 0.0)
    fracsn = dTisnmelt / (dTmax + C.small_value)
    snmeltice = dTmax * y / (C.row0 * C.Lwi) * fracsn
    dhihigh = dhihigh + snmeltice * C.roi_d_row0

    # --- impoundment of ice from above by water through cracks ------------
    dTmax = max((st.snowmass + l1 * (C.roi - C.row0)) * C.Lwi / y, 0.0)
    dTiimp = min(dTmax, x + C.T_phase_threshold - Ti1[1] - dTisnmelt)
    dTiimp = max(dTiimp, 0.0)
    fracimp = dTiimp / (dTmax + C.small_value)
    dhiimp = C.icetopfr * dTmax * y / (C.Lwi * C.row0) * fracimp
    dhihigh = dhihigh + dhiimp * C.roi_d_row0
    dhwhigh = dhwhigh - dhiimp

    # --- snowmelt water reaching the water column -------------------------
    dhwsnmelt = (C.snmeltwat * st.snmelt * dt
                 + (1.0 - C.snmeltwat) * st.snmelt * dt * (1.0 - fracsn))

    if st.ls1 != 0.0:
        flux1 = (-st.lamw[M] * (Tw1[M + 1] - Tw1[M]) / (st.ddz[M] * h1)
                 + st.rad_integr_water[M]
                 + cw * (st.dhw - st.dhw0) * (Tw1[M + 1] - Tw1[M])
                 / (2.0 * dt)
                 + cw * st.PEMF[M] * (st.pt_down_f[M]
                                      - 0.5 * (Tw1[M + 1] + Tw1[M])))
        flux2 = (-C.lami * (st.Tis1[2] - st.Tis1[1]) / (st.ddzi[1] * st.ls1)
                 + st.rad_integr_deepice[1]
                 + C.ci_m_roi * st.dls0 * (st.Tis1[2] - st.Tis1[1])
                 / (2.0 * dt))
        dhwls = dt * (flux1 - flux2) / (row0_m_Lwi)
        dls = -dhwls * C.roi_d_row0
        dhw = (dhwhigh + dhwlow + st.dhwfsoil + dhwls + st.dhwtrib
               + dhwsnmelt)
        dhw0 = dhwlow + dhwhigh + dhwsnmelt
        dls0 = dls
    else:
        dls = 0.0
        dhwls = 0.0
        dhw = dhwhigh + dhwlow + st.dhwfsoil + st.dhwtrib + dhwsnmelt
        dhw0 = dhwlow + dhwhigh + dhwsnmelt
        dls0 = st.dls0

    dhi = dhilow + dhihigh
    dhi0 = dhihigh

    st.dhwlow, st.dhilow = dhwlow, dhilow
    st.dhwhigh, st.dhihigh = dhwhigh, dhihigh
    st.dhwsnmelt, st.dhwls = dhwsnmelt, dhwls
    st.dhi, st.dhi0 = dhi, dhi0
    st.dls, st.dls0 = dls, dls0
    st.snmeltice, st.dhiimp = snmeltice, dhiimp
    st.dhw, st.dhw0 = dhw, dhw0

    return dict(dhwlow=dhwlow, dhilow=dhilow, dhwhigh=dhwhigh,
                dhihigh=dhihigh, dhwsnmelt=dhwsnmelt, dhwls=dhwls,
                dhi=dhi, dhi0=dhi0, dls=dls, dls0=dls0,
                snmeltice=snmeltice, dhiimp=dhiimp, fracsn=fracsn,
                fracimp=fracimp, dhw=dhw, dhw0=dhw0)


def ice_increments_nosnow(st: IceState, dt, *, precip, tempair, Elatent):
    """``lake.f90:1657-1717``: case 2.2 (water + ice, no snow) increments.

    Same quantities as :func:`ice_increments` plus ``dhis`` (ice growth from
    the surface heat loss ``-Elatent/(roi*Liv)*dt``) and ``dhip`` (precipitation
    turned into ice / snow-to-ice).  ``dhw``/``dhw0`` do not contain a snowmelt
    term here.

    The flag bookkeeping of the source is reproduced: precipitation below
    freezing temperature is stored in ``snowmass`` and, once it exceeds
    ``3*dzmin`` of fresh snow, switches the model to case 2.1
    (``flag_snow = 1``, ``itop = ms-2``).  The caller must then run the
    ``SNOWTEMP``/``SNOW_COND_HEAT_COEF`` sequence instead.

    ``dhwhigh``/``dhihigh`` are *not* reset when the ice top is not melting and
    the ``dhis`` branch does not fire (``Ti1(1) == x + T_phase_threshold``
    exactly); the stale values are kept, as in the source.
    """
    Mice, M, ms = st.Mice, st.M, st.ms
    Ti1, Tw1 = st.Ti1, st.Tw1
    l1, h1 = st.l1, st.h1
    cw = C.cw_m_row0
    x = meltpnt(0.0, 0.0, st.nmeltpoint)

    if Ti1[1] > x + C.T_phase_threshold:
        dhwhigh = ((Ti1[1] - x - C.T_phase_threshold)
                   * st.ci_m_roi_v[1] * st.ddzi[1] * l1 * 0.5 / C.row0_m_Lwi)
        dhis = 0.0
        dhihigh = -dhwhigh * C.roi_d_row0
        if st.saltice == 1:
            dhwhigh = dhwhigh * (1.0 + st.porice[1] * C.roi_d_row0
                                 / (1.0 - st.porice[1]))
            dhihigh = dhihigh * (1.0 / (1.0 - st.porice[1]))
        Ti1[1] = x + C.T_phase_threshold
    else:
        dhwhigh = st.dhwhigh
        dhihigh = st.dhihigh
        dhis = st.dhis

    # --- initial thin snow layer, to be used at the next timestep ---------
    if precip > 0.0 and tempair < 0.0 and l1 > 0.05:
        st.snowmass = st.snowmass + precip * dt * C.row0
        if st.snowmass / C.rofrsn >= 3.0 * C.dzmin:
            st.flag_snow = 1
            st.hs1 = st.snowmass / C.rofrsn + C.small_value
            st.itop = ms - 2
        dhip = 0.0
    else:
        dhip = precip * C.row0_d_roi * dt
    if tempair > 0.0 and st.snowmass > 0.0:
        dhip = dhip + st.snowmass / C.row0
        st.snowmass = 0.0

    flux1 = (-st.lami_v[Mice] * (Ti1[Mice + 1] - Ti1[Mice])
             / (st.ddzi[Mice] * l1)
             + st.rad_integr_ice[Mice]
             + st.ci_m_roi_v[Mice + 1] * (st.dhi - st.dhi0)
             * (Ti1[Mice + 1] - Ti1[Mice]) / (2.0 * dt))
    flux2 = (-st.lamw[1] * (Tw1[2] - Tw1[1]) / (st.ddz[1] * h1)
             + st.rad_integr_water[1]
             + cw * st.dhw0 * (Tw1[2] - Tw1[1]) / (2.0 * dt)
             + cw * st.PEMF[1] * (st.pt_down_f[1] - 0.5 * (Tw1[2] + Tw1[1])))
    dhwlow = min(max(dt * (flux1 - flux2) / C.row0_m_Lwi,
                     -st.ddz[1] * h1),
                 st.ddzi[Mice] * l1 * C.roi_d_row0)
    dhilow = -dhwlow * C.roi_d_row0
    if st.saltice == 1:
        dhwlow = dhwlow * (1.0 + st.porice[Mice + 1] * C.roi_d_row0
                           / (1.0 - st.porice[Mice + 1]))
        dhilow = dhilow * (1.0 / (1.0 - st.porice[Mice + 1]))

    if Ti1[1] < meltpnt(0.0, 0.0, st.nmeltpoint) + C.T_phase_threshold:
        dhis = -Elatent / (C.roi * C.Liv) * dt
        dhwhigh = 0.0
        dhihigh = 0.0

    dhi = dhihigh + dhilow + dhis + dhip
    dhi0 = dhis + dhihigh + dhip

    if st.ls1 != 0.0:
        flux1 = (-st.lamw[M] * (Tw1[M + 1] - Tw1[M]) / (st.ddz[M] * h1)
                 + st.rad_integr_water[M]
                 + cw * (st.dhw - st.dhw0) * (Tw1[M + 1] - Tw1[M])
                 / (2.0 * dt)
                 + cw * st.PEMF[M] * (st.pt_down_f[M]
                                      - 0.5 * (Tw1[M + 1] + Tw1[M])))
        flux2 = (-C.lami * (st.Tis1[2] - st.Tis1[1]) / (st.ddzi[1] * st.ls1)
                 + st.rad_integr_deepice[1]
                 + C.ci_m_roi * st.dls0 * (st.Tis1[2] - st.Tis1[1])
                 / (2.0 * dt))
        dhwls = dt * (flux1 - flux2) / C.row0_m_Lwi
        dls = -dhwls * C.roi_d_row0
        dls0 = dls
        dhw = dhwhigh + st.dhwfsoil + dhwls + dhwlow + st.dhwtrib
        dhw0 = dhwhigh + dhwlow
    else:
        dhw = dhwhigh + st.dhwfsoil + dhwlow + st.dhwtrib
        dhw0 = dhwhigh + dhwlow
        dls = 0.0
        dls0 = st.dls0
        dhwls = st.dhwls  # the source leaves dhwls untouched here

    st.dhwlow, st.dhilow = dhwlow, dhilow
    st.dhwhigh, st.dhihigh = dhwhigh, dhihigh
    st.dhwls = dhwls
    st.dhi, st.dhi0 = dhi, dhi0
    st.dls, st.dls0 = dls, dls0
    st.dhw, st.dhw0 = dhw, dhw0
    st.dhwsnmelt = 0.0
    st.dhis = dhis
    st.dhip = dhip

    return dict(dhwlow=dhwlow, dhilow=dhilow, dhwhigh=dhwhigh,
                dhihigh=dhihigh, dhis=dhis, dhip=dhip, dhwls=dhwls,
                dhi=dhi, dhi0=dhi0, dls=dls, dls0=dls0, dhw=dhw, dhw0=dhw0)


# ---------------------------------------------------------------------------
# UPDATE_ICE_SALINITY
# ---------------------------------------------------------------------------

def salicebot(dhbdt, salwat, por):
    """``SALICEBOT`` (``phys_func.f90:1569-1587``): ice bottom bulk salinity.

    ``w`` is the freezing rate in mm/h (``max(dhbdt,0)*3.6E+6``), then
    ``(row0*por + (1-por)*roi) * salwat*alpha*sqrt(w)/(beta*sqrt(w)+gamma_)``
    with ``alpha = beta = 7.`` and ``gamma_ = 10.3`` (default real).
    """
    w = max(dhbdt, 0.0) * _s(3.6e6)
    return ((C.row0 * por + (1.0 - por) * C.roi) * salwat
            * _s(7.0) * np.sqrt(w)
            / (_s(7.0) * np.sqrt(w) + _s(10.3)))


def update_ice_salinity(st: IceState, dt, Sal1, isoilcol=None):
    """``UPDATE_ICE_SALINITY`` (``salinity_mod.f90:306-395``), in place.

    Solves the bulk ice salinity with :func:`diff_coef_ice_sal`, then

    * top b.c. (``salinity_mod.f90:361-366``): a flux condition driven by the
      water-layer thickness change ``dhw0``;
    * bottom b.c.: at the water-ice interface the pore salinity follows
      ``max(SALICEBOT((dhw-dhw0)/dt, Sal1(1), porice(Mice+1)),
      Sal1(1)*row0*poricebot)``; with no water it is a zero-gradient condition;
    * ``salice(i) = max(salice(i), sal_min)`` for ``i = 1..Mice`` with
      ``sal_min = 1.e-2`` (default real);
    * ``porice(Mice+1) = salice(Mice+1)/(Sal1(1)*row0 + small_value)`` and the
      porosity rescaling ``porice(i) *= salice(i)/salprev(i)``.

    Only the ``h1 > 0`` branch is reachable with water present; both are
    transcribed.  ``isoilcol`` is accepted for signature symmetry and unused.
    """
    Mice = st.Mice
    nn = st.nveclen
    a = np.zeros(nn + 1)
    b = np.zeros(nn + 1)
    c = np.zeros(nn + 1)
    d = np.zeros(nn + 1)
    _S001 = _s(1.0e-2)  # sal_min

    salprev = st.salice.copy()

    diff_coef_ice_sal(a, b, c, d, 2, Mice, 2, Mice, dt, st)

    # 2. Salinity equation discretized at the top boundary
    b[1] = -0.5 * st.dhw0 / (st.h1 * dt * st.ddzi[1])
    c[1] = 1.0 / dt + b[1]
    d[1] = (st.salice[1] * (1.0 / dt - C.asalice)
            - 0.5 * st.dhw0 / (st.h1 * dt * st.ddzi[1])
            * (st.salice[2] - st.salice[1]))

    if st.h1 > 0.0:
        # Water-ice interface
        a[Mice + 1] = 0.0
        c[Mice + 1] = 1.0
        d[Mice + 1] = max(
            salicebot((st.dhw - st.dhw0) / dt, Sal1[1], st.porice[Mice + 1]),
            Sal1[1] * C.row0 * C.poricebot)
    else:
        # Ice-soil interface: zero salinity gradient
        a[Mice + 1] = 1.0
        c[Mice + 1] = 1.0
        d[Mice + 1] = 0.0

    y = _progonka(a, b, c, d, 1, Mice + 1)
    st.salice[1:Mice + 2] = y[1:Mice + 2]

    for i in range(1, Mice + 1):
        st.salice[i] = max(st.salice[i], _S001)

    # Brine salt concentration at the ice bottom equals the water salinity
    st.porice[Mice + 1] = (st.salice[Mice + 1]
                           / (Sal1[1] * C.row0 + C.small_value))

    # Rescaling of the water-filled porosity after the bulk salinity update
    for i in range(1, Mice + 1):
        st.porice[i] = st.porice[i] * st.salice[i] / salprev[i]

    return st

