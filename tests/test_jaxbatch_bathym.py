"""B5: JAX batched case-1 bathymetry matches the NumPy reference.

The reference state is the real ``ref/v1/summer60d`` lake (``M = 40``,
25-point morphometry table, ``nsoilcols = 5``, ``lakeform = 2`` rectangle,
``cellipt = 20``), evaluated at the initial water depth and at ``h1 - 1`` /
``h1 + 1`` m.

* :func:`lakematrix.jaxbatch.bathym.bathymetry_water` is compared with
  :func:`lakematrix.physics.bathym.bathymetry` for all twelve ``Bathym``
  fields;
* :func:`lakematrix.jaxbatch.bathym.zsoilcols_jax` /
  :func:`lakematrix.jaxbatch.bathym.soil_bathymetry_jax` with
  ``grid.zsoilcols`` / :func:`lakematrix.physics.bathym.soil_bathymetry`
  (the 1-based ``bathymsoil`` list); floating fields use ``1e-10`` relative
  *and* absolute, the ``itop``/``ibot``/``icent`` level indices must be exact.

The four-lake batch mixes ``h1`` values and is also compared lane by lane with
the single-lake calls (bit-identical on CPU) and with ``jax.jit`` (XLA
reorders the area/volume sums, so the jitted kernel keeps the 1e-10
tolerance).  The last test exercises the per-lake branches that the real case
never reaches: ``lakeform = 1`` (ellipse) and a lake without a morphometry
table (constant ``area_lake``).
"""

import copy
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.constants import C  # noqa: E402
from lakematrix.grid import create_grid, zsoilcols  # noqa: E402
from lakematrix.jaxbatch.bathym import (bathymetry_water,  # noqa: E402
                                        soil_bathymetry_jax, zsoilcols_jax)
from lakematrix.physics.bathym import (bathymetry,  # noqa: E402
                                       soil_bathymetry)
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.state import init_state  # noqa: E402

ATOL = 1e-10
RTOL = 1e-10

WATER_KEYS = ("z_full", "z_half", "preswat", "area_int", "area_half", "vol",
              "botar", "Lx", "Ly", "rad_int", "Lx_half", "Ly_half")
SOIL_FLOAT = ("area_int", "area_half", "Lx", "Ly", "dzSL", "dzSLc")
SOIL_INT = ("itop", "ibot", "icent")


def _case():
    case = read_case(ROOT / "ref/v1/summer60d")
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    return case, grid, init_state(case, grid)


CASE, GRID, STATE = _case()
H0 = float(STATE.h1[0])


def _batched(h1s, lakeform=None, morphometry_len=None):
    """Replicate the summer60d lake ``L`` times with one ``h1`` per lake."""
    h1s = np.asarray(h1s, dtype=float)
    L = h1s.size
    ml = (np.repeat(CASE.morphometry_len, L) if morphometry_len is None
          else np.asarray(morphometry_len, dtype=int))
    form = (np.repeat(CASE.lakeform, L) if lakeform is None
            else np.asarray(lakeform))
    params = dataclasses.replace(
        CASE, L=L, h10=h1s,
        morphometry=np.repeat(CASE.morphometry, L, axis=0),
        morphometry_len=ml,
        area_lake=np.full(L, float(np.atleast_1d(CASE.area_lake)[0])),
        lakeform=form,
        cellipt=np.full(L, float(np.atleast_1d(CASE.cellipt)[0])))
    state = copy.deepcopy(STATE)
    state.h1 = h1s.copy()
    state.ls1 = np.zeros(L)
    return params, state


def _numpy_water(h1s):
    params, state = _batched(h1s)
    return bathymetry(params, GRID, state, C.pref)


def _jax_water(h1s):
    params, state = _batched(h1s)
    return bathymetry_water(state.h1, state.ls1, params.area_lake,
                            params.morphometry, params.morphometry_len,
                            params.lakeform, params.cellipt, C.pref,
                            GRID.dzeta_int, GRID.dzeta_05int)


def _compare_water(out, ref, L):
    for key in WATER_KEYS:
        want = np.asarray(getattr(ref, key), dtype=float)
        got = np.asarray(out[key], dtype=float)
        assert got.shape == (L,) + want.shape[1:], key
        np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL,
                                   err_msg=key)


def _numpy_soil(h1s):
    """Per-lane ``soil_bathymetry`` list and ``zsoilcols`` of the reference."""
    params, state = _batched(h1s)
    nc = params.nsoilcols
    cols = [soil_bathymetry(params, GRID, state, l) for l in range(len(h1s))]
    zsc = np.stack([
        zsoilcols(state.h1[l],
                  params.morphometry[l, :int(params.morphometry_len[l])],
                  GRID, nc)
        for l in range(len(h1s))])
    expect = {key: np.array([[getattr(b[i], key) for i in range(nc + 2)]
                             for b in cols])
              for key in SOIL_FLOAT + SOIL_INT}
    expect["zsoilcols"] = zsc
    return expect


def _jax_soil(h1s, use_jit=False):
    params, state = _batched(h1s)
    args = (state.h1, params.morphometry, params.morphometry_len,
            params.lakeform, params.cellipt)
    if use_jit:
        # ``grid`` and ``nsoilcols`` are static; a Grid instance is not a JAX
        # pytree, so it is closed over instead of passed as a traced argument.
        def fn(h1, morph, ml, form, cell):
            return soil_bathymetry_jax(h1, morph, ml, form, cell, GRID,
                                       params.nsoilcols)
        return fn(*args)
    return soil_bathymetry_jax(*args, GRID, params.nsoilcols)


def _compare_soil(out, expect, L):
    nzc = expect["zsoilcols"].shape[1]          # nsoilcols + 1
    got = np.asarray(out["zsoilcols"])
    assert got.shape == (L, nzc)
    np.testing.assert_allclose(got, expect["zsoilcols"], rtol=RTOL, atol=ATOL,
                               err_msg="zsoilcols")
    for key in SOIL_FLOAT:
        got = np.asarray(out[key], dtype=float)
        assert got.shape == (L, nzc + 1), key
        np.testing.assert_allclose(got, expect[key], rtol=RTOL, atol=ATOL,
                                   err_msg=key)
    for key in SOIL_INT:
        got = np.asarray(out[key])
        assert got.shape == (L, nzc + 1), key
        np.testing.assert_array_equal(got, expect[key], err_msg=key)


# ---------------------------------------------------------------------------
# water bathymetry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dh", [0.0, -1.0, 1.0],
                         ids=["initial", "minus1m", "plus1m"])
def test_bathymetry_water_one_lake(dh):
    h1s = [H0 + dh]
    out = _jax_water(h1s)
    _compare_water(out, _numpy_water(h1s), 1)


def test_bathymetry_water_four_lakes():
    h1s = [H0, H0 - 1.0, H0 + 1.0, H0 + 1.0]
    out = _jax_water(h1s)
    _compare_water(out, _numpy_water(h1s), 4)

    # A per-lake pressure (L,) must broadcast to the lake axis, not mix it
    # with the level axis; with equal values the result is unchanged.
    params, state = _batched(h1s)
    out_p = bathymetry_water(state.h1, state.ls1, params.area_lake,
                             params.morphometry, params.morphometry_len,
                             params.lakeform, params.cellipt,
                             np.full(4, C.pref), GRID.dzeta_int,
                             GRID.dzeta_05int)
    for key in WATER_KEYS:
        np.testing.assert_array_equal(np.asarray(out_p[key]),
                                      np.asarray(out[key]), err_msg=key)

    # Batch consistency: every lane is what the single-lake kernel gives, and
    # the two identical lakes are bit-identical members of the batch.
    for lane, h1 in enumerate(h1s):
        one = _jax_water([h1])
        for key in WATER_KEYS:
            np.testing.assert_array_equal(np.asarray(out[key])[lane],
                                          np.asarray(one[key])[0],
                                          err_msg=f"{key}, lake {lane}")
    for key in WATER_KEYS:
        np.testing.assert_array_equal(np.asarray(out[key])[2],
                                      np.asarray(out[key])[3], err_msg=key)


def test_bathymetry_water_jit():
    import jax

    h1s = [H0, H0 - 1.0, H0 + 1.0, H0 + 1.0]
    params, state = _batched(h1s)

    def fn(h1, ls1, morph):
        return bathymetry_water(
            h1, ls1, params.area_lake, morph, params.morphometry_len,
            params.lakeform, params.cellipt, C.pref, GRID.dzeta_int,
            GRID.dzeta_05int)

    out = jax.jit(fn)(state.h1, state.ls1, params.morphometry)
    _compare_water(out, _numpy_water(h1s), 4)


def test_bathymetry_water_per_lake_branches():
    """Ellipse form and a lake without a table, next to the rectangle lake.

    The reference is the NumPy loop over the same batch (lake 0 has a table, so
    its "no morphometry" flag is set the same way and the per-lake branch is
    taken identically).
    """
    h1s = [H0, H0 - 1.0, H0 + 1.0, H0 - 0.5]
    form = np.array([2, 1, 2, 2])
    ml = np.array([CASE.morphometry_len[0]] * 4)
    ml[2] = 0                                    # no table -> constant area
    params, state = _batched(h1s, lakeform=form, morphometry_len=ml)
    ref = bathymetry(params, GRID, state, C.pref)
    out = bathymetry_water(state.h1, state.ls1, params.area_lake,
                           params.morphometry, params.morphometry_len,
                           params.lakeform, params.cellipt, C.pref,
                           GRID.dzeta_int, GRID.dzeta_05int)
    _compare_water(out, ref, 4)
    area_lake = float(np.atleast_1d(CASE.area_lake)[0])
    np.testing.assert_array_equal(np.asarray(out["area_int"])[2],
                                  np.full(GRID.M + 1, area_lake))
    np.testing.assert_array_equal(np.asarray(out["area_half"])[2],
                                  np.full(GRID.M, area_lake))


# ---------------------------------------------------------------------------
# soil-column bathymetry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dh", [0.0, -1.0, 1.0],
                         ids=["initial", "minus1m", "plus1m"])
def test_soil_bathymetry_one_lake(dh):
    h1s = [H0 + dh]
    out = _jax_soil(h1s)
    _compare_soil(out, _numpy_soil(h1s), 1)
    # zsoilcols_jax on its own
    params, state = _batched(h1s)
    zsc = zsoilcols_jax(state.h1, params.morphometry, GRID, params.nsoilcols)
    np.testing.assert_allclose(np.asarray(zsc),
                               _numpy_soil(h1s)["zsoilcols"],
                               rtol=RTOL, atol=ATOL)


def test_soil_bathymetry_four_lakes():
    h1s = [H0, H0 - 1.0, H0 + 1.0, H0 + 1.0]
    expect = _numpy_soil(h1s)
    out = _jax_soil(h1s)
    _compare_soil(out, expect, 4)

    for lane, h1 in enumerate(h1s):
        one = _jax_soil([h1])
        for key in ("zsoilcols",) + SOIL_FLOAT + SOIL_INT:
            np.testing.assert_array_equal(np.asarray(out[key])[lane],
                                          np.asarray(one[key])[0],
                                          err_msg=f"{key}, lake {lane}")


def test_soil_bathymetry_jit():
    h1s = [H0, H0 - 1.0, H0 + 1.0, H0 + 1.0]
    out = _jax_soil(h1s, use_jit=True)
    _compare_soil(out, _numpy_soil(h1s), 4)
