"""B3c: batched JAX ``MIXED_LAYER_CALC`` / ``TURB_SCALES`` match NumPy.

The reference arguments are the ones :func:`lakematrix.step.step_case1` really
builds for the ``ref/v1/summer60d`` case (captured by spying on
``physics.turb.turb_scales`` during a few warm-up steps), so the density
profile, the k-epsilon diffusivities, the radiation integrals and the
``DDENS_*`` constants are all realistic rather than synthetic.

The batched kernels are compared with the NumPy function **per lake** for
``L = 1`` and ``L = 4`` (identical lanes, lanes with different geometry /
forcing, and lanes that force the ``jnp.where`` branches: well-mixed fallback,
inverted profile, no 14 degC / 8 degC crossing, repeated crossings).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.forcing import read_ascii                          # noqa: E402
from lakematrix.grid import create_grid                            # noqa: E402
from lakematrix.physics.turb import (                              # noqa: E402
    mixed_layer_calc as np_mixed_layer_calc,
    turb_scales as np_turb_scales,
)
from lakematrix.setup_io import read_case                          # noqa: E402
from lakematrix.step import build_step_state, step_case1           # noqa: E402

CASE = ROOT / "ref" / "v1" / "summer60d"
NSTEPS = 3             # warm-up steps of the real case
ATOL = 1e-10           # target tolerance of the port

pytestmark = pytest.mark.skipif(not CASE.is_dir(),
                                reason="reference case ref/v1/summer60d missing")

#: names of the ``turb_scales`` return values
SCALARS = ("w_conv_scale", "T_conv_scale", "Wedderburn", "LakeNumber",
           "Rossby_rad", "ThermThick", "ReTherm", "RiTherm")

#: positional argument names of ``physics.turb.turb_scales``
ARGS = ("row", "ddz", "ddz2", "dzeta_int", "dzeta_05int", "h1", "M", "tw2",
        "sal2", "u2", "v2", "lamw", "lamsal", "k_turb_T_flux", "T_massflux",
        "eflux0_kinem", "turb_density_flux", "Buoyancy0", "tau", "kor",
        "rad_integr", "area_int", "Lx", "Ly", "vol", "z_full", "ddz05",
        "DDENS_DTEMP0", "DDENS_DSAL0")
M_POS = ARGS.index("M")


@pytest.fixture(scope="module")
def real_args():
    """The last ``turb_scales`` argument tuple of ``NSTEPS`` case-1 steps."""
    import lakematrix.physics.turb as turb_mod

    case = read_case(CASE)
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(CASE / "meteo" / "test.dat")
    ss = build_step_state(case, grid)
    orig = turb_mod.turb_scales
    calls = []

    def spy(*a, **kw):
        calls.append(a)
        return orig(*a, **kw)

    turb_mod.turb_scales = spy
    try:
        for k in range(1, NSTEPS + 1):
            step_case1(ss, case, grid, rows, k)
    finally:
        turb_mod.turb_scales = orig
    assert len(calls) == NSTEPS
    args = tuple(np.asarray(a, dtype=float) if np.ndim(a) else a
                 for a in calls[-1])
    assert int(args[M_POS]) == case.M
    return args


def _lane(args, **over):
    """Copy one unbatched argument tuple, overriding a few entries by index."""
    out = list(args)
    for name, value in over.items():
        out[ARGS.index(name)] = value
    return tuple(out)


def _stack(per_lane):
    """Stack per-lane unbatched tuples along a new leading lake axis."""
    M = int(per_lane[0][M_POS])
    out = []
    for j in range(len(per_lane[0])):
        if j == M_POS:
            assert all(int(t[j]) == M for t in per_lane)
            out.append(M)
        else:
            out.append(np.stack([np.asarray(t[j], dtype=float)
                                 for t in per_lane]))
    return tuple(out)


def _compare(out, ref):
    """Compare one batched ``turb_scales`` dict with a list of NumPy results."""
    L = len(ref)
    assert int(np.asarray(out["i_maxN"]).shape[0]) == L
    for l in range(L):
        assert int(np.asarray(out["i_maxN"])[l]) == int(ref[l]["i_maxN"]), \
            f"i_maxN lake {l}"
        np.testing.assert_allclose(np.asarray(out["H_mixed_layer"])[l],
                                   ref[l]["H_mixed_layer"], rtol=0, atol=ATOL,
                                   err_msg=f"H_mixed_layer lake {l}")
        np.testing.assert_allclose(np.asarray(out["maxN"])[l], ref[l]["maxN"],
                                   rtol=0, atol=ATOL,
                                   err_msg=f"maxN lake {l}")
        for k in SCALARS:
            np.testing.assert_allclose(np.asarray(out[k])[l], ref[l][k],
                                       rtol=0, atol=ATOL,
                                       err_msg=f"{k} lake {l}")
        for k in ("Rp", "Rpdens"):
            np.testing.assert_allclose(np.asarray(out[k])[l], ref[l][k],
                                       rtol=0, atol=ATOL,
                                       err_msg=f"{k} lake {l}")


def _same_jit(jgot, got, what):
    """Eager vs traced: same formulas, at most a few ulp apart.

    XLA may reassociate ``-x2/x1*dsal/x`` while the eager path keeps the
    Python evaluation order, so a 1-ulp difference is expected (never more).
    """
    for k in got:
        np.testing.assert_allclose(np.asarray(jgot[k]), np.asarray(got[k]),
                                   rtol=1.0e-13, atol=0.0,
                                   err_msg=f"{what} {k}")


# ---------------------------------------------------------------------------
# MIXED_LAYER_CALC
# ---------------------------------------------------------------------------
def test_mixed_layer_calc_matches_numpy(real_args):
    from lakematrix.jaxbatch.turb import (jitted_mixed_layer_calc,
                                          mixed_layer_calc)

    M = int(real_args[M_POS])
    row, ddz = real_args[ARGS.index("row")], real_args[ARGS.index("ddz")]
    dzeta = real_args[ARGS.index("dzeta_05int")]
    h1 = float(real_args[ARGS.index("h1")])
    want = np_mixed_layer_calc(row, ddz, dzeta, h1, M)

    # --- one lake -----------------------------------------------------------
    ml, H, maxN = mixed_layer_calc(row[None, :], ddz[None, :], dzeta[None, :],
                                   np.array([h1]), M)
    assert np.asarray(ml).shape == (1,) and np.asarray(H).shape == (1,)
    assert int(np.asarray(ml)[0]) == want[0]          # 1-based, exact
    np.testing.assert_allclose(np.asarray(H)[0], want[1], rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.asarray(maxN)[0], want[2], rtol=0, atol=ATOL)
    # eager and jitted agree bit for bit
    jml, jH, jmaxN = jitted_mixed_layer_calc(row[None, :], ddz[None, :],
                                             dzeta[None, :], np.array([h1]), M)
    np.testing.assert_array_equal(np.asarray(jml), np.asarray(ml))
    np.testing.assert_array_equal(np.asarray(jH), np.asarray(H))
    np.testing.assert_array_equal(np.asarray(jmaxN), np.asarray(maxN))

    # --- four lakes: identical lanes + three synthetic profiles ------------
    rows = [row, row, 1000.0 + 1.0e-6 * np.arange(M + 1),
            1000.0 - 0.05 * np.arange(M + 1)]     # well mixed / unstable
    rows[3] = rows[3] - rows[3][0] + row[0]
    rows[2] = rows[2] * (row[0] / 1000.0)
    h1s = np.array([h1, h1 * 1.35, h1, h1 * 0.7])
    got = mixed_layer_calc(np.stack(rows),
                           np.repeat(ddz[None, :], 4, axis=0),
                           np.repeat(dzeta[None, :], 4, axis=0),
                           h1s, M)
    for l in range(4):
        w = np_mixed_layer_calc(rows[l], ddz, dzeta, float(h1s[l]), M)
        assert int(np.asarray(got[0])[l]) == w[0], f"i_ML lake {l}"
        np.testing.assert_allclose(np.asarray(got[1])[l], w[1], rtol=0,
                                   atol=ATOL, err_msg=f"H lake {l}")
        np.testing.assert_allclose(np.asarray(got[2])[l], w[2], rtol=0,
                                   atol=ATOL, err_msg=f"maxN lake {l}")
    # the well-mixed lane falls back to `ml = M`, the unstable one too
    assert int(np.asarray(got[0])[2]) == M
    assert int(np.asarray(got[0])[3]) == M
    assert float(np.asarray(got[2])[3]) == 0.0     # all bvf < 0 -> maxN = 0


# ---------------------------------------------------------------------------
# TURB_SCALES
# ---------------------------------------------------------------------------
def test_turb_scales_one_lake(real_args):
    from lakematrix.jaxbatch.turb import jitted_turb_scales, turb_scales

    M = int(real_args[M_POS])
    batched = _stack([real_args])                    # L = 1
    want = np_turb_scales(*real_args)
    got = turb_scales(*batched)
    _compare(got, [want])
    assert np.asarray(got["Rp"]).shape == (1, M)
    assert np.asarray(got["Rpdens"]).shape == (1, M)
    assert np.asarray(got["i_maxN"]).dtype.kind == "i"

    jgot = jitted_turb_scales(*batched)
    _same_jit(jgot, got, "jit L=1")


def test_turb_scales_four_lakes(real_args):
    from lakematrix.jaxbatch.turb import turb_scales

    M = int(real_args[M_POS])
    idx = {n: ARGS.index(n) for n in ARGS}
    row, tw2 = real_args[idx["row"]], real_args[idx["tw2"]]
    sal2, h1 = real_args[idx["sal2"]], float(real_args[idx["h1"]])
    z_full, dzeta_int = real_args[idx["z_full"]], real_args[idx["dzeta_int"]]
    vol, tau = float(real_args[idx["vol"]]), float(real_args[idx["tau"]])
    eflux0, Buoy = (float(real_args[idx["eflux0_kinem"]]),
                    float(real_args[idx["Buoyancy0"]]))

    # lane 0: the reference state itself
    lanes = [real_args]
    # lane 1: deeper lake (h1, z_full), stronger wind, different Coriolis
    h11 = h1 * 1.25
    lanes.append(_lane(real_args, h1=h11, z_full=dzeta_int * h11, tau=tau * 4.0,
                       vol=vol * 0.6, kor=1.0e-4, eflux0_kinem=eflux0 * 2.0,
                       Buoyancy0=Buoy * 0.5))
    # lane 2: +0.75 K everywhere, artificial salinity profile (fresh water has
    # Sal = 0, which would leave Rp/Rpdens = 0 untested)
    sal = np.linspace(0.0, 1.0e-3, M + 1)
    lanes.append(_lane(real_args, tw2=tw2 + 0.75, sal2=sal,
                       lamw=real_args[idx["lamw"]] * 1.1,
                       tau=tau * 0.25))
    # lane 3: different DDENS_* constants and a scaled density profile
    lanes.append(_lane(real_args, row=row * (1.0 + 1.0e-4),
                       DDENS_DTEMP0=float(real_args[idx["DDENS_DTEMP0"]]) * 1.05,
                       DDENS_DSAL0=float(real_args[idx["DDENS_DSAL0"]]) * 0.95,
                       vol=vol * 2.0, kor=4.2e-5))

    batched = _stack(lanes)
    assert batched[idx["row"]].shape == (4, M + 1)
    assert batched[idx["h1"]].shape == (4,)
    got = turb_scales(*batched)
    _compare(got, [np_turb_scales(*t) for t in lanes])
    # the lanes are independent: the mixed-layer index or H must move
    assert len({int(x) for x in np.asarray(got["i_maxN"])}) > 1 or \
        len({float(x) for x in np.asarray(got["H_mixed_layer"])}) > 1
    assert not np.allclose(np.asarray(got["Rpdens"])[0],
                           np.asarray(got["Rpdens"])[2])
    # the jitted kernel gives the same numbers as the eager one
    from lakematrix.jaxbatch.turb import jitted_turb_scales
    jgot = jitted_turb_scales(*batched)
    _same_jit(jgot, got, "jit L=4")


def test_turb_scales_branches(real_args):
    """Lanes that force the ``jnp.where`` branches of the port."""
    from lakematrix.jaxbatch.turb import make_turb_scales_kernel

    M = int(real_args[M_POS])
    idx = {n: ARGS.index(n) for n in ARGS}
    row, tw2 = real_args[idx["row"]], real_args[idx["tw2"]]
    ddz, h1 = real_args[idx["ddz"]], float(real_args[idx["h1"]])
    z_full = real_args[idx["z_full"]]
    row0 = row[0]

    # lane 0: constant density (ml = M), constant 20 degC (no 14/8 crossing)
    lanes = [_lane(real_args, row=np.full(M + 1, row0),
                   tw2=np.full(M + 1, 20.0),
                   sal2=np.zeros(M + 1),
                   # w_conv_scale < 0 -> clipped to 0 by `max(..., 0)`
                   Buoyancy0=-1.0,
                   rad_integr=np.linspace(500.0, 50.0, M + 1))]
    # lane 1: statically unstable profile and a reversed surface-to-bottom
    # density difference (-> the `row(M+1) - row(1)` terms change sign)
    lanes.append(_lane(real_args, row=row0 + 0.2 * np.arange(M + 1),
                       tw2=tw2[::-1].copy(), sal2=np.zeros(M + 1),
                       tau=1.0e-9))
    # lane 2: repeated 14 degC crossings -- the *last* one must win
    tw2b = tw2.copy()
    tw2b[M // 3:M // 2 + 1] = np.maximum(tw2b[M // 3:M // 2 + 1], 15.5)
    lanes.append(_lane(real_args, tw2=tw2b))
    # lane 3: every value below 8 degC -> no crossing, defaults kept
    lanes.append(_lane(real_args, tw2=np.linspace(7.9, 4.0, M + 1),
                       sal2=np.linspace(1.0e-3, 0.0, M + 1)))

    batched = _stack(lanes)
    kernel = make_turb_scales_kernel(M)
    got = kernel(*[a for j, a in enumerate(batched) if j != M_POS])
    ref = [np_turb_scales(*t) for t in lanes]
    _compare(got, ref)
    # the assumed branch behaviour, i.e. the NumPy reference's own semantics
    assert int(np.asarray(got["i_maxN"])[0]) == M
    assert float(np.asarray(got["maxN"])[0]) == 0.0
    assert float(np.asarray(got["w_conv_scale"])[0]) == 0.0
    # no 14 degC crossing anywhere -> x1 = 0; no 8 degC crossing -> x2 = h1
    np.testing.assert_allclose(np.asarray(got["ThermThick"])[1], h1, rtol=0,
                               atol=ATOL)
    np.testing.assert_allclose(np.asarray(got["ThermThick"])[3], h1, rtol=0,
                               atol=ATOL)
    # the repeated-crossing lane still lands on exactly one crossing, and it
    # is the only lane with one (lanes 0/1/3 keep x1 = 0 and x2 = h1)
    assert np.isfinite(float(np.asarray(got["ThermThick"])[2]))
    assert 0.0 < float(np.asarray(got["ThermThick"])[2]) < h1
    np.testing.assert_allclose(np.asarray(got["ThermThick"])[[0, 1, 3]],
                               np.full(3, h1), rtol=0, atol=ATOL)


def test_turb_scales_rejects_bad_shapes(real_args):
    from lakematrix.jaxbatch.turb import turb_scales

    batched = list(_stack([real_args]))
    batched[ARGS.index("ddz")] = batched[ARGS.index("ddz")][:, :-1]
    with pytest.raises(ValueError, match="ddz"):
        turb_scales(*batched)
    # `ddz2`, `dzeta_int` and the k_turb/mass-flux arrays are not read at all
    batched = list(_stack([real_args]))
    for name in ("ddz2", "dzeta_int", "k_turb_T_flux", "T_massflux",
                 "turb_density_flux"):
        batched[ARGS.index(name)] = None
    turb_scales(*batched)
