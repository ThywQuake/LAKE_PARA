"""B5/B4: batched JAX ``K_EPSILON`` (case 1) matches the NumPy reference.

The reference state is a real ``ref/v1/summer60d`` run advanced a few
``step_case1`` steps, i.e. after the k-epsilon closure has been warmed up, so
both the ``init_keps = .true.`` (one pass) and ``init_keps = .false.`` (two
passes) paths are exercised.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.forcing import read_ascii                      # noqa: E402
from lakematrix.grid import create_grid                        # noqa: E402
from lakematrix.physics.kepsilon import (                      # noqa: E402
    KEpsilonParams,
    k_epsilon as np_k_epsilon,
)
from lakematrix.setup_io import read_case                      # noqa: E402
from lakematrix.step import build_step_state, step_case1       # noqa: E402

CASE = ROOT / "ref" / "v1" / "summer60d"
NSTEPS = 3            # warm-up steps of the real case
ATOL = 1e-8           # target tolerance of the port

#: the full state interface of the task statement (all must be accepted)
KEYS = (
    "e1", "eps1", "k2", "k2t", "k5", "k2_mid", "k3_mid", "k5_mid", "l", "tf",
    "kt", "gamt", "gamu", "gamv", "row", "row2", "z_half", "u1", "v1", "u2",
    "v2", "um", "vm", "tw1", "tw2", "sal1", "sal2", "knum", "veg_sw", "lamw",
    "lamsal", "h1", "vol", "botar", "b0", "tau_air", "tau_wav", "tau_i",
    "tau_gr", "uwind", "vwind", "fetch", "roughness",
    # grid / bathymetry needed by the closure itself
    "ddz", "ddz05", "dzeta_int", "dzeta_05int", "area_int", "area_half",
    # water-depth increments of the current model step
    "dhw", "dhw0",
)


@pytest.fixture(scope="module")
def warm_case():
    """``(case, KEpsilonState)`` after ``NSTEPS`` real case-1 steps."""
    case = read_case(CASE)
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(CASE / "meteo" / "test.dat")
    ss = build_step_state(case, grid)
    for k in range(1, NSTEPS + 1):
        step_case1(ss, case, grid, rows, k)
    assert bool(ss.ke.init_keps)
    return case, ss.ke


def _batched(ke, L):
    """Replicate one lake state ``L`` times into the batched input dict."""
    state = {}
    for k in KEYS:
        a = np.asarray(getattr(ke, k), dtype=float)
        if a.ndim == 0:
            state[k] = np.full(L, float(a))
        else:
            state[k] = np.repeat(a[None, :], L, axis=0)
    return state


def _reference(ke, case, init_keps=None):
    st = copy.deepcopy(ke)
    if init_keps is not None:
        st.init_keps = init_keps
    return np_k_epsilon(st, case.dt, dt_keps=case.dt, nstep_keps=1,
                        params=KEpsilonParams(M=case.M))


def _compare(out, ref, L, M, names, atol=ATOL):
    for name in names:
        got = np.asarray(out[name])
        want = np.asarray(getattr(ref, name))
        assert got.shape[-1] == want.shape[-1], f"{name}: {got.shape}"
        for l in range(L):
            np.testing.assert_allclose(
                got[l], want, rtol=0, atol=atol,
                err_msg=f"{name} lake {l}")


@pytest.mark.parametrize("L", [1, 4])
def test_jax_kepsilon_matches_numpy(warm_case, L):
    from lakematrix.jaxbatch.kepsilon import k_epsilon

    case, ke = warm_case
    M, dt = case.M, case.dt
    state = _batched(ke, L)
    state["init_keps"] = np.ones(L)

    out = k_epsilon(state, dt, dt, 1, KEpsilonParams(M=M))
    ref = _reference(ke, case)

    # the four quantities named by the task ...
    _compare(out, ref, L, M, ("e1", "eps1", "k2", "kt"))
    # ... and the rest of the closure output
    _compare(out, ref, L, M, ("k2t", "k5", "k2_mid", "k3_mid", "k5_mid",
                              "l", "tf", "ce", "lam_e", "lam_eps"))

    # identical lakes -> identical result
    if L > 1:
        for name in ("e1", "eps1", "k2", "kt"):
            got = np.asarray(out[name])
            assert np.array_equal(got, np.broadcast_to(got[0], got.shape))


def test_jax_kepsilon_two_pass_init(warm_case):
    """``init_keps = .false.`` reproduces the two ``keps_mode`` passes."""
    from lakematrix.jaxbatch.kepsilon import k_epsilon

    case, ke = warm_case
    M, dt = case.M, case.dt
    L = 2
    state = _batched(ke, L)
    state["init_keps"] = np.zeros(L)

    out = k_epsilon(state, dt, dt, 1, KEpsilonParams(M=M))
    ref = _reference(ke, case, init_keps=False)

    _compare(out, ref, L, M, ("e1", "eps1", "k2", "kt", "k2t", "k5", "l"))


def test_jax_kepsilon_sub_steps(warm_case):
    """``nstep_keps = 4`` / ``dt_keps = dt/4`` (``dhw_keps`` scaling)."""
    from lakematrix.jaxbatch.kepsilon import k_epsilon

    case, ke = warm_case
    M, dt = case.M, case.dt
    L = 2
    state = _batched(ke, L)
    state["init_keps"] = np.ones(L)

    out = k_epsilon(state, dt, dt / 4.0, 4, KEpsilonParams(M=M))
    ref = np_k_epsilon(copy.deepcopy(ke), dt, dt_keps=dt / 4.0, nstep_keps=4,
                       params=KEpsilonParams(M=M))

    _compare(out, ref, L, M, ("e1", "eps1", "k2", "kt"))


def test_jax_kepsilon_heterogeneous_batch(warm_case):
    """Per-lake states and per-lake ``init_keps`` flags in one call."""
    from lakematrix.jaxbatch.kepsilon import k_epsilon

    case, ke = warm_case
    M, dt = case.M, case.dt
    L = 4
    state = _batched(ke, L)
    states = []
    for l in range(L):
        st = copy.deepcopy(ke)
        st.e1 = ke.e1 * (1.0 + 0.3 * l)
        st.eps1 = ke.eps1 * (2.0 - 0.2 * l)
        st.u1 = ke.u1 * (1.0 + 0.5 * l)
        st.u2 = ke.u2 * (1.0 + 0.5 * l)
        st.tau_air = ke.tau_air * (1.0 + 10.0 * l) + 1.0e-3 * l
        st.tau_gr = ke.tau_gr * (1.0 + 3.0 * l) + 1.0e-4
        st.dhw = ke.dhw + 1.0e-4 * l
        # lakes 1 and 3 are not initialised yet -> two passes
        st.init_keps = bool(l % 2)
        states.append(st)
        state["e1"][l] = st.e1
        state["eps1"][l] = st.eps1
        state["u1"][l] = st.u1
        state["u2"][l] = st.u2
        state["tau_air"][l] = st.tau_air
        state["tau_gr"][l] = st.tau_gr
        state["dhw"][l] = st.dhw
    state["init_keps"] = np.array([float(s.init_keps) for s in states])

    out = k_epsilon(state, dt, dt, 1, KEpsilonParams(M=M))
    for l, st in enumerate(states):
        ref = np_k_epsilon(st, dt, dt_keps=dt, nstep_keps=1,
                           params=KEpsilonParams(M=M))
        for name in ("e1", "eps1", "k2", "kt"):
            np.testing.assert_allclose(
                np.asarray(out[name])[l], getattr(ref, name), rtol=0,
                atol=ATOL, err_msg=f"{name} lake {l}")


def test_jax_kepsilon_jitted_kernel(warm_case):
    """``make_kernel`` (jit, static dt/dt_keps/nstep_keps) matches eager."""
    from lakematrix.jaxbatch.kepsilon import k_epsilon, make_kernel

    case, ke = warm_case
    M, dt = case.M, case.dt
    L = 2
    state = _batched(ke, L)
    state["init_keps"] = np.ones(L)

    eager = k_epsilon(state, dt, dt, 1, KEpsilonParams(M=M))
    jitted = make_kernel(KEpsilonParams(M=M))(state, dt, dt, 1)
    # jit vs eager: XLA is free to reassociate, so allow a few ulps
    for name in ("e1", "eps1", "k2", "kt"):
        np.testing.assert_allclose(np.asarray(jitted[name]),
                                   np.asarray(eager[name]),
                                   rtol=1e-12, atol=1e-9)
    ref = _reference(ke, case)
    for name in ("e1", "eps1", "k2", "kt"):
        np.testing.assert_allclose(np.asarray(jitted[name])[0],
                                   getattr(ref, name), rtol=0, atol=ATOL)


def test_jax_kepsilon_scope_checks(warm_case):
    """Switches outside the case-1 path are rejected, not silently ignored."""
    from lakematrix.jaxbatch.kepsilon import k_epsilon

    case, ke = warm_case
    M, dt = case.M, case.dt
    state = _batched(ke, 1)
    state["init_keps"] = np.ones(1)

    for bad in (dict(kepsbc=2), dict(stabfunc=1), dict(semi_implicit_scheme=False),
                dict(smooth=True), dict(contrgrad=True), dict(keps_coef=2),
                dict(grav_wave_Gill=1), dict(seiches_Goudsmit=1)):
        with pytest.raises(NotImplementedError):
            k_epsilon(state, dt, dt, 1, KEpsilonParams(M=M, **bad))

    floating_ice = dict(state)
    floating_ice["l1"] = np.array([0.5])
    with pytest.raises(NotImplementedError):
        k_epsilon(floating_ice, dt, dt, 1, KEpsilonParams(M=M))

# UNIQUE-MARKER-12345
