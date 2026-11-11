"""Batched JAX port of T_SOLVER (surface scan + Chorde iteration)."""

import jax
import jax.numpy as jnp


def _where_state(cond, new, old):
    out = {}
    for k in new:
        a, b = new[k], old[k]
        nd = a.ndim
        c = cond.reshape(cond.shape + (1,) * (nd - 1)) if nd > cond.ndim else cond
        out[k] = jnp.where(c, a, b)
    return out


def t_solver(fn, nstep, surftyp, surftyp_prev, Tsurf_prev, maxiter=20,
             dt_scan=5.0, resid=1.0, ctx=None):
    """Vectorised ``T_SOLVER``; ``fn(Tsurf) -> (HB, state_dict)``.

    ``nstep`` is a scalar model step (all lakes share it); ``surftyp`` and
    ``surftyp_prev`` are ``(L,)`` integer arrays.  Lanes that have converged
    are frozen with ``jnp.where`` so the whole batch follows the Fortran
    per-lake iteration.
    """
    nstep = int(nstep)
    L = Tsurf_prev.shape[0]
    Tsurf2 = jnp.where((nstep > 1) & (surftyp == surftyp_prev),
                       Tsurf_prev - 10.0, -90.0)
    Bal1 = jnp.ones(L)
    Bal2 = jnp.ones(L)
    Tsurf1 = Tsurf2
    state1 = None
    state2 = None
    Tsurf3 = Tsurf2
    state3 = None

    def scan_cond(carry):
        Tsurf1, Tsurf2, Bal1, Bal2, Tsurf3, state1, state2, state3, it = carry
        return (jnp.any(it < 200) & jnp.any(Bal1 * Bal2 > 0.0))

    def scan_body(carry):
        Tsurf1, Tsurf2, Bal1, Bal2, Tsurf3, state1, state2, state3, it = carry
        active = (Bal1 * Bal2 > 0.0)
        # Update the active lanes only; inactive lanes keep their values.
        new_T1 = jnp.where(active, Tsurf2, Tsurf1)
        new_T2 = jnp.where(active, Tsurf2 + dt_scan, Tsurf2)
        hb1, s1 = fn(new_T1, ctx) if ctx is not None else fn(new_T1)
        hb2, s2 = fn(new_T2, ctx) if ctx is not None else fn(new_T2)
        Tsurf1 = jnp.where(active, new_T1, Tsurf1)
        Tsurf2 = jnp.where(active, new_T2, Tsurf2)
        Bal1 = jnp.where(active, hb1, Bal1)
        Bal2 = jnp.where(active, hb2, Bal2)
        state1 = _where_state(active, s1, state1) if state1 is not None else s1
        state2 = _where_state(active, s2, state2) if state2 is not None else s2
        return (Tsurf1, Tsurf2, Bal1, Bal2, Tsurf3, state1, state2, state3,
                it + 1)

    # A dummy state with the right structure is needed before the first call.
    _, _s0 = fn(Tsurf2, ctx) if ctx is not None else fn(Tsurf2)
    state1 = _s0
    state2 = _s0
    state3 = _s0
    Tsurf1, Tsurf2, Bal1, Bal2, Tsurf3, state1, state2, state3, _ = jax.lax.while_loop(
        scan_cond,
        scan_body,
        (Tsurf1, Tsurf2, Bal1, Bal2, Tsurf3, state1, state2, state3,
         jnp.array(0)))

    Bal3 = jnp.full(L, resid + 1.0)
    it = jnp.array(0)
    Tsurf3 = Tsurf2

    def chord_cond(carry):
        Tsurf1, Tsurf2, Tsurf3, Bal1, Bal2, Bal3, state1, state2, state3, it = carry
        active = (jnp.abs(Bal3) > resid) & (Bal1 != 0.0) & (Bal2 != 0.0) & (Bal1 * Bal2 <= 0.0)
        return jnp.any(active) & (it <= maxiter)

    def chord_body(carry):
        Tsurf1, Tsurf2, Tsurf3, Bal1, Bal2, Bal3, state1, state2, state3, it = carry
        active = (jnp.abs(Bal3) > resid) & (Bal1 != 0.0) & (Bal2 != 0.0) & (Bal1 * Bal2 <= 0.0)
        new_T3 = (Tsurf1 * Bal2 - Tsurf2 * Bal1) / (Bal2 - Bal1)
        hb3, s3 = fn(new_T3, ctx) if ctx is not None else fn(new_T3)
        Tsurf3 = jnp.where(active, new_T3, Tsurf3)
        Bal3 = jnp.where(active, hb3, Bal3)
        state3 = _where_state(active, s3, state3)
        move2 = active & (Bal1 * hb3 < 0.0)
        move1 = active & (Bal2 * hb3 < 0.0) & ~move2
        Tsurf2 = jnp.where(move2, new_T3, Tsurf2)
        Bal2 = jnp.where(move2, hb3, Bal2)
        state2 = _where_state(move2, s3, state2)
        Tsurf1 = jnp.where(move1, new_T3, Tsurf1)
        Bal1 = jnp.where(move1, hb3, Bal1)
        state1 = _where_state(move1, s3, state1)
        return (Tsurf1, Tsurf2, Tsurf3, Bal1, Bal2, Bal3, state1, state2,
                state3, it + 1)

    Tsurf1, Tsurf2, Tsurf3, Bal1, Bal2, Bal3, state1, state2, state3, it = jax.lax.while_loop(
        chord_cond, chord_body,
        (Tsurf1, Tsurf2, Tsurf3, Bal1, Bal2, Bal3, state1, state2, state3,
         it))
    return dict(Tsurf3=Tsurf3, state=state3, iter=it, Bal3=Bal3)


def make_solver(nstep, maxiter=20, dt_scan=5.0, resid=1.0):
    """Return a jitted solver bound to the chain of ``T_SOLVER`` switches."""
    from .surface import heatbalance_water

    @jax.jit
    def solve(ctx, Tsurf_prev, surftyp, surftyp_prev):
        return t_solver(heatbalance_water, nstep, surftyp, surftyp_prev,
                        Tsurf_prev, maxiter=maxiter, dt_scan=dt_scan,
                        resid=resid, ctx=ctx)
    return solve
