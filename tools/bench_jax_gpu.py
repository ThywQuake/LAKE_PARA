"""Benchmark the batched JAX case-1 step on the active backend (CPU/GPU).

A ``lax.scan`` chunk is jitted as one computation, so the per-step kernel
launch overhead is amortised; the first call includes XLA compilation.
"""
import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.forcing import read_ascii  # noqa: E402
from lakematrix.grid import create_grid  # noqa: E402
from lakematrix.jaxbatch.step import (  # noqa: E402
    _compiled, _split, from_numpy_state, make_params)
from lakematrix.setup_io import read_case  # noqa: E402
from lakematrix.step import build_step_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lakes", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--case", default="ref/v1/summer1d")
    a = ap.parse_args()
    case = read_case(ROOT / a.case)
    grid = create_grid(case.M, case.Mice, case.ns, case.d_surf, case.d_bot,
                       case.soil_depth, 0.02)
    rows = read_ascii(ROOT / a.case / "meteo" / "test.dat")
    params = make_params(case, grid, a.lakes)
    base = from_numpy_state(build_step_state(case, grid))
    state = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.asarray(x), a.lakes, axis=0)
        if np.ndim(x) else jnp.asarray(x), base)

    fn, stat = _compiled({**params, "rows": rows})
    _, dyn = _split({**params, "rows": rows})
    dyn = {**dyn, "first": jnp.asarray(False)}
    n = int(a.steps)

    def body(carry, nstep):
        return fn(carry, dyn, nstep)

    chunk = jax.jit(lambda s: jax.lax.scan(
        body, s, jnp.arange(2, n + 1, dtype=jnp.float64))[0])
    for run in range(2):
        t0 = time.perf_counter()
        out = chunk(state)
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0
        print(f"{jax.default_backend()} run={run} lakes={a.lakes} steps={n} "
              f"time={dt:.3f}s lake-steps/s={a.lakes * n / dt:.1f}",
              flush=True)


if __name__ == "__main__":
    main()
