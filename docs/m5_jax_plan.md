# M5 (vectorised JAX batch): remaining work

Current state (2026-11-11): the physics is a correct NumPy single-lake port.
`run.py` advances one `StepState` per lake; `cli batch --jobs N` runs lakes in
separate processes.  `tools/bench.py` measures ~89 lake-steps/s per process on
this VM (Fortran check build: ~6600 lake-steps/s).

The plan's vectorised JAX runner is **not implemented**.  It requires:

1. **State as a pytree.**  Register `State`, `SoilState`, `IceState`,
   `MomentumState`, `KEpsilonState` as JAX pytrees.  Every leaf gets a leading
   `(L, ...)` axis; no Python scalars or per-lake objects remain.
2. **Step functions in JAX.**  Replace `for` loops over the vertical with
   `lax.scan`; replace `if` on lake-dependent scalars with `jnp.where`;
   replace `T_SOLVER`/`K_EPSILON`/momentum iterations with
   `lax.while_loop` that masks converged lakes (plan §6.2).
3. **Case selection per lake.**  Evaluate the case-1/2/3 updates for the whole
   batch and select with `tree_map(jnp.where(layer_case == k, ...))`; the
   transitions become array updates.
4. **Chunked scan.**  `run.py` jits a chunk (e.g. one day) and `lax.scan`s
   over it; output between chunks.
5. **Batch consistency.**  Compare L lakes in one call with L calls of L=1;
   on CPU the result should be bit-identical (currently it is, because the
   process driver reuses the same code for each lake).

The 60-day and 425-day pointwise comparisons are chaos-limited (see
`NOTES.md`); the vectorised runner must be validated first with the step-level
probes and the revised chaos-aware criteria in plan §5.

## B1 delivered (2026-11-11)

`src/lakematrix/jaxbatch/diffusion.py` implements the case-1
`T_DIFF` water+soil kernel with a leading lake axis and `jax.jit`.  It matches
the NumPy reference to 3.6e-15 and is bit-identical across batch members.
Benchmark (one T_DIFF call, CPU):

| Lakes | JAX | NumPy (per-lake loop) | Speedup |
|---|---|---|---|
| 1 | 0.20 ms | 3.22 ms | 16x |
| 8 | 0.18 ms | 0.67 ms | 3.7x |
| 64 | 0.30 ms | 5.51 ms | 18x |

The next B2 step is the surface heat balance (`SURF_CHAR`/`SENSLATMOM_FLUXES`
/`RADBALANCE`/`HEATBALANCE`) and the batched `T_SOLVER`; then soil columns,
momentum and K_EPSILON.

## B2 delivered (2026-11-11)

`src/lakematrix/jaxbatch/sfcflx.py` (open-water SfcFlx) and
`src/lakematrix/jaxbatch/surface.py` (SURF_CHAR/HEATFLUXSURF/RADBALANCE/
HEATBALANCE) reproduce the NumPy values to ~1e-8 or better.
`src/lakematrix/jaxbatch/tsolver.py` implements the batched scan + Chorde
iteration.  One `T_SOLVER` call on the step-1 state:

| Lakes | JAX | NumPy per-lake loop | Speedup |
|---|---|---|---|
| 1 | 0.69 ms | 6.43 ms | 9.3x |
| 8 | 1.47 ms | 51.4 ms | 34.9x |
| 64 | 7.22 ms | 415 ms | 57.5x |

`Tsurf3`, `Tw2` and `Tsoil2` match the NumPy solver to 1e-10.
The next B3 is the soil-column preprocessing (`SOIL_COND_HEAT_COEF`,
`SOILCOLSTEMP`, `LATERHEAT`) and postprocessing (`SOILFORLAKE`).

## B5 delivered (2026-11-11)

`src/lakematrix/jaxbatch/step.py` assembles the full case-1 step from the
B1-B4 kernels and provides `run_case1_jax(params, rows, nsteps)` (`lax.scan`).
Parity vs NumPy `step_case1`/`run_case1` on `summer1d`:

- one step and ten steps, L=1 and L=4: `Tw1/Tsoil1` <=1e-13 K, `h1` 0 m,
  `E1/eps1/lamw` <=1e-9 relative.
- `lax.scan` vs a Python JAX loop is bit-identical.
- Throughput (case-1 step): **4.9x** at L=1 (196 vs 39.8 lake-steps/s),
  **14x** at L=8, **51x** at L=64 (3846 vs 75 lake-steps/s).

Heavy compile tests are marked `slow` (the coupled step compiles to several GB
per signature); run them individually, e.g.
`pytest -q tests/test_jaxbatch_step.py::test_one_step_matches_numpy_L1`.
