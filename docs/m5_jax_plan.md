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
