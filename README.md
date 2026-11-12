# lakematrix

The LAKE 3.2 thermodynamic core rewritten in Python + JAX, with the **lake as an
array dimension**: all state is `(lake, ...)`, every process advances all lakes
at once, and output is `(lake, time[, depth])` NetCDF.

The specification is `LAKEMATRIX-PLAN.en.md` (authoritative) and
`LAKEMATRIX-PLAN.md` (Chinese). Milestones M0–M6, reference cases and the
acceptance tolerances are defined there. `NOTES.md` records deviations from the
specification and the measured errors.

Status: M3 is at the revised §5 tolerances (one-day 3.7e-4 K, step-3
`Tw1` 9e-9 after the bathymetry fix); the 60-day check passes the chaos-aware §5 criteria (0.263 K pointwise, 0.049 K mean bias, 0.058 K RMSE). M4 has the
case-2 no-snow and snow steps, validated one-step against the Fortran probes,
plus freeze-up/melt transitions; case 3 is not covered by the stable reference
set (see `docs/reference_cases.md`). M5 has a ~5x single-lake NumPy speed-up
(radiation) and a vectorised JAX step for case 1 (`jaxbatch/step.py`, 4.9x/14x/51x at
1/8/64 lakes); JAX cases 2/3 exist as single steps, checked only against the
NumPy port. M6 has
`(lake, time, depth)` NetCDF output, `run`/`batch` CLI commands and a serial or
per-process batch driver. `PROGRESS.md` has the measured errors, data locations
and the resume checklist.

**JAX runs are open-water only.** The JAX runner (`tools/ref/run_jax_case.py`,
`run_case1_jax*`) has no case selection: it never switches from case 1 to the
ice cases (freeze-up) or back (melt), and it does not stop when the water
drops below freezing. Use it only for periods that stay ice-free. Runs that
freeze, including the 425-day acceptance cases, must use the NumPy driver
(`lakematrix run`/`batch`, `tools/ref/run_case.py`), which has the 1->2 and
2->1 transitions (case 3 is still not ported there either).


## Performance (M5, partial)

`tools/bench.py` measures the current NumPy/JAX-kernel single-lake driver.  On this VM:

| Driver | lake-steps/s |
|---|---|
| `tools/bench.py --lakes 1 --steps 100` | 122-137 |
| Fortran check build (full425d, 1 lake) | ~6,600 |

The radiation rewrite (NumPy `cumprod`) and JIT-compiled tridiagonal/block
solvers removed most of the solver overhead; the remaining cost is the
Python k-epsilon/momentum/soil loops.  The vectorised JAX `lax.scan` batch
required by plan M5 is **not yet implemented**; `run.py` currently advances
one lake per process (`cli batch --jobs N`).  Long runs are checkpointed daily
(`run_case.py --resume`).
