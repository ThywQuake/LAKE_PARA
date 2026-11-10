# lakematrix

The LAKE 3.2 thermodynamic core rewritten in Python + JAX, with the **lake as an
array dimension**: all state is `(lake, ...)`, every process advances all lakes
at once, and output is `(lake, time[, depth])` NetCDF.

The specification is `../LAKEMATRIX-PLAN.en.md` (authoritative) and
`../LAKEMATRIX-PLAN.md` (Chinese). Milestones M0–M6, reference cases and the
acceptance tolerances are defined there. `NOTES.md` records deviations from the
specification and the measured errors.

Status: M3 in progress: M2 done; `DIFF_COEF`/`T_DIFF` case 1 matches the Fortran probe exactly (49 calls). T_SOLVER next. Case 3 is not covered by the natural reference set; see `docs/reference_cases.md`.
