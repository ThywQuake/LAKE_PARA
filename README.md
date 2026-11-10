# lakematrix

The LAKE 3.2 thermodynamic core rewritten in Python + JAX, with the **lake as an
array dimension**: all state is `(lake, ...)`, every process advances all lakes
at once, and output is `(lake, time[, depth])` NetCDF.

The specification is `../LAKEMATRIX-PLAN.en.md` (authoritative) and
`../LAKEMATRIX-PLAN.md` (Chinese). Milestones M0–M6, reference cases and the
acceptance tolerances are defined there. `NOTES.md` records deviations from the
specification and the measured errors.

Status: M0 (preparation) in progress.
