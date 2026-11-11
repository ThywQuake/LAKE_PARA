# lakematrix

The LAKE 3.2 thermodynamic core rewritten in Python + JAX, with the **lake as an
array dimension**: all state is `(lake, ...)`, every process advances all lakes
at once, and output is `(lake, time[, depth])` NetCDF.

The specification is `LAKEMATRIX-PLAN.en.md` (authoritative) and
`LAKEMATRIX-PLAN.md` (Chinese). Milestones M0–M6, reference cases and the
acceptance tolerances are defined there. `NOTES.md` records deviations from the
specification and the measured errors.

Status: M3 is at the revised §5 tolerances (one-day 3.7e-4 K, step-3
`Tw1` 9e-9 after the bathymetry fix); the 60-day check is running. M4 has the
case-2 no-snow and snow steps, validated one-step against the Fortran probes,
plus freeze-up/melt transitions; case 3 is not covered by the stable reference
set (see `docs/reference_cases.md`). M5 has a ~5x single-lake NumPy speed-up
(radiation); the vectorised JAX batch is still pending. M6 has
`(lake, time, depth)` NetCDF output, `run`/`batch` CLI commands and a serial or
per-process batch driver. `PROGRESS.md` has the measured errors, data locations
and the resume checklist.
