# lakematrix

The LAKE 3.2 thermodynamic core rewritten in Python + JAX, with the **lake as an
array dimension**: all state is `(lake, ...)`, every process advances all lakes
at once, and output is `(lake, time[, depth])` NetCDF.

The specification is `../LAKEMATRIX-PLAN.en.md` (authoritative) and
`../LAKEMATRIX-PLAN.md` (Chinese). Milestones M0–M6, reference cases and the
acceptance tolerances are defined there. `NOTES.md` records deviations from the
specification and the measured errors.

Status: M2 complete (solvers, density, radiation, bathymetry, SfcFlx, RADBALANCE, HEATFLUXSURF, SENSLATMOM_FLUXES, HEATBALANCE and SURF_CHAR, 18 tests). M3 next. Case 3 is not covered by the natural reference set; see `docs/reference_cases.md`.
