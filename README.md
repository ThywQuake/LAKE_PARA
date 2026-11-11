# lakematrix

The LAKE 3.2 thermodynamic core rewritten in Python + JAX, with the **lake as an
array dimension**: all state is `(lake, ...)`, every process advances all lakes
at once, and output is `(lake, time[, depth])` NetCDF.

The specification is `../LAKEMATRIX-PLAN.en.md` (authoritative) and
`../LAKEMATRIX-PLAN.md` (Chinese). Milestones M0–M6, reference cases and the
acceptance tolerances are defined there. `NOTES.md` records deviations from the
specification and the measured errors.

Status: M3 in progress (step 1/2 near bit-exact; step 3 under debug). See
`PROGRESS.md` for the current measured errors, data locations and the resume
checklist. M4 has the ice/snow module and surface solver but no case-2 step yet;
M6 has NetCDF/CLI. Case 3 is not covered by the natural reference set; see
`docs/reference_cases.md`.
