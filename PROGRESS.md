# lakematrix — progress snapshot (2026-11-11)

This file is the hand-off note for moving development to another machine.
Authoritative plan: `LAKEMATRIX-PLAN.en.md` (English) and
`LAKEMATRIX-PLAN.md` (Chinese).

## Current commit

`main` at the commit that adds this file; see `git log -1`.

## Milestones

| Milestone | State |
|---|---|
| M0: lakebatch phase-1 closure | done (documented in `NOTES.md`) |
| M1: case-1 state, forcing, grid | done |
| M2: bathymetry, radiation, surface fluxes, density, tridiagonal | done |
| M3: case-1 step (`SOILFORLAKE`, `LATERHEAT`, `T_DIFF`, `T_SOLVER`, `K_EPSILON`, momentum, layers) | **done at revised §5**: 1 day 3.7e-4 K; step-3 `Tw1` 8.97e-9, `Tsoil1` 2.0e-6 after the per-step bathymetry fix; 60-day rerun in progress |
| M4: ice/snow (cases 2/3) | **cases 1/2 done**: case-2 no-snow and snow steps (one-step probes: `Tw1` 5.5e-7 / 1.5e-6, `Tsoil1` 5.5e-7 / 7.2e-7, `Ti1` 2.3e-5 / 8.7e-6), freeze-up transition validated (`Tw1` 2.2e-8); 425-day runs: one-step probes pass, but the first full runs become NaN in the snow branch (day 98-157); zero-thickness guards added and `full425d_v2` rerunning. Case 3 is not covered by the stable reference set |
| M5: CPU/JAX batch performance | **partial**: NumPy radiation `cumprod` gives ~5x single-lake step speed-up (89 lake-steps/s vs Fortran ~6.6k); vectorised `lax.scan` batch still pending |
| M6: NetCDF output + CLI | **partial**: `output.py`, `cli.py` (`run`/`batch`), `tools/compare.py`, `run.py` batch loop with `(lake,time,...)` NetCDF; true vectorised lake dimension pending |

Tests: 27 passed, 17 skipped (probe-dependent; probe data is under `scratch/`, not in git).

## Measured accuracy (latest)

| Level | Current | Plan §5 target |
|---|---|---|
| routine / probe | 1e-10 or better for standalone modules | 1e-10 |
| step 1 | `Tw1` 3.5e-13, `Tsoil1` 1.8e-15 | 1e-10 |
| step 2 | `Tw1` 7.1e-10, `Tsoil1` 2.1e-14 | 1e-10 |
| step 3 | `Tw1` 8.97e-9, `Tsoil1` 2.04e-6 (after per-step BATHYMETRY) | 1e-10 / solver floor |
| 1 day | 3.7e-4 K | 1e-3 (revised) |
| 60 days | 0.349 K pointwise, 0.054 K time-mean, 0.071 K RMSE — **PASS** | 0.5 / 0.1 / 0.15 (revised, chaos-aware) |

The older step-3 accumulation was traced to stale bathymetry and fixed (per-step `BATHYMETRY`); see `NOTES.md`.  Last finding: at step 1
`K_EPSILON` was being fed the **post-momentum** velocity; it must use the old
`u1/v1`.  After fixing this and setting `K_EPSILON.Buoyancy0`,
`kt` errors are 1.08e-8 (step 1) and 1.92e-8 (step 2) instead of 4e-4/3e-4.
Re-run the 1-day/3.3-day/20-day comparisons next.

## Data that is *not* in git

The code repo is small (~3 MB).  These are regenerable and were deliberately
gitignored:

| Path | Size | Content |
|---|---|---|
| `ref/` | 3.8 GB | Fortran reference cases (setup, meteo, results) |
| `scratch/` | 2.2 GB | probe builds, probe runs, 20-day runs |
| `../lakebatch/` build trees | 65 MB | Fortran builds + probe builds |

The Fortran **source** and the testlake forcing are vendored in the repo:
`vendor/lakebatch/core/source`, `vendor/LAKE3.2/meteo` (3.4 MB), so a clean
clone can rebuild the check binary and rerun the reference cases.

For the new machine, either copy these directories over the network/USB, or
regenerate:

```bash
# 1. Fortran check build (needs CMake/gfortran)
cd vendor/lakebatch/core
CHECK=1 ./build.sh
# 2. reference cases
cd ../../lakematrix
.venv/bin/python tools/ref/make_ref_cases.py   # see its --help
```

## Resume checklist on the new machine

```bash
git clone https://github.com/ThywQuake/LAKE_PARA.git
cd LAKE_PARA
UV_CACHE_DIR=$PWD/.uvcache uv venv .venv
UV_CACHE_DIR=$PWD/.uvcache uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest -q
```

Then:

1. Copy/regenerate `ref/` and `scratch/` (see above).
2. Continue from `NOTES.md`, section "M3: K_EPSILON uses pre-momentum
   velocity": rerun 1-day, 3.3-day and 20-day comparisons:
   ```bash
   .venv/bin/python tools/ref/run_nd.py 20
   ```
3. If step-3 is still off, build the v43 probe (`tools/ref/make_probe_build.py`)
   and compare `TD2` coefficients / `K_EPSILON` inputs.
4. Then finish M4 (`step_case2`), M5 (batch), M6 (4-lake 425-day).

## Known deviations / plan review

- `snowsoil_mod.f90` does not exist in this checkout; the port uses
  `snowcalc.f90` + `snowtemp.f90` (documented in `physics/ice.py`).
- `addPrecipToSnow` is not ported (needed if snowfall is modelled).
- v1 excludes biogeochemistry, BUBBLE, tributaries, FLake, restart.
- The plan §5 1e-10 step tolerance is below the `T_SOLVER` convergence noise
  (`resid=1 W/m2`); this needs a documented review if it blocks acceptance.
