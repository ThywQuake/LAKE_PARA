# lakematrix — progress snapshot (2026-11-11)

This file is the hand-off note for moving development to another machine.
Authoritative plan: `../LAKEMATRIX-PLAN.en.md` (English) and
`../LAKEMATRIX-PLAN.md` (Chinese).

## Current commit

`main` at the commit that adds this file; see `git log -1`.

## Milestones

| Milestone | State |
|---|---|
| M0: lakebatch phase-1 closure | done (documented in `NOTES.md`) |
| M1: case-1 state, forcing, grid | done |
| M2: bathymetry, radiation, surface fluxes, density, tridiagonal | done |
| M3: case-1 step (`SOILFORLAKE`, `LATERHEAT`, `T_DIFF`, `T_SOLVER`, `K_EPSILON`, momentum, layers) | **in progress** |
| M4: ice/snow (cases 2/3) | partial: `physics/ice.py` (`t_diff_ice`, snow_calc, increments), `IceSurfaceContext`/`heatbalance_ice`; no `step_case2` yet |
| M5: CPU/JAX batch performance | not started |
| M6: NetCDF output + CLI | partial: `output.py`, `cli.py`, `tools/compare.py`; no batch lake dimension |

Tests: 37 passed.

## Measured accuracy (latest)

| Level | Current | Plan §5 target |
|---|---|---|
| routine / probe | 1e-10 or better for standalone modules | 1e-10 |
| step 1 | `Tw1` 3.5e-13, `Tsoil1` 1.8e-15 | 1e-10 |
| step 2 | `Tw1` 7.1e-10, `Tsoil1` 2.1e-14 | 1e-10 |
| step 3 | `Tw1` 5.2e-7, `Tsoil1` 2.8e-4 | 1e-10 |
| 1 day | 0.00049 K (with dhwtrib) / 0.0052 K (without) | 1e-6 |
| 60 days | not yet re-run after the latest fixes | 0.01 K |

The dominant remaining issue is step-3 accumulation.  Last finding: at step 1
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
| `../lakebatch/` | 2.6 MB source | Fortran model + probe build |
| `../LAKE3.2/meteo/` | 0.8 MB | testlake forcing |

For the new machine, either copy these directories over the network/USB, or
regenerate:

```bash
# 1. Fortran check build (needs CMake/gfortran)
cd lakebatch/core
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
