# lakematrix: rewrite the LAKE thermodynamic core in Python + JAX (lake as an array dimension, CPU first)

This is an engineering specification for an implementer who has **no access to the earlier
discussion**. Every fact needed is written here. If the code disagrees with this document, the
code wins: record the difference in `lakematrix/NOTES.md`.

The Chinese version (`LAKEMATRIX-PLAN.md`) has the same content; where the two differ, this
English version is authoritative.

---

## 0. Background and goal

- LAKE 3.2 is a one-dimensional (vertical column) lake model written in Fortran, about 23,000
  lines of code excluding comments and blank lines. Each time step calls
  `SUBROUTINE LAKE` (`lakebatch/core/source/model/lake.f90:169-3212`) **once per lake**. With
  several lakes, the driver loops over them and calls `LAKE` for one lake at a time
  (`lakebatch/core/source/driver/driver.f90`, around line 765).
- Goal: a new model in which **the lake is an array dimension**. Input is (lake, time), every
  state variable is a (lake, ...) array, every physical process updates all lakes in one
  call, and output is written directly as (lake, time[, depth]) NetCDF. There is no loop over
  lakes anywhere.
- Decisions already made (do not revisit):
  1. Python + JAX in **float64** (`jax.config.update("jax_enable_x64", True)`). CPU only for
     now; a GPU version comes later.
  2. Correctness is judged **within tolerances** (section 5) against the existing Fortran model.
  3. Version 1 covers the **thermodynamic core** only: water temperature, salinity diffusion
     (fresh water), k-epsilon turbulence, momentum, ice, snow, soil (including several soil
     columns and lateral heat exchange), surface heat and momentum fluxes, shortwave radiation.
     **Out of scope**: methane, oxygen, CO2, phosphorus, chlorophyll, bubbles (`BUBBLE`),
     tributary inflow and outflow, FLake, data assimilation, restart files.
  4. A **separate new package**, `lakematrix`. It must not import lakebatch's Python code.
- Global variables: every Fortran module variable and every `SAVE` local becomes an explicit
  field of the State, with a leading lake dimension, updated by broadcasting array operations.
  No per-lake swapping mechanism is needed.

## 1. Environment and conventions (mandatory)

- Root directory: `/Users/mac/文档/研究生/LAKE/` (shared between the user's Mac and a Linux VM).
  - `lakebatch/`: existing project (Fortran core in `core/`, Python package in `src/lakebatch/`,
    tools). This plan treats it as **read-only**, except for the wrap-up in section 2 and the
    reference builds in section 4, which live under `lakebatch/scratch/`.
  - `LAKE3.2/`: the original model directory. Its testlake example is the source of the setup
    files used by `make_cases.py` (section 4.1).
  - The new package goes in `/Users/mac/文档/研究生/LAKE/lakematrix/` (create it; it is its own
    git repository).
- Machines: the Linux VM is arm64, 5 cores, no GPU, with network access (`pip install jax`
  works). The user also runs things on the Mac.
  - The VM's Fortran build outputs are `lakebatch/core/build-linux/` (production) and
    `lakebatch/core/build-check-linux/` (check). **Never touch** the Mac's build outputs in
    `lakebatch/core/build/` or `lakebatch/core/lake.out`.
  - lakebatch's VM virtual environment is `lakebatch/.venv-linux/`. lakematrix gets its own,
    `lakematrix/.venv-linux/`.
- Fortran builds:
  - `lakebatch/core/build.sh`: production build (`-O3`, with fused multiply-add contraction).
  - `CHECK=1 lakebatch/core/build.sh`: check build (`-O3 -ffp-contract=off`). **All reference
    data is produced with the check build**, because FMA contraction makes the last bits depend
    on code layout.
- Files: **do not use `rm -rf`**. Write every experiment into a new directory (e.g.
  `.../ref/v2/`) instead of clearing an old one. If something really must be deleted, state
  first what will be deleted.
- git: commit to the lakematrix repository once per milestone, after its checks pass. Set the
  author with the environment variables
  `GIT_AUTHOR_NAME="陈少春" GIT_AUTHOR_EMAIL="116076400+ThywQuake@users.noreply.github.com"`
  (and `GIT_COMMITTER_*` with the same values). Do not change git config.
- Tests: `pytest`. Long tests (60 and 425 days) are marked `@pytest.mark.slow`, skipped by
  default, and run with `pytest -m slow`.

## 2. Step 0: wrap up the lakebatch branch (small)

lakebatch is on branch `rewrite/phased-step`; the latest commit is `11790ea` (stage 1b). The
working tree has one uncommitted change: the section "BUBBLE's share of run time" appended to
`vector/rewrite/README.md`.

1. Commit that change. The message should say that BUBBLE takes 26–29% of run time and that
   stages 2–3 will not be done.
2. Add a paragraph at the top of `lakebatch/HANDOVER.md` and of `lakebatch/vector/STATUS.md`:
   stage 1 (several lakes in one process, each bit-identical to its single run) is complete;
   stages 2–3 are cancelled; work continues in `../lakematrix` (JAX rewrite with the lake as
   an array dimension). The stage 1 multi-lake Fortran stays available as a reference tool.
3. Commit.

## 3. Fortran code to translate

### 3.1 Structure of one step (`SUBROUTINE LAKE` in `lake.f90`)

| Part | Lines (approx.) | Content |
|---|---|---|
| First-call setup | 530–810 | Forcing assignment, `INIT_VAR` (`init_var_mod.f90:7-500`, first call only), `kor`, radiation extinction coefficients (653–659) |
| Load current state | 811 | `UPDATE_CURRENT_TIMESTEP` (`lake_modules.f90:1138-1335`): copies this lake's state out of the `*_2d` arrays |
| Grid and case selection | 855–912 | `BATHYMETRY` (`bathym_mod.f90:7-382`), advance `time`/`nstep`, **`layer_case` selection (881–912, see 3.2)** |
| Case 1: water, no ice | 916–1238 | `BATHYMETRY`, `DENSITY_W`, `MIXED_LAYER_CALC`, `RadWater%RAD_UPDATE` (1000), `MOMENTUM_SOLVER` (1049), `T_SOLVER` (1070) + `SOILFORLAKE` (1073), `S_DIFF` (1091), biogeochemistry (1095–1153, **skip**), `DENSITY_W`, `K_EPSILON` (1162), `TURB_SCALES` (1216) |
| Case 2: ice over water | 1239–1897 | Ice/snow radiation, `T_SOLVER` (1463), `SNOWTEMP` (1491), `MOMENTUM_SOLVER` (1549), `K_EPSILON` (1559), etc.; ice growth and melt, snow |
| Case 3: ice only (frozen to the bottom) | 1898–2086 | `T_SOLVER` (1960), `SNOWTEMP` (1969), etc. |
| Case 4 | 2087– | **Not operational** (the original code stops with an error). Do not translate. |
| Layer thickness updates | about 2153–2270 | Changes of water/ice/snow layer thickness, minimum thickness handling |
| Store state | 2575 | `UPDATE_NEXT_TIMESTEP` (`lake_modules.f90:1340-1534`) |
| Output | 2640–3150 | Do not translate (the new package has its own output, see M6) |

The implementer's first task (M0.3) is to expand this table into
`lakematrix/docs/step_map.md`: for cases 1–3, every call in order, the condition (`if`) it is
under, and the state variables it reads and writes. This document is the roadmap for all
later translation.

### 3.2 `layer_case` selection (from `lake.f90:881-912`)

```
layer_case = 1
if (h1 >  0 .and. l1 >  0) layer_case = 2
if (h1 == 0 .and. l1 >  0) layer_case = 3
if (h1 == 0 .and. l1 == 0) layer_case = 4
case 1, surface water at freezing point (WATER_FREEZE_MELT(Tw1(1), ..., +1)) and water thick enough -> case 2
case 3, ice bottom can melt (WATER_FREEZE_MELT(Ti1(Mice+1), ..., -1))                              -> case 2
case 3, ice top can melt and there is no snow -> create a thin water layer
        (h1 = min_water_thick, Tw1 = melting point + T_phase_threshold, ...)                     -> case 1
```

(`h1` water depth, `l1` ice thickness, `Tw1` water temperature profile, `Ti1` ice temperature
profile; read the original lines for the exact thresholds.) In JAX, first compute `layer_case`
per lake as an integer array of shape (L,). The third rule rewrites state; apply it with
`jnp.where` under the per-lake mask.

### 3.3 Thermodynamic-core routines (file:lines)

Numerics and utilities: `PROGONKA` numerics_mod.f90:243-273, `MATRIXPROGONKA`
numerics_mod.f90:108-172, `DZETA`/`DZETAI`/`VARMEAN` (dzeta_mod.f90), `DIFFUSION`
(diffusion_lake_mod), `VERTADV_UPDATE`/`MUSCL_FLUXES`/`FLUX_LIMITER` (vertadv),
`LININTERPOL` (grid_operations_lake), `IND_STAB_FACT_DB`, `STEP`, `KRON`, `INVERSMATRIX`,
`MATRIXMULT*`, `MATRIXSUM`, `VECTORSUM` (numerics).

Physics:
- Temperature: `T_SOLVER` t_solver_mod.f90:15-216, `T_DIFF` 219-656, `DIFF_COEF` 662-937,
  `SURF_CHAR` 940-1034.
- Surface: `HEATBALANCE` heatbalsurf_mod.f90:8-72, `SENSLATMOM_FLUXES` 167-384, `RADBALANCE`
  388-508, `HEATFLUXSURF`; `TSKIN_SOLVER` 644-748 is not used (testlake has `skin 0`);
  `SfcFlx_*` in shared/SfcFlx_mod.F90 (`momsenlat`, `roughness`, `rhoair` 623-687,
  `satwvpres` 855-936, `spechum` 949-1012, `lwradwsfc` 1019-1081).
- Turbulence: `K_EPSILON` turb_mod.f90:10-1158, `CE_CANUTO`/`CET_CANUTO`/
  `INTERPOLATE_TO_INTEGER_POINT` (turb_lake_mod), `TURB_SCALES` phys_func.f90:914-1069,
  `MIXED_LAYER_CALC` 1072-1154, `RICHARDSON_NUMBER`, `TURB_DENS_FLUX`, `DIFFMIN_HS`,
  `DOMWAVESPEED`.
- Momentum: `MOMENTUM_SOLVER` momentum_mod.f90:15-1611.
- Density: `DENSITY_W` phys_func.f90:160-201; module water_density (`WATER_DENS_TS`,
  `DDENS_DTEMP`, `DDENS_DSAL`, `SET_DDENS_*`); `SET_DENS_CONSTS` phys_func.f90:60-86.
  testlake uses `eos 5`.
- Radiation: `RAD_UPDATE` lake_modules.f90:386-403, `RAD_INIT` 406-418, `WATER_ALBEDO`.
- Ice and snow: `SNOWTEMP` snowtemp.f90:6-97, `SNOW_CALC` snowcalc.f90:5-497, and from
  phys_func `MELTPNT`/`MELTINGPOINT`/`WATER_FREEZE_MELT`/`WL_MAX`/`SOIL_COND_JOHANSEN`.
- Soil: `SOILFORLAKE` soil_mod.f90:215-709, `COMSOILFORLAKE` 8-212, `LATERHEAT` 1385-1628,
  `SOILCOLSTEMP`, `TSURFSOILCOL`, `AVERSOILCOL`, `SOIL_COND_HEAT_COEF`.
- Other: `S_DIFF` salinity_mod.f90:14-228; `ZERODIM_MODEL` t_0dim.f90:4-113 (about 5 calls
  per step; find out what it does before deciding whether it is needed); `BATHYMETRY`
  bathym_mod.f90:7-382; `JULIAN_DAY` (time_lake_mod).
- Initialization and constants: `INIT_LAKE` lake.f90:30-159, `INIT_VAR`
  init_var_mod.f90:7-500, `DEFINE_PHYS_CONSTANTS` lake_modules.f90:195-309,
  `DEFINE_PHYS_PARAMETERS` 119-128, `NUMERIC_PARAMS_SET` 99-104, `TURB_CONST_DERIVED_SET`
  903-937, `PBLDAT_LAKE` surf_scheme1_mod.f90:6-100, `PBLDAT_INM`
  surf_scheme_inm_mod.f90:4-185, `ALLOCARRAYS` init.f90:177-483, `DEFINE_DRIVING_PARAMS`
  driving_params_mod.f90:211-476 (reads the setup keywords).

This list comes from the gprof call record of one testlake run. Calls per time step, for
orientation: `HEATBALANCE` 8, `DIFF_COEF` 27, `T_DIFF` 13, `PROGONKA` 32, `SOILFORLAKE` 5,
`LATERHEAT` 4, `MOMENTUM_SOLVER`/`K_EPSILON`/`T_SOLVER` 1 each. More than one call per step
means an iteration or a loop over soil columns or media inside; the translation must
reproduce that loop structure exactly.

### 3.4 Complete list of state carried between steps (nothing may be missed)

The state carried from one step to the next is the union of the three groups below. The State
design (M1) must cover all of it, or `NOTES.md` must explain why a variable is not needed.

1. The `*_2d`/`*_3d` arrays read by `UPDATE_CURRENT_TIMESTEP` and written by
   `UPDATE_NEXT_TIMESTEP` (thermodynamic ones):
   `Tw1 Ti1 Tis1 T Tsoil1 wi1 wl1 Sal1 Sals1 salice porice dens lamw u v E eps h1 l1 hs1 ls1
   hx1 hx2 hy1 hy2 hx1t hx2t hy1t hy2t hx1ml hx2ml hy1ml hy2ml gradpx_tend gradpy_tend dhw dhw0
   dhi dhi0 dls0 dhwfsoil snmelt snowmass fl_sn fl_sn_init Tskin roughness cdm velfrict
   eflux0_kinem Elatent k_turb_T_flux i_maxN itherm itop dz wl time nstep Eseiches`.
   Leave out the biogeochemical ones: `qwater oxyg DIC DOC POCL POCD phosph carbdi qsoil
   oxygsoil febul0 tot_ice_meth_bubbles`.
2. Module variables and `SAVE` locals of routines. lakebatch stage 1 already generated the
   complete list:
   - `lakebatch/core/source/model/lb_point_ctx.f90`: each subroutine `LBC_MOD_<module>` lists
     the variables of that module that persist between calls;
   - `lakebatch/core/source/model/lb_ctx_<routine>.inc`: the `SAVE` locals of `LAKE`,
     `K_EPSILON`, `MOMENTUM_SOLVER`, `T_SOLVER`, `T_DIFF`, `DIFF_COEF`, `SURF_CHAR`,
     `SNOWTEMP`, `SNOW_CALC`, `SOILFORLAKE`, `COMSOILFORLAKE`, `SENSLATMOM_FLUXES`.

   Most of these are work arrays recomputed every step. Only variables that are **read before
   they are written** within a step are real carried state. Classify each one and record the
   result in `docs/state.md`.
3. Constants computed in first-call blocks (`firstcall`): these go into Params or constants,
   not into State.

## 4. Reference data (Fortran)

### 4.1 Reference configuration

- Base: `make_case(dst, forcings, tinteg)` in `lakebatch/vector/rewrite/make_cases.py` builds
  a case from the `LAKE3.2` testlake (with `hourly 0`, `npoints`, `dataname 'test'`,
  `spinup_times 0`). `_rows()` reads the testlake forcing; `_cold(rows)` lowers air
  temperature by 2 K.
- testlake facts: starts 2016-08-01 03:00; `dt 20` seconds; hourly forcing; `M 40`,
  `Mice 10`, `ns 10`, `nsoilcols 5`, `Turbpar 2`, `soilswitch 1`, `eos 5`, `skin 0`,
  `carbon_model 2`, `ifbubble 1`, `tribheat 3`, `phi 56`; depth 22.66 m; `lakeform 2` with a
  25-row morphometry table.
- Forcing file: 8 comma-separated columns: 1 air temperature (K), 2 pressure (Pa),
  3 downward longwave, 4 downward shortwave, 5 u wind, 6 v wind, 7 specific humidity (kg/kg),
  8 precipitation.
- Thermodynamic reference configuration: write `lakematrix/tools/ref/make_ref_cases.py`,
  which calls `make_case` and then edits the setup: `tribheat 0` (no tributary heat);
  everything else unchanged.
- **First verify that biogeochemistry does not feed back into thermodynamics** (M0.4). The
  biogeochemistry calls (`lake.f90:1095-1153` and the corresponding parts of cases 2 and 3)
  and `BUBBLE_BLOCK` are executed unconditionally; there is no master switch. Steps:
  1. Copy `lakebatch/core/source` to `lakebatch/scratch/thermo-only/`.
  2. Comment out those calls.
  3. Build with the check flags (build method in 4.3).
  4. Run 425 days and compare with the unmodified check build: the time series `water_temp`,
     `ice_temp`, `soil_temp`, `layers`, `u`, `v`, `epsilon`, `lamw`, `water_dens_layers`, etc.
  - If identical: use the unmodified check build for the references; JAX only needs to match
    the thermodynamic outputs.
  - If not identical: find the feedback path (possibly light extinction, density or soil),
    record it in `NOTES.md`, use the build with biogeochemistry commented out for the
    references, and make that build reproducible as a script in `tools/ref/`.

### 4.2 Reference case set (`lakematrix/ref/v1/`)

| Name | Lake | Days | Purpose |
|---|---|---|---|
| `summer1d` | testlake | 1 | Quick check for M3 |
| `summer60d` | testlake | 60 (August–September) | M3 |
| `full425d` | testlake | 425 | M4 (freezing, melting, snow) |
| `cold425d` | testlake, air temperature −2 K | 425 | M4; batched with `full425d`, the two lakes are in different `layer_case`s on the same step |
| `shallow425d` | testlake, `h10` set to 5 m, morphometry table scaled to match | 425 | Shallow lake (freezes to the bottom earlier; may reach case 3) |
| `warm425d` | testlake, air temperature +3 K, `phi 40` | 425 | Different latitude, little ice |

Each case is a single-lake Fortran run with the check build. For batched comparisons, JAX runs
these lakes together as one batch, and each lake is compared with its own single-lake
reference. If `shallow425d` never reaches case 3, reduce depth or air temperature further
until it does (the aim is to cover all three cases); record the final parameters in
`NOTES.md`.

### 4.3 Routine-level references: the probe build

Purpose: compare each JAX routine with Fortran on its own, without waiting for the whole step.

- Under `lakebatch/scratch/probe-build/`:
  1. Copy `lakebatch/core/source` to `source/`.
  2. Copy `lakebatch/core/build/CMakeLists.txt` to `build/CMakeLists.txt` and apply with sed
     the two substitutions that the CHECK branch of `core/build.sh` applies:
     `RUNTIME_OUTPUT_DIRECTORY "../"` → `"."`, and `list(APPEND FLAGS_ "-O3")` →
     `list(APPEND FLAGS_ "-O3" "-ffp-contract=off")`.
  3. Build: `cd build && cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5
     -DCMAKE_Fortran_FLAGS="-I$PWD/../source/model" . && cmake --build . --target lake.out`.

  `lakebatch/scratch/bubble-share/` was built this way and can serve as an example.
- Write a new Fortran module `source/model/lm_probe.f90` and add it to the model source list
  in the CMakeLists. It provides:
  - A generic interface `LM_PROBE(tag, name, x)` for real(8), integer and logical scalars
    and 1- to 3-dimensional arrays. When the current step number `nstep` is listed in the
    environment variable `LM_PROBE_STEPS` (comma-separated), it appends one record to
    `probe/<nstep>/<tag>.bin`: the name (64 bytes, space-padded), a dtype code (1 byte: `f`,
    `i` or `b`), the rank (int64), lower and upper bound of each dimension (int64 each), then
    the data (little-endian, Fortran column-major order).
  - A Python reader `lakematrix/tools/ref/probe.py`: `read_probe(path) -> dict[name,
    np.ndarray]`. It reshapes in Fortran order and keeps the lower bounds, so 0-based and
    1-based arrays can be aligned.
- In the probe build's source, insert `LM_PROBE("<routine>_in", ...)` immediately **before**
  and `LM_PROBE("<routine>_out", ...)` immediately **after** each routine call to be checked.
  Record all of the routine's inputs (including module variables it reads) and all of its
  outputs (including module variables and `SAVE` variables it modifies). The insertion is done
  by a script (`tools/ref/make_probe_build.py`, a text patch of the copied source), never by
  hand, so it can be repeated.
- Sampled steps: several summer steps (case 1), the step on which ice forms, several winter
  steps under ice (case 2), the step on which ice melts, steps with snow, and steps in case 3
  if any. Choose the step numbers in M0.4 from the `layers` output of `full425d` and
  `shallow425d`, and write them to `tools/ref/steps.toml`.
- Lower bounds: Fortran arrays often start at 0 or 1 (for example `fbbleflx(0:M+1)`,
  `Tw1(1:M+1)`, `ddz05(0:M)`). JAX arrays always start at 0. Record the mapping for every array
  in `docs/state.md`.

## 5. Acceptance criteria (tolerances)

| Level | Compared | Criterion |
|---|---|---|
| Routine | Probe records before/after, one step | For each output array: `max|a-b| <= 1e-10 * max(1, max|b|)`. With the same formulas in the same order this should be close to bit-identical |
| Whole step | Probe records after `UPDATE_CURRENT_TIMESTEP` and before `UPDATE_NEXT_TIMESTEP` | Same as above, 1e-10 |
| 1 day | `summer1d` output | Water temperature ≤ 1e-6 K; u, v ≤ 1e-6 m/s |
| 60 days | `summer60d` | Water temperature profile ≤ 0.01 K; surface temperature ≤ 0.01 K |
| 425 days | `full425d` and the other 425-day cases | Water temperature profile ≤ 0.05 K, excluding ±1 day around freeze-up and break-up; ice thickness and snow thickness ≤ 5 mm; freeze-up and break-up dates within 1 day |
| Batch consistency | Batch of N lakes vs each lake run alone (N = 1) in JAX | Every output of every lake ≤ 1e-12 (proves lakes do not interact; on CPU this should be bit-identical) |

Review the long-run criteria once at the end of M3, using the measured errors. Any loosening
must be justified in `NOTES.md` (for example, differences caused by iteration convergence
tests).

Comparison script: `lakematrix/tools/compare.py REF_CASE_DIR OUT.nc LAKE_INDEX`. It reads the
Fortran time series `results/test/time_series/<var>  1  1.dat` (the first 5 columns are year,
month, day, hour, days since start; then depth/value pairs; the file names contain spaces and
must be quoted) and `layers  1  1.dat` (columns described by the `Col. N - ...` header lines).
It aligns them in time and depth with the JAX output NetCDF and prints, per variable, the
maximum error and pass/fail.

## 6. Package design

```
lakematrix/
  pyproject.toml          # deps: jax[cpu]>=0.4.30, numpy, netCDF4, xarray; optional [test]: pytest
  README.md  NOTES.md
  docs/step_map.md docs/state.md
  src/lakematrix/
    __init__.py           # enables x64
    constants.py          # values from DEFINE_PHYS_CONSTANTS/PARAMETERS, TURB_CONST, NUMERIC_PARAMS (frozen dataclass or dict)
    config.py             # Params: per-lake parameters, a pytree whose leaves are (L,) or (L, k) arrays
    setup_io.py           # reads LAKE setup/driver .dat keywords (one lake) into Params; stacks several lakes into a batch
    grid.py               # dimensionless grid (DZETA/DZETAI), ddz/ddz05/ddzi..., morphometry interpolation (static part of BATHYMETRY)
    state.py              # State pytree (see below) + init_state(params, grid) (thermodynamic part of INIT_VAR)
    forcing.py            # forcing as (L, T) arrays, read from NetCDF or LAKE forcing .dat; per-step interpolation identical to Fortran
    physics/
      tridiag.py          # batched tridiagonal solver: a, b, c, d of shape (L, n); lax.scan along n
      blocktridiag.py     # MATRIXPROGONKA (2x2 block-tridiagonal system of the momentum equations)
      density.py  radiation.py  surface.py  tsolver.py  soil.py  snow.py  ice.py
      turb.py  momentum.py  salinity.py  layers.py  bathym.py
    step.py               # step(state, forcing_t, params, grid) -> state, for all lakes at once
    run.py                # run(...): jit the step; lax.scan over chunks of steps (e.g. 1 day = 4320 steps); output between chunks
    output.py             # (lake, time[, depth]) NetCDF
    cli.py                # lakematrix run <setup list | input.nc> <out.nc>
  tools/ref/              # make_ref_cases.py, make_probe_build.py, probe.py, steps.toml
  tools/compare.py
  tests/
```

### 6.1 Data conventions

- Dimension order: lake first. L = number of lakes. Water levels `(L, M+1)`, ice levels
  `(L, Mice+1)`, soil `(L, nsoilcols, ns)`, scalars `(L,)`.
- Version 1 requires all lakes in a batch to have the same `M`, `Mice`, `ns` and `nsoilcols`.
  (Fortran already uses a dimensionless vertical coordinate, so a different depth only changes
  `h1`.) `setup_io` raises an error on mismatches.
- State, Params and Grid are pytrees (`typing.NamedTuple`, or a dataclass registered with
  `jax.tree_util.register_dataclass`). State contains no Python scalars or strings.
- All functions are pure; no mutable globals. Each Fortran `SAVE` variable either becomes a
  State field or is shown to be recomputable every step.

### 6.2 Branches and iterations

- `layer_case`: computed every step per section 3.2 as an (L,) integer array. **Version 1**
  runs the case 1, case 2 and case 3 updates for all lakes and selects per lake with
  `jnp.where(layer_case == k, ...)` (a `tree_map` over every State leaf). This is correct and
  has no data-dependent control flow. M5 optimizes it (unified formulation, or grouping lakes
  by case).
- `if` statements inside routines: rewrite all of them as `jnp.where`. Both sides must be safe
  to evaluate: masked-out branches are still computed, so a division by zero or a log of a
  negative number would create NaN. Make the inputs valid with `jnp.where` first, then compute.
- Iterations (inside `K_EPSILON`, the surface temperature iteration, momentum, etc.): use
  `lax.while_loop` with the condition "some lake has not converged and the maximum count is
  not reached". Each round updates only the unconverged lakes (`jnp.where(done, old, new)`).
  The convergence test and the maximum count are exactly those of Fortran, so every lake gets
  the same result as if it iterated alone.
- Recurrences along the vertical (tridiagonal sweeps, layer-by-layer integration): `lax.scan`
  along the level dimension, vectorized over the lake dimension.

### 6.3 Output

- NetCDF dimensions: `lake`, `time`, `depth` (fixed output depths; default the 26 testlake
  `ngrid_out` depths from 0 to 24 m; configurable), `ice_depth`, `soil_depth`.
- Variables (version 1): `water_temp(lake,time,depth)`, `ice_temp`, `soil_temp`, `u`, `v`,
  `tke`, `epsilon`, `h_water`, `h_ice`, `h_snow`, `h_deepice`, `t_surface`, `sensible`,
  `latent`, `layer_case`.
- Default output interval: hourly (as Fortran `time_series` with `dt_out 1.`), for
  hour-by-hour comparison.

## 7. Milestones and tasks

At the end of every milestone: all tests pass → update `NOTES.md` (what was done, measured
errors, open issues) → commit.

### M0 Preparation (no JAX physics yet)
1. Step 0 (section 2): wrap up lakebatch.
2. Create the `lakematrix/` repository, `pyproject.toml` and `.venv-linux`; run
   `pip install -e .[test]`. Check that
   `python -c "import jax; jax.config.update('jax_enable_x64', True); print(jax.devices())"`
   runs.
3. Write `docs/step_map.md` (section 3.1 expanded): every call in cases 1, 2 and 3, its
   condition, and the variables it reads and writes.
4. Verify that biogeochemistry has no feedback (section 4.1); generate the reference case set
   (section 4.2); choose the probe steps (section 4.3); build the probe build and write
   `probe.py`. Check: `Tw1` read with `read_probe` matches the Fortran `water_temp` output of
   the same step (after interpolation to the output depths).
5. Write `docs/state.md`: the classification from section 3.4 and the lower-bound mapping of
   every array.

### M1 Skeleton
- `constants.py`, `config.py`, `setup_io.py`, `grid.py`, `state.py` (`init_state`),
  `forcing.py`.
- Tests:
  - For testlake, `init_state` equals the probe record taken after
    `UPDATE_CURRENT_TIMESTEP` on step 1 (1e-12).
  - Grid quantities (`ddz` etc.) equal the probe values.
  - Interpolated forcing at each step equals the forcing scalars recorded at `LAKE` entry.
  - Two setups (testlake and cold) stacked into a batch give, for each lake, the same fields
    as building each lake alone.

### M2 Basic computations
- `tridiag.py`, `blocktridiag.py`, `density.py`, `radiation.py`, `surface.py` (`SfcFlx_*` +
  `SENSLATMOM_FLUXES` + `RADBALANCE` + `HEATBALANCE` + `HEATFLUXSURF`), `bathym.py`.
- Tests:
  - Each function, fed the probe inputs, matches the probe outputs (1e-10).
  - `tridiag` additionally matches `numpy.linalg.solve` on random diagonally dominant systems.
  - Batch consistency (N different lakes vs each with N = 1).

### M3 Ice-free case (case 1)
- `tsolver.py` (the branches of `T_SOLVER`/`T_DIFF`/`DIFF_COEF`/`SURF_CHAR` used in case 1),
  `soil.py` (`SOILFORLAKE`, `LATERHEAT`, several soil columns), `salinity.py`, `momentum.py`,
  `turb.py` (`K_EPSILON` + `TURB_SCALES` + `MIXED_LAYER_CALC`), `layers.py` (case 1 thickness
  changes), the case 1 path in `step.py`, `run.py`, `output.py`, `tools/compare.py`.
- Tests: every routine on summer probe steps (1e-10); the whole step (1e-10); `summer1d` and
  `summer60d` end to end (section 5).
- Review the long-run tolerances (end of section 5).

### M4 Ice and snow (cases 2 and 3) and case transitions
- `ice.py`, `snow.py` (`SNOWTEMP`/`SNOW_CALC`), the ice/snow/deep-ice branches of
  `tsolver.py`, the case 2 and case 3 paths, and the transition rules of section 3.2
  (including the state rewrite of case 3 → 1).
- Tests:
  - Winter, freeze-up, break-up, snow and (if any) case 3 probe steps (1e-10).
  - `full425d`, `cold425d`, `shallow425d`, `warm425d` end to end (section 5).
  - The four lakes in one batch for 425 days: each lake matches its own reference, and matches
    JAX with N = 1 bit for bit.

### M5 Performance (CPU)
- `run.py`: `jax.jit` a whole chunk (`lax.scan` over many steps) to avoid per-step Python
  dispatch. Time with `jax.block_until_ready`; report the first compilation time separately.
- Benchmark: testlake replicated to L = 1, 8, 64, 512, 4096 lakes (small forcing
  perturbations allowed), 60 days. Metric: lake-steps per second. Baselines: the Fortran check
  and production builds, one lake in one process; and the total throughput of 5 Fortran
  processes running at once on the 5 cores.
- Profile (`jax.profiler`, or `XLA_FLAGS=--xla_dump_to=...`) and fix the bottlenecks. First
  candidates: the waste of computing all three cases, and `while_loop` overhead.
- Put the results in a table in `README.md`.

### M6 Output and command line
- `cli.py`: `lakematrix run <setup list | input.nc> <out.nc> [--days N] [--chunk-steps K]`.
- Inputs: (a) a list of LAKE setup/driver directories (the reference cases work directly);
  (b) a (lake, time) NetCDF in lakebatch's schema version 1 (documented in the module
  docstring of `lakebatch/src/lakebatch/inputs.py`); implement only the keywords the
  thermodynamic core needs.
- End-to-end test: run 4 lakes for 425 days through the CLI; `tools/compare.py` passes for all.

## 8. Risks and how to handle them

- **Large scope** (about 10,000 lines of Fortran): compare routine by routine with the probe
  build. Do not move on while a routine fails the 1e-10 criterion.
- **Hidden state**: `SAVE` and module variables are easy to miss. The list in section 3.4 is
  complete (it comes from the lakebatch stage 1 generator); use it as the checklist.
- **NaN under `where`**: masked branches are still evaluated; keep their inputs valid
  (section 6.2).
- **`while_loop` overhead on CPU**: measure in M5. If needed, switch to a fixed iteration
  count, but only after showing the results stay within tolerance.
- **Limited CPU speedup**: about 34% of Fortran run time is in math library functions (exp,
  log, ...). M5's measurements decide how soon to build the GPU version.
- **Floating-point order**: JAX/XLA may reorder sums, so routine-level results may not be
  bit-identical; hence the 1e-10 criterion. If a routine exceeds it, first look for a
  translation error in the formula or order of operations, and only then attribute it to
  reordering.
