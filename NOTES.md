# Notes: deviations, decisions and measured errors

The plan (`../LAKEMATRIX-PLAN.en.md`) is authoritative. This file records
places where the code or this checkout differs from it, plus measured errors
for every milestone.

## Environment

- 2026-11-11, M0.2: this checkout runs on the **Mac**. The plan names
  `lakematrix/.venv-linux/` for the Linux VM; on the Mac the environment is
  `lakematrix/.venv` (Python 3.12, so that `jaxlib` wheels for `jax>=0.4.30`
  are available). The VM keeps its own `.venv-linux`; both are git-ignored.
- JAX version: see `pip show jax jaxlib`; record the versions used for every
  reference comparison below. M0.4 used `jax==0.12.0.1` / `jaxlib==0.12.0.1`
  (Python 3.12).

## M0.4: does biogeochemistry feed back into thermodynamics?

**Method.** `tools/ref/make_thermo_only.py` copies `lakebatch/core/source` to
`lakebatch/scratch/thermo-only/source`, comments out the 35 top-level calls to
`CHLOROPHYLLA`, `OXYGEN_PRODCONS`, `METHANE`, `ADDOXPRODCONS`, `OXYGEN`,
`CARBON_DIOXIDE`, `PHOSPHORUS`, `METHANE_OXIDATION` and `BUBBLE_BLOCK` in all
layer cases (the soil `SOILCOLSTEMP`/`LATERHEAT` calls inside the same blocks
are kept, because they also update soil temperature), and builds with the
check flags. The unmodified check build and this build ran the same 425-day
testlake case (`tribheat 0`; `Turbpar 2`, `ifbubble 1`, `carbon_model 2`).

**Result.** All 506 output files compared; all thermodynamic products are
byte-identical (425 days, including ice formation, snow and melt):

```
Profiles fluxes scalars Ri Rp Rpdens TKE_budget_terms_series T_fluxes
conv_series epsilon ice_temp lamw layers sal_water soil_temp u v w
water_dens_layers water_temp wi_soil wl_soil
```

Only the biogeochemistry products differ (`DOC`, `POCD`, `POCL`,
`carbon_series`, `chla`, `co2_water`, `fbblflx_*`, `methane_*`,
`oxygen_water`, `phosph_water`, `prodox`, `rprod_soil`, `sod`, and the mixed
`scalars` product). **Conclusion:** for the testlake configuration the
biogeochemistry and BUBBLE have no feedback on the thermal solution, so the
reference is the unmodified check build and v1 only has to match the
thermodynamic outputs. This is consistent with the plan's assumption; it is
not a proof for other switch combinations (M0.4 used testlake's defaults).

The 1-day comparison is the same: 44 thermodynamic files identical, 25
biogeochemistry files differ.

## M0.4: reference cases

The check-build reference set is in `ref/v1/` (git-ignored); see
`docs/reference_cases.md` for the case table, event times and the regeneration
command. `summer1d`, `summer60d`, `full425d`, `cold425d`, `shallow425d` (5 m)
and `warm425d` all ran to completion. Two attempts to reach `layer_case 3`
with a very shallow lake failed: `h10 = 0.5 m` gives `Tw1 = NaN` after 5
hours, `h10 = 1 m` after 5651 hours (235 days), and neither reaches case 3
(the water thickness never becomes zero). **Case 3 is therefore not covered
by the natural reference set.** M4 must add either a stable shallow setup or
a synthetic case-3 probe; record the choice here.

The 5 m `shallow425d` does reach case 1 and 2 with earlier freeze-up (first
ice at step 416160) and is kept as the shallow reference.

## M0.4: probe build and reader

- Build: `tools/ref/make_probe_build.py --out <dir> --build`. On this macOS
  checkout CMake also needs `-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0` and
  `-DCMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path)`; the script adds them. The
  plan's §4.3 command is enough on the Linux VM.
- The probe module is `LM_PROBE_MOD` (the generic interface is `LM_PROBE`);
  the module name cannot equal the generic name. `nstep` lives in module
  `ARRAYS` (inside `lake_modules.f90`), not in a module named `LAKE_MODULES`.
- Probes at `LAKE_entry` (after `UPDATE_CURRENT_TIMESTEP`) and `LAKE_exit`
  (before `UPDATE_NEXT_TIMESTEP`) record `Tw1`, `Ti1`, `Tis1`, `Tsoil1`,
  `h1`, `l1`, `hs1`, `ls1`, `snowmass` in `probe/<nstep>/<tag>.bin`.
- **Instrumentation self-check:** with `LM_PROBE_STEPS` unset, the probe build
  reproduces the check build on the 1-day case: 69/69 files byte-identical.
- **Reader check:** at `nstep = 180` (2016-08-01 04:00), `Tw1(1)` from
  `probe/180/LAKE_entry.bin` is `24.7207385328519`; the Fortran `water_temp`
  output at depth 0 is `0.2472073853E+002` (24.72073853). The probe path and
  the output path agree to the printed precision.

## Differences from the plan

- The plan's §3.4 says the `lb_point_ctx` list is complete; the step-map and
  `state.md` work uses it as a starting point but also the full state dump,
  because the swap generator has known exclusions (see `docs/state.md` when
  written).
- The plan's §4.1 says BUBBLE has no master switch; the code has
  `ifbubble%par` (`lake.f90:3181`). The M0.4 experiment removed the call
  instead of setting the switch, to test the strongest case; a run with
  `ifbubble 0` is equivalent for the thermodynamic comparison.


## M1: skeleton and the first state/grid/forcing checks

**The first `LAKE` call is at `nstep = 0`.**  `INIT_VAR` sets `nstep = 0`; the
first call initialises and advances one step, and the next call (`nstep = 1`)
loads that state.  The probe tag `INITVAR` (inside `INIT_VAR`) is therefore the
raw initial state, while `LAKE_entry` at step 1 is the state **after** the
first step.  M1 compares `init_state` with `INITVAR` at step 0; the first JAX
`step` produces the step-1 state.

**Fortran default-real literals are single precision.**  In
`Tsoil1(i,j) = Tw1(k) + (Tbb0-Tw1(k))*(1.-exp(-10.*float(i-1)/float(ns-1)))`
the argument of `exp` is single precision; a float64-only port differs by up to
5e-7 K.  `state.py` reproduces the single-precision argument and the
single-precision `1 - exp`; the side soil columns then match the probe exactly.
This is a warning for the plan's 1e-10 routine tolerance: many expressions in
the Fortran source use default-real literals and `float()`, so a pure-float64
port will differ at the 1e-7 level unless each expression is checked.  M1 uses
1e-12 for `Tw1`/`Sal1`, 1e-6 for `Tsoil1`/`zsoil`, and 1e-5 for `tempair`; the
per-routine tolerances will be revisited in M2/M3.

**Forcing time.**  Model step `nstep` uses the forcing at `t = (nstep+1)*dt`
(step 1: 40 s), interpolating between the hourly records `ti` and `ti+1`.
Verified at steps 1, 2, 3, 180: shortwave/longwave match exactly, `tempair`
within 6.1e-6 K (constant offset, likely single/pow rounding in the Fortran
reader).

**M1 tests:** `tests/test_m1.py` — initial state (`Tw1` 3.6e-15, `Sal1`
exact, `Tsoil1` 1e-6), grid (`ddz`/`dzeta` 1e-12, `zsoil` 1e-6, `zsoilcols`
exact), forcing, and batch stacking.  The probe files live under
`lakematrix/scratch/m0.4/` (git-ignored); the tests skip when they are absent.

## M2: tridiagonal, block-tridiagonal, density and radiation

- `physics/tridiag.py`: `PROGONKA` ported with the Fortran 1-based recurrence
  (internal leading zero pad); NumPy and JAX (`lax.scan`) versions agree with a
  dense solve to 5.6e-17, including an inner `K..N` window.
- `physics/blocktridiag.py`: `MATRIXPROGONKA` ported; the Fortran recurrence
  indexes `alpha(k)` from block `k-1`, with a separate final block `N`.  After
  matching that indexing the result agrees with a dense block solve to
  1.7e-16.
- `physics/density.py`: `WATER_DENS_TS` (McCutcheon, `eos 5`, `lindens 0`);
  `DENS_975` and `DENS_1192` probes at step 1 match **exactly** (byte-equal
  values, max diff 0.0).
- `physics/radiation.py`: `RAD_UPDATE` Beer-Lambert integration; tested against
  a direct loop (1e-14).  A Fortran probe for radiation will be added with the
  surface/M3 work.
- Tests: `tests/test_m2_tridiag_density.py`,
  `tests/test_m2_block_radiation.py`; the full suite is 10 passed.
- The density probe was added to the probe build (`patch_density`); tags
  `DENS_<line>_in/out` at every `call DENSITY_W` in `lake.f90`.

## M2: bathymetry

- `physics/bathym.py`: water part of `BATHYMETRY` for the testlake
  configuration (`soilcolconjtype=2`, linear depth-area interpolation;
  `lakeform=2`, rectangle).  The water surface is placed relative to the
  deepest table point: `work1 = depth - max(depth) + h1 + ls1`; the area at
  model depth `z` is the table area at `z + max(depth) - h1 - ls1`.
- Probe `BATHY_902_out` at step 1: `z_full` 3.6e-15, `area_int` 3.7e-9 m2,
  `vol` 3.0e-8 m3, `botar` exact, `Lx/Ly/rad` 1e-12, `preswat` 2.9e-3 Pa
  (tolerance 1e-2).  The residual is the table interpolation / pressure
  rounding, not a formula error.
- The soil-column geometry (`bathymsoil`, `itop`/`ibot`/`icent`, `ksoilcol`,
  `isoilcolc`) is not ported yet; it is needed by `SOILCOLSTEMP`/`LATERHEAT`
  in M3.
- Test: `tests/test_m2_bathym.py`; suite 11 passed.

## M2: what remains

- `surface.py`: `SfcFlx_*`, `SENSLATMOM_FLUXES`, `RADBALANCE`, `HEATBALANCE`,
  `HEATFLUXSURF`.  This is the last M2 module and the entry point to M3.

## M2: surface radiation and heat-flux scaffolding

- `physics/surface.py`: `radbalance` (water/ice/snow/soil; `skin=0`,
  `ifrad=1`) and `heatfluxsurf_water` (the water branch of `HEATFLUXSURF`,
  including `eflux`, `dHdt`, `convect_flux` and `botflux`).
- Probe `SFC_in`/`SFC_out` added inside `SfcFlx_momsenlat`; at step 1 it is
  called 49 times (441 input records / 9, 196 output records / 4).  The
  sequence reader `probe.read_probe_sequence` returns all records in file
  order, because the per-step tag is reused.
- Tests: `tests/test_m2_surface.py` (RADBALANCE water and the water heat-flux
  branch); suite 13 passed.
- `SfcFlx_momsenlat` is being transcribed to `physics/sfcflx.py`; once it is in,
  `SENSLATMOM_FLUXES`/`HEATBALANCE` can be assembled and tested against the
  SFC probe.

## M2: SfcFlx (surface fluxes)

- `physics/sfcflx.py` is a direct port of `shared/SfcFlx_mod.F90`
  (`SfcFlx_momsenlat`, `rhoair`, `roughness`, `satwvpres`, `spechum`,
  `lwradwsfc`), preserving the two Newton iterations and every branch.
- **Validation:** the probe records 49 calls of `SfcFlx_momsenlat` at step 1
  (including the first, unphysical `T_s = 183 K` bracket).  For all 49 calls
  the Python outputs match the Fortran outputs **exactly** (0.0 absolute
  difference).  `tests/test_m2_sfcflx.py`.
- Port notes: the FLake constant `tpl_grav = 9.81` differs from the LAKE-core
  `g = 9.814`; the port keeps FLake's value locally, otherwise the fluxes would
  change.  The Fortran module's `*_sf` side effects are dropped (pure
  functions).
- Next: wrap `SENSLATMOM_FLUXES` (PBLpar 3 path) and `HEATBALANCE`, then
  compare against `HEATBALANCE` probes.

## M2: surface flux assembly

- Added probes `SLM_in`/`SLM_out` (`SENSLATMOM_FLUXES`), `HFS_in`/`HFS_out`
  (`HEATFLUXSURF`), `RB_in`/`RB_out` (`RADBALANCE`); at step 1 each is called
  49 times.
- `physics/surface.py`: `senslatmom_fluxes` (PBLpar 3), `heatbalance`.
  - `HFS`: all 49 calls **exact** (eflux/dHdt/convect/botflux, 0.0).
  - `SLM`: hflux/Elatent 4–8e-6 relative, tau/cdmw 1e-9, velfrict 5e-8,
    wind10/wr exact.  The residual comes from the known 6e-6 K `tempair`
    offset between the probes and the `SfcFlx` inputs; the SfcFlx core itself
    matched 49/49 exactly.
  - `RB`: Radbal/Radbal_surf 2.7e-6 relative, surfrad 1.3e-7, ShortBal/swup
    exact.
  - `heatbalance` composition (skin=0): probe terms reproduce `HEATBALANCE`,
    `eflux0`, `eflux0_kinem` to 1e-9.
- Tests: `tests/test_m2_surface.py`, `tests/test_m2_sfcflx.py`; suite 16
  passed.  **M2 is complete.**

## Constants: Fortran default-real literals are float32-rounded

The probe exposed a systematic difference: `DEFINE_PHYS_CONSTANTS` assigns
default-real (single precision) literals to `real(8)` variables, so the stored
double is `float32(literal)`, not the exact decimal.  `constants.py` now uses
`_s(x) = float(np.float32(x))` for those values (`g`, `Kelvin0`, `Rd`, `Rwv`,
`aMagw`, `bMagw`, `emissivityofwater`, `albedoofwater_lw`, ...).  Literals with
a `d` exponent (`sigma`, `pref`, `T_phase_threshold`, ...) stay double.
Effects: `tempair` now matches the probe **exactly** (0.0; before 6.1e-6 K) and
`preswat` differs by 5.8e-11 Pa instead of 2.9e-3 Pa.

## M2: SURF_CHAR

- `surface.py`: `julian_day`, `sinh0`, `water_albedo`, `surf_char` (all four
  surface types; water uses `SfcFlx_roughness` and the variable albedo).
- Probe `SC_in`/`SC_out` at step 1: all 8 outputs match to 1e-12
  (`albedo=0.33333333333333337`, `aM=float32(7.6326)`, ...).
- M2 complete: 18 tests passed.

## M3: T_DIFF probe

- Probe `TD_in`/`TD_out` inserted in the `water == 1` (case 1) branch of
  `T_DIFF`: inputs `Tsurf`, `dt`, `Tw1`, `Tsoil1`, `lamw`, `area_int`,
  `area_half`, `h1`, `ddz`, `dhw`, `dhw0`, `PEMF`, `pt_down_f`, `lamsoil`,
  `csoil`, `rosoil`, `dzs`, `RadWater%integr`; outputs `Tw2`, `Tsoil2`.
- At step 1 `T_DIFF` is called 49 times; the first calls have the scanning
  surface temperatures `-90, -85, ...`; `PEMF`/`pt_down_f` are zero at this
  step.  This is the target for the `DIFF_COEF`/`T_DIFF` port.

## M3: DIFF_COEF / T_DIFF (case 1)

- `physics/tsolver.py`: `diff_coef_water`, `diff_coef_soil`, `t_diff_water`
  (water column + water-soil interface + soil column, Dirichlet surface
  temperature), using the 1-based Fortran indexing internally.
- Validated against the 49 `T_DIFF` calls at step 1: `Tw2` max diff 3.6e-15,
  `Tsoil2` 0.0.  The earlier 0.007 K discrepancy was entirely `lsh%water`
  (lateral heat from the soil columns); the probe now records it and the port
  uses it.
- Test: `tests/test_m3_tdiff.py`; suite 19 passed.
- Next: `T_SOLVER` (scan + Chorde iteration), and the source of `lsh%water`
  (`LATERHEAT`/`SOILFORLAKE`).

## M3: T_SOLVER and the water heat balance

- `tsolver.t_solver`: the scan + Chorde surface-temperature iteration,
  including the `Tsurf_prev-10` warm start, `dt_scan=5`, `resid=1 W/m2`,
  `maxiter=20` and the exact `Bal1 /= Bal3` comparisons.
- `surface.WaterSurfaceContext` + `surface.heatbalance_water`: runs
  `T_DIFF`, `HEATFLUXSURF`, `SENSLATMOM_FLUXES`, `RADBALANCE` and the
  `HEATBALANCE` composition for a case-1 water surface.
- Probe validation at step 1: all 49 `HEATBALANCE` calls match to 5.7e-14;
  `t_solver` reproduces `Tsurf3=24.96465085113388`, `iter=1`,
  `Bal3=-0.18064846780376964` exactly.  This is the first complete
  surface-temperature solve of the rewritten model.
- Tests: `tests/test_m3_tsolver.py` (synthetic scan/Chorde) and
  `tests/test_m3_surface_solver.py` (Fortran probes); suite 23 passed.
- Next: `lsh%water` source (`LATERHEAT`/`SOILFORLAKE`), then the soil,
  momentum, turbulence and layer-thickness ports for the full case-1 step.

## M3: MIXED_LAYER_CALC

- `physics/turb.py`: `mixed_layer_calc` (max BV frequency, well-mixed
  fallback, `H_mixed_layer = dzeta_05int(ml)*h`), tests in
  `tests/test_m3_mixed_layer.py`.
- Next: `TURB_SCALES` (the same `MIXED_LAYER_CALC` plus Wedderburn, Lake
  number, thermocline diagnostics and `Rp/Rpdens`), then `K_EPSILON`.

## M3: TURB_SCALES

- `physics/turb.py`: `turb_scales` (MIXED_LAYER_CALC plus w_conv/T_conv
  scales, Wedderburn, Lake number, Rossby radius, thermocline thickness,
  Re/Ri, Rp/Rpdens), and `density.ddens_dtemp/ddens_dsal/set_dens_consts`
  for eos 5.
- Probe `TSC_in`/`TSC_out` at step 1: **every output matches exactly (0.0)**,
  including the `kor=0` limits (Rossby radius 3.8e19, Lake number clamped).
- Test: `tests/test_m3_turb_scales.py`; suite 26 passed.

## M3: salinity and layer update (case 1)

- `physics/salinity.py`: `diff_coef_salinity` and `s_diff` (water branch;
  fresh-water testlake yields zero, but the full branch is transcribed).
- `physics/layers.py`: `advance_case1` (`h2=h1+dhw`, `ls2=ls1+dls`, `l2=0`).
- Tests: `tests/test_m3_salinity_layers.py`; suite 28 passed.

## M3: first-pass ports of K_EPSILON / momentum / soil

Three large modules were transcribed by parallel subagents and import cleanly:

- `physics/kepsilon.py` (1814 lines): `k_epsilon(state, dt, dt_keps=None,
  nstep_keps=None, params=None)` with `KEpsilonState` (126 fields) and
  `KEpsilonParams`; `Turbpar 2`/`kepsbc 1`/`stabfunc 2` path; helpers
  `CE_CANUTO`, `VSMOP_LAKE`, `TKE_FLUX_*`, `H13`, `SEICHE_ENERGY`.
- `physics/momentum.py` (~62 KB): `momentum_solver(state, params, dt)` with
  `MomentumState`/`MomentumParams`/`MomentumResult`; uses the ported
  `blocktridiag.matrixprogonka`.
- `physics/soil.py` (~68 KB): `comsoilforlake`, `soil_cond_heat_coef`,
  `soilforlake`, `soilcolstemp`, `laterheat`, with `SoilState`/`SoilParams`.

These are **not yet validated against the Fortran probes** and not yet wired
into `step.py`; the next step is the integration test.  The subagents' own
smoke tests and branch scans are recorded in their reports, not in this repo.

## M3: soil initial water/ice from INIT_VAR

- `config.Params` now carries `soiltype`, `nmeltpoint`, `tricemethhydr`,
  `soilbotflx` (read in `setup_io`).
- `state.init_state` fills `Sals1` (`Salb0*row0/(rosdry(1)*(1-por(1)))`, as
  in INIT_VAR), then `wl1`/`wi1` from `MELTINGPOINT`/`WL_MAX`/`UNFRWAT`/
  `WI_MAX`.  Validation against the step-1 T_DIFF probe:
  `csoil` 2e-10 relative, `rosoil` 1e-10, `lamsoil` exact.
- `SOILFORLAKE`/`LATERHEAT`/`SOILCOLSTEMP` (the subagent port) still need
  probe validation; `soil_cond_heat_coef` is validated.

## M3: soil port probe-validated (bit-exact heat path)

The soil subagent's `physics/soil.py` was validated against the M0.4 probes at
step 1: `zsoil/dzs`, `csoil/rosoil/lamsoil` and the side-column `Tsoil1`
after `SOILCOLSTEMP` are all **bit-exact (0.0)**.  The last 2e-10 residual was
`state.py`'s `- 0.01`, which INIT_VAR treats as a default-real literal; fixed
to `- float(np.float32(0.01))`.  Moisture/phase-change (`wl3/wl4/wi2/
dhwfsoil`) is transcribed but not yet probe-validated.

## M3: momentum probe validation

- `MOM_1105_in/out` probe at the case-1 `MOMENTUM_SOLVER` call.  Driving the
  port from the probe inputs gives: `u2` <= 9.4e-9 m/s, `v2` <= 1.6e-6 m/s,
  `k2` **exact**, `Ri` <= 2.4e-6.  The residual is in the tiny near-surface
  velocities (u2 ~ 1e-5 m/s), negligible for the temperature acceptance.
- Test: `tests/test_m3_momentum.py`; suite 29 passed.
- `K_EPSILON` still needs a probe validation; then `step.py`.

## M3: soil bathymetry scaffold (unvalidated)

- Added `bathym.soil_bathymetry` (area_int/area_half/itop/dzSL/ibot/icent/
  dzSLc from `BATHYMETRY`).  Import/smoke only.
- Driving `soilcolstemp` + `laterheat` from `init_state` and this geometry
  currently gives `soilflux` values of order 1e20 in side columns 2-4 and
  `lsh%water` differs from the probe by a factor ~2, so the geometry /
  `tsurfsoilcol` wiring is **not yet correct**.  The soil subagent's
  bit-exact validation used the probe's `BATHY_902_out`/`LAKE_exit`, not this
  function; the mismatch is in the newly written geometry or in the
  `SoilState` initialization, not in the validated heat coefficients.

## M3: soil geometry fixed; LATERHEAT matches the probe

The 1e20 `soilflux` / 2x `lsh` mismatch was **missing `Lx`/`Ly` on the soil
`Bathym` objects**: `TSURFSOILCOL` needs the ellipse semi-axes of each soil
column, and `bathym_levels` did not fill them.  `soil_bathymetry` now sets
`Lx, Ly = LXLY(area_int, lakeform, cellipt)` for every soil column.

After the fix, `Tsoilsurf = (24.34744, 19.28719, 10.03557, 7.99107)` for the
four side columns (matching the probe's top `Tsoil1`), `soilflux` is finite,
and `lsh%water` matches `TD_in` to 3.4e-12.  Test:
`tests/test_m3_laterheat.py`; suite 30 passed.

## M3: K_EPSILON probe (first comparison)

- Added `KE_in`/`KE_out` probes at the case-1 `call K_EPSILON`: 46 input
  arrays/scalars and 8 outputs (`E2/eps2/k2/kt/H_entrainment/Buoyancy0/
  Ri_bulk/Eseiches`).
- First replay from the probe state (filling `KEpsilonState` and the grid
  arrays `ddz/ddz2/ddz05/ddz052/ddz054/dzeta*`):
  `e2` max diff 1.5e-4, `eps2` 1.9e-3, `k2` 2.6e-3, `kt` 3.4e3.
  The `kt` difference is probably a missing `cw_m_row0` scaling or a
  first-call state field; `e2/eps2` need debugging.
- The K_EPSILON port is therefore **not yet validated**; the probe and replay
  setup are in place for the next debugging round.

## M3: K_EPSILON debugging note

The first replay (v25) differs in `e2` (1.5e-4), `eps2` (1.9e-3),
`k2` (2.6e-3) and `kt` (3.4e3).  An attempt to probe `CE/CEt/s/gen/f` at the
`lake.f90` call site failed: those are `turb_mod.f90` locals/module arrays and
are not in `lake.f90` scope (`c_eps3`/`cet` have no IMPLICIT type there).  The
next debugging step is a probe **inside** `K_EPSILON` (`turb_mod.f90`) for
`CE/CEt/s/gen/f/c_eps*` and a comparison of the stability functions; the
v25 `KE_in/KE_out` probes remain the entry-level reference.

## M3: density derivative bug and K_EPSILON divergence

- **Fixed `ddens_dsal`**: the Fortran multiplies the Canuto salinity
  derivative by `1.d+3` (`DDENS_DSAL = DDENS_DSAL*1.d+3`); the Python port was
  missing it.  `rhotemp` matched the probe exactly; `rhosal` now does too.
- K_EPSILON first-iteration `work1/work2` reproduce the probe to 4e-12 /
  3e-20, and `E12/eps12` match.  The remaining `e2/eps2` divergence must be in
  the E/eps tridiagonal update; the next probe is after the two `PROGONKA`
  calls inside `iterat_keps` (`E_it2`/`eps_it2`).

## M3: K_EPSILON replay helper

`tools/ref/replay_kepsilon.py` builds a `KEpsilonState` from a `KE_in` probe
and runs the port; `replay(dir)` is the debugging entry point.  First-iteration
`work1/work2` and `rhotemp` match the probe; the E/eps solve is the remaining
divergence.  Next probes: `FTKES/FTKEB/FS/FB` and the two `PROGONKA` results.

## M3: K_EPSILON mode-1 E solve still differs

- Added probes `KE5` (E_it2/eps_it2 after each PROGONKA) and `KE6`
  (ksurf1/Esurf1/CE(1)/k2(1)/tau_air after FS), plus a `trace` list in the
  Python port that stores `e_it2` after each `keps_mode` pass.
- Result: **mode 1** (`time_deriv=0`) `E_it2` = 6.28e-4 (port) vs 5.41e-4
  (Fortran); the first-iteration `work1/work2` and `rhotemp` match, so the
  divergence is in the mode-1 E tridiagonal coefficients or the `FTKES`
  boundary (probe `FTKES=0`, so the boundary term should vanish).
- `FS` differs because it inherits the mode-1/mode-2 `E_it1`; fixing mode 1
  is expected to fix `FS`/`eps_it2`/`E2`.
- Test suite still 30 passed.

## M3: K_EPSILON validated (root cause was a probe-harness bug)

The replay harness padded **`ddz05`** with a leading zero, but `ddz05` is a
Fortran **0-based** array (`0:M`); the padding shifted every `ddz05(i)` by one
and doubled the top-boundary coefficient `xx(1)`.  The K_EPSILON port itself
was correct.  Keeping `ddz05` 0-based (`s.ddz05 = grid.ddz05`):

- `e2` 4.9e-19, `eps2` 1.8e-18, `k2` 3.5e-18, `kt` 9e-12;
- `H_entrainment`/`Buoyancy0` exact.
- Test: `tests/test_m3_kepsilon.py`; suite **31 passed**.

This is the last major physics module of case 1.  Next: the full `step.py`
(`VERTADV`, radiation, soil, T_SOLVER, K_EPSILON, layers).

## M3: step integration scaffold

- `src/lakematrix/step.py`: `StepState` + `build_step_state` assemble the M1
  state, soil state/geometry, momentum state and K_EPSILON state (radiation
  still zero until the next step of the wiring).  Import/smoke only.
- `step_case1` is not written yet; the next step is to wire
  `BATHYMETRY → RAD_UPDATE → DENSITY → MOMENTUM → SOIL → T_SOLVER → S_DIFF →
  K_EPSILON → TURB_SCALES → layers` and compare probe step 1 → step 2.
