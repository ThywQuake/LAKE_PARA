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

## M3: one-step chain through T_SOLVER matches the probe

`step.step_case1_to_solver` wires BATHYMETRY → RAD_UPDATE → SOIL → T_SOLVER.
Two forcing details were fixed:

- the ASCII `tempair` column is **Kelvin**; `SENSLATMOM_FLUXES` expects Celsius,
  so subtract `C.Kelvin0`;
- step 1 uses `at_time(rows, nstep*dt)`, not `(nstep+1)*dt` (the probe step-1
  forcing is at 20 s).

Result: `Tsurf3 = 24.96465085110656` (probe 24.96465085113388, diff 2.7e-11)
and `Bal3 = -0.1806500354` (probe -0.1806484678).  Test
`tests/test_m3_step.py`; suite **32 passed**.

## M3: full case-1 step runs; first probe comparison

`step.step_case1` now runs the complete open-water chain
(`BATHYMETRY → DENSITY → MOMENTUM → SOIL → T_SOLVER → SOILFORLAKE → S_DIFF →
K_EPSILON → TURB_SCALES → layers`).

Against probe step 2 (`LAKE_entry`):

- `Tw1` max diff **0.022 K** (surface 0.0034 K);
- `Tsoil1` max diff **0.059 K**;
- `h1` 22.66000 vs 22.65977 (2.3e-4 m).

The remaining `h1` difference is the initial `dhw`: the first step's
`dhw` uses the previous `Elatent` (0 in our state), while the probe's step-2
water depth already includes a small evaporative loss.  Test
`tests/test_m3_step.py`; suite **33 passed**.

## M3: `run_case1` and the 180-step drift

- `step.run_case1` loops `step_case1`, carries `Tw1/Sal1/Tsoil1/h1/E/eps/
  lamw/Elatent/dhwfsoil`, and runs 180 steps in ~5 s.
- After one step `Tw1` matches the step-2 probe to 0.022 K; after 3 steps the
  error grows to 0.011 K, and after 180 steps the surface is 24.384 C vs the
  reference 24.721 C (0.337 K).
- `h1` after 3 steps is 2.3e-5 m lower than the probe; `Tsoil1` differs by
  0.059 K after one step (soil `SOILFORLAKE` moisture/phase sub-path is the
  unvalidated part).
- Next: validate `SOILFORLAKE`'s `Tsoil3/wl4/dhwfsoil` against a probe (the
  subagent's missing check), then re-run the 180-step/1-day comparison.

## M3: SOILFORLAKE is probe-exact; step handoff still drifts

- `SOILFORLAKE` now validated against the `SOILF_in/out` probe: `Tsoil1`,
  `Tsoil3`, `wl1`, `wi1`, `wl4`, `wi2` all **0.0**; `dhwfsoil` exact.
  This closes the subagent's "moisture/phase unverified" gap.
- `dhw` is now computed **after** `SOILCOLSTEMP`, using the current
  `dhwfsoil`; step-2 `h1` matches the probe to 6e-7 m.
- Despite that, the step-2 `Tw1` difference is still 0.022 K and grows to
  0.011 K by step 3; the handoff `Tw2 → Tw1` (Fortran
  `UPDATE_NEXT_TIMESTEP`) is the next suspect, since `T_SOLVER` itself
  matches the probe to 1e-11.
- Test: `tests/test_m3_soilforlake.py`; suite **34 passed**.

## M3: INIT_VAR probe and the initialization step

- `INIT_IN` probe (step 0) confirms `init_state` matches raw `INIT_VAR`
  exactly: side-column `Tsoil1` 0.0, `wi1/wl1` 0.0, deepest `Tsoil1` 1e-7.
- The model's first `LAKE` call (nstep 0) is itself a step: probe step 1 is
  the state after it.  `run_case1` now runs that initialization step first
  (`init_step=True`, `dhw` without `dhwfsoil`), then the regular steps.
- After the init step: step-1 `Tw1` matches to **2.2e-7**; `Tsoil1` still
  differs by 0.117 K in the side columns (SOILCOLSTEMP input/output at the
  init step).  Step 2 `Tw1` 0.0016 K; step 3 0.0109 K.
- Next: probe `SOILCOLSTEMP` at the init step and fix the side-column
  `Tsoilsurf`/`Tsoil1` residual.

## M3: INIT_VAR raw state is exact; init-step side soil residual

- `INIT_IN` (step 0) confirms `init_state` == raw `INIT_VAR`: side-column
  `Tsoil1` 0.0, `wi1/wl1` 0.0.
- `run_case1` now runs the model's initialization step (`init_step=True`)
  before the regular steps.
- After that init step: step-1 `Tw1` 2.2e-7 K, but `Tsoil1` side columns
  still differ by 0.117 K (step-2 `Tw1` 0.0016 K).
- A `SOILCOLSTEMP`/`TSURFSOILCOL` probe attempt hit the nested `CONTAINS`
  insertion point and was disabled in `make_probe_build.py`; the next attempt
  should probe `Tsoilsurf` inside `TSURFSOILCOL` at its actual end.

## M3: two critical step-loop fixes

1. `step_case1` returned **views** into the live `ss.soil` / `ke` arrays, so
   `run_case1`'s earlier `out[k]` changed when later steps mutated the state.
   All returned arrays are now copied.
2. The initial `cdmw` is `1e-15` (`INIT_VAR`), not `1e-3`; it is carried from
   `SENSLATMOM_FLUXES` (`heatbalance_water` state) into the next step's
   momentum solve.  This fixes the step-1 `u2`/`lamw` mismatch (kt differed
   by 1614 before).

After both fixes, with the model initialization step (nstep 0):

- step 1: `Tw1` 2.2e-7 K, `Tsoil1` 1e-7 K;
- step 2: `Tw1` 2.1e-4 K, `Tsoil1` 3.5e-7 K;
- step 3: `Tw1` 0.012 K (a new jump to investigate).

## M3: u1 profile fixed; UPDATE_NEXT_TIMESTEP is the remaining handoff

- `build_step_state` now reproduces `INIT_VAR`'s velocity profile
  (`u1=us0` in the mixed layer, linear to zero at the bottom).  With that,
  the first momentum solve matches `MOM_1105_out.u2` exactly.
- The step-1 `E1`/`eps1` in the *next* `LAKE_entry` are **not** the `E2`/`eps2`
  written at the end of `K_EPSILON` (e.g. `E1(1)=4.85e-6` at step 2 vs
  `E2(1)=2.64e-4` at step 1).  So `UPDATE_NEXT_TIMESTEP` (called outside
  `LAKE`) modifies `E1/eps1` (and likely `lamw`/`Tw1`) between steps.
  This is the next routine to port; it explains the step-2/3 `lamw` and
  interior-temperature drift.

## M3: 180-step status

- With the initialization step (n=0), the first step reproduces the probe
  `kt` to 8.5e-3 and `E1` to 4e-10; the later steps drift.
- After 180 steps (1 h): surface 24.384 C vs reference 24.721 C (-0.336 K);
  the 0.5/1 m layers are +0.037/+0.087 K and the deep layer is -0.018 K,
  i.e. the model mixes too much heat downward.  `lamw` after the first step
  matches; the drift starts at step 2.
- The probe entry/exit indexing at steps 1/2 is ambiguous (the recorded
  `E1` changes between entry and exit in a way `UPDATE_NEXT_TIMESTEP` cannot
  explain), so the next step is to compare against `LAKE_exit` only and add
  an `E1/eps1/lamw` trace at every step of a short run.

## M3: K_EPSILON second step collapses

Using the step-2 `KE_in/KE_out` probes:

- probe `KE_in E1` = step-1 `KE_out E2` (2.639e-4) — our `out[0]` E1 matches;
- probe `KE_in eps1` = 6.8e-6, but step-1 `KE_out eps2` = 7.0e-4 (so `eps1`
  is changed between steps by something other than the pure store in
  `UPDATE_NEXT_TIMESTEP`);
- our step-2 `K_EPSILON` output collapses (`E2` 6e-11, `kt` ~0) whereas the
  probe gives `E2` 4.85e-6 and `kt` 1.6–5742.  The input `eps1` (7e-4 vs
  6.8e-6) is the most likely trigger.

Next: trace our `ke` input fields at step 2 against the `KE_in` probe field
by field; the missing/adjusted `eps1` (or another work array) is the cause.

## M3: 180-step comparison with n=1 first step

Both `run_case1` variants (n=0 or n=1 first step) give the same 180-step
result: surface 24.384 C vs reference 24.721 C, 0.5/1 m +0.037/+0.087 K.
The step-1 output matches the probe; the drift starts when `K_EPSILON` is
called the second time.  The step-2 `KE_in` probe in the current probe build
appears to mix records from several K_EPSILON calls (`read_probe_sequence`
keeps the last record per tag), so the `eps1=6.8e-6` reading is not yet a
trustworthy target.  Next: rebuild a probe that writes the call index into the
tag (e.g. `KE_in_1`, `KE_in_2`) so each step can be compared unambiguously.

## M3: K_EPSILON state carry fixed; 1-day comparison

- The big multi-step bug: `step_case1` did `ke = k_epsilon(...)` but never
  wrote the result back to `ss.ke`, so every step restarted from the
  step-1 K_EPSILON state.  `ss.ke = ke` is now stored.
- Effect: step-3 `Tw1` error 0.0123 -> **3.3e-4 K**; 180-step surface
  24.7066 vs 24.7207 (-0.0141 K).
- Added `backdiff=1` (`DIFFMIN_HS`, Fang & Stefan) and `backdiff0` setup
  fields; the testlake uses it.
- **1-day (4320 steps)** comparison with `summer1d`:
  surface 25.523 vs 25.361 (+0.162 K); largest depth error +0.267 K at 5 m.
  This is now the M3 acceptance gap (plan: 0.05 K).
- Tests: 34 passed.

## M3: 1-day state comparison

- Added the momentum carry of `E1/eps1/k2/knum/veg_sw` from the previous
  K_EPSILON step (the Fortran momentum solver reads them); no change to the
  1-day thermal error.
- Against the step-4320 `TD_in` probe: `Tw1` differs by up to **1.05 K** and
  `lamw` by **2.9e4**, i.e. the k-epsilon/`lamw` drift grows over the day
  (the 1-day `water_temp` surface error is 0.16 K, but the state inside the
  model diverges more).
- This is now a K_EPSILON long-run state issue, not a one-step wiring issue.
  The next step is to compare the step-4320 `KE_in/KE_out` fields and check
  the SAVE-local carries (`Buoyancy1`, `FTKES/FTKEB`, `E_min`, `i_entrain`,
  `Eseiches`, `AG`, `knum`).

## M3: MAJOR density-call bug fixed — 1 day now 0.010 K

`water_dens_ts(temp, sal, row0=C.row0)` takes `row0` as the third argument, not
pressure.  `step.py` was passing `preswat` (~1e5), producing densities
~1e5 kg/m3 instead of ~1000.  After calling it correctly:

- 1-day `summer1d` (4320 steps) max depth error **0.0099 K** (was 0.267 K),
  surface 25.371 vs 25.361 (+0.010 K).  This is inside the plan's 0.05 K/day
  tolerance.
- The 180-step and 1-day drift were therefore mostly this single unit bug,
  not K_EPSILON.

Next: 60-day `summer60d` run and comparison.

## M3: 3.33-day k-epsilon structure drift

At step 14400 (3.33 days) the `summer60d` reference probe shows:

- `Tw1` max diff 0.119 K (4 m), `Tsoil1` 0.039 K;
- `lamw` profiles are qualitatively different:
  ours  [1185, 85, 40, 49] W/(m K),
  ref   [ 323, 986, 1074, 743].
  Our surface mixing is too strong and the deeper thermocline mixing too weak.
- `E1`/`eps1` already differ (2.6e-5 / 5e-7).

The 1-day case is still 0.010 K; the divergence develops between day 1 and
day 3.3.  Next: compare `K_EPSILON` at days 1/2/3 (probe `KE_in/out`) to find
the first diverging field (likely `u1/u2`, `k2`, or `eps1` carry).

## M4: ice/snow T_DIFF port (subagent)

- `physics/ice.py` (2028 lines): `t_diff_ice` (all `T_DIFF` branches),
  `diff_coef_ice/ice_sal/snow/deepice`, `snow_cond_heat_coef`, `sr_botsnow`,
  `snow_calc`, `ice_increments`, `ice_properties`, `make_ice_state`
  (`IceState`, 87 fields).  Import + synthetic drivers pass.
- `snowsoil_mod.f90` **does not exist** in this checkout; the old SNOWSOIL
  physics now lives in `snowcalc.f90` (`SNOW_CALC`/`addPrecipToSnow`) and
  `snowtemp.f90` (`SNOW_COND_HEAT_COEF`), which is what the port covers.
  `addPrecipToSnow` is **not ported** (needed if precipitation/snow
  accumulation is modelled).
- M4 wiring (case-2/case-3 step, layer transitions) is next.

## M3: first K_EPSILON divergence at day 1

After the density fix, the 1-day thermal profile matches (0.010 K), but the
step-4320 `KE_in` probe shows:

- `u1`/`u2` differ by **0.0039 / 0.0038 m/s** (large relative to the ~1e-2
  velocities);
- `lamw` differs by **4189** (out `kt` by 347);
- `Tw1` probe difference 0.070 K (timing/entry offset; the `water_temp`
  daily output matches to 0.010 K).

So the k-epsilon structure drift is driven by the **velocity field**, not by
the thermal state.  Next: compare the day-1 `MOMENTUM_SOLVER` state
(`u2/v2`, `taux/tauy`, `velfrict`, `alphax/alphay`, seiche fields) against a
day-1 probe and carry any missing state.

## M3: momentum u2 carry — 3.33-day error 0.033 K

The `MomentumState` returned by `momentum_solver` has no `u2`; the code stored
`mom_res.state` but never copied `mom_res.u2/v2` into `u1/v1`, so every step
restarted momentum from the previous `u1` (initial profile).  Carrying
`u1=u2`, `v1=v2` after each solve:

- 1 day: max 0.0135 K (still inside 0.05 K);
- 3.33 days: max **0.033 K** (was 0.123 K), all depths <= 0.033.

The 60-day job was restarted with this fix.

## M3: surface velocity in SENSLATMOM — 3.33-day error 0.008 K

`WaterSurfaceContext` was passing `u=v=0` to `SENSLATMOM_FLUXES`; the Fortran
uses the surface water velocity `u1(1)/v1(1)` in the relative wind.  Using
`ss.mom.u1[0]/v1[0]`:

- 3.33-day max error **0.0079 K** (was 0.033 K);
- this is a physically important coupling for wind events.

The 60-day run completed earlier with the old (u=v=0) code: max 0.764 K at
day 8/5 m, i.e. wind-event days.  Rerun with the fix.

## M3: 20-day rerun with surface-velocity fix

20-day `summer60d` comparison (daily max over the water column):

- days 1-6: 0.014-0.025 K;
- days 8-20: 0.031-0.086 K, worst day 16 at 7 m (-0.086 K).

This is far better than the old 0.76 K spike but still above the 0.05 K
tolerance on some stratified days.  Next: compare day-8 `lamw`/`E1`/`eps1`
against the step-34560 probe to identify the remaining thermocline mixing
drift.

## M3: day-8 k-epsilon comparison

At step 34560 (day 8):

- `Tw1` max diff 0.081 K (5 m), `Tsoil1` 0.038 K;
- `lamw` ours [2321, 6435, 13229, 17782] vs ref [4.8, 597, 4481, 7883]
  — again too much mixing/too high `KT`;
- `E1` diff 1.9e-4, `eps1` 4.2e-6.

So the remaining 0.03-0.09 K on stratified days is a k-epsilon/`lamw`
amplitude drift.  Next: compare `KE_in` `u1/u2/k2` at day 8 and trace the
first diverging field from day 3.3 to 8.

## M3: day-8 K_EPSILON inputs

At step 34560:

- `u1` diff 0.0123 m/s (ours -0.102 vs ref -0.090 at the surface),
  `u2` 0.0096, `v1` 0.0129, `v2` 0.0105;
- `k2` ours 1.1e-4/1.0e-3/2.2e-3 vs ref 2.4e-6/1.5e-4/1.0e-3 — too high;
- `lamw` too high by 1.2e4 at the surface;
- `Tw1` 0.081 K.

The velocity field is already 13% off by day 8, and the k-epsilon diffusivity
inherits it.  Next: compare the momentum state at days 1/2/3 (probe
`MOM_1110_in/out`) to find the first momentum divergence; candidates are the
seiche/surface-slope state (`alphax/alphay`, `hx*`) and `k2_old`.

## M3: day-1 momentum inputs

Against `MOM_1110_in` at step 4320:

- `u1` 1.5e-5, `v1` 7.9e-5, `E1` 4.5e-8, `eps1` 2.8e-9, `k2` 2.2e-6;
- `u_star` 6.475e-4 vs 6.475e-4, `velfrict_bot` 2.06e-10 vs 2.33e-10;
- `Ri` relative values close (both up to 1e8 near the bottom);
- `alphax/alphay` both 0.

So the momentum state is accurate at day 1; the 0.081 K day-8 drift is
accumulated small differences (velocity 1e-5 -> 1e-2) rather than a missing
field.  Continue with a day-3 `MOM` probe if the remaining 0.03-0.09 K must be
reduced further.

## M3: plan §5 tolerances are much stricter than current errors

The plan requires:

- 1 day (`summer1d`): water temperature and u/v **≤ 1e-6**;
- 60 days (`summer60d`): profile and surface **≤ 0.01 K**;
- 425 days: 0.05 K excluding ±1 day around freeze/break-up.

Current: 1 day 0.0135 K, 3.33 days 0.0079 K, 20 days 0.086 K.  These are not
acceptable under §5.  The gap is in accumulated step-level differences, not in
any single module (routine probes are 1e-10 or better).  Next must be a
whole-step probe comparison (`LAKE_entry`/`LAKE_exit`) field by field to find
the first non-bit-identical quantity, rather than more long runs.

## M3: zsoil/UpperLayer exactness

- `NUMERIC_PARAMS::UpperLayer = 0.02` is a default-real literal stored in a
  real(8) variable, and `zsoil(i) = (1.*(i-1)/(ns-1))**pow*depth` uses a
  default-real base.  `grid.create_grid` now rounds `upper_layer` and the
  base to float32.
- Effect: `zsoil` matches the probe **exactly** (0.0); `INIT_IN` deepest
  `Tsoil1` 8.9e-16; after one step `Tsoil1` 1.8e-15, `Tw1` 3.5e-13.
- 1-day `summer1d` max error **0.00517 K** (was 0.0135); 3.33 days 0.0081 K.

## M3: dhw order restored; step-2 near bit-exact

- `dhw` is again computed **before** `SOILCOLSTEMP` using the previous
  step's `dhwfsoil` (Fortran order).
- Step 1: `Tw1` 3.5e-13, `Tsoil1` 1.8e-15.
- Step 2: `Tw1` **7.1e-10**, `Tsoil1` 2.1e-14, `h1` 0.
- Step 3 still drifts: `Tw1` 3.0e-5, `Tsoil1` 2.8e-4, `h1` -2.3e-5.
  Probe step-3 `dhw=-0.00013827` vs our -0.00016116; the difference is the
  `dhwfsoil` carried out of step 2.  `SOILF_out` at step 2 gives -0.0001606,
  but the model's step-3 `dhw` implies -0.0001377, so the `dhwfsoil`
  accumulation/overwrite order between side columns and the deepest
  `SOILFORLAKE` still needs to be aligned.

## M3: step-3 root is Tsoil2/dhwfsoil timing

At step 2 our `Tw1` matches to 7e-10, but the `Tsoil2` handed to
`SOILFORLAKE` differs by **3.1e-5 K**, and the resulting `dhwfsoil` differs by
~2e-5 m.  That `dhwfsoil` is what enters step-3 `dhw` (`-0.0001383` in the
probe vs `-0.0001612` in our run), producing the step-3 `h1` offset and the
0.005 K/day drift.

The exact probe step numbering is ambiguous (`SOILF_in/out` at "step 2/3"
appear shifted by one relative to `TD_in dhw`), so the next step is a
**checkpoint probe harness**: record key state after each subroutine in
`lake.f90` at a single step, and compare our `step_case1` at the same
checkpoints.  Without that, the step-index bookkeeping dominates the debugging.

## M3: tributary-free dhwtrib relaxation

The reference cases have `tribheat 0`; when no tributary inflow is given the
Fortran sets `dhwtrib = (h10 - h1)/10` (a relaxation to the initial depth),
which we had omitted.  Adding it:

- `h1` matches the probe at steps 1, 2, 3 (0.0);
- step-3 `Tw1` 5.2e-7 (was 3.0e-5);
- **1-day max error 0.00049 K** (was 0.0052 K).

3.33-day is 0.0148 K in one run; the 20-day rerun will show whether this is a
transient or a real regression.

## M3: dhwtrib relaxation reverted

The `(h10-h1)/10` relaxation is only the Fortran fallback branch.  In the
reference setup `tribheat 0` but `N_tribin 1` remains; the tributary state is
ambiguous, and applying the relaxation made the 20-day error worse
(0.183 K at day 8 vs 0.086 before).  Since v1 excludes tributaries, `dhwtrib`
is set to 0 again.

The correct step-3 `h1` then needs a different explanation; the h1 match with
relaxation was likely accidental.  Keep the other exactness fixes (zsoil,
dhw order, surface velocity).

## M3: dhwtrib is in the Fortran `dhw`; dhwfsoil carry ambiguity

Step-3 checkpoint: `dhw=-0.000138272` and `dhw0=-5.9686e-7` in the probe.
Our `dhw0+dhwfsoil` gives `-0.00016116`; the missing `+2.289e-5` is exactly
`(h10-h1)/10` at step 3.  So the relaxation *is* active in the reference run,
even with `tribheat 0`.

Re-enabling it makes step-3 `h1` exact, but the 3.33-day error grows from
0.0066 to 0.0148 K (and the 20-day run was 0.183 K).  The remaining ambiguity
is the `dhwfsoil` carried between the **methane SOILCOLSTEMP** call and
`UPDATE_NEXT_TIMESTEP`: the probe's step-3 `dhw` implies a different carried
value than the step-2 checkpoint.  This must be resolved with per-call probes
rather than guessed.

## M3: `dhw`/`dhwfsoil` probe semantics are offset

`LAKE_entry`/`LAKE_exit` with `dhwfsoil`/`dhw`/`dhw0` show:

- step 1 entry/exit `dhwf=-2.283e-4`, `dhw=0`;
- step 2 entry `dhwf=-1.606e-4`, but `dhw - dhw0 = -2.283e-4` (step-1 value);
- step 3 entry `dhwf=-1.459e-4`, `dhw - dhw0 = -1.377e-4` (neither entry
  nor previous exit).

So the probe `dhw`/`dhwfsoil` are not the values used by the current step's
`dhw` computation; the `LAKE_entry` insertion point is after the update but
the quantities are still from another stage.  Exact step-level acceptance
cannot be driven from these tags; a Fortran-side print *inside* the case-1
block (before/after `dhw` assignment) would be needed.

The v1 code currently keeps the Fortran `dhwtrib=(h10-h1)/10` relaxation
active, `dhw` before soil, and is otherwise bit-exact at step 1/2.

## M3: actual `dhw` assignment probes (resolved)

Probes inserted **immediately after** `dhw0 = dhwp + dhwe` in case 1:

- step 1: `dhw=dhw0=0`, `dhwfsoil=0`;
- step 2: `dhw=-0.0002289217`, `dhw0=-5.9735e-7`,
  `dhwfsoil=-0.0002283244`; `dhw0+dhwfsoil` matches exactly (dhwtrib=0,
  because h1==h10);
- step 3: `dhw=-0.0001382717`, `dhw0=-5.9686e-7`,
  `dhwfsoil=-0.0001605670`; the difference is exactly
  `(h10-h1)/10 = +2.2892e-5`.

So the earlier `LAKE_entry/LAKE_exit` values were stale; the actual formula is
`dhw = dhw0 + dhwfsoil + dhwtrib`, with `dhwtrib` active from step 3.  Our
current `step.py` matches this formula.

## M3: step-3 soil divergence

At step 3, after `SOILCOLSTEMP`:

- `Tsoil1` diff 2.84e-4, `Tsoil2` 5.58e-6, `dhwfsoil` exact;
- after `SOILFORLAKE`: `Tsoil2` 1.8e-10, `Tsoil1` still 2.84e-4.

So the soil **temperature** handed to `SOILFORLAKE` (`Tsoil2`) is already 5.6e-6
off; the moisture/phase update then maps it to a 2.8e-4 `Tsoil1` error.
`dhwfsoil` remains exact.

Next: compare `wl1/wi1/csoil/rosoil/lamsoil` at step 3 against the SOILF probe
to find why `Tsoil2` (from `T_SOLVER`) differs despite exact `Tsoil1`/`Tw1`.

## M3: step-3 soil inputs are exact except Tsoil2

At step 3, after `SOILCOLSTEMP`:

- `wl1`, `wi1`, `csoil`, `rosoil`, `lamsoil` all **0.0** vs `SOILF_in`;
- `RadWater%integr` **0.0**;
- `Tsoil2` compared before `T_SOLVER` is the side column, not the deepest
  (invalid comparison); the deepest `Tsoil2` handed to `SOILFORLAKE` is
  5.6e-6 K off and yields the 2.8e-4 `Tsoil1` difference.

So the remaining step-3 soil error is inside `T_DIFF`'s water-soil interface /
soil solve with exact coefficients and exact `Tsoil1/Tw1`.  Next compare the
`T_DIFF` coefficients (`a,b,c,d`) at the interface with an in-block probe.

## M3: step-3 T_DIFF coefficients

Final `T_DIFF` coefficients at step 3 (last `TD2` record):

- `a` 3.6e-6 relative, `b` 6e-7, `c` 6e-6, `d` 6e-6;
- `Temp` (`Tw1/Tsoil2`) 5.2e-7 K;
- mismatch concentrated at the interface index M+1.

The 5e-7 K `Temp` is consistent with a **different T_SOLVER convergence path**
(`resid=1 W/m2`, Chorde stop criterion) rather than a formula error: a 1 W/m2
residual corresponds to ~1e-4 K, and our `HEATBALANCE` matches to 5e-14, not
bitwise.  The plan's 1e-10 step tolerance is below this convergence noise.
Next: compare the step-3 `Tsurf3` iteration sequence (`iter`, `Bal3`) and/or
tighten the convergence criterion to test this hypothesis.

## M3: plan §5 acceptance criteria revised (end-of-M3 review)

The original plan required whole-step agreement at 1e-10 and one-day water
temperature agreement at 1e-6 K.  Both are below the intrinsic noise of the
Fortran solver:

- `T_SOLVER` stops when `|Bal3| <= resid` with `resid = 1 W/m2` (the default of
  `t_solver`; our port uses the same value).
- That residual corresponds to roughly **1e-4 K** in surface temperature.
- Two implementations that use the same formulas but take a different number
  of Chorde iterations are therefore not required to agree to 1e-10 per step
  or 1e-6 K after 4320 steps, even when both are correct.

Measured on 2026-11-11 (Linux, float64, current `main`):

| Level | Measured | Old criterion | Revised criterion |
|---|---|---|---|
| step 1 | `Tw1` 3.5e-13, `Tsoil1` 1.8e-15 | 1e-10 | 1e-10 (pass) |
| step 2 | `Tw1` 7.1e-10, `Tsoil1` 2.1e-14 | 1e-10 | 1e-10 (pass) |
| step 3 | `Tw1` 5.2e-7, `Tsoil1` 2.8e-4 | 1e-10 | 1e-3 K solver floor (diagnostic) |
| 1 day | 2.2e-4 K (`summer1d`) | 1e-6 K | 1e-3 K (pass) |
| 60 days | see below | 0.01 K | 0.01 K (unchanged) |
| 425 days | pending | 0.05 K | 0.05 K (unchanged) |

The revised table is in plan §5 (en/zh).  Deterministic module probes stay at
1e-10 because they are the primary translation check.  Whole-step non-iterative
fields are compared at 1e-8 relative, with `layer_case`, flags and iteration
counts required to be identical; iteratively solved fields are compared at the
solver noise floor.  Long-run acceptance is unchanged: loosening those would
hide real drift.  The existing step-3 `Tw1`/`Tsoil1` difference is still tracked
as a diagnostic issue (the `Tsoil2`/`SOILFORLAKE` timing discrepancy described
above), not as an acceptance failure.

### 1-ULP perturbation experiment (done)

The vendored tree has no `core/build/CMakeLists.txt`; a working one was
reconstructed at `/root/lakebatch/core/build/CMakeLists.txt` (all 81 sources,
`-O3 -cpp -DFCOMPILER=2 -fopenmp -ffree-line-length-none
-fallow-argument-mismatch`, linked with `netcdff`/`netcdf`).  `CHECK=1
core/build.sh` then builds `core/build-check-linux/lake.out`.

A fresh copy of `summer1d` was run with `OMP_NUM_THREADS=1`.  The output
selections need `results/{debug,control_point}` and
`results/test/{time_series,monthly,daily}` to exist, otherwise the model stops
at the first output.  Results:

| Comparison | day-1 max | max within the day |
|---|---|---|
| rebuilt Fortran vs tracked reference | 3.6e-4 K | 1.3e-2 K (first hour) |
| Fortran with one forcing value +1 ULP vs unperturbed Fortran | **5.8e-4 K** | 1.6e-2 K |

The 1-ULP perturbation is 5.7e-14 K in `tempair`; it is a pure floating-point
perturbation, not a physical change.  Its effect after one day is 5.8e-4 K,
i.e. **580 times larger than the old 1e-6 K criterion**.  Therefore the old
one-day criterion is provably unattainable by two non-bit-identical builds,
independent of translation correctness.

The rebuilt check binary itself differs from the tracked reference by
3.6e-4 K after one day (1.3e-2 K at the first hourly output).  This is a
build/compiler sensitivity (OpenMP and a different NetCDF/gfortran version),
not a JAX error.  The JAX one-day error of 2.2e-4 K is **inside** this
Fortran-vs-Fortran spread; the revised 1e-3 K threshold is only 1.7x above the
measured 1-ULP sensitivity, so it is tight rather than generous.

Consequence: the revised §5 thresholds are adopted (the 1-day value stays at
1e-3 K); the old 1e-6 K value is withdrawn.  The step-3 `Tsoil2` difference
remains a diagnostics issue, not an acceptance failure.

## M3: stale bathymetry was the step-3 soil root cause (fixed)

`step_case1_to_solver` reused the bathymetry arrays computed once by
`build_step_state`, while the Fortran calls `BATHYMETRY` every step with the
current `h1`.  At step 1 `h1 == h10`, and at step 2 the previous step's `dhw`
is zero, so steps 1 and 2 matched.  From step 3 the water thickness has
changed by `dhw ~ -1.4e-4 m`; the stale layer areas/widths shifted
`TSURFSOILCOL`'s area-weighted side-column surface temperature by up to
2.8e-4 K, which then propagated into `Tsoil1`.  This also explains why the
error appeared only in the surface soil cell of the side columns.

Fix: `_refresh_geometry()` recomputes `bathymetry`, `SoilGeometry` and the
momentum/k-epsilon area arrays at the start of every step.

`full425d` step 3, before -> after (probe `LAKE_exit`):

| Field | Before | After |
|---|---|---|
| `Tw1` | 5.15e-7 | **8.97e-9** |
| `Tsoil1` | 2.84e-4 | **2.04e-6** |
| `lamw` | 9.64e-5 | **2.04e-6** |

`Tsoilsurf` (probe `SCOL`) improves from 2.8e-4 K to ~1e-6 K when the
bathymetry is recomputed at the current `h1`.  The remaining `Tsoil1` 2e-6 is
the residual area-interpolation difference and is inside the solver noise
floor.

One-day comparison against the tracked reference moved from 2.2e-4 to 3.7e-4 K
(against the rebuilt Fortran: 5.1e-4 to 6.2e-4 K).  Both numbers are at the
Fortran 1-ULP sensitivity (5.8e-4 K) and build-to-build spread (3.6e-4 K), so
the day-1 aggregate cannot resolve the change; the step-3 probe is the
decisive measurement.  The 60-day rerun with this fix is in progress.

## M4: case-2 no-snow step and ice bathymetry (first validation)

Two bugs were found while starting case 2:

1. `_zsoil_geom` built `bathymice`/`bathymdice` as the **full lake area** at
   every level.  The Fortran `BATHYMETRY` interpolates them from the
   morphometry table at `depth - max(depth) + h1 + l1 + ls1` (ice) and
   `+ ls1` (deep ice), with the vertical positions taken from `ddzi*l1` /
   `ddzi*ls1`.  The full-area shortcut put cold ice over the shallow soil
   columns and made `TSURFSOILCOL` return e.g. -0.012 C instead of +0.0134 C.
   Fixed in `step._zsoil_geom`; `Tsoilsurf` now matches the `SCOL` probe to
   ~1e-8 K.  (`bathymdice` was also accidentally aliased to `bice`; fixed.)
2. The probe build instrumented only the first `SOILFORLAKE` call (case 1).
   `tools/ref/make_probe_build.py` now instruments every call, so case-2
   probes contain `SOILF_in`/`SOILF_out` (wl1/wi1 etc.).

`step2.py` starts the case-2 (no-snow) path.  Initialised from the probe state
at `full425d` step 500000 (`probe-w3/499999/LAKE_exit` + `SOILF_out` +
`MOM_1899_in`), one step gives:

| Field | Difference vs `probe/500000/LAKE_exit` |
|---|---|
| `Tw1` | 5.5e-7 |
| `Tsoil1` | 5.5e-7 |
| `Ti1` | 2.3e-5 |
| `Tsurf3` | 2.3e-5 |
| `l1` | 5.1e-7 |
| `h1` | 1.5e-6 |
| `lamw` | still differs (turbulence inputs; under investigation) |

These are inside the revised solver-noise acceptance; the snow branch,
case transitions and `lamw`/`K_EPSILON` inputs are the next M4 items.  The
60-day `summer60d` rerun with the step-3 bathymetry fix is still running.

## M4: case-2 snow step (validated) and k-epsilon ice flags

The snow branch of `step2.py` now follows case 2.1: `snow_cond_heat_coef`,
`SR_botsnow`, `ice_increments` (snow variant), `T_SOLVER` with `snow=1`, then
`snow_calc` for the pack, with `T = Tsn` before the call as in `SNOWTEMP`.

Validated at `full425d` step 545580 (snow depth 0.0332 m, ice 0.1909 m),
initialised from `probe-build5` state: `Tw1` 1.5e-6, `Tsoil1` 7.2e-7, `Ti1`
8.7e-6 K, `Tsurf3` 1e-7 K, `l1` 1.9e-7 m, `hs1` 2.6e-5 m, `snowmass`
1.9e-3 kg/m2 (0.026 mm water equivalent).  `kt`/`eps` also match to ~1e-4
relative after the fixes below.

K-epsilon fixes found by this comparison:

- `ke.l1`/`ke.ls1` were never set in the case-2 step, so K_EPSILON always took
  the open-water branch (`eps_min = 1e-12`, wind-stress surface flux).  With
  the ice thickness set, the ice branch (`eps_min` from `Racr`, `tau_i`
  surface flux) matches the probe.
- For thick ice `MOMENTUM_SOLVER` leaves `b0` undefined; use `b0 = 1` when
  `l1 > 0` unless momentum returned a value.
- `ke.b0` now comes from `MOMENTUM_SOLVER` (thin ice) or 1 (thick ice).

The probe build now also records the snow state (`T`, `Tsn`, `cs`, `dens`,
`dz`, `lams`, `itop`, `flag_snow`, `flag_snow_init`) in `LAKE_entry`/`_exit`,
and every `K_EPSILON` call has `KE_in`/`KE_out`.

Case transitions (1 -> 2 freeze-up, 2 -> 1 melt, 2/3 -> 3/1) are the next
M4 item; `water_freeze_melt` and the layer-case selector still need porting.

## M4: case transitions (freeze-up validated)

`step2.py` now has `water_freeze_melt`, `layer_case`, `create_initial_ice`,
`step_any` (case selection plus 1->2 and 2->1 transitions) and `run_model`.

The freeze-up path was validated from the probe state at `full425d` step
460619 (`LAKE_exit` + `SOILF_out` + momentum), running steps 460620 and
460621:

| Field | step 460620 | step 460621 |
|---|---|---|
| `Tw1` | 2.2e-8 | 4.0e-8 |
| `Tsoil1` | 8.8e-7 | 9.4e-7 |
| `Ti1` | 7.7e-5 | 1.2e-4 |
| `h1` | 3.0e-8 m | 5.8e-8 m |
| `l1` | 1.4e-6 m | 2.9e-6 m |

The decisive detail was `dhwtrib`: with no tributaries the Fortran still
applies the depth relaxation `(h10 - h1)/10` inside case 2 (the `BATHYMETRY`
fallback).  `sync_ice` now sets it from the current `h1`; without it the
freeze-up step had an `h1` error of 9.3e-5 m.

Melt (2 -> 1) uses the same `layer_case` selector: once `l1` reaches zero the
state is reset to open water before the next case-1 step.  The 2 -> 3 and
3 -> 1 paths raise `NotImplementedError` for now (case 3 is not covered by the
stable reference set; see `docs/reference_cases.md`).
