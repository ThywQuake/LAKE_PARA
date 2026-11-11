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
