# State inventory and classification

Plan §3.4: everything a step carries to the next step must be in the JAX
`State`, or `NOTES.md` must explain why it is not. This document records the
inventory, the classification and the Fortran → JAX index mapping.

Sources (authoritative, generated from the declarations):

- `lakebatch/core/source/model/lb_restart_io.f90`: every variable the restart
  saves, 561 records in 13 state modules. This is a **superset** of the true
  state: it also saves work arrays so that a restart is bit-exact.
- `lakebatch/core/source/model/lb_restart_lake.inc`: the 80 `SAVE` locals of
  `SUBROUTINE LAKE`.
- `lakebatch/core/source/model/lb_point_ctx.f90` and `lb_ctx_<routine>.inc`:
  the per-point swap list (plan stage 1). This is a superset too; it is the
  list of variables that persist between calls, not only between steps.

The plan's rule: a variable is **carried state** only if it is read before it
is written within a step; otherwise it is a **work** array. v1 starts with the
prognostic arrays and the module/`SAVE` state marked `S` below; every `W`
variable is recomputed. The M1 test (JAX `init_state` vs the probe at the
entry of step 1) and the per-routine probes in M2/M3 are the check: a missing
read-before-write state shows up as a mismatch and is then added.

## 1. Prognostic arrays (State, `S`)

Shapes follow `M`, `Mice`, `ns`, `nsoilcols`. `(M+1)` is the water column
including the surface and bottom half-levels; `(Mice+1)` is the ice column;
soil is `(ns, nsoilcols)`.

| Fortran | Shape | JAX shape | Notes |
|---|---|---|---|
| `Tw1`, `Tw2` | `(M+1)` | `(L, M+1)` | water temperature |
| `Ti1`, `Ti2` | `(Mice+1)` | `(L, Mice+1)` | ice temperature |
| `Tis1` | `(Mice+1)` | `(L, Mice+1)` | snow/ice interface temperature |
| `Sal1`, `Sal2` | `(M+1)` | `(L, M+1)` | water salinity (fresh in v1) |
| `Sals1` | `(ns, nsoilcols)` | `(L, ns, nsoilcols)` | soil salinity |
| `Tsoil1` | `(ns, nsoilcols)` | `(L, ns, nsoilcols)` | soil temperature (probe: 10×5) |
| `u`, `v` | `(M+1)` | `(L, M+1)` | horizontal velocities |
| `E`, `eps` | `(M+1)` | `(L, M+1)` | TKE and dissipation |
| `lamw` | `(M)` | `(L, M)` | vertical heat diffusivity |
| `k_turb_T_flux` | `(M+1)` | `(L, M+1)` | turbulent heat flux |
| `dz` | `(M+1)` | `(L, M+1)` | layer thickness (from `h1`) |
| `wl`, `dens` | `(M+1)` | `(L, M+1)` | layer thickness/density working copy |
| `Tskin` | `(M+1)` | `(L, M+1)` | skin temperature (skin 0 in v1) |
| `salice`, `porice` | `(Mice+1)` | `(L, Mice+1)` | ice salinity/porosity |
| `h1`, `l1`, `hs1`, `ls1` | scalar | `(L,)` | water/ice/snow/deep-ice thickness |
| `snowmass`, `snmelt` | scalar | `(L,)` | snow mass and melt |
| `dhw`, `dhw0`, `dhi`, `dhi0`, `dls0`, `dhwfsoil` | scalar | `(L,)` | thickness increments |
| `roughness`, `cdm`, `velfrict`, `eflux0_kinem`, `Elatent` | scalar | `(L,)` | surface exchange state |
| `i_maxN`, `itherm`, `itop` | integer | `(L,)` | indices from turbulence/mixed layer |
| `Eseiches` | scalar | `(L,)` | seiche energy |
| `time`, `nstep` | scalar | `(L,)` | clock (see §4); `nstep` is an integer |
| `fl_sn`, `fl_sn_init` | logical | `(L,)` | snow flags |

`T` (soil temperature in the `UPDATE_*` argument list) and `wl` need their
exact declaration confirmed when `state.py` is written; the probe records the
shapes once they are added to the entry probe.

## 2. Module-level state (`S` / `W` / `C`)

The full variable list is in `lb_restart_io.f90`. This table classifies the
modules and the important names; the rest of each module is `W` until the
probe proves otherwise.

| Module | Classification | Important names |
|---|---|---|
| `ARRAYS_WATERSTATE` | `S` | `Tw1`, `Ti1`, `Tis1`, `Sal1`, `Tskin`, `k_turb_T_flux` (the prognostic arrays) |
| `ARRAYS` | `S`/`W` | `S`: `Tsoil1`, `Sals1`, `wi1`, `wl1`, `soilflux`, `Tsoil3`, `salice`, `porice`; `W`: `Tsoil2`, `Sals2`, `wi2`, `wl2`, `Tsoil2` scratch, `work` arrays |
| `ARRAYS_TURB` | `S`/`W` | `S`: `E`, `eps`, `k2` (previous diffusivity), `E2`, `eps2`; `W`: `KT`, `Ri`, `Rp`, `Rpdens`, `row`, `CE`, `CEt`, `lam_E`, `lam_eps`, most of the 101 restart names |
| `ARRAYS_BATHYM` | `W`/`C` | `W`: `bathymwater`, `bathymice`, `bathymdice`, `bathymsoil`, `area_int`, `area_half`, `vol`, `depth_area`, `cellipt` (recomputed by `BATHYMETRY` each step); `C`: the morphometry table and `area_lake` (Params/Grid) |
| `EVOLUTION_VARIABLES` | `S`/`W` | `S`: `i_maxN`, `itherm`, `itop`, flags that survive a step (`arrays_allocated` is config); `W`: output accumulators (`total*`, `*_sum`), diagnostics |
| `ATMOS` | `S`/`W` | `S`: `SurfTemp` if it is an input that survives; `W`: forcing scalars and fluxes recomputed each step (`tempair`, `humair`, `pressure`, `uwind`, `vwind`, `precip`, `longwave`, `shortwave`, `netrad`, `sensible`, `latent`, `SensFlux`, `LatentFlux`, `surfrad`, `cloud`, ...) |
| `BL_MOD_LAKE` | `S` | `i_ML`, `H_mixed_layer`, `maxN`, `itherm` (mixed-layer state; recomputed by `MIXED_LAYER_CALC` but used later in the step) |
| `SNOWSOIL_MOD` | `S` | snow/soil state: `snowmass`, `hs1`, `ls1`, `Tis1`, soil flags; see the 19 restart names |
| `ARRAYS_GRID` | `C`/`W` | `C`: `M`, `Mice`, `ns`, `nsoilcols`, grid `dzeta*`, `dz_*`, `z_*`, morphometry; `W`: `gs%ix/iy/isoilcol`, per-step indices |
| `LB_SAVED_LOCALS` | `S` | `keps_init_keps`, `tsolver_Tsurf_prev`, `tsolver_surftyp_prev`, `tsolver_total_Bal` (the oxygen variables are out of scope) |
| `ARRAYS_METHANE`, `ARRAYS_OXYGEN` | out of scope | not translated; their outputs do not feed back (M0.4) |

## 3. `SAVE` locals of routines (from `lb_ctx_*.inc`)

| Routine | `S` (carried) | `W` (work) / `C` (constant) |
|---|---|---|
| `LAKE` | `snowmass`, `snowmass_init`, `layer_case` (recomputed but used), `flag_snow`, `i_ML`, `kor`, `dt`, `b0`, `tau_*`, `eflux0_kinem_water`, `total*` accumulators (output), `add_to_winter` | `work1..zzz`, `x..zzz`, `flux1/2`, `i..k`, `dnx/dny`, `npoint_calibr`, `calibr_par_*`, `control_point_read`, `worklog`, ... |
| `K_EPSILON` | `keps_init` (in `LB_SAVED_LOCALS`), `i_entrain`, `i_entrain_old`, `cycle_keps`, `next_tstep_keps`, `iter`? | `a..d`, `AG`, `dt05`, `wr`, `dhw_keps`, `pi`, `ufr`, `ACC*`, `dE_it`, `deps_it`, `xx..`, coefficients `C_eps*`, `CE`, `CEt`, `lam_E`, `lam_eps` (constants/config), `FS/FTKES/...`, `iter/iterat`, `numt`, `tp_name` |
| `MOMENTUM_SOLVER` | `k2_old` (in `ARRAYS_TURB` per point), `itherm_new`? | `CE`, `a..d`, `am_..dm_`, `row1/2`, `wr`, `ufr`, all coefficients, `iter`, `numt`, `Amatrix*`, `Lxi..`, `time1/2` |
| `T_SOLVER` | `Tsurf1/2/3`, `Bal1/2/3` (bracket the surface iteration inside a step), `firstcall` (per point), `tsolver_*` in `LB_SAVED_LOCALS` | `dt_scan`, `resid_Bal_max`, `iter`, `maxiter`, `Tbc_switch`, `n_unit`, `iterate` |
| `T_DIFF` | — | `a,b,c,d,Temp,cc,xx,i,j` |
| `DIFF_COEF` | — | `lam1/2`, `dzmean`, `aa,bb`, `i,j` |
| `SURF_CHAR` | — | `x1..x5` |
| `SNOWTEMP` | — | `xx`, `dzsn`, `rofresh`, `t2`, `pheat`, `pmelt`, `CCT` (constants), `i`, dates |
| `SNOW_CALC` | — | `i,j,k` |
| `SOILFORLAKE` | `firstcall`? (per point) | `dzmean`, `ma,mi`, `wflow`, `wfhigh`, `lammoist1/2`, `surplus`, `Potphenergy`, `watavailtofreeze`, `filtr_low`, `dhwfsoil1..4`, `max1`, `work`, `mhsep`, `alp`, `psiit`, `pressoil`, `iter` |
| `COMSOILFORLAKE` | — | `b`, `psi_max`, `Porosity`, `gamma_max`, `lambda_max`, `W_0`, `W_m`, `pow` (constants/config), `i,j`, `firstcall` |
| `SENSLATMOM_FLUXES` | — | `TET2`, `TET1`, `ro`, `esatsurf`, `humsurf`, `bx`, `bix`, `c_u`, `c_t`, `ksw`, `hwave`, `SHF`, `LHF`, `bx4`, `bix4`, `xx..`, `deltaR/deltaT`, `wr_st`, `wr_n` |
| `TRIBUTARIES` | out of scope v1 | `itriblev`, `inout_fluxes`, `trib_allocated`; tributaries are off |

`S` here means "start in State; confirm with the probe". `W` means "recompute
each step"; `C` means "constant or setup keyword → Params/Grid".

## 4. Fortran → JAX index mapping

JAX arrays are 0-based. The rule is `jax_index = fortran_index - lower_bound`.

| Fortran | Bounds | Length | JAX |
|---|---|---|---|
| `Tw1`, `Sal1`, `u`, `v`, `E`, `eps` | `1:M+1` | `M+1` | `[0:M+1]` (JAX index = Fortran index − 1) |
| `Ti1`, `Tis1`, `salice`, `porice` | `1:Mice+1` | `Mice+1` | `[0:Mice+1]` |
| `Tsoil1`, `Sals1` | `1:ns, 1:nsoilcols` | `ns × nsoilcols` | `[0:ns, 0:nsoilcols]` |
| `lamw` | `1:M` | `M` | `[0:M]` |
| `ddz05` | `0:M` | `M+1` | `[0:M+1]` (JAX index = Fortran index) |
| `ddzi05` | `0:Mice` | `Mice+1` | `[0:Mice+1]` |
| `bathymsoil` | `0:nsoilcols+1` (per `ix,iy`) | `nsoilcols+2` | `[0:nsoilcols+2]`; index 0 is the surface? confirm with `BATHYMETRY` |
| `fbbleflx_*` | `0:M+1` | `M+2` | out of scope v1 |
| `z_full`, `z_half` | 1-based in the model | — | `[0:]`; used for output depth interpolation |

`storage_size`/probe records keep the Fortran lower bounds, so
`tools/ref/probe.py:read_probe_full` can reconstruct the mapping when a shape
is uncertain.

## 5. Verification plan

1. **M1:** build `init_state` from `INIT_VAR` + the first-call constants, then
   compare every entry-probe variable at step 1 (1e-12). Add any mismatch to
   `S`.
2. **M2/M3:** for each translated routine, insert per-call probes (before and
   after) and compare the routine's inputs/outputs at the steps in
   `tools/ref/steps.toml` (1e-10).
3. **Whole step:** compare the entry/exit probe state at the selected steps;
   any variable that differs is either a translation error or a missing `S`.
4. **Long run:** the 425-day cases; a missing `S` usually shows up at the
   first case change (freeze-up, snow, melt).

## 6. Open items

- `k2`/`E2`/`eps2`/`Ri` in `ARRAYS_TURB`: `k2` is very likely state (previous
  diffusivity, the per-point `k2_old` fix); `Ri`/`Rp`/`Rpdens`/`row` look like
  work. Confirm with the probe at a summer step and at a step just after ice
  formation.
- `MOMENTUM_SOLVER`'s `CE`, `itherm_new`, `time1`, `time2`, `numt`, `iter`:
  the stage-1 audit called them "most likely work, not proven"; probe them.
- `T_SOLVER`'s `Tsurf1/2/3`, `Bal1/2/3`: they bracket the surface iteration
  within one call; if the iteration is restarted from them on the next call,
  they are state. `tsolver_Tsurf_prev`/`surftyp_prev` in `LB_SAVED_LOCALS` are
  definitely state.
- `SNOWTEMP`'s `firstcall`/`CCT` and `SOILFORLAKE`'s `firstcall`: first-call
  constants vs per-point flags; compare the first call of two lakes.
- `ATMOS::SurfTemp`: is it an input that survives a step, or a per-step
  diagnostic? Check `UPDATE_CURRENT_TIMESTEP`/`UPDATE_NEXT_TIMESTEP`.
- The output accumulators in `EVOLUTION_VARIABLES` and `out_mod` (hourly,
  daily, monthly) are not needed by v1's own output, but the reference
  comparison reads the Fortran output; they do not enter State.
