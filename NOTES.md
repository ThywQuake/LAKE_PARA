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
