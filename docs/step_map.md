# Step map: one `SUBROUTINE LAKE` call

Source: `lakebatch/core/source/model/lake.f90` at lakebatch commit `6657ebc`
(branch `rewrite/phased-step`, 2026-11-11). Line numbers below are from that
file. `SUBROUTINE LAKE` starts at 169 and ends at 3212.

This map covers what version 1 of `lakematrix` must reproduce: the
thermodynamic core, cases 1–3, and the shared prologue. Case 4 (2087–2162) is
not operational in the original model (it stops with an error) and is not
translated.

## How to read this map

- **State** means the per-lake arrays listed in plan §3.4, loaded at 811
  (`UPDATE_CURRENT_TIMESTEP`) and stored at 2575 (`UPDATE_NEXT_TIMESTEP`).
- The “state” column lists the primary arrays read (R) and written (W) by a
  call or block. It is a roadmap, not a complete dataflow: the probe build
  (M0.4) records the exact module-variable I/O and is authoritative.
- **In scope** calls are in the v1 thermodynamic core. *Out of scope* calls
  (biogeochemistry, BUBBLE, tributaries, FLake, data assimilation) are kept in
  the map because they sit between in-scope calls and may initialize arrays
  that are read later; the M0.4 experiment decides whether any of them feeds
  back into thermodynamics.
- Conditions are abbreviated. The full expression is on the source line.
- Fortran lower bounds: `Tw1(1:M+1)`, `Ti1(1:Mice+1)`, `ddz05(0:M)`,
  `ddzi05(0:Mice)`, `fbbleflx(0:M+1)`, `bathymsoil(0:nsoilcols+1,ix,iy)`.
  The JAX (0-based) mapping is in `state.md`.
- A call may occur more than once per step (inside iterations or soil-column
  loops). The step map lists the call site; the internal loop structure is in
  plan §3.3 and must be reproduced exactly.

## 0. Shared prologue (all layer cases), 530–915

| Line | Call / block | Condition | State |
|---|---|---|---|
| 532–538 | `RESTART_OPEN` / `LB_MODULES_IO` / `RESTART_CLOSE` | `firstcall .and. RESTART_IN_PRESENT()` | restart file; **out of scope** v1 |
| 618–621 | `dataread` check | `.not.dataread .and. PBLpar%par /= 0` | STOP; setup validation |
| 633–637 | `SurfTemp` from `SurfTemp1` | `SurfTemp1 /= missing_value` | W `SurfTemp` |
| 644–648 | `kor` from `kor_in` or `phi` | `kor_in == -999` | W `kor` |
| 652–657 | radiation extinction coefficients | `nbands > 1` vs 1 band | W `RadWater%extinct`, `RadIce%extinct`, `RadDeepIce%extinct` (from `extwat`, `extice`, `extwatbands`) |
| 660–666 | shortwave/longwave selection | `ifrad%par` 0 or 1 | R `longwave1`, `shortwave1`; W `longwave`, `shortwave`; **v1: `ifrad 1` only** |
| 668–697 | net radiation / missing-radiation path | `longwave1` / `shortwave1` / `cloud1` missing | calls `SHORTWAVE_RADIATION`, `LONGWAVE_RADIATION`; W `netrad`, `shortwave`, `longwave` |
| 705–730 | forcing range checks | per variable | STOP on out-of-range/non-finite forcing; no state change |
| 752–796 | `INIT_VAR` | `init(ix,iy) == 0` (first call) | initial `Tw1`, `Ti1`, `Tis1`, `Tsoil1`, `Sal1`, `h1`, `l1`, `hs1`, `ls1`, `snowmass`, `flag_snow`, `flag_snow_init`, ...; **first call only** |
| 801–806 | `CONTROL_POINT_IN` | control-point mode | **out of scope** v1 |
| 811–843 | `UPDATE_CURRENT_TIMESTEP` | always | R `*_2d`/`*_3d` state; W the 1-D working arrays (`Tw1`, `Ti1`, `h1`, `l1`, `hs1`, `ls1`, `u`, `v`, `E`, `eps`, `lamw`, ...) |
| 846–854 | `Tw1` sanity loop | always | STOP if NaN or `|Tw1| > 100` |
| 859–864 | `CONTROL_POINT_OUT` | `icp == 1` | **out of scope** v1 |
| 868–870 | `BATHYMETRY` | always | R `month`, `day`, `lakeform`, `hour`, `dt`, `trib_inflow`; W `depth_area`, `area_lake`, `cellipt`, `dhwtrib`, `vol`, `botar`, `multsoil`, `bathym*` |
| 872–876 | tributary inflow flag | `trib_inflow(1) /= -9999` | W `dhwtrib`; **v1: turn tributaries off** |
| 881–912 | `layer_case` decision | see §3.2 of the plan | W `layer_case`, `flag_ice_surf`, `flag_snow`; may create a thin water layer in case 3 → 1 |

## 1. Case 1: open water (`layer_case == 1`), 916–1237

| Line | Call / block | Condition | State |
|---|---|---|---|
| 920–928 | bottom freezing check (`WATER_FREEZE_MELT`) | bottom water below freezing, `ls1 == 0`, enough water | W `ls1`, `Tis1`, `h1` (thin ice layer while still in case 1) |
| 931–934 | `BATHYMETRY` | always | R `depth_area`, `area_lake`, `cellipt`, `multsoil`, `trib_inflow`, `dhwtrib`; W `vol`, `botar`, `bathymwater`, `bathymsoil`, `area_int`, `area_half` |
| 936–938 | water temperature floor | `i = 2..M` | W `Tw1` (at least the melt point) |
| 941 | `DENSITY_W` | always | R `Tw1`, `Sal1`, `preswat`; W `row` |
| 942 | `MIXED_LAYER_CALC` | always | R `row`, `ddz`, `ddz2`, `dzeta_int`, `dzeta_05int`, `h1`; W `i_ML`, `H_mixed_layer`, `maxN` |
| 952–957 | `TRIBTEMP`, `VERTADV_UPDATE` | `tribheat%par > 0` and not the `-9999` no-inflow case | R/W `Tw1`, `Sal1`, `qwater`, `h1`, `dhwtrib`; vertical advection of heat/salinity/gases. **v1: tributaries off, so `TRIBTEMP` skipped; `VERTADV_UPDATE` stays** |
| 989–994 | albedo | `varalb%par` 0/1 | W `albedo` (or `albedoofice`/`albedo`); **v1: `varalb 1`** |
| 1000 | `RadWater%RAD_UPDATE` | always | R `ddz05(0:M)*h1`, `radflux`, `extwat`; W `RadWater%flux`, `RadWater%integr` |
| 1002–1012 | photic-zone depth | `RadWater%integr(1) > 0` | W `H_photic_zone`, `i_photic` (or `itherm`-like index) |
| 1013–1017 | `RadDeepIce%RAD_UPDATE` | `ls1 /= 0` | R `ddzi05(0:Mice)*ls1`; W `RadDeepIce%flux`, `RadDeepIce%integr` |
| 1020–1045 | layer thickness increments | `ls1 == 0` vs not | R `precip`, `Elatent`, `dhwfsoil`, `dhwtrib`, `lamw`, `Tw1`, `Tis1`, `RadWater%integr`, `RadDeepIce%integr`, `PEMF`, `pt_down_f`; W `dhw`, `dhw0`, `dls`, `dls0`, `dhwls` |
| 1048–1052 | `MOMENTUM_SOLVER` | `Turbpar%par == 2 .or. 5` | R `u`, `v`, `h1`, `Tw1`, `Sal1`, `row`, `wind`, `tau_air`, `fetch`; W `u2`, `v2`, `hx1/hx2/hy1/hy2`, `hx1t/hx2t/hy1t/hy2t`, `gradpx_tend`, `gradpy_tend`, `Eseiches`, `cdm`, `velfrict` |
| 1054–1056 | `SOIL_COND_HEAT_COEF` | `soilswitch%par == 1` | R `Tsoil1`, `por`, `wst`, `gas`; W soil heat coefficients (`csoil`, `rosoil`, `dsoil`?) |
| 1057–1068 | `SOILCOLSTEMP`, `LATERHEAT` | `multsoil` | R/W `Tsoil1`, `soilflux`, `fdiffbot`, `fdiff_lake_surf`, `bathymsoil`, `wst`, `gsp`, `ftot` |
| 1070–1071 | `T_SOLVER` | always | R `Tw1`, `Sal1`, `h1`, `RadWater`, `RadIce`, `SurfTemp`, `fetch`, `dt`; W `Tw1`, `Tbar`, surface temperature, `T_fluxes`; internally calls `T_DIFF` (13×), `DIFF_COEF` (27×), `SURF_CHAR` |
| 1072–1074 | `SOILFORLAKE` | `soilswitch%par == 1` | R `Tsoil1`, `a`, `b`, `c`, `d`; W `Tsoil1`, `Tsoil3`, `soilflux`, `fdiffbot` |
| 1077–1089 | molecular diffusivity arrays | always | W `lamsal`, `lammeth`, `lamoxyg`, `lamcarbdi`, `lampart` (scratch; used by biogeochemistry and `S_DIFF`) |
| 1091 | `S_DIFF` | always | R `Tw1`, `Sal1`, `lamw`, `lamsal`, `dt`, `ls`, `salice`; W `Sal1`, `Sals1`, `salice` (fresh water: `saltice 0`) |
| 1095–1153 | biogeochemistry | always (no master switch confirmed) | `CHLOROPHYLLA`, `OXYGEN_PRODCONS`, `METHANE`, `ADDOXPRODCONS`, `OXYGEN`, `CARBON_DIOXIDE`, `PHOSPHORUS`, `METHANE_OXIDATION`; **out of scope v1**; M0.4 decides whether any output feeds back into thermodynamics |
| 1104 | `SOILCOLSTEMP` (methane soil) | `soilswitch == 1 .and. multsoil` | R/W `Tsoil1`, `soilflux`; **inside the biogeochemistry block; out of scope, check feedback** |
| 1111 | `BUBBLE_BLOCK` | `soilswitch%par == 1`; inside, `ifbubble%par == 1` | R `Tw1`, `Sal1`, `h1`, `qwater`, `DIC`, `oxyg`; W `qwater`, `DIC`, `oxyg`, `fbbleflx_*`; **out of scope v1; use `ifbubble 0`** |
| 1117–1123 | `LATERHEAT` (CO2/O2 soil fluxes) | `multsoil` inside the biogeochemistry block | R/W `Tsoil1`, `soilflux`; **out of scope, check feedback** |
| 1158 | `DENSITY_W` | always | R `Tw2`, `Sal2`, `preswat`; W `row2` |
| 1160–1165 | `K_EPSILON` | `Turbpar%par == 2`, loop `i = 1, nstep_keps` | R `Tw1`, `u`, `v`, `h1`, `row`, `radflux`, `tau_air`, `tau_wav`, `tau_i`, `tau_gr`, `b0`, `fetch`, `roughness`; W `E`, `eps`, `k2`, `KT`, `Ri`, `i_ML`, `H_mixed_layer`, `maxN`, `itherm`, `itop`; internally calls `CE_CANUTO`, `CET_CANUTO`, `INTERPOLATE_TO_INTEGER_POINT` |
| 1166–1193 | other turbulence schemes | `Turbpar 5/8/9/1` | `RICHARDSON_NUMBER`, `EDDY_DIFF_MOST_RICH`, `ED_TEMP_HOSTETLER`, `ED_TEMP_HOSTETLER2`, `UnDenGrMix`; **v1: `Turbpar 2` only, reject others** |
| 1196–1208 | background diffusivity | `backdiff%par` 0/1/2/3 | `DIFFMIN_HS` (case 1) or `DIFFMIN_KPP` (case 2); W `lamw_back`; **v1: `backdiff 1`** |
| 1211–1213 | `lamw` assembly | always | W `lamw` (`lamw0 + KT + lamw_back`) |
| 1216–1220 | `TURB_SCALES` | always | R `gsp`, `ls`, `bathymwater`, `wst`, `RadWater`, `E`, `eps`, `h1`, `u`, `v`; W `KT`, `k2`, `E`, `eps`, `Ri`, `itherm`, `H_mixed_layer`, `maxN` |
| 1221–1231 | `ZERODIM_MODEL` | `zero_model%par == 1` | diagnostic 0-D model; **v1: `zero_model 0`, skip** |
| 1233–1236 | end-of-step thickness bookkeeping | always | W `h2 = h1 + dhw`, `ls2`, `l2` |

## 2. Case 2: ice over water (`layer_case == 2`), 1239–1894

Case 2 splits into **2.1 snow on ice** (`flag_snow == 1`, 1334) and **2.2 no
snow** (otherwise, 1613). The call sequence is nearly the same in both; the
inline ice/snow growth and melt code differs. The table lists the sequence
once, with the 2.1/2.2 line numbers. The duplicated bodies are a translation
risk: use the same JAX functions for both, with `flag_snow` as a mask.

### 2.0 Initial ice layer and common start

| Line | Call / block | Condition | State |
|---|---|---|---|
| 1246–1281 | create initial thin ice | `l1 == 0` | W `l1`, `Ti1`, `h1`, `porice`, `salice`, `dhi`, `dhi0`, `dhilow`, `dhihigh`; `tempair < 0` sets a linear `Ti1` profile, else `Ti1 = x - T_phase_threshold` |
| 1284–1289 | `BATHYMETRY` | always | as case 1 |
| 1290 | `DENSITY_W` | always | R `Tw1`, `Sal1`; W `row` |
| 1308–1313 | `TRIBTEMP`, `VERTADV_UPDATE` | `tribheat > 0` | as case 1; v1 tributaries off |
| 1316–1329 | ice salinity/packing | `saltice%par` 0/1 | W `porice`, `salice`; **v1: `saltice 0`** |

### 2.1 Snow on ice (1334–1612) and 2.2 no snow (1613–1877)

| 2.1 line | 2.2 line | Call / block | Condition | State |
|---|---|---|---|---|
| 1336 | 1617 | `MIXED_LAYER_CALC` | always | R `row`, `ddz`, `h1`; W `i_ML`, `H_mixed_layer`, `maxN` |
| 1346 | 1622 | `RadIce%RAD_UPDATE` | always | R `ddzi05(0:Mice)*l1`; W `RadIce%flux`, `RadIce%integr` |
| 1348 | 1624 | `RadWater%RAD_UPDATE` | always | R `ddz05(0:M)*h1`; W `RadWater%flux`, `RadWater%integr` |
| 1363 | 1639 | `RadDeepIce%RAD_UPDATE` | `ls1 /= 0` | R `ddzi05(0:Mice)*ls1`; W `RadDeepIce%flux`, `RadDeepIce%integr` |
| 1376–1394 | 1645–1655 | ice top melt/freeze | `Ti1(1) > x + T_phase_threshold` (melt) vs else | W `Ti1`, `dhi`, `dhilow`, `dhihigh`, `dhi0`, `salice`, `porice`; `saltice 0` |
| — | 1658–1668 | snow creation | `precip > 0 .and. tempair < 0 .and. l1 > 0.05` | W `snowmass`, `hs1`, `fl_sn`, `flag_snow`, `flag_snow_init`; sets `snowmass/rofrsn >= 3*dzmin` |
| — | 1669–1672 | snow melt | `tempair > 0 .and. snowmass > 0` | W `snowmass`, `hs1`, `snmelt`, `fl_sn` |
| 1382 | 1688–1692 | ice surface temperature check | `Ti1(1) < Meltpnt(0,0) + T_phase_threshold` | W `Ti1(1)` |
| 1399 | — | `SNOW_COND_HEAT_COEF` | `flag_snow == 1` | W snow heat capacity/conductivity (`cs`, `rosn`, ...) |
| 1420–1443 | 1695–1713 | snow/ice thickness updates | `ls1 /= 0` vs else | W `dls`, `dls0`, `hs2`, `ls2`, `dhw`, `snowmass`, `snmelt` |
| 1448 | 1716 | `SOIL_COND_HEAT_COEF` | `soilswitch == 1` | as case 1 |
| 1453 | 1721 | `SOILCOLSTEMP` | `multsoil` | as case 1 |
| 1458 | 1726 | `LATERHEAT` | `multsoil` | as case 1 |
| 1463 | 1731 | `T_SOLVER` | always | as case 1; the ice/snow/deep-ice branches of `T_DIFF`/`DIFF_COEF` are used |
| 1466 | 1734 | `SOILFORLAKE` | `soilswitch == 1` | as case 1 |
| 1477 | 1746 | `S_DIFF` | always | as case 1 |
| 1483 | 1759 | `SOILCOLSTEMP` (methane) | `soilswitch == 1 .and. multsoil` | **out of scope; check feedback** |
| 1491 | — | `SNOWTEMP` | `flag_snow == 1` | R `snowmass`, `snowmass_init`, `T`, `Ti1`, `phi`, `fetch`; W `Tis1`, `hs1`, `ls1`, `snowmass`, `snmelt`; **v1 in scope** |
| 1496 | 1750 | `CHLOROPHYLLA` | always | **out of scope** |
| 1497 | 1751 | `OXYGEN_PRODCONS` | always | **out of scope** |
| 1502 | 1767 | `BUBBLE_BLOCK` | `soilswitch == 1` | **out of scope; `ifbubble 0`** |
| 1505–1511 | 1770–1776 | `LATERHEAT` (CO2/O2) | `multsoil` inside the block | **out of scope; check feedback** |
| 1517 | 1782 | `METHANE` | always | **out of scope** |
| 1534–1541 | 1799–1806 | `ADDOXPRODCONS`, `OXYGEN`, `CARBON_DIOXIDE`, `PHOSPHORUS`, `METHANE_OXIDATION` | always | **out of scope** |
| 1546–1552 | 1811–1817 | `MOMENTUM_SOLVER` | `Turbpar /= 1` and (`2` or `5`) | as case 1 |
| 1555 | 1820 | `DENSITY_W` | always | R `Tw2`, `Sal2`; W `row2` |
| 1559 | 1824 | `K_EPSILON` | `Turbpar == 2` | as case 1 |
| 1565–1589 | 1830–1854 | other turbulence | `Turbpar 5/8/9/1` | **v1 rejects; `Turbpar 2` only** |
| 1889 | 1889 | `TURB_SCALES` | after both branches, always | as case 1 |

Note: in 2.1 the `TURB_SCALES` call is outside the `if (flag_snow == 1)` block
(1889), so it runs for both 2.1 and 2.2 exactly once.

## 3. Case 3: ice only, frozen to the bottom (`layer_case == 3`), 1898–2083

Case 3 splits into **3.1 snow on ice** (`flag_snow == 1`, 1918) and **3.2 no
snow** (otherwise, 1997).

| 3.1 line | 3.2 line | Call / block | Condition | State |
|---|---|---|---|---|
| 1901–1914 | 1901–1914 | ice salinity/packing | `saltice%par` 0/1 | **v1: `saltice 0`** |
| 1928 | 2004 | `RadIce%RAD_UPDATE` | always | R `ddzi05(0:Mice)*l1`; W `RadIce%flux`, `RadIce%integr` |
| 1932–1944 | 2020–2033 | ice top melt/freeze | `Ti1(1) > x + T_phase_threshold` | W `Ti1`, `dhi`, `dhilow`, `dhihigh`, `dhi0`, `salice`, `porice` |
| — | 2006–2018 | snow creation | `precip > 0 .and. tempair < 0` | W `snowmass`, `hs1`, `fl_sn`, `flag_snow`, `flag_snow_init` |
| 1951 | — | `SNOW_COND_HEAT_COEF` | `flag_snow == 1` | W snow heat coefficients |
| 1953 | 2040 | `SOIL_COND_HEAT_COEF` | `soilswitch == 1` | as case 1 |
| 1960 | 2046 | `T_SOLVER` | always | ice/deep-ice branch only (`h1 == 0`) |
| 1963 | 2049 | `SOILFORLAKE` | `soilswitch == 1` | as case 1 |
| 1967 | 2053 | `S_DIFF` | always | as case 1 |
| 1969 | — | `SNOWTEMP` | `flag_snow == 1` | as case 2 |
| 1972 | 2056 | `BUBBLE_BLOCK` | `soilswitch == 1` | **out of scope; `ifbubble 0`** |
| 1975 | 2059 | `METHANE` | always | **out of scope** |
| — | — | no `MOMENTUM_SOLVER`, no `K_EPSILON`, no `TURB_SCALES` in case 3 | — | the water column is gone; `u`, `v`, `E`, `eps` are carried unchanged |

## 4. Case 4: snow on soil, no water or ice (`layer_case == 4`), 2087–2162

The original code stops with an error (not operational). Do not translate. The
JAX model should raise a clear error if a lake reaches case 4, rather than
silently doing nothing.

## 5. After the cases: layer-thickness corrections and state store, 2163–2580

| Line | Block | Condition | State |
|---|---|---|---|
| 2163–2168 | dead-volume correction | `deadvol%par > 0 .and. h2 < deadvol` | W `h2`, `voldef`; v1: `deadvol` not used |
| 2170–2178 | thin water over snow | `h2 < min_water_thick .and. ls2 /= 0 .and. l2 == 0` | W `ls2`, `Tis2`, `h2`, `Tw2`, `Sal2` |
| 2180–2189 | thin water over ice | `h2 < min_water_thick .and. h2 > 0 .and. l2 /= 0` | W `l2`, `Ti2`, `h2`, `Tw2`, `Sal2` |
| 2191–2199 | thin water over ice+snow | `h2 < min_water_thick .and. l2 /= 0 .and. ls2 /= 0` | W `Ti2`, `l2`, `ls2`, `Tis2`, `h2`, `Tw2`, `Sal2` |
| 2201–2209 | thin ice over water | `l2 < 0.5*min_ice_thick .and. h2 /= 0` | W `h2`, `l2`, `Ti2` |
| 2211–2216 | thin water only | `h2 < min_water_thick .and. l2 == 0` | W `h2`, `Tw2`, `Sal2` |
| 2218–2221 | thin ice only | `l2 < 0.5*min_ice_thick .and. h2 == 0` | W `l2`, `Ti2` |
| 2223–2237 | thin snow | `hs1 < dzmin .and. hs1 > 0` or `hs1 > 0 .and. l2 == 0 .and. h2 /= 0` | W `h2`, `l2`, `hs1`, `snowmass`, `flag_snow`, `flag_snow_init` |
| 2239–2242 | snow flag reset | `hs1 == 0 .and. flag_snow == 1` | W `flag_snow`, `flag_snow_init` |
| 2244–2252 | thin snow ice | `ls2 < 0.5*min_ice_thick .and. h2 /= 0` | W `h2`, `ls2`, `Tis2` |
| 2254–2262 | snow ice only | `ls2 > 0 .and. h2 == 0` | W `l2`, `Ti2`, `ls2`, `Tis2` |
| 2264–2267 | effluent check | `outflpar%par == 2 .and. tribheat > 0` | STOP; v1 tributaries off |
| 2269–2278 | `Tw2` sanity loop | always | STOP if NaN or `|Tw2| > 100` |
| 2280–2560 | diagnostics, output, control point | mostly `runmode`/output switches | **not translated**; v1 has its own output (M6) |
| 2575–2580 | `UPDATE_NEXT_TIMESTEP` | always | W `*_2d`/`*_3d` state from the 1-D working arrays; this is the end of the step |

## 6. `layer_case` decision (`lake.f90:881-912`)

The decision is made once per step, before case 1:

```
layer_case = 1
if (h1 >  0 .and. l1 >  0) layer_case = 2
if (h1 == 0 .and. l1 >  0) layer_case = 3
if (h1 == 0 .and. l1 == 0) layer_case = 4
```

Then three state-dependent overrides, in this order:

1. case 1 and surface water at the freezing point (`WATER_FREEZE_MELT(Tw1(1), ...,
   +1)`) and enough water → case 2;
2. case 3 and the ice bottom can melt (`WATER_FREEZE_MELT(Ti1(Mice+1), ..., -1)`)
   → case 2;
3. case 3 and the ice top can melt and there is no snow → create a thin water
   layer (`h1 = min_water_thick`, `Tw1 = melting point + T_phase_threshold`,
   ...) → case 1.

In JAX, compute `layer_case` as `(L,)` integers, then apply the overrides with
`jnp.where` under the per-lake masks. The third override rewrites `h1`, `Tw1`,
`Sal1`, `l1`, `Ti1`, `ls1`, `Tis1`, so it must be applied to every affected
State leaf, not just `layer_case`.

## 7. Open points to confirm with the probe / source

- The exact state written by `INIT_VAR` (first call) versus recomputed every
  step: the M1 `init_state` test compares against the probe after
  `UPDATE_CURRENT_TIMESTEP` on step 1.
- The `flag_snow` / `flag_snow_init` lifecycle across case changes (case 1 →
  2 → 3 → 1); the probe steps must include each transition.
- Whether `CHLOROPHYLLA` changes `RadWater%extinct` (light extinction) or any
  other thermodynamic input; M0.4 decides.
- Whether the methane soil `SOILCOLSTEMP` and `LATERHEAT` calls inside the
  biogeochemistry block change `Tsoil1`/`soilflux` in a way that feeds back
  into the thermal solution; M0.4 decides.
- `S_DIFF` with `saltice 0`: confirm it is a no-op for salinity in the water
  column or whether it still diffuses `Sal1`; testlake has `saltice 0`.
- The exact iteration counts and convergence tests inside `K_EPSILON`,
  `T_SOLVER`/`T_DIFF`, and `MOMENTUM_SOLVER`; these are internal to the
  routines and are captured by the probe at each call, not by this top-level
  map.

## 8. Internal call counts per step (from plan §3.3)

`HEATBALANCE` 8, `DIFF_COEF` 27, `T_DIFF` 13, `PROGONKA` 32, `SOILFORLAKE` 5,
`LATERHEAT` 4, `MOMENTUM_SOLVER` / `K_EPSILON` / `T_SOLVER` 1 each. More than
one call means an internal iteration or a loop over soil columns or media; the
translation must reproduce that loop structure, and the per-call probe records
are the check.
