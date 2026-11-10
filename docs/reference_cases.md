# Reference cases (M0.4)

Reference model: `lakebatch` at commit `6657ebc`, **check build**
(`CHECK=1 core/build.sh`, `-O3 -ffp-contract=off`), one lake per run, no
tributaries (`tribheat 0`; the testlake driver already has `trib_inflow -9999`,
so there is no tributary mass either). Created by
`tools/ref/make_ref_cases.py`; each case is a copy of the testlake setup with
the stated change.

The reference outputs live in `lakematrix/ref/v1/<case>/results/test/` and
are git-ignored (they are reproducible; about 0.8 GB per 425-day case). The
comparison in `tools/compare.py` reads the time-series files from there.

## Cases and results

`rows` is the number of hourly `layers` rows (425 days = 10200). `first_ice`
and `last_ice` are hours since the start (the model step is `hours × 180`).

| Case | Change | Rows | First ice | First snow | Last ice | Status |
|---|---|---|---|---|---|---|
| `summer1d` | 1 day, no ice | 24 | — | — | — | ok |
| `summer60d` | 60 days, no ice | 1440 | — | — | — | ok |
| `full425d` | none | 10200 | 2559 h (step 460620) | 3031 h | 5947 h | ok |
| `cold425d` | air temperature −2 K | 10200 | 2527 h (step 454860) | 2827 h | 5968 h | ok |
| `shallow425d` | `h10 = 5 m`, morphometry depth column scaled | 10200 | 2312 h (step 416160) | 2347 h | 5941 h | ok |
| `warm425d` | air temperature +3 K, `phi = 40` | 10200 | 2910 h (step 523800) | 3034 h | 5429 h | ok |
| `shallow1m425d` | `h10 = 1 m`, table scaled | 5651 | 2066 h (step 371880) | 2260 h | 5651 h | **NaN at h 5651** |
| `shallow05m425d` | `h10 = 0.5 m`, table scaled | 5 | — | — | — | **NaN at h 5** |

The 1 m and 0.5 m cases were attempts to reach `layer_case 3` (water gone,
ice present). The 0.5 m case fails almost immediately; the 1 m case runs
235 days but hits `Tw1 = NaN` at h 5651, and neither reaches case 3 (the water
layer never reaches zero while ice is present). **The stable reference set
therefore does not cover case 3.** M4 must add a case-3 check; the options are
(a) a stable very shallow or cold-shallow setup, once one is found, or (b) a
synthetic case-3 state fed to the probe build and to JAX. The plan's §4.2
allows this ("if `shallow425d` never reaches case 3, reduce depth or air
temperature further ... record the final parameters"), but a crash is not a
usable reference; record the choice in `NOTES.md` when M4 starts.

`full425d`/`cold425d` are the pair used for a batch with two different layer
cases on the same step (plan §4.2). They cover case 1, case 2, ice formation,
snow, and melt; case 3 is the gap.

## How to regenerate

```bash
cd lakematrix
.venv/bin/python tools/ref/make_ref_cases.py            # build the cases
.venv/bin/python tools/ref/make_ref_cases.py --run --jobs 4 --only full425d ...
```

`make_ref_cases.py` refuses to overwrite an existing case (the project rule is
no `rm -rf`); use a new `ref/v2/` after changing a parameter. The check-build
binary path is `lakebatch/core/build-check/lake.out` on the Mac and
`lakebatch/core/build-check-linux/lake.out` on the Linux VM.

## Probe steps

`tools/ref/steps.toml` lists the steps chosen from `full425d` for the
per-routine probes (summer, ice formation, under ice, snow, ice melt, first
no-ice step). The step `nstep = 180 × hours` was verified: at `nstep 180`,
`Tw1(1)` from the probe equals the `water_temp` output at depth 0 (see
`NOTES.md`).
