# GPU deployment notes

## Environment used

- GPU: NVIDIA GeForce RTX 4060 Ti 16 GB, driver 610.57 / CUDA 13.3
- System Python 3.10 (`python3.10-venv` installed via apt)
- JAX 0.6.2 with the CUDA 12 plugin:

```bash
python3.10 -m venv /root/.venv-gpu
/root/.venv-gpu/bin/pip install --upgrade pip
/root/.venv-gpu/bin/pip install "jax[cuda12]==0.6.2" numpy netCDF4 xarray pytest
cd /root/LAKE_PARA
/root/.venv-gpu/bin/pip install -e . --no-deps
/root/.venv-gpu/bin/python -c "import jax; print(jax.devices())"
```

`jax.devices()` returns `[CudaDevice(id=0)]`; the CPU-only Python 3.14 venv
reports the missing CUDA jaxlib warning.

## Correctness on GPU

`python -m pytest tests/test_jaxbatch_step*.py` passes on the GPU after
relaxing one exact (`atol=0`) integration-helper assertion to `1e-12`
(GPU fusion changes the last 1–2 ULPs).  Physics tolerances are unchanged.

## Throughput (`tools/bench_jax_gpu.py`)

A 100-step `lax.scan` chunk is compiled once (first `run=0`), then measured
again (`run=1`):

| backend | lakes | lake-steps/s (steady) |
|---|---:|---:|
| CPU (JAX 0.12) | 8 | 3,219 |
| GPU (JAX 0.6.2) | 1 | 19 |
| GPU (JAX 0.6.2) | 64 | ~2,000 |
| GPU (JAX 0.6.2) | 512 | 17,178 |

The GPU is **launch-bound** for one/few lakes: one step costs ~50 ms even
though the arrays are tiny.  A 425-day single-lake run would take ~15 h on
this GPU, versus ~1.5–3 h on the CPU JAX path.  The GPU only wins for large
ensembles (hundreds of lakes in one batch), where the same 425-day window
takes ~15 h for 512 lakes.

## Recommended long-run strategy

- **Single reference case / acceptance run:** CPU JAX
  (`tools/ref/run_jax_case.py` with `.venv`, not `.venv-gpu`).
- **Large ensemble:** GPU with `--lakes 512` and a chunked `lax.scan`
  runner; do not use `run_case1_jax` from a Python day loop, because its
  scan is traced/compiled on every call (the benchmark's `chunk` is jitted
  as one computation).
- The 425-day acceptance itself still needs to be run; GPU deployment is
  ready but is not the fastest path for a single lake.
