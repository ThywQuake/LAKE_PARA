"""M6 test: NetCDF output round trip."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.output import read_profiles, save_profiles  # noqa: E402


def test_output_roundtrip(tmp_path):
    times = np.array([0.0, 86400.0])
    z = np.array([0.0, 1.0, 2.0])
    Tw = np.array([[20.0, 19.0, 18.0], [21.0, 20.0, 19.0]])
    p = save_profiles(tmp_path / "out.nc", times, z, Tw)
    ds = read_profiles(p)
    assert ds.water_temp.shape == (2, 3)
    np.testing.assert_allclose(ds.water_temp.values, Tw)
    np.testing.assert_allclose(ds.time.values, [0.0, 1.0])


def test_run_batch_single_and_two_lakes():
    import numpy as np
    from lakematrix.run import run_batch

    d1 = run_batch([ROOT / "ref/v1/summer1d"], 0.005)
    assert d1["water_temp"].shape[0] == 1
    d2 = run_batch([ROOT / "ref/v1/summer1d", ROOT / "ref/v1/summer1d"], 0.005)
    assert d2["water_temp"].shape[0] == 2
    np.testing.assert_allclose(d2["water_temp"][0], d1["water_temp"][0],
                               rtol=0, atol=0)
    np.testing.assert_allclose(d2["water_temp"][1], d1["water_temp"][0],
                               rtol=0, atol=0)


def test_run_batch_different_depths_are_independent():
    import numpy as np
    from lakematrix.run import run_batch

    cases = [ROOT / "ref/v1/full425d", ROOT / "ref/v1/shallow425d"]
    dep = np.arange(0, 24.01, 1.0)
    d = run_batch(cases, 0.002, out_depth=dep)
    for i, case in enumerate(cases):
        one = run_batch([case], 0.002, out_depth=dep)
        np.testing.assert_allclose(d["water_temp"][i], one["water_temp"][0],
                                   rtol=0, atol=0, equal_nan=True)


def test_run_batch_parallel_matches_serial(tmp_path):
    import numpy as np
    import xarray as xr
    from lakematrix.run import run_batch, run_batch_parallel, save_netcdf

    cases = [ROOT / "ref/v1/summer1d", ROOT / "ref/v1/summer1d"]
    ser = run_batch(cases, 0.005)
    par = run_batch_parallel(cases, 0.005, jobs=2)
    assert set(par) == set(ser)
    for key in ser:
        np.testing.assert_array_equal(par[key], ser[key], err_msg=key)
    out = save_netcdf(tmp_path / "par.nc", par, names=cases)
    with xr.open_dataset(out) as ds:
        assert ds.sizes["lake"] == 2
