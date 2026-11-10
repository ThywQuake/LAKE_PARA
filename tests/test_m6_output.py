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
