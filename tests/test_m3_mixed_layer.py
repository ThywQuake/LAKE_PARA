"""M3 test: MIXED_LAYER_CALC."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.constants import C  # noqa: E402
from lakematrix.physics.turb import mixed_layer_calc  # noqa: E402


def test_mixed_layer_max_bvf():
    M = 4
    row = np.array([1000.0, 1000.5, 1001.5, 1002.0, 1002.1])  # 1-based length 5
    ddz = np.full(M, 0.25)
    dzeta05 = np.array([0.125, 0.375, 0.625, 0.875])
    ml, H, maxN = mixed_layer_calc(row, ddz, dzeta05, 10.0, M)
    bvf = C.g / C.row0 * (row[1:] - row[:-1]) / (10.0 * ddz)
    assert ml == int(np.argmax(bvf)) + 1
    assert maxN == np.sqrt(bvf.max())
    assert H == dzeta05[ml - 1] * 10.0


def test_mixed_layer_well_mixed_fallback():
    M = 4
    row = np.array([1000.0, 1000.0001, 1000.0002, 1000.0003, 1000.0004])
    ddz = np.full(M, 0.25)
    dzeta05 = np.array([0.125, 0.375, 0.625, 0.875])
    ml, H, _ = mixed_layer_calc(row, ddz, dzeta05, 10.0, M)
    assert ml == M
    assert H == dzeta05[M - 1] * 10.0
