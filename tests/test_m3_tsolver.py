"""M3 test: the T_SOLVER scan/Chorde algorithm on synthetic heat balances."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakematrix.physics.tsolver import t_solver  # noqa: E402


def test_t_solver_linear_root():
    root = 24.9
    calls = []

    def fn(t):
        calls.append(t)
        return t - root, ("state", t)

    r = t_solver(fn, nstep=1, surftyp=4, surftyp_prev=4, Tsurf_prev=0.0)
    assert r["Tsurf3"] == root
    assert r["iter"] == 1
    assert r["state"] == ("state", root)
    # scan starts at -90 and steps by 5: the bracket is [20, 25]
    assert calls[0] == -90.0 and calls[1] == -85.0


def test_t_solver_quadratic_root():
    root = 12.3

    def fn(t):
        return (t - root) * (t + 100.0), t

    r = t_solver(fn, nstep=1, surftyp=4, surftyp_prev=4, Tsurf_prev=0.0)
    assert abs(r["Tsurf3"] - root) < 0.01  # Chorde stops at |Bal| <= 1 W/m2
    assert r["iter"] >= 1


def test_t_solver_uses_previous_temperature():
    def fn(t):
        return t - 10.0, t

    r = t_solver(fn, nstep=2, surftyp=4, surftyp_prev=4, Tsurf_prev=18.0)
    # starts at 8, next 13; bracket [8,13], root 10
    assert abs(r["Tsurf3"] - 10.0) < 1e-12
