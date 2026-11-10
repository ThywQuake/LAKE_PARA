"""Layer-thickness update for the open-water case (case 1)."""

import numpy as np


def advance_case1(state, dhw, dls=0.0):
    """Return the next-step thicknesses for case 1 (no ice or snow).

    Fortran `lake.f90:1233-1236` (`h2 = h1 + dhw`, `ls2 = ls1 + dls`, `l2=0`)
    plus the post-case corrections that are inactive for a deep lake.
    """
    h2 = np.asarray(state.h1) + np.asarray(dhw)
    ls2 = np.asarray(state.ls1) + np.asarray(dls)
    l2 = np.zeros_like(h2)
    return dict(h1=h2, l1=l2, ls1=ls2, hs1=np.zeros_like(h2),
                snowmass=np.zeros_like(h2))
