"""Per-lake parameters (Params) and batch stacking.

`Params` mirrors the setup/driver keywords used by the thermodynamic core.
Grid sizes and the vertical stretching are static and must be identical for all
lakes in a batch (plan section 6.1); the per-lake values are arrays with a
leading lake dimension.
"""

from dataclasses import dataclass, fields
from typing import List

import numpy as np


@dataclass
class Params:
    L: int
    # static grid / setup
    M: int
    Mice: int
    ns: int
    nsoilcols: int
    d_surf: float
    d_bot: float
    soil_depth: float
    dt: float
    interval: float
    year0: int
    month0: int
    day0: int
    hour0: float
    # per-lake scalars, shape (L,)
    tinteg: np.ndarray
    h10: np.ndarray
    l10: np.ndarray
    hs10: np.ndarray
    ls10: np.ndarray
    Ts0: np.ndarray
    Tb0: np.ndarray
    Tbb0: np.ndarray
    Tm: np.ndarray
    h_ML0: np.ndarray
    Sals0: np.ndarray
    Salb0: np.ndarray
    us0: np.ndarray
    vs0: np.ndarray
    init_T: np.ndarray
    phi: np.ndarray
    extwat: np.ndarray
    extice: np.ndarray
    fetch: np.ndarray
    lam: np.ndarray
    kor: np.ndarray
    area_lake: np.ndarray
    cellipt: np.ndarray
    lakeform: np.ndarray
    # profiles, padded with NaN; *_len is the real length
    T_profile: np.ndarray        # (L, nT, 7)
    T_profile_len: np.ndarray    # (L,)
    T_soilprofile: np.ndarray    # (L, nS, 2)
    T_soilprofile_len: np.ndarray
    morphometry: np.ndarray      # (L, nM, 2)
    morphometry_len: np.ndarray
    # source case (for diagnostics)
    name: List[str]
    # soil / lake setup options (thermodynamic core)
    soiltype: int = 5
    nmeltpoint: int = 1
    tricemethhydr: float = 0.0
    soilbotflx: float = 0.0
    backdiff: int = 0
    backdiff0: float = -999.0

    @property
    def static(self):
        return dict(M=self.M, Mice=self.Mice, ns=self.ns, nsoilcols=self.nsoilcols,
                    d_surf=self.d_surf, d_bot=self.d_bot,
                    soil_depth=self.soil_depth, dt=self.dt, interval=self.interval)


def _pad_stack(arrays, fill=np.nan):
    n = max(a.shape[0] for a in arrays)
    rest = arrays[0].shape[1:]
    out = np.full((len(arrays), n) + rest, fill, dtype=float)
    for i, a in enumerate(arrays):
        out[i, :a.shape[0]] = a
    return out


def stack_params(ps):
    """Stack single-lake (or already batched) Params into one batch."""
    if len(ps) == 1:
        return ps[0]
    first = ps[0]
    for p in ps[1:]:
        if p.static != first.static:
            raise ValueError(f"batched lakes have different static parameters: "
                             f"{first.static} vs {p.static}")
    scalar_fields = [f.name for f in fields(Params)
                     if f.name not in ("L", "name", "T_profile", "T_profile_len",
                                       "T_soilprofile", "T_soilprofile_len",
                                       "morphometry", "morphometry_len",
                                       "M", "Mice", "ns", "nsoilcols",
                                       "d_surf", "d_bot", "soil_depth", "dt",
                                       "interval", "year0", "month0", "day0",
                                       "hour0")]
    kw = {k: getattr(first, k) for k in
          ("M", "Mice", "ns", "nsoilcols", "d_surf", "d_bot", "soil_depth",
           "dt", "interval", "year0", "month0", "day0", "hour0")}
    for k in scalar_fields:
        kw[k] = np.concatenate([np.atleast_1d(getattr(p, k)) for p in ps])
    kw["T_profile"] = _pad_stack([p.T_profile[0] for p in ps])
    kw["T_profile_len"] = np.array([p.T_profile_len[0] for p in ps])
    kw["T_soilprofile"] = _pad_stack([p.T_soilprofile[0] for p in ps])
    kw["T_soilprofile_len"] = np.array([p.T_soilprofile_len[0] for p in ps])
    kw["morphometry"] = _pad_stack([p.morphometry[0] for p in ps])
    kw["morphometry_len"] = np.array([p.morphometry_len[0] for p in ps])
    kw["L"] = len(ps)
    kw["name"] = [p.name[0] for p in ps]
    return Params(**kw)


def batch_size(p):
    return p.L
