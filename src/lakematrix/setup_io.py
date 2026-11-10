"""Read LAKE setup/driver files into `Params`.

The files are the classic LAKE keyword files: one `key value` per line, `#`
comments, and a few block keywords followed by a count and that many rows.
Only the keywords the thermodynamic core needs are interpreted; everything else
is kept in the returned dictionary for later milestones.
"""

from pathlib import Path

import numpy as np

from .config import Params, stack_params

TABLES = {
    "t_profile": 7,
    "t_soilprofile": 2,
    "morphometry": 2,
}


def _clean(line):
    return line.split("#", 1)[0].strip()


def _value(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def read_dat(path):
    """Return (scalars, tables) from one LAKE keyword file."""
    lines = Path(path).read_text(errors="replace").splitlines()
    scalars, tables = {}, {}
    i = 0
    while i < len(lines):
        line = _clean(lines[i])
        i += 1
        if not line:
            continue
        parts = line.split(None, 1)
        key = parts[0].lower()
        val = parts[1] if len(parts) > 1 else ""
        if key in TABLES:
            n = int(_value(val).split()[0])
            rows = []
            while len(rows) < n and i < len(lines):
                row = _clean(lines[i])
                i += 1
                if not row:
                    continue
                cols = row.replace(",", " ").split()
                rows.append([float(x) for x in cols[:TABLES[key]]])
            tables[key] = np.array(rows, dtype=float)
        else:
            scalars[key] = _value(val)
    return scalars, tables


def _get(d, key, default=None):
    key = key.lower()
    if key in d and d[key] not in ("", None):
        return d[key]
    return default


def _f(d, key, default=0.0):
    v = _get(d, key)
    return float(v) if v is not None else float(default)


def _i(d, key, default=0):
    v = _get(d, key)
    return int(float(v)) if v is not None else int(default)


def params_from_dicts(driver, setup, tables, name="lake"):
    """Build a single-lake `Params` from parsed driver/setup dictionaries."""
    def arr(x):
        return np.array([x], dtype=float)

    Tp = tables["t_profile"]
    Tsp = tables["t_soilprofile"]
    morph = tables.get("morphometry", np.zeros((0, 2)))
    return Params(
        L=1,
        M=_i(setup, "M"), Mice=_i(setup, "Mice"), ns=_i(setup, "ns"),
        nsoilcols=_i(setup, "nsoilcols", 1),
        soiltype=_i(setup, "soiltype", 5),
        nmeltpoint=_i(setup, "nmeltpoint", 1),
        tricemethhydr=_f(setup, "tricemethhydr", 0.0),
        soilbotflx=_f(setup, "soilbotflx", 0.0),
        d_surf=_f(setup, "d_surf", 1e-2), d_bot=_f(setup, "d_bot", 1e-2),
        soil_depth=_f(setup, "soil_depth", 10.0),
        dt=_f(driver, "dt", 20.0), interval=_f(driver, "interval", 1.0),
        year0=_i(driver, "year0", 2016), month0=_i(driver, "month0", 8),
        day0=_i(driver, "day0", 1), hour0=_f(driver, "hour0", 0.0),
        tinteg=arr(_f(driver, "tinteg", 1.0)),
        h10=arr(_f(driver, "h10", 0.0)), l10=arr(_f(driver, "l10", 0.0)),
        hs10=arr(_f(driver, "hs10", 0.0)), ls10=arr(_f(driver, "ls10", 0.0)),
        Ts0=arr(_f(driver, "Ts0", 0.0)), Tb0=arr(_f(driver, "Tb0", 0.0)),
        Tbb0=arr(_f(driver, "Tbb0", 0.0)), Tm=arr(_f(driver, "Tm", 0.0)),
        h_ML0=arr(_f(driver, "h_ML0", 0.0)),
        Sals0=arr(_f(driver, "Sals0", 0.0)), Salb0=arr(_f(driver, "Salb0", 0.0)),
        us0=arr(_f(driver, "us0", 0.0)), vs0=arr(_f(driver, "vs0", 0.0)),
        init_T=np.array([_i(driver, "init_T", 1)]),
        phi=arr(_f(driver, "phi", 0.0)),
        extwat=arr(_f(driver, "extwat", 0.0)),
        extice=arr(_f(driver, "extice", 0.0)),
        fetch=arr(_f(driver, "fetch", 0.0)),
        lam=arr(_f(driver, "lam", 0.0)),
        kor=arr(_f(driver, "kor", 0.0)),
        area_lake=arr(_f(driver, "area_lake", 0.0)),
        cellipt=arr(_f(driver, "cellipt", 1.0)),
        lakeform=arr(_f(driver, "lakeform", 0.0)),
        T_profile=Tp[None, :, :], T_profile_len=np.array([len(Tp)]),
        T_soilprofile=Tsp[None, :, :], T_soilprofile_len=np.array([len(Tsp)]),
        morphometry=(morph[None, :, :] if len(morph) else np.zeros((1, 0, 2))),
        morphometry_len=np.array([len(morph)]),
        name=[name],
    )


def read_case(case_dir):
    """Read one LAKE case directory (the reference cases use `setup/*.dat`)."""
    case_dir = Path(case_dir)
    driver = case_dir / "setup" / "testlake_driver.dat"
    setup = case_dir / "setup" / "testlake_setup.dat"
    if not driver.exists() or not setup.exists():
        raise FileNotFoundError(f"{case_dir}: expected setup/testlake_driver.dat and testlake_setup.dat")
    ds, dt = read_dat(driver)
    ss, st = read_dat(setup)
    tables = {}
    tables.update(dt)
    tables.update(st)
    return params_from_dicts(ds, ss, tables, name=case_dir.name)


def read_cases(case_dirs):
    return stack_params([read_case(p) for p in case_dirs])
