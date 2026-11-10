"""Reader for the binary probe files written by `lm_probe.f90`.

Record layout (little-endian):
    name       char[64]
    dtype      char[1]      f = float64, i = int32, b = int8 (logical)
    rank       int64
    lower[rank] int64
    upper[rank] int64
    data       Fortran column-major

`read_probe(path)` returns `{name: ndarray}` with 0-based arrays.
`read_probe_full(path)` also returns the Fortran lower/upper bounds, for
aligning 0-based JAX arrays with 1-based (or 0-based) Fortran arrays.
"""

from pathlib import Path

import numpy as np

DTYPES = {"f": np.float64, "i": np.int32, "b": np.int8}


def _i64(buf, pos):
    return int.from_bytes(buf[pos:pos + 8], "little", signed=True)


def read_probe_full(path):
    """Return {name: (array, lower_bounds, upper_bounds)}."""
    buf = Path(path).read_bytes()
    pos = 0
    out = {}
    while pos < len(buf):
        name = buf[pos:pos + 64].decode("ascii").rstrip()
        pos += 64
        dtype = chr(buf[pos])
        pos += 1
        rank = _i64(buf, pos)
        pos += 8
        lower, upper = [], []
        for _ in range(rank):
            lower.append(_i64(buf, pos))
            pos += 8
        for _ in range(rank):
            upper.append(_i64(buf, pos))
            pos += 8
        shape = tuple(u - l + 1 for l, u in zip(lower, upper))
        n = int(np.prod(shape)) if rank else 1
        np_dtype = np.dtype(DTYPES[dtype])
        arr = np.frombuffer(buf, dtype=np_dtype, count=n, offset=pos).copy()
        pos += n * np_dtype.itemsize
        arr = arr.reshape(shape, order="F") if rank else arr.reshape(())
        out[name] = (arr, lower, upper)
    return out


def read_probe(path):
    """Return {name: ndarray} with 0-based arrays."""
    return {name: arr for name, (arr, _, _) in read_probe_full(path).items()}


def read_probe_sequence(path):
    """All records in file order, as [(name, ndarray), ...].

    Use this when the same tag was written several times in one step (for
    example every call of SfcFlx_momsenlat); `read_probe` keeps only the last
    record of each name.
    """
    buf = Path(path).read_bytes()
    pos = 0
    out = []
    while pos < len(buf):
        name = buf[pos:pos + 64].decode("ascii").rstrip()
        pos += 64
        dtype = chr(buf[pos]); pos += 1
        rank = _i64(buf, pos); pos += 8
        lower, upper = [], []
        for _ in range(rank):
            lower.append(_i64(buf, pos)); pos += 8
        for _ in range(rank):
            upper.append(_i64(buf, pos)); pos += 8
        shape = tuple(u - l + 1 for l, u in zip(lower, upper))
        n = int(np.prod(shape)) if rank else 1
        np_dtype = np.dtype(DTYPES[dtype])
        arr = np.frombuffer(buf, dtype=np_dtype, count=n, offset=pos).copy()
        pos += n * np_dtype.itemsize
        arr = arr.reshape(shape, order="F") if rank else arr.reshape(())
        out.append((name, arr))
    return out
