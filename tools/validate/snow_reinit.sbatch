#!/usr/bin/env bash
# Regression for the second-snowfall NaN (flag_snow_init never reset to 1).
#
# Before the fix, every 425-day case died on the first snowfall that came
# after a snow pack had melted away: SNOW_COND_HEAT_COEF "zero active layer"
# (dz = 0), divide by zero in heatfluxsurf_snow, then NaN.  shallow425d hit it
# on day 98, so 100 days is enough; set DAYS=157 CASE=full425d for the longest.
#
#   sbatch tools/validate/snow_reinit.sbatch             # from the repo root
#   CASE=cold425d DAYS=130 sbatch tools/validate/snow_reinit.sbatch
#   tools/validate/snow_reinit.sbatch                    # or run directly
#
# Always starts from scratch in a fresh directory (the old *.ckpt.pkl files
# already contain the NaN state and cannot be resumed).  The last line of
# <OUT>/status is PASS or FAIL.
#SBATCH --job-name=lakematrix-snow
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=validation-snowfix/slurm-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ROOT=$PWD
PY=${PY:-$ROOT/.venv/bin/python}
CASE=${CASE:-shallow425d}
DAYS=${DAYS:-100}
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${OUT:-$ROOT/validation-snowfix/${CASE}_${DAYS}d_$STAMP}
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

{
    echo "== snow_reinit $CASE ${DAYS}d  $(date -Is)  host $(hostname)"
    echo "commit $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
    grep -n "flag_snow_init = 1" src/lakematrix/step2.py || echo "WARNING: reset not found in step2.py"
} | tee "$OUT/log.txt"

[[ -d ref/v1/$CASE ]] || { echo "ERROR missing ref/v1/$CASE" | tee "$OUT/status"; exit 1; }

"$PY" -W always tools/ref/run_case.py "ref/v1/$CASE" "$DAYS" \
    --out "$OUT/run.npz" --ckpt "$OUT/run.ckpt.pkl" >> "$OUT/log.txt" 2>&1
rc=$?

"$PY" -I - "$OUT" "$DAYS" "$rc" <<'EOF' | tee "$OUT/check.txt"
import re, sys
import numpy as np
from pathlib import Path

out, days, rc = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
log = (out / "log.txt").read_text(errors="replace")
bad = []
if rc != 0:
    bad.append(f"model exit {rc}")
zero = len(re.findall(r"zero active layer", log))
div0 = len(re.findall(r"surface\.py:\d+: RuntimeWarning: divide by zero", log))
nan_msg = re.search(r"^NaN at step.*$", log, re.M)
if zero or div0:
    bad.append(f"zero-layer warnings {zero}, divide-by-zero {div0}")
if nan_msg:
    bad.append(nan_msg.group(0))
try:
    d = np.load(out / "run.npz")
except Exception as e:                       # noqa: BLE001
    print(f"cannot read run.npz: {e}")
    print("FAIL")
    sys.exit(0)

n = len(d["n"])
if n != days:
    bad.append(f"{n} daily records, expected {days}")
fin = {k: bool(np.isfinite(d[k]).all()) for k in ("Tw1", "Ti1", "Tsoil1", "h1", "l1", "hs1")}
if not all(fin.values()):
    bad.append("non-finite: " + ",".join(k for k, v in fin.items() if not v))

# the fix only matters if a snow pack formed, vanished, and formed again
hs = d["hs1"].reshape(n, -1)[:, 0] > 0
episodes = int(np.sum(hs[1:] & ~hs[:-1]) + (1 if hs[0] else 0))
print(f"snow episodes: {episodes}; days with snow: {int(hs.sum())}; "
      f"layer_case last: {int(d['layer_case'].reshape(n, -1)[-1, 0])}")
if episodes < 2:
    print("NOTE: fewer than 2 snow episodes, so the re-init path was not exercised")
    bad.append("re-init path not exercised (raise DAYS or pick another CASE)")

for b in bad:
    print("problem:", b)
print("FAIL" if bad else "PASS")
EOF

verdict=$(tail -n 1 "$OUT/check.txt")
printf '%s\n%s\n' "$verdict" "$CASE ${DAYS}d" > "$OUT/status"
echo "== $verdict  $OUT"
[[ $verdict == PASS ]]
