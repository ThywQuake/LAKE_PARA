#!/bin/bash
# Build the LAKE Fortran core.
#
#   core/build.sh           production build: core/lake.out (macOS) or
#                           core/build-linux/lake.out (Linux)
#   CHECK=1 core/build.sh   check build with -ffp-contract=off:
#                           core/build-check[-linux]/lake.out
#
# The check build exists for exact (bit-for-bit) comparisons.  With
# floating-point contraction on, gfortran -O3 fuses a*b+c into FMA instructions
# depending on the surrounding code, so editing a routine can change its last
# bits even when no formula changes; with contraction off the result depends
# only on the formulas (see ../lake-refactor/ARITHMETIC-REPORT.md).
#
# Every build except the Mac production one gets its own directory, so a
# checkout shared between the Mac and Linux keeps each platform's build; all use
# core/build/CMakeLists.txt.
set -euo pipefail
CORE="$(cd "$(dirname "$0")" && pwd)"

export FC="${FC:-gfortran}"
if [[ "$OSTYPE" == darwin* ]]; then
  # Homebrew GCC on recent macOS needs an explicit SDK and deployment target.
  export SDKROOT="${SDKROOT:-$(xcrun --show-sdk-path)}"
  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"
  SUFFIX=""
else
  SUFFIX="-linux"
fi

if [[ "${CHECK:-0}" == 1 ]]; then
  DIR="$CORE/build-check$SUFFIX"
elif [[ -n "$SUFFIX" ]]; then
  DIR="$CORE/build$SUFFIX"
else
  DIR="$CORE/build"
fi

if [[ "$DIR" == "$CORE/build" ]]; then
  cd "$DIR"
  EXE="$CORE/lake.out"
else
  mkdir -p "$DIR"
  cd "$DIR"
  sed 's|RUNTIME_OUTPUT_DIRECTORY "../"|RUNTIME_OUTPUT_DIRECTORY "."|' "$CORE/build/CMakeLists.txt" > CMakeLists.txt.new
  if [[ "${CHECK:-0}" == 1 ]]; then
    sed -i.bak 's|list(APPEND FLAGS_ "-O3")|list(APPEND FLAGS_ "-O3" "-ffp-contract=off")|' CMakeLists.txt.new
    rm -f CMakeLists.txt.new.bak
    grep -q 'ffp-contract=off' CMakeLists.txt.new || { echo "check build: could not add -ffp-contract=off" >&2; exit 1; }
  fi
  # replace only when changed, so cmake does not rebuild everything
  cmp -s CMakeLists.txt.new CMakeLists.txt 2>/dev/null && rm CMakeLists.txt.new || mv CMakeLists.txt.new CMakeLists.txt
  EXE="$DIR/lake.out"
fi

cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_Fortran_FLAGS="-I$CORE/source/model" . > cmake.log 2>&1
cmake --build . --target lake.out --parallel "${JOBS:-8}" > build.log
echo "built $EXE"
