#!/usr/bin/env bash
# Build bench_kernels under several compiler-flag variants.
#   Os                 : what the Python extension actually gets (nanobind appends -Os after -O3)
#   O3                 : plain -O3
#   O3native           : -O3 tuned for the CPU (M1 / Cortex-A72 with NEON)
#   O3native_noerrno   : + -fno-math-errno (lets libm calls such as sqrt inline)
#   aarch64_O3native   : (RPi only, if a cross compiler exists) 64-bit static build that runs on
#                        the Pi's 64-bit kernel: estimates the gain of a 64-bit OS userland.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../../.. && pwd)
INC="-I$REPO/sdsl/include -DSDSL_CPP_ONLY -DFMT_HEADER_ONLY"
mkdir -p bin
case "$(uname -s)" in
  Darwin)
    CXX=clang++
    STD=-std=c++23
    OMP="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp"
    INC="$INC -I/opt/homebrew/include"
    NATIVE="-mcpu=apple-m1"
    ;;
  Linux)
    CXX=g++
    STD=-std=c++2b
    OMP="-fopenmp"
    # conan2 puts downloaded packages in p/<pkg>/p and source-built ones in p/b/<pkg>/p
    FMT_INC=$(ls -d "$HOME"/.conan2/p/fmt*/p/include "$HOME"/.conan2/p/b/fmt*/p/include 2>/dev/null | head -1 || true)
    [ -n "$FMT_INC" ] && INC="$INC -I$FMT_INC"
    NATIVE="-mcpu=cortex-a72 -mfpu=neon-fp-armv8 -mfloat-abi=hard"
    ;;
esac
VARIANTS=(
  "Os|-Os -DNDEBUG"
  "O3|-O3 -DNDEBUG"
  "O3native|-O3 -DNDEBUG $NATIVE"
  "O3native_noerrno|-O3 -DNDEBUG $NATIVE -fno-math-errno"
)
for entry in "${VARIANTS[@]}"; do
  v=${entry%%|*}; fl=${entry#*|}
  echo "== building $v: $fl"
  $CXX $STD $fl $INC bench_kernels.cpp -o bin/bench_$v $OMP
done
if [ "$(uname -s)" = Linux ] && command -v aarch64-linux-gnu-g++ >/dev/null; then
  echo "== building aarch64_O3native (static)"
  A64_FMT=""
  [ -n "${FMT_INC:-}" ] && A64_FMT="-I$FMT_INC"
  aarch64-linux-gnu-g++ $STD -O3 -DNDEBUG -mcpu=cortex-a72 -I$REPO/sdsl/include -DSDSL_CPP_ONLY -DFMT_HEADER_ONLY $A64_FMT \
      bench_kernels.cpp -o bin/bench_aarch64_O3native -static -fopenmp
fi
ls -la bin
