#!/usr/bin/env bash
# Phase 2, run after the original suite finished on this host:
#   equivalence test -> optimized-variant benchmarks -> native C++ variants -> compile/test checks.
# Usage: bash benchmarks/active_sdsl/phase2.sh <mac|rpi> [--wait-pid PID]
set -uo pipefail
HOST=$1
cd "$(dirname "$0")/../.."
R=benchmarks/active_sdsl/results/$HOST
mkdir -p "$R"
if [ "${2:-}" = "--wait-pid" ]; then
  while kill -0 "$3" 2>/dev/null; do sleep 20; done
fi
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$R/phase2.log"; }
log "phase2 start on $HOST"

log "1/5 equivalence test (orig vs opt examples)"
EQ=""
[ "$HOST" = rpi ] && EQ="--quick"   # the original code is ~13x slower on the Pi; quick keeps it to minutes
python3 benchmarks/active_sdsl/test_equivalence.py $EQ > "$R/equivalence.log" 2>&1
log "equivalence exit=$? : $(tail -1 "$R/equivalence.log")"

log "2/5 optimized-variant benchmarks"
python3 benchmarks/active_sdsl/run_all.py --host-kind "$HOST" --variant opt \
    --only choose_movement,profile,episodes_deployed,episodes_fixed >> "$R/phase2_opt.log" 2>&1
log "opt suite exit=$?"

log "3/5 native C++ kernels"
EXTRA=""
[ "$HOST" = rpi ] && EXTRA="--cross-aarch64"
python3 benchmarks/active_sdsl/run_native.py --host-kind "$HOST" $EXTRA >> "$R/phase2_native.log" 2>&1
log "native exit=$?"

log "4/5 compile + run the standalone C++ demo app (apps/demo_lidar_localization_pgm)"
if [ "$(uname -s)" = Darwin ]; then
  clang++ -std=c++23 -O2 -DSDSL_CPP_ONLY -DFMT_HEADER_ONLY -Isdsl/include -I/opt/homebrew/include \
    -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp \
    apps/demo_lidar_localization_pgm/main.cpp -o /tmp/sdsl_demo_app > "$R/compile_app.log" 2>&1
else
  # conan2 puts downloaded packages in p/<pkg>/p and source-built ones in p/b/<pkg>/p
  FMT_INC=$(ls -d "$HOME"/.conan2/p/fmt*/p/include "$HOME"/.conan2/p/b/fmt*/p/include 2>/dev/null | head -1)
  g++ -std=c++2b -O2 -DSDSL_CPP_ONLY -DFMT_HEADER_ONLY -Isdsl/include -I"$FMT_INC" -fopenmp \
    apps/demo_lidar_localization_pgm/main.cpp -o /tmp/sdsl_demo_app > "$R/compile_app.log" 2>&1
fi
rc=$?
log "demo app compile exit=$rc"
[ $rc = 0 ] && { /tmp/sdsl_demo_app >> "$R/compile_app.log" 2>&1; log "demo app run exit=$?"; }

log "5/5 pytest"
python3 -m pytest -q tests > "$R/pytest.log" 2>&1
log "pytest exit=$? : $(grep -E '[0-9]+ (passed|failed)' "$R/pytest.log" | tail -1)"
log "phase2 done"
