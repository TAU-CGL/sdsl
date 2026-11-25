#!/bin/zsh
# Validate the top grid-search combinations with full runs (4 maps, 200 steps,
# best vs random) on fresh seeds. One process per (candidate, seed), all in parallel.
cd "$(dirname "$0")/.." || exit 1
OUT=validation
REPEATS=30
SEEDS=(1 2 3)
#              name  min_clearance alpha fsm  entropy_eps
CANDIDATES=(  "A     0.0           1.0   0.2  0.05"
              "B     0.10          0.15  0.2  0.1"
              "C     0.05          0.05  0.15 0.1" )

for cand in "${CANDIDATES[@]}"; do
  read name clr alpha fsm eps <<< "$cand"
  for seed in "${SEEDS[@]}"; do
    tag="${name}_seed${seed}"
    python tests/experiment_movement.py --seed "$seed" --repeats "$REPEATS" --max-steps 200 \
      --min-clearance "$clr" --alpha "$alpha" --forward-safety-margin "$fsm" --entropy-eps "$eps" \
      --out "$OUT/${tag}.csv" > "$OUT/${tag}.txt" 2>/dev/null &
  done
done
wait
echo "all validation runs finished"
