#!/usr/bin/env bash
# Run all six binary configs at three seeds each. Skips the seed=42 run if
# the canonical (un-suffixed) artifact already exists, then runs seeds 7
# and 1337 with --seed-override.
#
# Usage:
#   ./scripts/run_multi_seed.sh
#
# Expects to be run from the repo root with the venv active.

set -euo pipefail

SEEDS=(7 1337)
CONFIGS=(
  configs/tfidf_davidson.yaml
  configs/tfidf_hatexplain.yaml
  configs/doc2vec_davidson.yaml
  configs/doc2vec_hatexplain.yaml
  configs/distilbert_davidson.yaml
  configs/distilbert_hatexplain.yaml
)

for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo
    echo "=============================================================="
    echo " $cfg   seed=$seed"
    echo "=============================================================="
    python -m hsd.train --config "$cfg" --seed-override "$seed"
  done
done

echo
echo "All multi-seed runs done. Aggregating ..."
python scripts/multi_seed_aggregate.py
