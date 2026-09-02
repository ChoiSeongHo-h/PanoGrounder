#!/usr/bin/env bash
#SBATCH --job-name=panogrounder-train
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${PANOGROUNDER_PRESET:?Set PANOGROUNDER_PRESET}"
: "${PANOGROUNDER_DATA:?Set PANOGROUNDER_DATA}"
: "${PANOGROUNDER_MODEL:?Set PANOGROUNDER_MODEL}"
: "${PANOGROUNDER_OUTPUT:?Set PANOGROUNDER_OUTPUT}"

mkdir -p logs
IFS=':' read -r -a data_paths <<< "$PANOGROUNDER_DATA"
data_args=()
for data_path in "${data_paths[@]}"; do
  data_args+=(--data "$data_path")
done
panogrounder train "$PANOGROUNDER_PRESET" \
  "${data_args[@]}" \
  --model-path "$PANOGROUNDER_MODEL" \
  --output-dir "$PANOGROUNDER_OUTPUT" \
  --execute
