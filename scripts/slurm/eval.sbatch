#!/usr/bin/env bash
#SBATCH --job-name=panogrounder-eval
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${PANOGROUNDER_PRESET:?Set PANOGROUNDER_PRESET}"
: "${PANOGROUNDER_DATA:?Set PANOGROUNDER_DATA}"
: "${PANOGROUNDER_CHECKPOINT:?Set PANOGROUNDER_CHECKPOINT}"
: "${PANOGROUNDER_OUTPUT:?Set PANOGROUNDER_OUTPUT}"

mkdir -p logs
panogrounder eval "$PANOGROUNDER_PRESET" \
  --data "$PANOGROUNDER_DATA" \
  --checkpoint "$PANOGROUNDER_CHECKPOINT" \
  --output-dir "$PANOGROUNDER_OUTPUT" \
  --execute
