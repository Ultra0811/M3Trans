#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-demo_examples/demo_predictions.json}
OUTPUT=${2:-outputs/evaluation/demo_metrics_summary.json}
SBERT_MODEL=${3:-BAAI/bge-large-zh-v1.5}

python evaluation/run_all_metrics.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --sbert_model "$SBERT_MODEL"
