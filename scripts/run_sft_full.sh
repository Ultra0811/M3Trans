#!/usr/bin/env bash
set -euo pipefail

python train/train_sft_full.py --config configs/sft_full_qwen25vl_7b.yaml
