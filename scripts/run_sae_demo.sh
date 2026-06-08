#!/usr/bin/env bash
set -euo pipefail

python preprocess/sae_enhancement.py \
  --input demo_examples/sample_001/reconstructed_sentence.png \
  --output demo_examples/sample_001/enhanced_sentence.png

python preprocess/sae_enhancement.py \
  --input demo_examples/sample_002/reconstructed_sentence.png \
  --output demo_examples/sample_002/enhanced_sentence.png
