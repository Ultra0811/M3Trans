#!/usr/bin/env bash
set -euo pipefail

python preprocess/pre_binarize_words.py \
  --input_dir demo_examples/sample_001/word_fragments_raw \
  --output_dir demo_examples/sample_001/word_fragments_binarized \
  --method adaptive

python preprocess/pre_binarize_words.py \
  --input_dir demo_examples/sample_002/word_fragments_raw \
  --output_dir demo_examples/sample_002/word_fragments_binarized \
  --method adaptive
