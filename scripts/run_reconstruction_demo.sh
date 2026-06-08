#!/usr/bin/env bash
set -euo pipefail

python preprocess/reconstruct_sentence_image.py \
  --input_dir demo_examples/sample_001/word_fragments_binarized \
  --metadata demo_examples/sample_001/metadata.json \
  --output_file demo_examples/sample_001/reconstructed_sentence.png \
  --direction vertical

python preprocess/reconstruct_sentence_image.py \
  --input_dir demo_examples/sample_002/word_fragments_binarized \
  --metadata demo_examples/sample_002/metadata.json \
  --output_file demo_examples/sample_002/reconstructed_sentence.png \
  --direction vertical
