# Data usage

MWTrans-118K was constructed from rare Manchu--Chinese archival materials and expert-validated sentence-level annotations. The full corpus is not redistributed in this repository because the original sources and annotations are subject to usage and redistribution restrictions.

This release provides two small real demonstration examples only to show the preprocessing pipeline and JSON schema. These examples should not be treated as the full benchmark and cannot reproduce the paper's reported scores. They are not covered by the MIT code license and should be used only for format inspection, pipeline demonstration, non-commercial research, and educational purposes.

Researchers who have legal access to the source materials can use the released scripts to prepare their own data in the same format:

1. segment Manchu sentence images or word-image fragments;
2. binarize word fragments with `preprocess/pre_binarize_words.py`;
3. reconstruct sentence-level images with `preprocess/reconstruct_sentence_image.py`;
4. apply SAE with `preprocess/sae_enhancement.py`;
5. build the multimodal SFT JSON file with `preprocess/build_demo_json.py`.

Do not redistribute restricted archival images, full annotations, or held-out-book materials unless the relevant rights and permissions allow redistribution.
