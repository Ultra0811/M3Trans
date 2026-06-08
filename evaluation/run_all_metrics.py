import argparse
import json
from pathlib import Path

from evaluate_bleu import load_pairs as load_bleu_pairs, corpus_bleu_char
from sacrebleu.metrics import CHRF
from sentence_transformers import SentenceTransformer
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Run BLEU, chrF++, and SBERT metrics.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sbert_model", default="BAAI/bge-large-zh-v1.5")
    parser.add_argument("--prediction_key", default="prediction")
    parser.add_argument("--reference_key", default="reference")
    args = parser.parse_args()

    pairs = load_bleu_pairs(Path(args.input), args.prediction_key, args.reference_key)
    metric = CHRF(word_order=2)
    preds = [pred for pred, _ in pairs]
    refs = [ref for _, ref in pairs]
    result = {
        "num_pairs": len(pairs),
        "BLEU": corpus_bleu_char(pairs) if pairs else 0.0,
        "chrF++": float(metric.corpus_score(preds, [refs]).score) if pairs else 0.0,
        "SBERT": 0.0,
    }
    if pairs:
        model = SentenceTransformer(args.sbert_model)
        pred_emb = model.encode(preds, normalize_embeddings=True, convert_to_numpy=True)
        ref_emb = model.encode(refs, normalize_embeddings=True, convert_to_numpy=True)
        result["SBERT"] = float(np.sum(pred_emb * ref_emb, axis=1).mean() * 100.0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
