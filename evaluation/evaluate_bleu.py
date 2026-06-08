import argparse
import json
from pathlib import Path

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu


def load_pairs(path: Path, prediction_key: str, reference_key: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    pairs = []
    for item in data:
        pred = str(item.get(prediction_key, "")).strip()
        ref = str(item.get(reference_key, "")).strip()
        if pred and ref:
            pairs.append((pred, ref))
    return pairs


def char_tokens(text: str):
    return list(text.strip())


def corpus_bleu_char(pairs, weights=(0.25, 0.25, 0.25, 0.25)):
    smooth = SmoothingFunction().method3
    hyps = [char_tokens(pred) for pred, _ in pairs]
    refs = [[char_tokens(ref)] for _, ref in pairs]
    return float(corpus_bleu(refs, hyps, weights=weights, smoothing_function=smooth)) * 100.0


def sentence_bleu_char(prediction: str, reference: str, weights=(0.25, 0.25, 0.25, 0.25)):
    smooth = SmoothingFunction().method3
    return float(sentence_bleu([char_tokens(reference)], char_tokens(prediction), weights=weights, smoothing_function=smooth)) * 100.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate character-level BLEU.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--prediction_key", default="prediction")
    parser.add_argument("--reference_key", default="reference")
    args = parser.parse_args()

    pairs = load_pairs(Path(args.input), args.prediction_key, args.reference_key)
    result = {
        "num_pairs": len(pairs),
        "BLEU4_corpus_char_nltk": corpus_bleu_char(pairs) if pairs else 0.0,
        "BLEU1_corpus_char_nltk": corpus_bleu_char(pairs, (1.0, 0.0, 0.0, 0.0)) if pairs else 0.0,
        "BLEU2_corpus_char_nltk": corpus_bleu_char(pairs, (0.5, 0.5, 0.0, 0.0)) if pairs else 0.0,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
