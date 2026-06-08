import argparse
import json
from pathlib import Path

from sacrebleu.metrics import CHRF


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate chrF++ with sacrebleu.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--prediction_key", default="prediction")
    parser.add_argument("--reference_key", default="reference")
    args = parser.parse_args()

    pairs = load_pairs(Path(args.input), args.prediction_key, args.reference_key)
    metric = CHRF(word_order=2)
    preds = [pred for pred, _ in pairs]
    refs = [ref for _, ref in pairs]
    result = {
        "num_pairs": len(pairs),
        "chrF++_corpus": float(metric.corpus_score(preds, [refs]).score) if pairs else 0.0,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
