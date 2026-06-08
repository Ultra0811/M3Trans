import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return np.sum(a * b, axis=1)


def char_sequence(text: str) -> str:
    return " ".join(ch for ch in str(text).strip() if not ch.isspace())


class SemanticReward:
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-large-zh-v1.5",
        lambda_sem: float = 0.50,
        device: Optional[str] = None,
    ):
        if lambda_sem < 0.0 or lambda_sem > 1.0:
            raise ValueError("lambda_sem must be in [0, 1]")
        self.lambda_sem = lambda_sem
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def rouge_l(self, prediction: str, reference: str) -> float:
        scores = self.rouge.score(char_sequence(reference), char_sequence(prediction))
        return float(scores["rougeL"].fmeasure)

    def score(self, predictions, references):
        pred_emb = self.model.encode(predictions, normalize_embeddings=True, convert_to_numpy=True)
        ref_emb = self.model.encode(references, normalize_embeddings=True, convert_to_numpy=True)
        semantic_cosine = cosine(pred_emb, ref_emb)
        rouge_l = np.array([self.rouge_l(pred, ref) for pred, ref in zip(predictions, references)], dtype=np.float32)
        return (self.lambda_sem * rouge_l + (1.0 - self.lambda_sem) * semantic_cosine).tolist()


def semantic_reward(prediction: str, reference: str, model: Optional[SemanticReward] = None) -> float:
    scorer = model or SemanticReward()
    return float(scorer.score([prediction], [reference])[0])


def main():
    parser = argparse.ArgumentParser(description="Compute semantic rewards from ROUGE-L and BGE/SBERT similarity.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="BAAI/bge-large-zh-v1.5")
    parser.add_argument("--lambda_sem", type=float, default=0.50)
    parser.add_argument("--prediction_key", default="prediction")
    parser.add_argument("--reference_key", default="reference")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    preds = [str(item.get(args.prediction_key, "")) for item in data]
    refs = [str(item.get(args.reference_key, "")) for item in data]
    scorer = SemanticReward(args.model, lambda_sem=args.lambda_sem)
    scores = scorer.score(preds, refs)
    for item, score in zip(data, scores):
        item["semantic_reward"] = float(score)
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
