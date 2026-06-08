import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class StyleReward:
    def __init__(self, model_name_or_path: str, positive_label: int = 1, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.positive_label = positive_label
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, trust_remote_code=True).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, texts, batch_size: int = 16):
        scores = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            label = min(self.positive_label, log_probs.shape[-1] - 1)
            scores.extend(log_probs[:, label].detach().cpu().tolist())
        return scores


def main():
    parser = argparse.ArgumentParser(description="Compute Classical Chinese style log-probability rewards with a sequence classifier.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prediction_key", default="prediction")
    parser.add_argument("--positive_label", type=int, default=1)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    texts = [str(item.get(args.prediction_key, "")) for item in data]
    scorer = StyleReward(args.model, args.positive_label)
    scores = scorer.score(texts)
    for item, score in zip(data, scores):
        item["style_reward"] = float(score)
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
