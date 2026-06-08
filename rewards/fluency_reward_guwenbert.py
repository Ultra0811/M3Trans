import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class FluencyReward:
    def __init__(self, model_name_or_path: str, device: Optional[str] = None, max_length: int = 128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True).to(self.device)
        self.model.eval()
        if self.tokenizer.mask_token_id is None:
            raise ValueError("the tokenizer must provide a mask token for pseudo-likelihood scoring")

    @torch.inference_mode()
    def score_one(self, text: str) -> float:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        input_ids = encoded["input_ids"]
        if input_ids.shape[1] <= 2:
            return 0.0
        log_probs = []
        for pos in range(1, input_ids.shape[1] - 1):
            masked = input_ids.clone()
            target = input_ids[0, pos].item()
            masked[0, pos] = self.tokenizer.mask_token_id
            logits = self.model(input_ids=masked, attention_mask=encoded.get("attention_mask")).logits[0, pos]
            log_prob = torch.log_softmax(logits, dim=-1)[target]
            log_probs.append(float(log_prob.detach().cpu()))
        return sum(log_probs) / max(len(log_probs), 1)

    def score(self, texts):
        return [self.score_one(text) for text in texts]


def main():
    parser = argparse.ArgumentParser(description="Compute fluency rewards with GuwenBERT pseudo log-likelihood scoring.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prediction_key", default="prediction")
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    texts = [str(item.get(args.prediction_key, "")) for item in data]
    scorer = FluencyReward(args.model, max_length=args.max_length)
    scores = scorer.score(texts)
    for item, score in zip(data, scores):
        item["fluency_reward"] = float(score)
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
