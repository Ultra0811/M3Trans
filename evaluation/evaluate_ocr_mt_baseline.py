import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import torch
from sacrebleu.metrics import CHRF
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_bleu import corpus_bleu_char, sentence_bleu_char


DEFAULT_PROMPT = "You are a Manchu-to-Classical-Chinese translator. Translate the following OCR text into Classical Chinese. Output only the translation."


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def clean_prediction(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def load_ocr_map(path: Optional[str], text_key: str):
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k): str(v.get(text_key, v) if isinstance(v, dict) else v) for k, v in data.items()}
    out = {}
    for item in data:
        key = item.get("image") or item.get("image_path") or item.get("path")
        if key:
            out[str(key)] = str(item.get(text_key, item.get("text", "")))
    return out


def get_reference(sample):
    return next((m.get("value", "") for m in sample.get("conversations", []) if m.get("from") == "gpt"), "")


def get_ocr_text(sample, image_paths, ocr_map, sample_ocr_key, concat_mode):
    if sample_ocr_key in sample:
        values = as_list(sample[sample_ocr_key])
    else:
        values = []
        for path in image_paths:
            values.append(ocr_map.get(str(path), ocr_map.get(Path(str(path)).name, "")))
    values = [str(v).strip() for v in values if str(v).strip()]
    if concat_mode == "space":
        return " ".join(values)
    if concat_mode == "newline":
        return "\n".join(values)
    return "".join(values)


class TextMT:
    def __init__(self, model_path: str, prompt: str):
        self.prompt = prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True)

    @torch.inference_mode()
    def translate(self, source_text: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        prompt = f"{self.prompt}\n\n{source_text}\n\nTranslation:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        kwargs = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            kwargs.update({"do_sample": False})
        output = self.model.generate(**inputs, **kwargs)
        generated = output[:, prompt_len:]
        return clean_prediction(self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0])


def main():
    parser = argparse.ArgumentParser(description="Evaluate an OCR→MT cascaded baseline from OCR outputs and a text MT model.")
    parser.add_argument("--mt_model_path", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ocr_json", default=None)
    parser.add_argument("--ocr_text_key", default="roman")
    parser.add_argument("--sample_ocr_key", default="ocr_texts")
    parser.add_argument("--concat_mode", choices=["join", "space", "newline"], default="space")
    parser.add_argument("--mt_prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.test_json).read_text(encoding="utf-8"))
    ocr_map = load_ocr_map(args.ocr_json, args.ocr_text_key)
    mt = TextMT(args.mt_model_path, args.mt_prompt)
    chrf = CHRF(word_order=2)
    pairs, items = [], []
    start = time.time()

    for idx, sample in enumerate(tqdm(data, desc="Evaluating OCR→MT")):
        image_paths = as_list(sample.get("images"))
        reference = get_reference(sample)
        source = get_ocr_text(sample, image_paths, ocr_map, args.sample_ocr_key, args.concat_mode)
        if not source:
            items.append({"idx": idx, "images": image_paths, "reference": reference, "prediction": "", "error": "missing OCR text"})
            continue
        prediction = mt.translate(source, args.max_new_tokens, args.temperature, args.top_p)
        pairs.append((prediction, reference))
        items.append({
            "idx": idx,
            "images": image_paths,
            "ocr_text": source,
            "reference": reference,
            "prediction": prediction,
            "bleu4_sentence_char_nltk": sentence_bleu_char(prediction, reference),
            "chrf_sentence": chrf.sentence_score(prediction, [reference]).score,
        })

    preds = [pred for pred, _ in pairs]
    refs = [ref for _, ref in pairs]
    summary = {
        "mt_model_path": args.mt_model_path,
        "num_examples_total": len(data),
        "num_scored": len(pairs),
        "metrics": {
            "BLEU4_corpus_char_nltk": corpus_bleu_char(pairs) if pairs else 0.0,
            "chrF++_corpus": chrf.corpus_score(preds, [refs]).score if pairs else 0.0,
        },
        "time_sec": time.time() - start,
    }
    (out_dir / "pred_detail_nltk_bleu.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics_summary_nltk_bleu.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
