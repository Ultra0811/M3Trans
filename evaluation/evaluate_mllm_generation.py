import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from sacrebleu.metrics import CHRF
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

from evaluate_bleu import corpus_bleu_char, sentence_bleu_char


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_instruction(conversations):
    human = next((m.get("value", "") for m in conversations if m.get("from") == "human"), "")
    return re.sub(r"^(?:<image>)+\s*", "", human.strip()).strip()


def extract_reference(conversations):
    return next((m.get("value", "") for m in conversations if m.get("from") == "gpt"), "")


def build_messages(num_images: int, instruction: str):
    content = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": instruction})
    return [{"role": "user", "content": content}]


def load_images(paths):
    return [Image.open(path).convert("RGB") for path in paths]


@torch.inference_mode()
def generate_one(model, processor, messages, images, max_new_tokens, temperature, top_p):
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    kwargs = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        kwargs.update({"do_sample": False})
    output = model.generate(**inputs, **kwargs)
    generated = output[:, prompt_len:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    text = re.sub(r"^<\|im_start\|>?\s*assistant\s*", "", text).strip()
    return re.sub(r"<\|im_end\|>.*$", "", text, flags=re.S).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate translations with an OCR-free MLLM and compute BLEU/chrF++.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForVision2Seq.from_pretrained(args.model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    data = load_json(args.test_json)

    items, pairs = [], []
    chrf = CHRF(word_order=2)
    start = time.time()
    for idx, sample in enumerate(tqdm(data, desc="Evaluating")):
        try:
            conversations = sample.get("conversations", [])
            image_paths = as_list(sample.get("images"))
            instruction = extract_instruction(conversations)
            reference = extract_reference(conversations)
            images = load_images(image_paths)
            prediction = generate_one(
                model,
                processor,
                build_messages(len(images), instruction),
                images,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )
            pairs.append((prediction, reference))
            items.append({
                "idx": idx,
                "images": image_paths,
                "instruction": instruction,
                "reference": reference,
                "prediction": prediction,
                "bleu4_sentence_char_nltk": sentence_bleu_char(prediction, reference),
                "chrf_sentence": chrf.sentence_score(prediction, [reference]).score,
            })
        except Exception as exc:
            items.append({"idx": idx, "error": str(exc), "images": as_list(sample.get("images"))})

    preds = [pred for pred, _ in pairs]
    refs = [ref for _, ref in pairs]
    summary = {
        "model_path": args.model_path,
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
