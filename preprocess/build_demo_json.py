import argparse
import json
from pathlib import Path


DEFAULT_PROMPT = "<image> Translate the Manchu archival sentence image into Classical Chinese. Output only the final translation."


def image_field(path: Path, root: Path):
    return [str(path.relative_to(root.parent))]


def build_item(sample_dir: Path, root: Path, image_name: str, prompt: str):
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    image_path = sample_dir / image_name
    if not image_path.exists():
        image_path = sample_dir / "reconstructed_sentence.png"
    if not image_path.exists():
        image_path = sample_dir / "raw_sentence.png"
    translation = metadata.get("classical_chinese") or metadata.get("translation") or metadata.get("target") or ""
    return {
        "id": metadata.get("sample_id", sample_dir.name),
        "conversations": [
            {"from": "human", "value": metadata.get("prompt", prompt)},
            {"from": "gpt", "value": translation},
        ],
        "images": image_field(image_path, root),
    }


def main():
    parser = argparse.ArgumentParser(description="Build a LLaMA-Factory-style multimodal JSON file from demo metadata.")
    parser.add_argument("--demo_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--image_name", default="binarized_sentence.png")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    root = Path(args.demo_dir)
    samples = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not samples:
        raise SystemExit(f"no sample directories found under {root}")
    items = [build_item(sample, root, args.image_name, args.prompt) for sample in samples]

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(items)} examples to {out}")


if __name__ == "__main__":
    main()
