import argparse
import json
from pathlib import Path
from typing import Optional

from PIL import Image


def load_order(input_dir: Path, metadata_path: Optional[Path], pattern: str):
    if metadata_path and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        order = metadata.get("word_order") or metadata.get("word_images") or metadata.get("word_fragments")
        if order:
            return [input_dir / item for item in order]
    return sorted(input_dir.glob(pattern))


def trim_white(image: Image.Image, margin: int) -> Image.Image:
    gray = image.convert("L")
    pixels = gray.load()
    w, h = gray.size
    xs, ys = [], []
    for y in range(h):
        for x in range(w):
            if pixels[x, y] < 250:
                xs.append(x)
                ys.append(y)
    if not xs:
        return image.convert("RGB")
    box = (
        max(min(xs) - margin, 0),
        max(min(ys) - margin, 0),
        min(max(xs) + margin + 1, w),
        min(max(ys) + margin + 1, h),
    )
    return image.crop(box).convert("RGB")


def reconstruct(images, direction: str, gap: int, background: int) -> Image.Image:
    if direction == "vertical":
        width = max(img.width for img in images)
        height = sum(img.height for img in images) + gap * (len(images) - 1)
        canvas = Image.new("RGB", (width, height), (background, background, background))
        y = 0
        for img in images:
            x = (width - img.width) // 2
            canvas.paste(img, (x, y))
            y += img.height + gap
        return canvas

    width = sum(img.width for img in images) + gap * (len(images) - 1)
    height = max(img.height for img in images)
    canvas = Image.new("RGB", (width, height), (background, background, background))
    x = 0
    for img in images:
        y = (height - img.height) // 2
        canvas.paste(img, (x, y))
        x += img.width + gap
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Reconstruct a sentence-level image from ordered word fragments.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--direction", choices=["vertical", "horizontal"], default="vertical")
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--trim_margin", type=int, default=2)
    parser.add_argument("--background", type=int, default=255)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    metadata = Path(args.metadata) if args.metadata else None
    paths = load_order(input_dir, metadata, args.pattern)
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit(f"no word images found under {input_dir}")

    images = [trim_white(Image.open(path).convert("RGB"), args.trim_margin) for path in paths]
    output = reconstruct(images, args.direction, args.gap, args.background)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)


if __name__ == "__main__":
    main()
