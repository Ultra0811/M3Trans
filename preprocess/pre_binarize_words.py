import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def ensure_white_background(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    border = np.concatenate([image[0, :], image[h - 1, :], image[:, 0], image[:, w - 1]])
    return 255 - image if float(border.mean()) < 127.5 else image


def binarize(image: np.ndarray, method: str, block_size: int, c_value: int) -> np.ndarray:
    image = ensure_white_background(image)
    if method == "otsu":
        _, out = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )


def iter_images(input_dir: Path, pattern: str, recursive: bool):
    yield from (input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description="Binarize Manchu word-image fragments.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--method", choices=["adaptive", "otsu"], default="adaptive")
    parser.add_argument("--block_size", type=int, default=99)
    parser.add_argument("--c_value", type=int, default=15)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(p for p in iter_images(input_dir, args.pattern, args.recursive) if p.is_file())
    if not files:
        raise SystemExit(f"no images matched {args.pattern} under {input_dir}")

    for src in tqdm(files, desc="Binarizing"):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        image = read_gray(src)
        out = binarize(image, args.method, args.block_size, args.c_value)
        cv2.imwrite(str(dst), out)


if __name__ == "__main__":
    main()
