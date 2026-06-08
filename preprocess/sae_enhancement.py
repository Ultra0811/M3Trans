import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def ensure_white_background(gray: np.ndarray) -> np.ndarray:
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return 255 - gray if float(np.mean(border)) < 127.5 else gray


def resize_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    if long_edge <= 0:
        return image
    height, width = image.shape[:2]
    scale = float(long_edge) / float(max(height, width))
    if abs(scale - 1.0) < 1e-6:
        return image
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, size, interpolation=interpolation)


def normalize_response(response: np.ndarray, epsilon: float) -> np.ndarray:
    response = response.astype(np.float32)
    minimum = float(np.min(response))
    maximum = float(np.max(response))
    return (response - minimum) / (maximum - minimum + epsilon)


def resize_to_square_canvas(image: np.ndarray, output_size: int, background: int = 255) -> np.ndarray:
    if output_size <= 0:
        return image
    height, width = image.shape[:2]
    scale = float(output_size) / float(max(height, width))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((output_size, output_size), background, dtype=resized.dtype)
    x = (output_size - resized_width) // 2
    y = (output_size - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def enhance(
    image: np.ndarray,
    window_size: int = 5,
    alpha: float = 0.60,
    lambda_sae: float = 1.20,
    epsilon: float = 1e-6,
    long_edge: int = 1024,
    output_size: int = 448,
) -> np.ndarray:
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1

    gray = ensure_white_background(to_gray(image))
    gray = resize_long_edge(gray, long_edge)
    gray01 = gray.astype(np.float32) / 255.0

    kernel = (window_size, window_size)
    local_mean = cv2.boxFilter(gray01, -1, kernel, normalize=True, borderType=cv2.BORDER_REFLECT)
    local_square_mean = cv2.boxFilter(gray01 * gray01, -1, kernel, normalize=True, borderType=cv2.BORDER_REFLECT)
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))
    local_contrast = normalize_response(np.abs(gray01 - local_mean) / (local_std + epsilon), epsilon)

    grad_x = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    gradient_response = normalize_response(np.sqrt(grad_x * grad_x + grad_y * grad_y), epsilon)

    structure_response = alpha * local_contrast + (1.0 - alpha) * gradient_response
    foreground = 1.0 - gray01
    enhanced_foreground = np.clip(foreground * (1.0 + lambda_sae * structure_response), 0.0, 1.0)
    enhanced = np.clip((1.0 - enhanced_foreground) * 255.0, 0.0, 255.0).astype(np.uint8)

    return resize_to_square_canvas(enhanced, output_size)


def process_file(src: Path, dst: Path, args):
    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"failed to read image: {src}")
    out = enhance(
        image,
        window_size=args.window_size,
        alpha=args.alpha,
        lambda_sae=args.lambda_sae,
        epsilon=args.epsilon,
        long_edge=args.long_edge,
        output_size=args.output_size,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out)


def main():
    parser = argparse.ArgumentParser(description="Apply structure-aware enhancement to Manchu sentence images.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument("--lambda_sae", type=float, default=1.20)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--long_edge", type=int, default=1024)
    parser.add_argument("--output_size", type=int, default=448)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if src.is_file():
        process_file(src, dst, args)
        return

    files = sorted(p for p in src.rglob(args.pattern) if p.is_file())
    if not files:
        raise SystemExit(f"no images matched {args.pattern} under {src}")
    for path in tqdm(files, desc="Enhancing"):
        process_file(path, dst / path.relative_to(src), args)


if __name__ == "__main__":
    main()
