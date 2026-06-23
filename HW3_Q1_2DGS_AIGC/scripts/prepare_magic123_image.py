"""Prepare a simple RGBA image for Magic123 by removing near-white background.

This is a deterministic helper for already generated images. It is not a full
segmentation model; inspect the result before using it for final training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def remove_white_background(src: Path, dst: Path, threshold: int, feather: int) -> None:
    img = Image.open(src).convert("RGBA")
    pixels = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            distance = max(255 - r, 255 - g, 255 - b)
            if distance <= threshold:
                alpha = 0
            elif distance <= threshold + feather:
                alpha = int(255 * (distance - threshold) / max(1, feather))
            else:
                alpha = a
            pixels[x, y] = (r, g, b, alpha)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("assets/magic123_input/main.png"))
    parser.add_argument("--output", type=Path, default=Path("assets/magic123_input/rgba.png"))
    parser.add_argument("--threshold", type=int, default=18)
    parser.add_argument("--feather", type=int, default=32)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = args.input if args.input.is_absolute() else root / args.input
    dst = args.output if args.output.is_absolute() else root / args.output
    if not src.exists():
        raise FileNotFoundError(src)
    remove_white_background(src, dst, args.threshold, args.feather)
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
