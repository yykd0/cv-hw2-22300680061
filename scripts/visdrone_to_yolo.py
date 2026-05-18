from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def split_dir(root: Path, split: str) -> Path:
    candidates = {
        "train": ["VisDrone2019-DET-train", "train"],
        "val": ["VisDrone2019-DET-val", "val", "valid"],
        "test": ["VisDrone2019-DET-test-dev", "test"],
    }
    for name in candidates[split]:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find VisDrone {split} directory under {root}")


def convert_annotation(src: Path, dst: Path, image_size: tuple[int, int]) -> int:
    width, height = image_size
    rows = []
    if not src.exists():
        dst.write_text("", encoding="utf-8")
        return 0
    for line in src.read_text(encoding="utf-8").strip().splitlines():
        parts = [float(x) for x in line.split(",")[:8]]
        x, y, w, h, score, category, truncation, occlusion = parts
        category = int(category)
        if category < 1 or category > 10 or w <= 1 or h <= 1:
            continue
        cls_id = category - 1
        cx = (x + w / 2.0) / width
        cy = (y + h / 2.0) / height
        rows.append(f"{cls_id} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")
    dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def convert_split(root: Path, out: Path, split: str) -> tuple[int, int]:
    src_split = split_dir(root, split)
    image_dir = src_split / "images"
    anno_dir = src_split / "annotations"
    out_images = out / "images" / split
    out_labels = out / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    image_count = 0
    box_count = 0
    for image_path in sorted(image_dir.glob("*.*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(image_path) as img:
            size = img.size
        label_path = out_labels / f"{image_path.stem}.txt"
        box_count += convert_annotation(anno_dir / f"{image_path.stem}.txt", label_path, size)
        target_image = out_images / image_path.name
        shutil.copy2(image_path, target_image)
        image_count += 1
    return image_count, box_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visdrone-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in args.splits:
        images, boxes = convert_split(args.visdrone_root, args.out, split)
        summary[split] = {"images": images, "boxes": boxes}
        print(f"{split}: {images} images, {boxes} boxes")
    yaml_path = args.out / "visdrone.yaml"
    names = "\n".join([f"  {i}: {name}" for i, name in enumerate(VISDRONE_CLASSES)])
    yaml_path.write_text(
        f"path: {args.out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
