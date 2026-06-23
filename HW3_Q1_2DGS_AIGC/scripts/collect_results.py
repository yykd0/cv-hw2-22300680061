"""Collect HW3 Q1 artifact status into results/manifest.json."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "manifest.json"


@dataclass
class Artifact:
    key: str
    description: str
    status: str
    paths: list[str]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def find_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    if not root.exists():
        return []
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.rglob(pattern))
    return sorted({p for p in found if p.is_file()})


def artifact(key: str, description: str, root: Path, patterns: Iterable[str]) -> Artifact:
    paths = find_files(root, patterns)
    return Artifact(
        key=key,
        description=description,
        status="present" if paths else "pending",
        paths=[rel(p) for p in paths[:12]],
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    artifacts = [
        artifact("object_a_2dgs_checkpoint", "Object A 2DGS trained Gaussian checkpoint / point cloud", RESULTS / "2dgs" / "bicycle", ["*.ply", "*.pth", "*.ckpt"]),
        artifact("object_a_2dgs_render", "Object A 2DGS rendered images or videos", RESULTS / "2dgs" / "bicycle", ["*.png", "*.jpg", "*.mp4"]),
        artifact("object_b_threestudio_checkpoint", "Object B threestudio checkpoint", RESULTS / "threestudio", ["*.ckpt", "*.pth"]),
        artifact("object_b_threestudio_mesh_or_video", "Object B threestudio mesh or turntable video", RESULTS / "threestudio", ["*.obj", "*.ply", "*.mp4", "*.png"]),
        artifact("object_c_aigc_input", "Object C generated single-view image and RGBA foreground", ROOT / "assets" / "magic123_input", ["main.png", "rgba.png"]),
        artifact("object_c_magic123_checkpoint", "Object C Magic123 checkpoint", RESULTS / "magic123", ["*.pth", "*.ckpt"]),
        artifact("object_c_magic123_mesh_or_video", "Object C Magic123 mesh or turntable video", RESULTS / "magic123", ["*.obj", "*.ply", "*.mp4", "*.png"]),
        artifact("unified_blender_scene", "Unified Blender scene render or turntable video", RESULTS / "blender", ["*.png", "*.jpg", "*.mp4"]),
        artifact("training_logs", "Training and rendering logs", RESULTS / "logs", ["*.log", "*.out", "*.err"]),
    ]

    manifest = {
        "project": "HW3_Q1_2DGS_AIGC",
        "root": str(ROOT),
        "artifacts": [asdict(a) for a in artifacts],
        "pending_count": sum(a.status != "present" for a in artifacts),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {MANIFEST}")
    print(f"Pending artifacts: {manifest['pending_count']}")


if __name__ == "__main__":
    main()
