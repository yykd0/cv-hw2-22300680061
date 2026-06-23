"""Blender script for a unified A/B/C turntable scene.

Run with:
  blender -b --python scripts/render_scene.py -- --project-root . --output-dir results/blender
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("results/blender"))
    parser.add_argument("--allow-proxy", action="store_true")
    return parser.parse_args(argv)


def find_mesh(root: Path, candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        base = root / candidate
        if base.is_file():
            return base
        if base.is_dir():
            for pattern in ("*.obj", "*.ply"):
                matches = sorted(base.rglob(pattern))
                if matches:
                    return matches[-1]
    return None


def import_mesh(bpy, path: Path):
    suffix = path.suffix.lower()
    if suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported mesh format: {path}")
    return bpy.context.object


def normalize_object(obj, x: float) -> None:
    obj.location = (x, 0, 0)
    obj.rotation_euler[2] = 0
    max_dim = max(obj.dimensions) if max(obj.dimensions) > 0 else 1.0
    scale = 1.8 / max_dim
    obj.scale = (scale, scale, scale)
    obj.location.z = 0


def create_proxy(bpy, name: str, x: float, primitive: str):
    if primitive == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1.6, location=(x, 0, 0.8))
    elif primitive == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.9, location=(x, 0, 0.9))
    else:
        bpy.ops.mesh.primitive_cone_add(vertices=6, radius1=0.9, depth=1.8, location=(x, 0, 0.9))
    obj = bpy.context.object
    obj.name = name
    return obj


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import bpy

    bpy.ops.object.delete()

    meshes = [
        ("A_2DGS_bicycle", -2.5, "cube", find_mesh(root, [Path("results/2dgs/bicycle"), Path("results/meshes/object_a")])) ,
        ("B_SDS_teapot", 0.0, "sphere", find_mesh(root, [Path("results/threestudio"), Path("results/meshes/object_b")])) ,
        ("C_Magic123_dragon", 2.5, "cone", find_mesh(root, [Path("results/magic123/dragon"), Path("results/meshes/object_c")])) ,
    ]

    for name, x, primitive, mesh_path in meshes:
        if mesh_path:
            obj = import_mesh(bpy, mesh_path)
            obj.name = name
            normalize_object(obj, x)
        elif args.allow_proxy:
            create_proxy(bpy, name + "_proxy", x, primitive)
        else:
            raise FileNotFoundError(f"Missing mesh for {name}")

    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0, 0, -0.02))
    plane = bpy.context.object
    plane.name = "matte_floor"

    bpy.ops.object.light_add(type="AREA", location=(0, -4, 5))
    light = bpy.context.object
    light.name = "softbox"
    light.data.energy = 450
    light.data.size = 5

    bpy.ops.object.camera_add(location=(0, -6.5, 2.6), rotation=(math.radians(68), 0, 0))
    bpy.context.scene.camera = bpy.context.object

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.render.resolution_x = 1600
    bpy.context.scene.render.resolution_y = 900
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 120
    bpy.context.scene.render.filepath = str(out_dir / "unified_scene.png")
    bpy.ops.render.render(write_still=True)

    empty = bpy.data.objects.new("turntable_center", None)
    bpy.context.collection.objects.link(empty)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and obj.name != "matte_floor":
            obj.parent = empty

    for frame in range(1, 121):
        bpy.context.scene.frame_set(frame)
        empty.rotation_euler[2] = 2 * math.pi * (frame - 1) / 120
        empty.keyframe_insert(data_path="rotation_euler", frame=frame)

    bpy.context.scene.render.filepath = str(out_dir / "unified_turntable.mp4")
    bpy.context.scene.render.image_settings.file_format = "FFMPEG"
    bpy.context.scene.render.ffmpeg.format = "MPEG4"
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
