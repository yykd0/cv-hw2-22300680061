# HW3 Q1: 2DGS and AIGC 3D Assets

This directory is the public code entry for Computer Vision HW3 Question 1.

## Pipelines

- Object A: Mip-NeRF 360 `bicycle` reconstructed with 2D Gaussian Splatting.
- Object B: text-to-3D generation with threestudio DreamFusion/SDS.
- Object C: image-to-3D generation with Magic123 from an AIGC single image.

Heavy training was run on an AutoDL Ubuntu instance with RTX 4090 24GB VRAM. The local Windows machine was used for checking, report generation, and packaging.

## Directory Layout

```text
HW3_Q1_2DGS_AIGC/
  configs/                 Experiment defaults and upload links.
  docs/                    AutoDL runbook and grading checklist.
  scripts/                 Setup, training, rendering, reporting, packaging.
  assets/                  Prompt files and Magic123 input image slot.
  results/                 Logs, figures, links, manifest and artifact index.
  submission/              Final PDF and lightweight submission package.
```

## Quick Start on AutoDL

```bash
cd HW3_Q1_2DGS_AIGC
bash scripts/setup_autodl.sh
bash scripts/run_all.sh
python scripts/collect_results.py
python scripts/make_report.py
python scripts/package_submission.py
```

## Individual Runs

```bash
bash scripts/run_2dgs_bicycle.sh
bash scripts/run_threestudio_teapot.sh
bash scripts/run_magic123_dragon.sh
python scripts/render_scene.py --project-root . --output-dir results/blender
```

## External Artifacts

Large files are intentionally not committed to GitHub. Model weights, videos, meshes, and the lightweight code package are listed in:

```text
results/links/UPLOAD_MANIFEST_FINAL.md
configs/submission_links.json
```

The final local submission zip also contains the complete lightweight code package `HW3_Q1_submission.zip`.
