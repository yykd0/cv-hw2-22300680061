"""Create a lightweight HW3_Q1_submission.zip for upload."""

from __future__ import annotations

import fnmatch
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "submission"
ZIP_PATH = SUBMISSION_DIR / "HW3_Q1_submission.zip"

EXCLUDE_DIRS = {"external", "data", "upload_parts", "__pycache__", ".git"}
EXCLUDE_PATTERNS = {
    "*.pyc",
    "*.pth",
    "*.ckpt",
    "*.pt",
    "*.ply",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.part*",
}


def excluded(path: Path) -> bool:
    parts = set(path.relative_to(ROOT).parts)
    if parts & EXCLUDE_DIRS:
        return True
    name = path.name
    return any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDE_PATTERNS)


def run(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)


def main() -> None:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    run("collect_results.py")
    run("make_report.py")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(ROOT.rglob("*")):
            if path.is_dir() or excluded(path):
                continue
            rel = path.relative_to(ROOT)
            zf.write(path, rel.as_posix())

    print(f"Wrote {ZIP_PATH}")


if __name__ == "__main__":
    main()
