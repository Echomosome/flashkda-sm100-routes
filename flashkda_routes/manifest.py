"""Freeze the runner source identity at correctness time, verify it at timing time."""

import json
from pathlib import Path

from .config import REPO_ROOT, sha256


def write_runner_manifest(build, package_root=None):
    build = Path(build)
    root = Path(package_root) if package_root else REPO_ROOT
    package = root / "flashkda_routes"
    files = sorted(list(package.rglob("*.py")) + list(package.rglob("*.cuh")))
    manifest = {str(p.relative_to(root)): sha256(p) for p in files if p.is_file()}
    (build / "runner_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_runner_manifest(build, package_root=None):
    build = Path(build)
    root = Path(package_root) if package_root else REPO_ROOT
    manifest = json.loads((build / "runner_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        path = root / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"code changed after correctness: {name}; rerun validation")
    return manifest
