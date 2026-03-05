"""Helpers for locating and reading pipeline run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path | None:
    """Walk upward from a starting path until a project root is found."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def default_runs_dir() -> Path:
    """Return the default directory used to store run artifacts."""
    cwd_root = find_project_root(Path.cwd())
    if cwd_root is not None:
        return cwd_root / "runs"

    module_root = find_project_root(Path(__file__).resolve().parent)
    if module_root is not None:
        return module_root / "runs"

    return Path.cwd() / "runs"


def resolve_runs_dir(runs_dir: Path | None = None) -> Path:
    return runs_dir or default_runs_dir()


def save_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def load_manifest(run_id: str, runs_dir: Path | None = None) -> dict[str, Any]:
    runs_root = resolve_runs_dir(runs_dir)
    path = runs_root / run_id / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Run {run_id!r} not found at {path}")
    return json.loads(path.read_text())


def list_runs(runs_dir: Path | None = None) -> list[dict[str, Any]]:
    runs_root = resolve_runs_dir(runs_dir)
    if not runs_root.exists():
        return []

    manifests: list[dict[str, Any]] = []
    for run_dir in sorted(runs_root.iterdir(), reverse=True):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifests.append(json.loads(manifest_path.read_text()))
        except Exception:
            continue
    return manifests
