"""Workflow helpers for the CLI's image-to-CAD pipeline."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ai import analyze_image, iterate_model
from .cad import build_geometry, export_all
from .schemas import CADModel
from .storage import load_manifest, resolve_runs_dir, save_manifest


def parse_formats(formats: str) -> list[str]:
    """Parse a comma-separated export format list."""
    format_list = [item.strip() for item in formats.split(",")]
    if not format_list or any(not item for item in format_list):
        raise ValueError("Formats must be a comma-separated list like 'step,stl'.")
    return format_list


def parse_known_dims(items: list[str]) -> dict[str, float]:
    """Parse repeatable key=value_mm entries into a normalized mapping."""
    known_dims: dict[str, float] = {}
    for raw in items:
        text = raw.strip()
        if "=" not in text:
            raise ValueError(f"Invalid --known-dim {raw!r}. Expected key=value_mm.")

        key, value_text = text.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --known-dim {raw!r}. Missing key.")

        normalized_value = value_text.strip().lower()
        if normalized_value.endswith("mm"):
            normalized_value = normalized_value[:-2].strip()

        try:
            value = float(normalized_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --known-dim {raw!r}. Value must be numeric in mm."
            ) from exc

        if value <= 0:
            raise ValueError(f"Invalid --known-dim {raw!r}. Value must be > 0.")

        known_dims[key] = value
    return known_dims


def build_dry_run_result(
    *,
    image: Path,
    model: str | None,
    formats: list[str],
    hint: str,
    known_dims: dict[str, float],
) -> dict[str, Any]:
    return {
        "status": "dry_run",
        "image": str(image.resolve()),
        "model": model,
        "formats": formats,
        "hint": hint,
        "known_dims": known_dims,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _failure_category(exc: Exception) -> str:
    category = getattr(exc, "category", None)
    if isinstance(category, str) and category:
        return category
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return "schema_invalid"
    return "unknown"


def run_pipeline(
    *,
    image: Path,
    model: str | None,
    hint: str,
    formats: list[str],
    known_dims: dict[str, float],
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute the core image-to-CAD workflow and persist run artifacts."""
    runs_root = resolve_runs_dir(runs_dir)
    run_id = uuid.uuid4().hex[:12]
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dest_image = run_dir / f"input{image.suffix}"
    shutil.copy2(image.resolve(), dest_image)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": _now(),
        "image_path": str(dest_image),
        "model": model,
        "hint": hint,
        "known_dims": known_dims,
        "status": "running",
        "cad_model": None,
        "exports": {},
        "failure_category": None,
    }
    save_manifest(run_dir, manifest)

    try:
        debug_dir = run_dir / "debug"
        cad_model = analyze_image(
            image_path=dest_image,
            model=model,
            hint=hint,
            known_dims=known_dims,
        )

        (run_dir / "cad_model.json").write_text(cad_model.model_dump_json(indent=2))
        manifest["cad_model"] = cad_model.model_dump()

        geometry = build_geometry(cad_model, debug_dir=debug_dir / "ops_pass_0")

        exported = export_all(geometry, run_dir / "output", cad_model.name, formats=formats)
        manifest["exports"] = {name: str(path) for name, path in exported.items()}
        manifest["status"] = "completed"
        save_manifest(run_dir, manifest)

        return {
            "status": "completed",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "cad_model": cad_model.model_dump(),
            "exports": manifest["exports"],
        }
    except Exception as exc:
        category = _failure_category(exc)
        manifest["status"] = "failed"
        manifest["failure_category"] = category
        manifest["error"] = str(exc)
        save_manifest(run_dir, manifest)
        return {
            "status": "failed",
            "error": str(exc),
            "failure_category": category,
            "run_id": run_id,
            "run_dir": str(run_dir),
        }


def export_run(
    *,
    run_id: str,
    formats: list[str],
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    """Rebuild and export geometry for an existing run."""
    runs_root = resolve_runs_dir(runs_dir)
    manifest = load_manifest(run_id, runs_dir=runs_root)
    if manifest.get("cad_model") is None:
        raise ValueError("No CAD model found in this run. Was the analyze stage completed?")

    cad_model = CADModel.model_validate(manifest["cad_model"])
    geometry = build_geometry(cad_model)
    run_dir = runs_root / run_id
    exported = export_all(geometry, run_dir / "output", cad_model.name, formats=formats)
    return {
        "run_id": run_id,
        "exports": {name: str(path) for name, path in exported.items()},
    }


def iterate_run(
    *,
    run_id: str,
    instruction: str,
    model: str | None,
    formats: list[str],
    runs_dir: Path | None = None,
) -> dict[str, Any]:
    """Create a new run by modifying an existing CAD model with AI."""
    runs_root = resolve_runs_dir(runs_dir)
    manifest = load_manifest(run_id, runs_dir=runs_root)
    if manifest.get("cad_model") is None:
        raise ValueError("No CAD model in run. Complete the analyze stage first.")

    current_model = CADModel.model_validate(manifest["cad_model"])
    new_model = iterate_model(
        current_model=current_model,
        instruction=instruction,
        model=model,
    )

    geometry = build_geometry(new_model)
    new_run_id = uuid.uuid4().hex[:12]
    new_run_dir = runs_root / new_run_id
    new_run_dir.mkdir(parents=True, exist_ok=True)

    exported = export_all(geometry, new_run_dir / "output", new_model.name, formats=formats)
    new_manifest: dict[str, Any] = {
        "run_id": new_run_id,
        "created_at": _now(),
        "image_path": manifest["image_path"],
        "model": model or manifest.get("model"),
        "hint": f"Iterated from {run_id}: {instruction}",
        "status": "completed",
        "cad_model": new_model.model_dump(),
        "exports": {name: str(path) for name, path in exported.items()},
    }
    save_manifest(new_run_dir, new_manifest)
    (new_run_dir / "cad_model.json").write_text(new_model.model_dump_json(indent=2))

    return {
        "status": "completed",
        "run_id": new_run_id,
        "parent_run_id": run_id,
        "instruction": instruction,
        "run_dir": str(new_run_dir),
        "exports": new_manifest["exports"],
    }
