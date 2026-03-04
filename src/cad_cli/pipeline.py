"""Pipeline orchestration: manages runs, artifacts, and stage execution."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .schemas import CADModel

RUNS_DIR = Path.cwd() / "runs"


class StageStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class StageRecord(BaseModel):
    name: str
    status: StageStatus = StageStatus.pending
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    artifacts: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class RunManifest(BaseModel):
    run_id: str
    created_at: str
    image_path: str
    provider: str
    model: Optional[str] = None
    hint: str = ""
    stages: list[StageRecord] = Field(default_factory=list)
    cad_model: Optional[dict] = None
    exports: dict[str, str] = Field(default_factory=dict)
    status: str = "created"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineRun:
    """Manages a single pipeline run with artifact tracking."""

    def __init__(
        self,
        image_path: Path,
        provider: str,
        model: Optional[str] = None,
        hint: str = "",
        run_id: Optional[str] = None,
        runs_dir: Optional[Path] = None,
    ):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.runs_dir = runs_dir or RUNS_DIR
        self.run_dir = self.runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Copy input image into run directory
        dest_image = self.run_dir / f"input{image_path.suffix}"
        if not dest_image.exists():
            shutil.copy2(image_path, dest_image)

        self.manifest = RunManifest(
            run_id=self.run_id,
            created_at=_now(),
            image_path=str(dest_image),
            provider=provider,
            model=model,
            hint=hint,
            stages=[
                StageRecord(name="analyze"),
                StageRecord(name="validate"),
                StageRecord(name="build"),
                StageRecord(name="export"),
            ],
        )
        self._save_manifest()

    def _save_manifest(self) -> None:
        path = self.run_dir / "manifest.json"
        path.write_text(self.manifest.model_dump_json(indent=2))

    def _get_stage(self, name: str) -> StageRecord:
        for s in self.manifest.stages:
            if s.name == name:
                return s
        raise ValueError(f"Unknown stage: {name}")

    def _start_stage(self, name: str) -> StageRecord:
        stage = self._get_stage(name)
        stage.status = StageStatus.running
        stage.started_at = _now()
        self._save_manifest()
        return stage

    def _complete_stage(self, name: str, artifacts: list[str] | None = None) -> None:
        stage = self._get_stage(name)
        stage.status = StageStatus.completed
        stage.completed_at = _now()
        if artifacts:
            stage.artifacts.extend(artifacts)
        self._save_manifest()

    def _fail_stage(self, name: str, error: str) -> None:
        stage = self._get_stage(name)
        stage.status = StageStatus.failed
        stage.completed_at = _now()
        stage.error = error
        self.manifest.status = "failed"
        self._save_manifest()

    def run_analyze(self) -> CADModel:
        """Stage 1 & 2: Send image to VLM, get validated CADModel."""
        from .vlm import analyze_image

        self._start_stage("analyze")
        try:
            image_path = Path(self.manifest.image_path)
            cad_model = analyze_image(
                image_path=image_path,
                provider=self.manifest.provider,
                model=self.manifest.model,
                hint=self.manifest.hint,
            )
        except Exception as e:
            self._fail_stage("analyze", str(e))
            raise

        # Save raw VLM output
        raw_path = self.run_dir / "cad_model.json"
        raw_path.write_text(cad_model.model_dump_json(indent=2))
        self.manifest.cad_model = cad_model.model_dump()
        self._complete_stage("analyze", [str(raw_path)])

        # Validate (trivial since Pydantic already validated)
        self._start_stage("validate")
        self._complete_stage("validate")

        return cad_model

    def run_build_and_export(
        self, cad_model: CADModel, formats: list[str] | None = None
    ) -> dict[str, Path]:
        """Stage 3 & 4: Build geometry and export."""
        from .cad import build_geometry, export_all

        formats = formats or ["step", "stl"]

        self._start_stage("build")
        try:
            geometry = build_geometry(cad_model)
        except Exception as e:
            self._fail_stage("build", str(e))
            raise
        self._complete_stage("build")

        self._start_stage("export")
        try:
            exported = export_all(
                geometry,
                self.run_dir / "output",
                cad_model.name,
                formats=formats,
            )
        except Exception as e:
            self._fail_stage("export", str(e))
            raise

        self.manifest.exports = {k: str(v) for k, v in exported.items()}
        self._complete_stage("export", [str(v) for v in exported.values()])
        self.manifest.status = "completed"
        self._save_manifest()
        return exported

    def execute(self, formats: list[str] | None = None) -> dict[str, Any]:
        """Run the full pipeline end to end."""
        self.manifest.status = "running"
        self._save_manifest()

        cad_model = self.run_analyze()
        exported = self.run_build_and_export(cad_model, formats)

        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "status": self.manifest.status,
            "cad_model": cad_model.model_dump(),
            "exports": {k: str(v) for k, v in exported.items()},
        }


def load_manifest(run_id: str, runs_dir: Optional[Path] = None) -> RunManifest:
    """Load a manifest from an existing run."""
    runs_dir = runs_dir or RUNS_DIR
    manifest_path = runs_dir / run_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run {run_id!r} not found at {manifest_path}")
    return RunManifest.model_validate_json(manifest_path.read_text())


def list_runs(runs_dir: Optional[Path] = None) -> list[RunManifest]:
    """List all runs, sorted by creation time descending."""
    runs_dir = runs_dir or RUNS_DIR
    if not runs_dir.exists():
        return []
    manifests = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        mf = d / "manifest.json"
        if mf.exists():
            try:
                manifests.append(RunManifest.model_validate_json(mf.read_text()))
            except Exception:
                continue
    return manifests
