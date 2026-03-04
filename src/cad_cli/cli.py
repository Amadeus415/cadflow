"""Agent-native CLI entry point."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(
    name="cad-cli",
    help="Agent-native CLI: image → parametric CAD via VLM-described geometry.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()

RUNS_DIR = Path.cwd() / "runs"


def _output_json(data) -> None:
    """Print machine-readable JSON to stdout."""
    print(json.dumps(data, indent=2, default=str))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Run helpers (replaces pipeline.py)
# ---------------------------------------------------------------------------


def _save_manifest(run_dir: Path, manifest: dict) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def _load_manifest(run_id: str, runs_dir: Path = RUNS_DIR) -> dict:
    path = runs_dir / run_id / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Run {run_id!r} not found at {path}")
    return json.loads(path.read_text())


def _list_runs(runs_dir: Path = RUNS_DIR) -> list[dict]:
    if not runs_dir.exists():
        return []
    manifests = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        mf = d / "manifest.json"
        if mf.exists():
            try:
                manifests.append(json.loads(mf.read_text()))
            except Exception:
                continue
    return manifests


# ---------------------------------------------------------------------------
# cad-cli run
# ---------------------------------------------------------------------------


@app.command()
def run(
    image: Path = typer.Argument(..., help="Path to input image", exists=True),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Override VLM model name"
    ),
    hint: str = typer.Option(
        "", "--hint", "-h", help="Text hint for the VLM (e.g. 'this is a bracket')"
    ),
    formats: str = typer.Option(
        "step,stl", "--formats", "-f", help="Comma-separated export formats"
    ),
    output_json: bool = typer.Option(
        False, "--json", "-j", help="Output results as JSON (agent-friendly)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate inputs without running the pipeline"
    ),
    runs_dir: Optional[Path] = typer.Option(
        None, "--runs-dir", help="Custom directory for run artifacts"
    ),
) -> None:
    """Run the full image → CAD pipeline."""
    from .cad import build_geometry, export_all
    from .vlm import analyze_image

    fmt_list = [f.strip() for f in formats.split(",")]
    rd = runs_dir or RUNS_DIR

    if dry_run:
        result = {
            "status": "dry_run",
            "image": str(image.resolve()),
            "model": model,
            "formats": fmt_list,
            "hint": hint,
        }
        if output_json:
            _output_json(result)
        else:
            rprint("[green]Dry run OK.[/green] Pipeline would run with:")
            rprint(f"  Image:    {image.resolve()}")
            rprint(f"  Model:    {model or 'default'}")
            rprint(f"  Formats:  {fmt_list}")
        return

    run_id = uuid.uuid4().hex[:12]
    run_dir = rd / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy input image
    dest_image = run_dir / f"input{image.suffix}"
    shutil.copy2(image.resolve(), dest_image)

    manifest = {
        "run_id": run_id,
        "created_at": _now(),
        "image_path": str(dest_image),
        "model": model,
        "hint": hint,
        "status": "running",
        "cad_model": None,
        "exports": {},
    }
    _save_manifest(run_dir, manifest)

    if not output_json:
        rprint(f"[bold blue]Starting pipeline[/bold blue]  run_id={run_id}")

    try:
        # Stage 1: Analyze image → CADModel
        cad_model = analyze_image(image_path=dest_image, model=model, hint=hint)

        # Save intermediate JSON
        (run_dir / "cad_model.json").write_text(cad_model.model_dump_json(indent=2))
        manifest["cad_model"] = cad_model.model_dump()

        # Stage 2: Build geometry + export
        geometry = build_geometry(cad_model)
        exported = export_all(geometry, run_dir / "output", cad_model.name, formats=fmt_list)

        manifest["exports"] = {k: str(v) for k, v in exported.items()}
        manifest["status"] = "completed"
        _save_manifest(run_dir, manifest)

    except Exception as e:
        manifest["status"] = "failed"
        manifest["error"] = str(e)
        _save_manifest(run_dir, manifest)
        if output_json:
            _output_json({"status": "failed", "error": str(e), "run_id": run_id})
        else:
            rprint(f"[bold red]Pipeline failed:[/bold red] {e}")
        raise typer.Exit(code=1)

    result = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": "completed",
        "cad_model": cad_model.model_dump(),
        "exports": manifest["exports"],
    }
    if output_json:
        _output_json(result)
    else:
        rprint(f"[bold green]Pipeline completed![/bold green]  run_id={run_id}")
        rprint(f"  Run dir: {run_dir}")
        for fmt, path in manifest["exports"].items():
            rprint(f"  {fmt.upper()}: {path}")


# ---------------------------------------------------------------------------
# cad-cli status
# ---------------------------------------------------------------------------


@app.command()
def status(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
) -> None:
    """Show the status of a pipeline run."""
    rd = runs_dir or RUNS_DIR
    try:
        manifest = _load_manifest(run_id, runs_dir=rd)
    except FileNotFoundError as e:
        if output_json:
            _output_json({"error": str(e)})
        else:
            rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    if output_json:
        _output_json(manifest)
        return

    rprint(f"[bold]Run {manifest['run_id']}[/bold]  status={manifest['status']}")
    rprint(f"  Created: {manifest['created_at']}")
    rprint(f"  Image:   {manifest['image_path']}")
    rprint(f"  Model:   {manifest.get('model') or 'default'}")

    if manifest.get("exports"):
        rprint("\n[bold]Exports:[/bold]")
        for fmt, path in manifest["exports"].items():
            rprint(f"  {fmt.upper()}: {path}")

    if manifest.get("error"):
        rprint(f"\n[bold red]Error:[/bold red] {manifest['error']}")


# ---------------------------------------------------------------------------
# cad-cli list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_runs_cmd(
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
) -> None:
    """List all pipeline runs."""
    rd = runs_dir or RUNS_DIR
    runs = _list_runs(runs_dir=rd)

    if output_json:
        _output_json(runs)
        return

    if not runs:
        rprint("[dim]No runs found.[/dim]")
        return

    table = Table(title="Pipeline Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Exports")

    for r in runs:
        status_style = {
            "completed": "[green]completed[/green]",
            "failed": "[red]failed[/red]",
            "running": "[yellow]running[/yellow]",
        }.get(r["status"], r["status"])
        exports = ", ".join(r.get("exports", {}).keys()) or "-"
        table.add_row(r["run_id"], status_style, r["created_at"][:19], exports)
    console.print(table)


# ---------------------------------------------------------------------------
# cad-cli export
# ---------------------------------------------------------------------------


@app.command()
def export(
    run_id: str = typer.Argument(..., help="Run ID to export from"),
    formats: str = typer.Option("step,stl", "--formats", "-f"),
    output_json: bool = typer.Option(False, "--json", "-j"),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
) -> None:
    """Re-export a completed run in additional formats."""
    from .cad import build_geometry, export_all
    from .schemas import CADModel

    rd = runs_dir or RUNS_DIR
    try:
        manifest = _load_manifest(run_id, runs_dir=rd)
    except FileNotFoundError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    if manifest.get("cad_model") is None:
        rprint("[red]No CAD model found in this run. Was the analyze stage completed?[/red]")
        raise typer.Exit(code=1)

    cad_model = CADModel.model_validate(manifest["cad_model"])
    geometry = build_geometry(cad_model)

    run_dir = rd / run_id
    fmt_list = [f.strip() for f in formats.split(",")]
    exported = export_all(geometry, run_dir / "output", cad_model.name, formats=fmt_list)

    if output_json:
        _output_json({k: str(v) for k, v in exported.items()})
    else:
        rprint("[green]Exported:[/green]")
        for fmt, path in exported.items():
            rprint(f"  {fmt.upper()}: {path}")


# ---------------------------------------------------------------------------
# cad-cli schema
# ---------------------------------------------------------------------------


@app.command()
def schema() -> None:
    """Print the CAD operations JSON schema (for agent consumption)."""
    from .schemas import CADModel

    print(json.dumps(CADModel.model_json_schema(), indent=2))


# ---------------------------------------------------------------------------
# cad-cli iterate
# ---------------------------------------------------------------------------


@app.command()
def iterate(
    run_id: str = typer.Argument(..., help="Run ID to iterate on"),
    instruction: str = typer.Argument(..., help="Modification instruction (e.g. 'make it 2mm thicker')"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    formats: str = typer.Option("step,stl", "--formats", "-f"),
    output_json: bool = typer.Option(False, "--json", "-j"),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
) -> None:
    """Iterate on an existing run with a modification instruction."""
    from .cad import build_geometry, export_all
    from .schemas import CADModel
    from .vlm import iterate_model

    rd = runs_dir or RUNS_DIR
    try:
        manifest = _load_manifest(run_id, runs_dir=rd)
    except FileNotFoundError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    if manifest.get("cad_model") is None:
        rprint("[red]No CAD model in run. Complete the analyze stage first.[/red]")
        raise typer.Exit(code=1)

    current_model = CADModel.model_validate(manifest["cad_model"])

    if not output_json:
        rprint(f"[bold blue]Iterating on run {run_id}[/bold blue]: {instruction}")

    new_model = iterate_model(
        current_model=current_model,
        instruction=instruction,
        model=model,
    )

    geometry = build_geometry(new_model)
    fmt_list = [f.strip() for f in formats.split(",")]

    # Create new run for the iteration
    new_run_id = uuid.uuid4().hex[:12]
    new_run_dir = rd / new_run_id
    new_run_dir.mkdir(parents=True, exist_ok=True)

    exported = export_all(geometry, new_run_dir / "output", new_model.name, formats=fmt_list)

    new_manifest = {
        "run_id": new_run_id,
        "created_at": _now(),
        "image_path": manifest["image_path"],
        "model": model or manifest.get("model"),
        "hint": f"Iterated from {run_id}: {instruction}",
        "status": "completed",
        "cad_model": new_model.model_dump(),
        "exports": {k: str(v) for k, v in exported.items()},
    }
    _save_manifest(new_run_dir, new_manifest)
    (new_run_dir / "cad_model.json").write_text(new_model.model_dump_json(indent=2))

    result = {
        "run_id": new_run_id,
        "parent_run_id": run_id,
        "instruction": instruction,
        "run_dir": str(new_run_dir),
        "status": "completed",
        "exports": {k: str(v) for k, v in exported.items()},
    }

    if output_json:
        _output_json(result)
    else:
        rprint(f"[bold green]Iteration complete![/bold green]  new_run_id={new_run_id}")
        for fmt, path in exported.items():
            rprint(f"  {fmt.upper()}: {path}")


if __name__ == "__main__":
    app()
