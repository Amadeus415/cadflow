"""Agent-native CLI entry point."""

from __future__ import annotations

import json
import sys
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


def _output_json(data: dict) -> None:
    """Print machine-readable JSON to stdout."""
    print(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# cad-cli run
# ---------------------------------------------------------------------------


@app.command()
def run(
    image: Path = typer.Argument(..., help="Path to input image", exists=True),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="VLM provider: openai or anthropic"
    ),
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
    from .pipeline import PipelineRun

    fmt_list = [f.strip() for f in formats.split(",")]

    if dry_run:
        result = {
            "status": "dry_run",
            "image": str(image.resolve()),
            "provider": provider or "openai",
            "model": model,
            "formats": fmt_list,
            "hint": hint,
        }
        if output_json:
            _output_json(result)
        else:
            rprint("[green]Dry run OK.[/green] Pipeline would run with:")
            rprint(f"  Image:    {image.resolve()}")
            rprint(f"  Provider: {provider or 'openai'}")
            rprint(f"  Model:    {model or 'default'}")
            rprint(f"  Formats:  {fmt_list}")
        return

    kwargs = {
        "image_path": image.resolve(),
        "provider": provider or "openai",
        "model": model,
        "hint": hint,
    }
    if runs_dir:
        kwargs["runs_dir"] = runs_dir

    pipeline = PipelineRun(**kwargs)

    if not output_json:
        rprint(f"[bold blue]Starting pipeline[/bold blue]  run_id={pipeline.run_id}")

    try:
        result = pipeline.execute(formats=fmt_list)
    except Exception as e:
        if output_json:
            _output_json({"status": "failed", "error": str(e), "run_id": pipeline.run_id})
        else:
            rprint(f"[bold red]Pipeline failed:[/bold red] {e}")
        raise typer.Exit(code=1)

    if output_json:
        _output_json(result)
    else:
        rprint(f"[bold green]Pipeline completed![/bold green]  run_id={result['run_id']}")
        rprint(f"  Run dir: {result['run_dir']}")
        for fmt, path in result["exports"].items():
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
    from .pipeline import load_manifest

    try:
        manifest = load_manifest(run_id, runs_dir=runs_dir)
    except FileNotFoundError as e:
        if output_json:
            _output_json({"error": str(e)})
        else:
            rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    if output_json:
        _output_json(manifest.model_dump())
        return

    rprint(f"[bold]Run {manifest.run_id}[/bold]  status={manifest.status}")
    rprint(f"  Created: {manifest.created_at}")
    rprint(f"  Image:   {manifest.image_path}")
    rprint(f"  Provider: {manifest.provider}  Model: {manifest.model or 'default'}")

    table = Table(title="Stages")
    table.add_column("Stage", style="cyan")
    table.add_column("Status")
    table.add_column("Artifacts")
    table.add_column("Error")

    for stage in manifest.stages:
        status_style = {
            "completed": "[green]completed[/green]",
            "failed": "[red]failed[/red]",
            "running": "[yellow]running[/yellow]",
            "pending": "[dim]pending[/dim]",
        }.get(stage.status.value, stage.status.value)

        table.add_row(
            stage.name,
            status_style,
            ", ".join(stage.artifacts) if stage.artifacts else "-",
            stage.error or "-",
        )
    console.print(table)

    if manifest.exports:
        rprint("\n[bold]Exports:[/bold]")
        for fmt, path in manifest.exports.items():
            rprint(f"  {fmt.upper()}: {path}")


# ---------------------------------------------------------------------------
# cad-cli list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_runs_cmd(
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
) -> None:
    """List all pipeline runs."""
    from .pipeline import list_runs

    runs = list_runs(runs_dir=runs_dir)

    if output_json:
        _output_json([r.model_dump() for r in runs])
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
        }.get(r.status, r.status)
        exports = ", ".join(r.exports.keys()) if r.exports else "-"
        table.add_row(r.run_id, status_style, r.created_at[:19], exports)
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
    from .pipeline import load_manifest
    from .schemas import CADModel

    try:
        manifest = load_manifest(run_id, runs_dir=runs_dir)
    except FileNotFoundError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    if manifest.cad_model is None:
        rprint("[red]No CAD model found in this run. Was the analyze stage completed?[/red]")
        raise typer.Exit(code=1)

    cad_model = CADModel.model_validate(manifest.cad_model)
    geometry = build_geometry(cad_model)

    run_dir = (runs_dir or Path.cwd() / "runs") / run_id
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
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    formats: str = typer.Option("step,stl", "--formats", "-f"),
    output_json: bool = typer.Option(False, "--json", "-j"),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir"),
) -> None:
    """Iterate on an existing run with a modification instruction."""
    from .cad import build_geometry, export_all
    from .pipeline import load_manifest, PipelineRun, RUNS_DIR
    from .schemas import CADModel
    from .vlm_iterate import iterate_model

    rd = runs_dir or RUNS_DIR
    try:
        manifest = load_manifest(run_id, runs_dir=rd)
    except FileNotFoundError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    if manifest.cad_model is None:
        rprint("[red]No CAD model in run. Complete the analyze stage first.[/red]")
        raise typer.Exit(code=1)

    current_model = CADModel.model_validate(manifest.cad_model)

    if not output_json:
        rprint(f"[bold blue]Iterating on run {run_id}[/bold blue]: {instruction}")

    new_model = iterate_model(
        current_model=current_model,
        instruction=instruction,
        provider=provider or manifest.provider,
        model=model,
    )

    geometry = build_geometry(new_model)
    fmt_list = [f.strip() for f in formats.split(",")]

    # Create new run for the iteration
    import uuid
    new_run_id = uuid.uuid4().hex[:12]
    new_run_dir = rd / new_run_id
    new_run_dir.mkdir(parents=True, exist_ok=True)

    exported = export_all(geometry, new_run_dir / "output", new_model.name, formats=fmt_list)

    # Save new manifest
    from .pipeline import RunManifest, StageRecord, StageStatus, _now
    new_manifest = RunManifest(
        run_id=new_run_id,
        created_at=_now(),
        image_path=manifest.image_path,
        provider=provider or manifest.provider,
        model=model or manifest.model,
        hint=f"Iterated from {run_id}: {instruction}",
        stages=[
            StageRecord(name="analyze", status=StageStatus.completed),
            StageRecord(name="validate", status=StageStatus.completed),
            StageRecord(name="build", status=StageStatus.completed),
            StageRecord(name="export", status=StageStatus.completed, artifacts=[str(v) for v in exported.values()]),
        ],
        cad_model=new_model.model_dump(),
        exports={k: str(v) for k, v in exported.items()},
        status="completed",
    )
    (new_run_dir / "manifest.json").write_text(new_manifest.model_dump_json(indent=2))
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
