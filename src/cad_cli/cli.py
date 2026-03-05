"""CLI entry point for the CAD learning workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from dotenv import load_dotenv

from .pipeline import (
    build_dry_run_result,
    export_run,
    iterate_run,
    parse_formats,
    parse_known_dims,
    run_pipeline,
)
from .schemas import CADModel
from .storage import list_runs, load_manifest, resolve_runs_dir

load_dotenv()

app = typer.Typer(
    name="cad-cli",
    help="Agent-native CLI: image → parametric CAD via VLM-described geometry.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _output_json(data: Any) -> None:
    """Print machine-readable JSON to stdout."""
    print(json.dumps(data, indent=2, default=str))


def _coerce_formats(formats: str) -> list[str]:
    try:
        return parse_formats(formats)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--formats") from exc


def _coerce_known_dims(items: list[str]) -> dict[str, float]:
    try:
        return parse_known_dims(items)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--known-dim") from exc


# ---------------------------------------------------------------------------
# cad-cli run
# ---------------------------------------------------------------------------


@app.command()
def run(
    image: Path = typer.Argument(..., help="Path to input image", exists=True),
    model: str | None = typer.Option(
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
    known_dim: list[str] = typer.Option(
        [],
        "--known-dim",
        help="Hard dimension constraint key=value_mm. Repeatable.",
    ),
    runs_dir: Path | None = typer.Option(
        None, "--runs-dir", help="Custom directory for run artifacts"
    ),
) -> None:
    """Run the full image → CAD pipeline."""
    format_list = _coerce_formats(formats)
    known_dims = _coerce_known_dims(known_dim)
    runs_root = resolve_runs_dir(runs_dir)

    if dry_run:
        result = build_dry_run_result(
            image=image,
            model=model,
            formats=format_list,
            hint=hint,
            known_dims=known_dims,
        )
        if output_json:
            _output_json(result)
        else:
            rprint("[green]Dry run OK.[/green] Pipeline would run with:")
            rprint(f"  Image:    {image.resolve()}")
            rprint(f"  Model:    {model or 'default'}")
            rprint(f"  Formats:  {format_list}")
            if known_dims:
                rprint(f"  Known dims: {known_dims}")
        return

    if not output_json:
        rprint("[bold blue]Starting pipeline[/bold blue]")

    result = run_pipeline(
        image=image,
        model=model,
        hint=hint,
        formats=format_list,
        known_dims=known_dims,
        runs_dir=runs_root,
    )
    if output_json:
        _output_json(result)
        if result["status"] == "failed":
            raise typer.Exit(code=1)
    elif result["status"] == "failed":
        rprint(
            f"[bold red]Pipeline failed ({result['failure_category']}):[/bold red] "
            f"{result['error']}"
        )
        raise typer.Exit(code=1)
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
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Show the status of a pipeline run."""
    try:
        manifest = load_manifest(run_id, runs_dir=resolve_runs_dir(runs_dir))
    except FileNotFoundError as exc:
        if output_json:
            _output_json({"error": str(exc)})
        else:
            rprint(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    if output_json:
        _output_json(manifest)
        return

    rprint(f"[bold]Run {manifest['run_id']}[/bold]  status={manifest['status']}")
    rprint(f"  Created: {manifest['created_at']}")
    rprint(f"  Image:   {manifest['image_path']}")
    rprint(f"  Model:   {manifest.get('model') or 'default'}")
    if manifest.get("known_dims"):
        rprint(f"  Known dims: {manifest['known_dims']}")

    if manifest.get("exports"):
        rprint("\n[bold]Exports:[/bold]")
        for fmt, path in manifest["exports"].items():
            rprint(f"  {fmt.upper()}: {path}")

    if manifest.get("failure_category"):
        rprint(f"\n[bold red]Failure category:[/bold red] {manifest['failure_category']}")

    if manifest.get("error"):
        rprint(f"\n[bold red]Error:[/bold red] {manifest['error']}")


# ---------------------------------------------------------------------------
# cad-cli list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_runs_cmd(
    output_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """List all pipeline runs."""
    runs = list_runs(runs_dir=resolve_runs_dir(runs_dir))

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
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Re-export a completed run in additional formats."""
    format_list = _coerce_formats(formats)
    try:
        result = export_run(
            run_id=run_id,
            formats=format_list,
            runs_dir=resolve_runs_dir(runs_dir),
        )
    except FileNotFoundError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except ValueError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    if output_json:
        _output_json(result["exports"])
    else:
        rprint("[green]Exported:[/green]")
        for fmt, path in result["exports"].items():
            rprint(f"  {fmt.upper()}: {path}")


# ---------------------------------------------------------------------------
# cad-cli schema
# ---------------------------------------------------------------------------


@app.command()
def schema() -> None:
    """Print the CAD operations JSON schema (for agent consumption)."""
    print(json.dumps(CADModel.model_json_schema(), indent=2))


# ---------------------------------------------------------------------------
# cad-cli iterate
# ---------------------------------------------------------------------------


@app.command()
def iterate(
    run_id: str = typer.Argument(..., help="Run ID to iterate on"),
    instruction: str = typer.Argument(..., help="Modification instruction (e.g. 'make it 2mm thicker')"),
    model: str | None = typer.Option(None, "--model", "-m"),
    formats: str = typer.Option("step,stl", "--formats", "-f"),
    output_json: bool = typer.Option(False, "--json", "-j"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Iterate on an existing run with a modification instruction."""
    format_list = _coerce_formats(formats)
    if not output_json:
        rprint(f"[bold blue]Iterating on run {run_id}[/bold blue]: {instruction}")

    try:
        result = iterate_run(
            run_id=run_id,
            instruction=instruction,
            model=model,
            formats=format_list,
            runs_dir=resolve_runs_dir(runs_dir),
        )
    except FileNotFoundError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except ValueError as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    if output_json:
        _output_json(result)
    else:
        rprint(f"[bold green]Iteration complete![/bold green]  new_run_id={result['run_id']}")
        for fmt, path in result["exports"].items():
            rprint(f"  {fmt.upper()}: {path}")


if __name__ == "__main__":
    app()
