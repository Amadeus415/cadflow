# cad-cli

**Agent-native CLI: image → parametric CAD via VLM-described geometry.**

Turn a photo of a physical object into a manufacturable STEP/STL file through a structured, reproducible pipeline designed for autonomous design workflows.

## How It Works

```
Image → VLM (GPT-4o / Claude) → Structured JSON → CadQuery → STEP / STL
```

1. **Analyze** — Send image to a VLM with a structured prompt describing CAD primitives
2. **Validate** — Output validated against a strict Pydantic/JSON schema
3. **Build** — Operations compiled into CadQuery parametric geometry
4. **Export** — Geometry exported as STEP (CAD-ready) and STL (3D printing)

Every run is tracked in `runs/` with a manifest, input image, intermediate JSON, and outputs.

## Quick Start

```bash
pip install -e .
export OPENAI_API_KEY=sk-...       # or ANTHROPIC_API_KEY
cad-cli run photo.png
```

## Commands

```
cad-cli run <image>              # Full pipeline: image → CAD
cad-cli run <image> --dry-run    # Validate inputs only
cad-cli run <image> --json       # Machine-readable JSON output
cad-cli status <run-id>          # Inspect a pipeline run
cad-cli list                     # List all runs
cad-cli export <run-id>          # Re-export in different formats
cad-cli iterate <run-id> "msg"   # Modify a design with natural language
cad-cli schema                   # Print JSON schema for agents
```

## Key Options

- `--provider` / `-p` — `openai` or `anthropic`
- `--model` / `-m` — Override model (e.g. `gpt-4o`, `claude-sonnet-4-20250514`)
- `--hint` / `-h` — Context hint for the VLM (e.g. "this is a mounting bracket")
- `--formats` / `-f` — Export formats, comma-separated (default: `step,stl`)
- `--json` / `-j` — Output JSON for agent consumption
- `--dry-run` — Validate without running

## Agent Interface

Every command supports `--json` for structured output. The `schema` command prints the full JSON schema so agents know exactly what operations are available. The `iterate` command lets agents refine designs conversationally.

## Architecture

```
src/cad_cli/
├── cli.py          # Typer CLI entry point
├── vlm.py          # Image → structured CAD JSON
├── vlm_iterate.py  # Iterate on existing models via VLM
├── schemas.py      # Pydantic models for CAD operations
├── cad.py          # JSON → CadQuery → STEP/STL
└── pipeline.py     # Run orchestration & artifact tracking
```
