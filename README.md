# cad-cli

**Agent-native CLI: image → parametric CAD via VLM-described geometry.**

Turn a photo of a physical object into a manufacturable STEP/STL file through a small, inspectable pipeline designed for learning and experimentation.

## How It Works

```
Image → VLM (Gemini) → Structured JSON → CadQuery → STEP / STL
```

1. **Analyze** — Send image to a VLM with a structured prompt describing CAD primitives
2. **Validate** — Output validated against a strict Pydantic/JSON schema
3. **Build** — Operations compiled into CadQuery parametric geometry
4. **Export** — Geometry exported as STEP (CAD-ready) and STL (3D printing)

Every run is tracked in repository-root `runs/` by default with a manifest, input image, intermediate JSON, and outputs. Use `--runs-dir` to override.

## Quick Start

```bash
pip install -e .
export GEMINI_API_KEY=...
cad-cli run photo.png
```

Gemini integration uses the current `google-genai` Python SDK.
Reference docs: https://ai.google.dev/gemini-api/docs

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

- `--model` / `-m` — Override Gemini model (default: `gemini-3.1-pro-preview`)
- `--hint` / `-h` — Context hint for the VLM (e.g. "this is a mounting bracket")
- `--formats` / `-f` — Export formats, comma-separated (default: `step,stl`)
- `--json` / `-j` — Output JSON for agent consumption
- `--dry-run` — Validate without running
- `--known-dim` — Hard size constraints like `--known-dim overall_width=50mm` (repeatable)

## Agent Interface

Every command supports `--json` for structured output. The `schema` command prints the full JSON schema so agents know exactly what operations are available. The `iterate` command lets agents refine designs conversationally.

## Architecture

```
src/cad_cli/
├── cli.py       # Typer commands + output formatting
├── pipeline.py  # Analyze → build → export workflow orchestration
├── storage.py   # Run artifact loading/saving
├── ai.py        # Gemini API: image analysis + model iteration
├── schemas.py   # Pydantic models for CAD operations
├── cad.py       # JSON → CadQuery (strict topology + debug checkpoints)
├── alignment.py # Optional future scoring utilities
└── errors.py    # Typed pipeline failure categories
```
