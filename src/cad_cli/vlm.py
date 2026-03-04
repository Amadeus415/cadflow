"""VLM integration: Gemini-powered image analysis and model iteration."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional

import google.generativeai as genai

from .schemas import CADModel

DEFAULT_MODEL = "gemini-3-flash"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANALYZE_PROMPT = """\
You are a CAD engineering assistant. Given an image of a physical object or part, \
describe it as a sequence of parametric CAD operations that would reproduce the geometry.

Rules:
1. Use millimeters as the unit unless told otherwise.
2. Estimate real-world dimensions from visual cues (e.g., a USB connector is ~12mm wide).
3. Build the object bottom-up: start with the main body, then add features (holes, fillets, etc.).
4. Output ONLY valid JSON matching the schema below. No markdown, no commentary.

JSON Schema:
{schema}

Respond with a single JSON object. Do not wrap it in code fences.
"""

ITERATE_PROMPT = """\
You are a CAD engineering assistant. You are given an existing CAD model as JSON \
and a modification instruction from the user.

Your job is to return a MODIFIED version of the CAD model JSON that applies the \
requested change. Preserve all existing operations that are not affected by the change.

Rules:
1. Keep the same JSON schema structure.
2. Only modify what the instruction asks for.
3. Output ONLY valid JSON. No markdown, no commentary.

Current CAD model:
{current_model}

JSON Schema:
{schema}

Respond with the complete modified JSON object.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_image(image_path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file."""
    suffix = image_path.suffix.lower()
    media_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    media_type = media_map.get(suffix, "image/png")
    data = base64.b64encode(image_path.read_bytes()).decode()
    return data, media_type


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _gemini_model(model: Optional[str] = None) -> genai.GenerativeModel:
    """Configure the Gemini SDK and return a GenerativeModel."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY env var is required.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=model or DEFAULT_MODEL,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_image(
    image_path: Path,
    model: Optional[str] = None,
    hint: str = "",
    **_kwargs,
) -> CADModel:
    """Analyze an image and return a validated CADModel."""
    b64, media = _encode_image(image_path)
    schema_text = json.dumps(CADModel.model_json_schema(), indent=2)
    system = ANALYZE_PROMPT.format(schema=schema_text)

    parts: list = [system]
    if hint:
        parts.append(hint)
    parts.append({"inline_data": {"mime_type": media, "data": b64}})

    gm = _gemini_model(model)
    resp = gm.generate_content(parts)
    return CADModel.model_validate(json.loads(_strip_fences(resp.text)))


def iterate_model(
    current_model: CADModel,
    instruction: str,
    model: Optional[str] = None,
    **_kwargs,
) -> CADModel:
    """Send the current model + instruction to Gemini, get back a modified CADModel."""
    schema_text = json.dumps(CADModel.model_json_schema(), indent=2)
    system = ITERATE_PROMPT.format(
        current_model=current_model.model_dump_json(indent=2),
        schema=schema_text,
    )

    gm = _gemini_model(model)
    resp = gm.generate_content([system, instruction])
    return CADModel.model_validate(json.loads(_strip_fences(resp.text)))
