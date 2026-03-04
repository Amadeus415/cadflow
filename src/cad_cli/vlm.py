"""VLM integration: Gemini-powered image analysis and model iteration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from .schemas import CADModel

DEFAULT_MODEL = "gemini-3-flash-preview"
MAX_GENERATION_RETRIES = 3

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
5. Keep the operation list concise; avoid overly fine repetitive detail unless essential.

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


def _read_image_bytes(image_path: Path) -> tuple[bytes, str]:
    """Return (image_bytes, media_type) for an image file."""
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
    return image_path.read_bytes(), media_type


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _gemini_client() -> genai.Client:
    """Build and return a Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY env var is required.")
    return genai.Client(api_key=api_key)


def _generation_config(system_instruction: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=8192,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        system_instruction=system_instruction,
    )


def _response_debug_info(resp: types.GenerateContentResponse) -> str:
    finish_reason = None
    if resp.candidates:
        finish_reason = resp.candidates[0].finish_reason
    usage = resp.usage_metadata
    total_tokens = usage.total_token_count if usage else None
    thoughts_tokens = usage.thoughts_token_count if usage else None
    return (
        f"finish_reason={finish_reason}, "
        f"total_tokens={total_tokens}, thoughts_tokens={thoughts_tokens}"
    )


def _parse_cad_model_response(resp: types.GenerateContentResponse) -> CADModel:
    """Parse Gemini response into CADModel, preferring structured parsed output."""
    if resp.parsed is not None:
        return CADModel.model_validate(resp.parsed)

    if not resp.text:
        raise ValueError(f"Gemini returned an empty response. {_response_debug_info(resp)}")

    text = _strip_fences(resp.text)
    try:
        return CADModel.model_validate(json.loads(text))
    except json.JSONDecodeError as e:
        snippet = text[:500].replace("\n", "\\n")
        raise ValueError(
            "Gemini returned invalid JSON: "
            f"{e}. {_response_debug_info(resp)}. "
            f"Response length={len(text)}. Response snippet: {snippet}"
        ) from e


def _generate_cad_model(
    client: genai.Client,
    model_name: str,
    contents: list[object],
    system_instruction: str,
) -> CADModel:
    last_error: Optional[ValueError] = None

    for attempt in range(1, MAX_GENERATION_RETRIES + 1):
        attempt_contents = contents
        if attempt > 1:
            attempt_contents = [
                "IMPORTANT: Return exactly one complete JSON object. No markdown, no truncation.",
                *contents,
            ]

        resp = client.models.generate_content(
            model=model_name,
            contents=attempt_contents,
            config=_generation_config(system_instruction),
        )
        try:
            return _parse_cad_model_response(resp)
        except ValueError as e:
            last_error = e

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini generation failed without an explicit parse error.")


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
    image_bytes, media = _read_image_bytes(image_path)
    schema_text = json.dumps(CADModel.model_json_schema(), indent=2)
    system = ANALYZE_PROMPT.format(schema=schema_text)

    parts: list[object] = []
    if hint:
        parts.append(hint)
    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=media))

    client = _gemini_client()
    return _generate_cad_model(
        client=client,
        model_name=model or DEFAULT_MODEL,
        contents=parts,
        system_instruction=system,
    )


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

    client = _gemini_client()
    return _generate_cad_model(
        client=client,
        model_name=model or DEFAULT_MODEL,
        contents=[instruction],
        system_instruction=system,
    )
