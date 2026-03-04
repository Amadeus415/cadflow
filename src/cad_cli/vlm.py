"""VLM integration: send an image and get structured CAD operations back."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .schemas import CADModel

# ---------------------------------------------------------------------------
# System prompt for VLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
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

# ---------------------------------------------------------------------------
# Image encoding helper
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


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _call_openai(
    image_path: Path,
    user_hint: str,
    model: str = "gpt-4o",
) -> dict:
    """Call OpenAI's vision model."""
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY env var
    b64, media = _encode_image(image_path)

    schema_text = json.dumps(CADModel.model_json_schema(), indent=2)
    system = SYSTEM_PROMPT.format(schema=schema_text)

    user_content: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media};base64,{b64}", "detail": "high"},
        },
    ]
    if user_hint:
        user_content.insert(0, {"type": "text", "text": user_hint})

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown code fences if the model wraps anyway
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def _call_anthropic(
    image_path: Path,
    user_hint: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Call Anthropic's vision model."""
    from anthropic import Anthropic

    client = Anthropic()  # uses ANTHROPIC_API_KEY env var
    b64, media = _encode_image(image_path)

    schema_text = json.dumps(CADModel.model_json_schema(), indent=2)
    system = SYSTEM_PROMPT.format(schema=schema_text)

    user_content: list[dict] = []
    if user_hint:
        user_content.append({"type": "text", "text": user_hint})
    user_content.append(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media, "data": b64},
        }
    )

    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_content}],
        temperature=0.2,
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_image(
    image_path: Path,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    hint: str = "",
) -> CADModel:
    """Analyze an image and return a validated CADModel.

    Args:
        image_path: Path to the input image.
        provider: "openai" or "anthropic". Falls back to CAD_CLI_VLM_PROVIDER env var.
        model: Override the default model name.
        hint: Optional text hint for the VLM (e.g. "this is a mounting bracket").

    Returns:
        Validated CADModel instance.

    Raises:
        ValidationError: If VLM output doesn't match schema.
        ValueError: If provider is unknown or API key missing.
    """
    provider = provider or os.getenv("CAD_CLI_VLM_PROVIDER", "openai")

    if provider == "openai":
        kwargs = {"image_path": image_path, "user_hint": hint}
        if model:
            kwargs["model"] = model
        raw = _call_openai(**kwargs)
    elif provider == "anthropic":
        kwargs = {"image_path": image_path, "user_hint": hint}
        if model:
            kwargs["model"] = model
        raw = _call_anthropic(**kwargs)
    else:
        raise ValueError(f"Unknown VLM provider: {provider!r}. Use 'openai' or 'anthropic'.")

    return CADModel.model_validate(raw)
