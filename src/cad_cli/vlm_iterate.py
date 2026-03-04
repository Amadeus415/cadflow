"""VLM-powered iteration: modify an existing CADModel based on natural language instructions."""

from __future__ import annotations

import json
import os
from typing import Optional

from .schemas import CADModel

ITERATE_SYSTEM_PROMPT = """\
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


def iterate_model(
    current_model: CADModel,
    instruction: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> CADModel:
    """Send the current model + instruction to a VLM, get back a modified CADModel."""
    provider = provider or os.getenv("CAD_CLI_VLM_PROVIDER", "openai")
    schema_text = json.dumps(CADModel.model_json_schema(), indent=2)
    system = ITERATE_SYSTEM_PROMPT.format(
        current_model=current_model.model_dump_json(indent=2),
        schema=schema_text,
    )

    if provider == "openai":
        raw = _call_openai_text(system, instruction, model)
    elif provider == "anthropic":
        raw = _call_anthropic_text(system, instruction, model)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")

    return CADModel.model_validate(raw)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _call_openai_text(system: str, user_msg: str, model: Optional[str] = None) -> dict:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model or "gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    return json.loads(_strip_fences(resp.choices[0].message.content))


def _call_anthropic_text(system: str, user_msg: str, model: Optional[str] = None) -> dict:
    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=model or "claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.2,
    )
    return json.loads(_strip_fences(resp.content[0].text))
