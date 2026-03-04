from __future__ import annotations

from types import SimpleNamespace
import unittest

from cad_cli.vlm import (
    INITIAL_MAX_OUTPUT_TOKENS,
    INITIAL_THINKING_BUDGET,
    RETRY_MAX_OUTPUT_TOKENS,
    RETRY_THINKING_BUDGET,
    _generate_cad_model,
    _generation_config,
)


def _fake_response(
    *,
    text: str = "",
    parsed: dict | None = None,
    finish_reason: str = "FinishReason.STOP",
    total_tokens: int = 100,
    thoughts_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        parsed=parsed,
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
        usage_metadata=SimpleNamespace(
            total_token_count=total_tokens,
            thoughts_token_count=thoughts_tokens,
        ),
    )


class _FakeModels:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = responses
        self.configs = []

    def generate_content(self, *, model, contents, config):  # noqa: ANN001
        self.configs.append(config)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.models = _FakeModels(responses)


class VLMConfigTests(unittest.TestCase):
    def test_generation_config_first_attempt_prioritizes_balanced_reasoning(self) -> None:
        cfg = _generation_config("system prompt", attempt=1)
        self.assertEqual(cfg.max_output_tokens, INITIAL_MAX_OUTPUT_TOKENS)
        self.assertEqual(cfg.thinking_config.thinking_budget, INITIAL_THINKING_BUDGET)
        self.assertFalse(cfg.thinking_config.include_thoughts)
        self.assertEqual(cfg.response_mime_type, "application/json")
        self.assertIsNotNone(cfg.response_json_schema)

    def test_generation_config_retry_disables_thinking_and_expands_output_budget(self) -> None:
        cfg = _generation_config("system prompt", attempt=2)
        self.assertEqual(cfg.max_output_tokens, RETRY_MAX_OUTPUT_TOKENS)
        self.assertIsNotNone(cfg.thinking_config)
        self.assertEqual(cfg.thinking_config.thinking_budget, RETRY_THINKING_BUDGET)
        self.assertFalse(cfg.thinking_config.include_thoughts)

    def test_generate_cad_model_retries_with_retry_budget_after_truncated_json(self) -> None:
        client = _FakeClient(
            [
                _fake_response(
                    text='{"name":"bad",',
                    finish_reason="FinishReason.MAX_TOKENS",
                    total_tokens=12000,
                    thoughts_tokens=7800,
                ),
                _fake_response(
                    parsed={
                        "name": "good",
                        "description": "ok",
                        "unit": "mm",
                        "operations": [
                            {
                                "op": "extrude",
                                "depth": 10,
                                "plane": "XY",
                                "origin": [0, 0, 0],
                                "sketch": {
                                    "type": "rectangle",
                                    "width": 10,
                                    "height": 10,
                                },
                            }
                        ],
                    }
                ),
            ]
        )

        out = _generate_cad_model(
            client=client,
            model_name="gemini-3.1-pro-preview",
            contents=["repair this model"],
            system_instruction="system",
        )

        self.assertEqual(out.name, "good")
        self.assertEqual(len(client.models.configs), 2)
        self.assertEqual(
            client.models.configs[0].thinking_config.thinking_budget,
            INITIAL_THINKING_BUDGET,
        )
        self.assertEqual(
            client.models.configs[1].thinking_config.thinking_budget,
            RETRY_THINKING_BUDGET,
        )
        self.assertEqual(client.models.configs[1].max_output_tokens, RETRY_MAX_OUTPUT_TOKENS)


if __name__ == "__main__":
    unittest.main()
