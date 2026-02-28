"""
LLM provider abstraction: propose(system_prompt, user_prompt) -> dict.

- ModalLLMProvider: direct Modal GPU function call (no API key).
- OpenAIProvider: optional; uses OpenAI-compatible API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Base for LLM backends that return mutation-style JSON dicts."""

    @abstractmethod
    def propose(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Return a dict with at least proposed_residue, confidence, reason."""
        ...


class ModalLLMProvider(LLMProvider):
    """Call the Modal GPU function generate_llm_json (direct, no HTTP)."""

    def __init__(self, app_name: str = "protein-swarm", function_name: str = "generate_llm_json"):
        self._app_name = app_name
        self._function_name = function_name

    def propose(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        import modal

        fn = modal.Function.from_name(self._app_name, self._function_name)
        return fn.remote(system_prompt, user_prompt)


class OpenAIProvider(LLMProvider):
    """Use OpenAI-compatible API; requires api_key."""

    def __init__(
        self,
        api_key: str,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self._api_key = api_key
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def propose(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        from protein_swarm.agents.llm_client import _call_llm_raw, _parse_json

        raw = _call_llm_raw(
            system_prompt,
            user_prompt,
            api_key=self._api_key,
            provider=self._provider,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return _parse_json(raw)
