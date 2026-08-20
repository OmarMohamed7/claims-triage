"""
LLM provider interface.

Agents call get_llm().generate(prompt) and never import a provider SDK
directly. Swapping providers (Ollama today, anything else later) is a
one-line change in .env (LLM_PROVIDER / LLM_MODEL), not a code change
across every agent.

To add a provider: subclass LLMProvider, implement generate(), and register
it in _PROVIDER_FACTORIES.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from src.config import config


class LLMProvider(ABC):
    """Minimal interface every provider implements."""

    @abstractmethod
    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Return the model's raw text completion for prompt."""


class OllamaProvider(LLMProvider):
    def __init__(self, model: str, base_url: str):
        import ollama

        self._client = ollama.Client(host=base_url)
        self._model = model

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        response = self._client.generate(
            model=self._model,
            prompt=prompt,
            options={"temperature": temperature},
        )
        return response["response"]


_PROVIDER_FACTORIES = {
    "ollama": lambda: OllamaProvider(
        model=config.models.llm_model,
        base_url=config.models.ollama_base_url,
    ),
}


@lru_cache(maxsize=None)
def get_llm() -> LLMProvider:
    """Return the configured LLMProvider (LLM_PROVIDER in .env), cached."""
    provider_name = config.models.llm_provider
    try:
        factory = _PROVIDER_FACTORIES[provider_name]
    except KeyError:
        raise ValueError(
            f"Unknown LLM_PROVIDER {provider_name!r}. "
            f"Available: {sorted(_PROVIDER_FACTORIES)}. "
            "Add a new LLMProvider subclass and factory entry in src/llm.py "
            "to support it."
        ) from None
    return factory()
