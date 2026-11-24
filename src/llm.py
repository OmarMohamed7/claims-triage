"""
LLM provider interface.

Agents call get_llm().generate(prompt) and never import a provider SDK
directly. Swapping providers (Ollama, Gemini, Groq) is a
one-line change in .env (LLM_PROVIDER / LLM_MODEL), not a code change
across every agent.

To add a provider: subclass LLMProvider, implement generate(), and register
it in _PROVIDER_FACTORIES.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx

from src.config import config


def _post_with_retry(url: str, headers: dict, payload: dict, attempts: int = 3) -> httpx.Response:
    for attempt in range(attempts):
        response = httpx.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code != 429 or attempt == attempts - 1:
            response.raise_for_status()
            return response
        time.sleep(float(response.headers.get("retry-after", 2**attempt)))
    raise RuntimeError("unreachable")


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


class GeminiProvider(LLMProvider):
    def __init__(self, model: str, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self._url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )
        self._api_key = api_key

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        response = _post_with_retry(
            self._url,
            {"x-goog-api-key": self._api_key},
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature},
            },
        )
        parts = response.json()["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts if not part.get("thought"))


class GroqProvider(LLMProvider):
    def __init__(self, model: str, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self._model = model
        self._api_key = api_key

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        response = _post_with_retry(
            "https://api.groq.com/openai/v1/chat/completions",
            {"Authorization": f"Bearer {self._api_key}"},
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            },
        )
        return response.json()["choices"][0]["message"]["content"]


_PROVIDER_FACTORIES = {
    "ollama": lambda: OllamaProvider(
        model=config.models.llm_model,
        base_url=config.models.ollama_base_url,
    ),
    "gemini": lambda: GeminiProvider(
        model=config.models.llm_model,
        api_key=config.models.gemini_api_key,
    ),
    "groq": lambda: GroqProvider(
        model=config.models.llm_model,
        api_key=config.models.groq_api_key,
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
