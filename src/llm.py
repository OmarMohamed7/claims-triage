"""
LLM provider interface.

Agents call get_llm().generate(prompt) and never import a provider SDK
directly. Swapping providers (Ollama, Gemini, Vertex, Groq) is a
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


def _gemini_text(response: httpx.Response) -> str:
    parts = response.json()["candidates"][0]["content"]["parts"]
    return "".join(part.get("text", "") for part in parts if not part.get("thought"))


def _gemini_payload(prompt: str, temperature: float) -> dict:
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }


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
            _gemini_payload(prompt, temperature),
        )
        return _gemini_text(response)


class VertexGeminiProvider(LLMProvider):
    def __init__(self, model: str, project: str, location: str):
        import google.auth

        self._credentials, default_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project = project or default_project
        if not project:
            raise ValueError("GCP_PROJECT is not set and no default project was found")
        host = (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        self._url = (
            f"https://{host}/v1/projects/{project}/locations/{location}"
            f"/publishers/google/models/{model}:generateContent"
        )

    def _token(self) -> str:
        from google.auth.transport.requests import Request

        if not self._credentials.valid:
            self._credentials.refresh(Request())
        return self._credentials.token

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        response = _post_with_retry(
            self._url,
            {"Authorization": f"Bearer {self._token()}"},
            _gemini_payload(prompt, temperature),
        )
        return _gemini_text(response)


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
    "vertex": lambda: VertexGeminiProvider(
        model=config.models.llm_model,
        project=config.models.gcp_project,
        location=config.models.gcp_location,
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
