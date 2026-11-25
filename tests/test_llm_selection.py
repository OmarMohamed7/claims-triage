import pytest

from src import llm
from src.config import config
from src.schemas import CustomLLM


def _custom(provider="groq", model="m-1", key="secret-key"):
    return CustomLLM(provider=provider, model=model, api_key=key)


def test_default_choice_is_allowed():
    assert config.models.default_llm in config.models.llm_choices


def test_disallowed_choice_raises():
    with pytest.raises(ValueError, match="not allowed"):
        llm.get_llm("groq:not-in-the-list")


def test_custom_provider_is_used_inside_context_only():
    with llm.use_llm(None, _custom()):
        provider = llm.get_llm()
        assert isinstance(provider, llm.GroqProvider)
        assert provider._api_key == "secret-key"
    assert llm._selected.get() is None


def test_custom_gemini_provider():
    with llm.use_llm(None, _custom("gemini", "gemini-x")):
        assert isinstance(llm.get_llm(), llm.GeminiProvider)


def test_custom_rejects_unknown_provider_and_bad_model_name():
    with pytest.raises(ValueError):
        CustomLLM(provider="evil", model="m", api_key="k")
    with pytest.raises(ValueError):
        CustomLLM(provider="groq", model="../../etc", api_key="k")
    with pytest.raises(ValueError):
        CustomLLM(provider="groq", model="m", api_key="")
