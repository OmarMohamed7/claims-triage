from unittest.mock import MagicMock, patch

from src import llm


def _response(text: str) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": text}, {"text": "!", "thought": True}]}}]
    }
    return response


def test_vertex_provider_uses_adc_bearer_token():
    credentials = MagicMock(valid=True, token="adc-token")
    with patch("google.auth.default", return_value=(credentials, "proj-1")), patch(
        "src.llm.httpx.post", return_value=_response('{"ok": true}')
    ) as post:
        provider = llm.VertexGeminiProvider("gemini-x", "", "europe-west1")
        assert provider.generate("hi") == '{"ok": true}'
    url = post.call_args.args[0]
    assert url.startswith("https://europe-west1-aiplatform.googleapis.com/v1/projects/proj-1/")
    assert url.endswith("/models/gemini-x:generateContent")
    assert post.call_args.kwargs["headers"] == {"Authorization": "Bearer adc-token"}


def test_vertex_global_location_uses_global_host():
    credentials = MagicMock(valid=True, token="t")
    with patch("google.auth.default", return_value=(credentials, "p")):
        provider = llm.VertexGeminiProvider("m", "explicit", "global")
    assert provider._url.startswith("https://aiplatform.googleapis.com/v1/projects/explicit/locations/global/")
