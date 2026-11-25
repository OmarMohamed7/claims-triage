import pytest
from fastapi.testclient import TestClient

import main
from src import graph
from src.schemas import DecisionStatus
from tests.test_graph import _claim, _decision, _fraud, _policy


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(graph, "extract_claim", lambda s: _claim())
    monkeypatch.setattr(graph, "score_fraud_risk", lambda c: _fraud())
    monkeypatch.setattr(graph, "retrieve_policy", lambda c: _policy())
    monkeypatch.setattr(
        graph, "adjudicate", lambda c, p, f: _decision(DecisionStatus.APPROVED, [])
    )
    main.limiter.reset()
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_submit_claim_returns_pipeline_state(client):
    response = client.post("/claims", json={"raw_text": "x", "policy_number": "P1"})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["status"] == "approved"
    assert body["errors"] == []


def test_submit_claim_requires_raw_text(client):
    assert client.post("/claims", json={}).status_code == 422


def test_rate_limit_returns_429(client):
    codes = [client.post("/claims", json={"raw_text": "x"}).status_code for _ in range(11)]
    assert codes[:10] == [200] * 10
    assert codes[10] == 429


def test_missing_artifacts_fail_startup(monkeypatch, tmp_path):
    monkeypatch.setattr(main.config.models, "fraud_model_path", str(tmp_path / "none.joblib"))
    with pytest.raises(RuntimeError, match="fraud model"):
        with TestClient(main.app):
            pass


def test_models_endpoint_lists_default(client):
    body = client.get("/models").json()
    assert body["default"] in body["choices"]


def test_disallowed_model_is_rejected(client):
    response = client.post("/claims", json={"raw_text": "x", "model": "groq:nope"})
    assert response.status_code == 400


def test_custom_llm_is_used_and_key_is_not_echoed(client, monkeypatch):
    from src import llm

    seen = {}

    def extract(submission):
        provider = llm.get_llm()
        seen["type"] = type(provider).__name__
        seen["key"] = provider._api_key
        return _claim()

    monkeypatch.setattr(graph, "extract_claim", extract)
    response = client.post(
        "/claims",
        json={
            "raw_text": "x",
            "custom_llm": {"provider": "groq", "model": "llama-x", "api_key": "sk-test-123"},
        },
    )
    assert response.status_code == 200
    assert seen == {"type": "GroqProvider", "key": "sk-test-123"}
    assert "sk-test-123" not in response.text
    assert "custom_llm" not in response.json()


def test_custom_llm_validation(client):
    bad = {"provider": "evil", "model": "m", "api_key": "k"}
    assert client.post("/claims", json={"raw_text": "x", "custom_llm": bad}).status_code == 422
