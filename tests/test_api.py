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
