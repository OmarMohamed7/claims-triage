import pytest

from src import graph
from src.schemas import (
    AdjudicationDecision,
    ClaimSubmission,
    DecisionStatus,
    EscalationQueue,
    EscalationReason,
    ExtractedClaim,
    FraudRiskResult,
    PipelineState,
    PolicyRetrievalResult,
)


def _claim(confidence: float = 0.95) -> ExtractedClaim:
    return ExtractedClaim(submission_id="s1", extraction_confidence=confidence)


def _fraud() -> FraudRiskResult:
    return FraudRiskResult(
        submission_id="s1", risk_score=0.9, risk_tier="high", model_version="t"
    )


def _policy() -> PolicyRetrievalResult:
    return PolicyRetrievalResult(
        submission_id="s1",
        policy_number="P1",
        is_covered=True,
        retrieval_confidence=0.9,
    )


def _decision(status: DecisionStatus, reasons: list[EscalationReason]):
    return AdjudicationDecision(
        submission_id="s1", status=status, reasoning="r", escalation_reasons=reasons
    )


def _run() -> dict:
    submission = ClaimSubmission(submission_id="s1", raw_text="x")
    return graph.build_pipeline().invoke(PipelineState(submission=submission))


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(graph, "extract_claim", lambda s: _claim())
    monkeypatch.setattr(graph, "score_fraud_risk", lambda c: _fraud())
    monkeypatch.setattr(graph, "retrieve_policy", lambda c: _policy())
    monkeypatch.setattr(
        graph,
        "adjudicate",
        lambda c, p, f: _decision(DecisionStatus.APPROVED, []),
    )
    return monkeypatch


def test_approved_claim_has_no_escalation_packet(patched):
    result = _run()
    assert result["decision"].status == DecisionStatus.APPROVED
    assert result.get("escalation_packet") is None
    assert result["errors"] == []


def test_high_fraud_escalation_goes_to_fraud_queue(patched):
    patched.setattr(
        graph,
        "adjudicate",
        lambda c, p, f: _decision(
            DecisionStatus.ESCALATED, [EscalationReason.HIGH_FRAUD_RISK]
        ),
    )
    packet = _run()["escalation_packet"]
    assert packet.queue == EscalationQueue.FRAUD_REVIEW
    assert packet.policy_result is not None


def test_other_escalation_goes_to_adjudicator_queue(patched):
    patched.setattr(
        graph,
        "adjudicate",
        lambda c, p, f: _decision(
            DecisionStatus.ESCALATED, [EscalationReason.HIGH_CLAIM_VALUE]
        ),
    )
    assert _run()["escalation_packet"].queue == EscalationQueue.ADJUDICATOR


def test_low_extraction_confidence_skips_fraud_and_policy(patched):
    patched.setattr(graph, "extract_claim", lambda s: _claim(0.1))

    def boom(c):
        raise AssertionError("should not run")

    patched.setattr(graph, "score_fraud_risk", boom)
    patched.setattr(graph, "retrieve_policy", boom)
    result = _run()
    assert result["errors"] == []
    assert result.get("fraud_result") is None
    assert result.get("policy_result") is None
    assert result["decision"].escalation_reasons == [
        EscalationReason.LOW_CONFIDENCE_EXTRACTION
    ]
    assert result["escalation_packet"] is not None


def test_intake_error_escalates_instead_of_dropping(patched):
    def boom(s):
        raise RuntimeError("bad json")

    patched.setattr(graph, "extract_claim", boom)
    result = _run()
    assert result["errors"] == ["intake: bad json"]
    assert result["escalation_packet"].escalation_reasons == [
        EscalationReason.PROCESSING_ERROR
    ]
    assert result["escalation_packet"].extracted_claim is None


def test_fraud_error_escalates_after_parallel_branches(patched):
    def boom(c):
        raise RuntimeError("model missing")

    patched.setattr(graph, "score_fraud_risk", boom)
    result = _run()
    assert result["errors"] == ["fraud_risk: model missing"]
    assert result["policy_result"] is not None
    assert result["escalation_packet"].escalation_reasons == [
        EscalationReason.PROCESSING_ERROR
    ]


def test_adjudication_error_escalates(patched):
    def boom(c, p, f):
        raise RuntimeError("llm down")

    patched.setattr(graph, "adjudicate", boom)
    result = _run()
    assert result["errors"] == ["adjudication: llm down"]
    assert result["escalation_packet"].escalation_reasons == [
        EscalationReason.PROCESSING_ERROR
    ]


def test_missing_input_is_recorded_not_raised(patched, caplog):
    patched.setattr(graph, "score_fraud_risk", lambda c: None)
    patched.setattr(graph, "retrieve_policy", lambda c: None)
    result = _run()
    assert result["errors"] == ["adjudication: missing policy_result"]
    assert result["escalation_packet"].escalation_reasons == [
        EscalationReason.PROCESSING_ERROR
    ]
    assert "adjudication failed for submission s1" in caplog.text
