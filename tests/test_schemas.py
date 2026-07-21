"""Sanity checks for the shared pipeline schemas. Run with: pytest tests/"""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from src.schemas import (
    ClaimSubmission,
    ClaimType,
    DecisionStatus,
    EscalationReason,
    ExtractedClaim,
    FraudRiskResult,
    PipelineState,
    PolicyRetrievalResult,
)


def test_claim_submission_minimal():
    sub = ClaimSubmission(submission_id="sub_001", raw_text="My car was hit.")
    assert sub.submission_id == "sub_001"
    assert sub.attachment_paths == []


def test_extracted_claim_valid():
    claim = ExtractedClaim(
        submission_id="sub_001",
        policy_number="POL-123",
        claimant_name="Jane Doe",
        claim_type=ClaimType.AUTO_COLLISION,
        incident_date=date(2026, 6, 1),
        reported_date=date(2026, 6, 3),
        claimed_amount=4200.00,
        incident_description="Rear-ended at a stoplight.",
        extraction_confidence=0.92,
    )
    assert claim.claim_type == ClaimType.AUTO_COLLISION
    assert claim.missing_fields == []


def test_extracted_claim_rejects_reported_before_incident():
    with pytest.raises(ValidationError):
        ExtractedClaim(
            submission_id="sub_002",
            policy_number="POL-124",
            claimant_name="John Roe",
            claim_type=ClaimType.PROPERTY_FIRE,
            incident_date=date(2026, 6, 10),
            reported_date=date(2026, 6, 1),  # before incident_date -> invalid
            claimed_amount=1000.0,
            incident_description="Kitchen fire.",
            extraction_confidence=0.5,
        )


def test_extracted_claim_rejects_zero_amount():
    with pytest.raises(ValidationError):
        ExtractedClaim(
            submission_id="sub_003",
            policy_number="POL-125",
            claimant_name="A B",
            claim_type=ClaimType.OTHER,
            incident_date=date(2026, 1, 1),
            reported_date=date(2026, 1, 2),
            claimed_amount=0,  # must be > 0
            incident_description="n/a",
            extraction_confidence=0.5,
        )


def test_fraud_risk_result_bounds():
    with pytest.raises(ValidationError):
        FraudRiskResult(
            submission_id="sub_001",
            risk_score=1.5,  # out of [0, 1] bounds
            risk_tier="high",
            model_version="v0.1",
        )


def test_policy_retrieval_result_inconclusive_allowed():
    result = PolicyRetrievalResult(
        submission_id="sub_001",
        policy_number="POL-123",
        is_covered=None,
        retrieval_confidence=0.4,
    )
    assert result.is_covered is None


def test_pipeline_state_builds_incrementally():
    sub = ClaimSubmission(submission_id="sub_001", raw_text="text")
    state = PipelineState(submission=sub)
    assert state.extracted_claim is None
    assert state.errors == []
