# Tests for the Adjudication Agent (src/agents/adjudication.py). See PLAN.md, Phase 9.
#
# TODO:
# - apply_rules() auto-approves a low-fraud, covered, high-confidence,
#   low-value claim without needing an LLM call.
# - apply_rules() auto-escalates when fraud_result.risk_score is above
#   config.thresholds.fraud_auto_escalate, and lists HIGH_FRAUD_RISK in
#   escalation_reasons.
# - apply_rules() auto-escalates a claim above
#   config.thresholds.high_value_claim_amount even when fraud risk is low.
# - apply_rules() returns None (ambiguous) for a case that doesn't cleanly
#   hit an approve or escalate rule, and adjudicate() then falls back to
#   the LLM path — mock the LLM call for this test.

import json

import pytest

from src.agents.adjudication import adjudicate, adjudicate_with_llm, apply_rule
from src.config import config
from src.schemas import (
    DecisionStatus,
    ExtractedClaim,
    FraudRiskResult,
    PolicyRetrievalResult,
)


def make_claim(**overrides) -> ExtractedClaim:
    defaults = dict(
        submission_id="CLM-TEST-0001",
        claimed_amount=1000.0,
        extraction_confidence=0.95,
    )
    defaults.update(overrides)
    return ExtractedClaim(**defaults) # pyright: ignore[reportArgumentType]


def make_policy_result(**overrides) -> PolicyRetrievalResult:
    defaults = dict(
        submission_id="CLM-TEST-0001",
        policy_number="POL-0001",
        is_covered=True,
        retrieval_confidence=0.95,
    )
    defaults.update(overrides)
    return PolicyRetrievalResult(**defaults) # pyright: ignore[reportArgumentType]


def make_fraud_result(**overrides) -> FraudRiskResult:
    defaults = dict(
        submission_id="CLM-TEST-0001",
        risk_score=0.05,
        risk_tier="low",
        model_version="test-1",
    )
    defaults.update(overrides)
    return FraudRiskResult(**defaults) # pyright: ignore[reportArgumentType]


def test_apply_rule_auto_approves_low_risk_covered_claim():
    decision = apply_rule(make_claim(), make_policy_result(), make_fraud_result())

    assert decision is not None
    assert decision.status == DecisionStatus.APPROVED


def test_apply_rule_auto_escalates_high_fraud_risk():
    fraud_result = make_fraud_result(
        risk_score=config.thresholds.fraud_auto_escalate + 0.01
    )

    decision = apply_rule(make_claim(), make_policy_result(), fraud_result)

    assert decision is not None
    assert decision.status == DecisionStatus.ESCALATED


def test_apply_rule_auto_escalates_high_value_claim():
    claim = make_claim(
        claimed_amount=config.thresholds.high_value_claim_amount + 1
    )

    decision = apply_rule(claim, make_policy_result(), make_fraud_result())

    assert decision is not None
    assert decision.status == DecisionStatus.ESCALATED


def test_apply_rule_returns_none_for_ambiguous_case():
    # Covered, confident, low-value — but fraud risk sits between the
    # auto-approve ceiling and the auto-escalate threshold, so neither
    # fast-path rule fires and the caller must fall back to the LLM.
    fraud_result = make_fraud_result(
        risk_score=(
            config.thresholds.fraud_auto_approve_ceiling
            + config.thresholds.fraud_auto_escalate
        )
        / 2
    )

    decision = apply_rule(make_claim(), make_policy_result(), fraud_result)

    assert decision is None


def test_adjudicate_with_llm_parses_json_response_into_decision(monkeypatch):
    """Regression test: LLMProvider.generate() returns a raw JSON string,
    not a structured object. adjudicate_with_llm must parse it, not treat
    the string itself as having .status/.reasoning/etc attributes."""

    llm_response = json.dumps(
        {
            "status": "escalated",
            "reasoning": "Coverage is ambiguous given the retrieved clauses.",
            "escalation_reasons": ["ambiguous_policy_coverage"],
            "approved_amount": None,
        }
    )

    class FakeLLM:
        def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
            return llm_response

    monkeypatch.setattr(
        "src.agents.adjudication.get_llm", lambda: FakeLLM()
    )

    claim = make_claim()
    policy_result = make_policy_result(is_covered=None)
    fraud_result = make_fraud_result()

    decision = adjudicate_with_llm(claim, policy_result, fraud_result)

    assert decision.submission_id == claim.submission_id
    assert decision.status == DecisionStatus.ESCALATED
    assert decision.reasoning == (
        "Coverage is ambiguous given the retrieved clauses."
    )
    assert decision.escalation_reasons == ["ambiguous_policy_coverage"]


def test_adjudicate_falls_back_to_llm_when_rules_ambiguous(monkeypatch):
    llm_response = json.dumps(
        {
            "status": "escalated",
            "reasoning": "Ambiguous — routing to human review.",
            "escalation_reasons": [],
            "approved_amount": None,
        }
    )

    class FakeLLM:
        def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
            return llm_response

    monkeypatch.setattr(
        "src.agents.adjudication.get_llm", lambda: FakeLLM()
    )

    claim = make_claim()
    policy_result = make_policy_result()
    fraud_result = make_fraud_result(
        risk_score=(
            config.thresholds.fraud_auto_approve_ceiling
            + config.thresholds.fraud_auto_escalate
        )
        / 2
    )

    decision = adjudicate(claim, policy_result, fraud_result)

    assert decision.status == DecisionStatus.ESCALATED
