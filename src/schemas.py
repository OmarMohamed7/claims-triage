"""
Shared data contracts for the claims triage pipeline.

Every agent reads and writes these schemas. Keeping them centralized means
the Intake Agent, Policy Retrieval Agent, Fraud Risk Agent, Adjudication
Agent, and Human Escalation Agent all speak the same language, and any
schema change is a one-file diff instead of a hunt through five agents.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ClaimType(str, Enum):
    AUTO_COLLISION = "auto_collision"
    AUTO_THEFT = "auto_theft"
    AUTO_COMPREHENSIVE = "auto_comprehensive"
    PROPERTY_FIRE = "property_fire"
    PROPERTY_WATER = "property_water"
    PROPERTY_THEFT = "property_theft"
    LIABILITY = "liability"
    OTHER = "other"


class DecisionStatus(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    ESCALATED = "escalated"
    PENDING_HUMAN_REVIEW = "pending_human_review"


class EscalationReason(str, Enum):
    HIGH_FRAUD_RISK = "high_fraud_risk"
    AMBIGUOUS_POLICY_COVERAGE = "ambiguous_policy_coverage"
    HIGH_CLAIM_VALUE = "high_claim_value"
    LOW_CONFIDENCE_EXTRACTION = "low_confidence_extraction"
    CONFLICTING_SIGNALS = "conflicting_signals"
    MANUAL_REVIEW_REQUESTED = "manual_review_requested"


# ---------------------------------------------------------------------------
# 1. Raw input -> Intake Agent
# ---------------------------------------------------------------------------


class ClaimSubmission(BaseModel):
    """Whatever arrives at the front door: a form, an email body, a PDF dump."""

    submission_id: str
    raw_text: str = Field(..., description="Unstructured claim narrative / form text")
    policy_number: Optional[str] = None
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attachment_paths: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Intake Agent output -> everything downstream
# ---------------------------------------------------------------------------


class ExtractedClaim(BaseModel):
    submission_id: str

    policy_number: Optional[str] = None
    claimant_name: Optional[str] = None
    claim_type: Optional[ClaimType] = None

    incident_date: Optional[date] = None
    reported_date: Optional[date] = None

    claimed_amount: Optional[float] = Field(
        default=None,
        gt=0,
    )

    incident_location: Optional[str] = None
    incident_description: Optional[str] = None

    other_parties_involved: bool = False

    extraction_confidence: float = Field(
        ...,
        ge=0,
        le=1,
    )

    missing_fields: list[str] = Field(
        default_factory=list,
    )

    @field_validator("reported_date")
    @classmethod
    def reported_after_incident(cls, v, info):
        incident = info.data.get("incident_date")

        if v is None or incident is None:
            return v

        if v < incident:
            raise ValueError(
                "reported_date cannot be before incident_date"
            )

        return v
# ---------------------------------------------------------------------------
# 3. Policy Retrieval Agent output
# ---------------------------------------------------------------------------


class PolicyClause(BaseModel):
    """A single retrieved clause with enough provenance to audit the decision."""

    document_id: str
    section: str
    text_excerpt: str = Field(..., max_length=1000)
    relevance_score: float = Field(..., ge=0, le=1)


class PolicyRetrievalResult(BaseModel):
    submission_id: str
    policy_number: str
    is_covered: Optional[bool] = Field(
        None, description="None means retrieval was inconclusive"
    )
    deductible: Optional[float] = None
    coverage_limit: Optional[float] = None
    relevant_clauses: list[PolicyClause] = Field(default_factory=list)
    exclusions_matched: list[str] = Field(default_factory=list)
    retrieval_confidence: float = Field(..., ge=0, le=1)


# ---------------------------------------------------------------------------
# 4. Fraud Risk Agent output (classical ML, not an LLM call)
# ---------------------------------------------------------------------------


class FeatureContribution(BaseModel):
    """One feature's contribution to the risk score, e.g. from SHAP."""

    feature_name: str
    value: float
    contribution: float = Field(
        ..., description="Signed contribution to the risk score (SHAP value)"
    )


class FraudRiskResult(BaseModel):
    submission_id: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_tier: str = Field(..., description="e.g. 'low', 'medium', 'high'")
    top_contributing_features: list[FeatureContribution] = Field(default_factory=list)
    model_version: str


# ---------------------------------------------------------------------------
# 5. Adjudication Agent output
# ---------------------------------------------------------------------------


class AdjudicationDecision(BaseModel):
    submission_id: str
    status: DecisionStatus
    approved_amount: Optional[float] = None
    reasoning: str = Field(..., description="Human-readable justification")
    escalation_reasons: list[EscalationReason] = Field(default_factory=list)
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# 6. Human Escalation Agent
# ---------------------------------------------------------------------------


class EscalationPacket(BaseModel):
    """Everything a human reviewer needs, assembled from every prior agent."""

    submission_id: str
    extracted_claim: ExtractedClaim
    policy_result: PolicyRetrievalResult
    fraud_result: FraudRiskResult
    planner_reasoning: str
    escalation_reasons: list[EscalationReason]
    status: DecisionStatus = DecisionStatus.PENDING_HUMAN_REVIEW


class HumanReviewOutcome(BaseModel):
    submission_id: str
    reviewer_id: str
    final_status: DecisionStatus
    approved_amount: Optional[float] = None
    reviewer_notes: str
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# 7. Full pipeline state (what LangGraph threads through the graph)
# ---------------------------------------------------------------------------


class PipelineState(BaseModel):
    """The shared state object passed between LangGraph nodes."""

    submission: ClaimSubmission
    extracted_claim: Optional[ExtractedClaim] = None
    policy_result: Optional[PolicyRetrievalResult] = None
    fraud_result: Optional[FraudRiskResult] = None
    decision: Optional[AdjudicationDecision] = None
    escalation_packet: Optional[EscalationPacket] = None
    human_outcome: Optional[HumanReviewOutcome] = None
    errors: list[str] = Field(default_factory=list)
