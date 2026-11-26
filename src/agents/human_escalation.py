# HUMAN ESCALATION AGENT  (Step 4 of 5, only for escalated claims)
# Uses an LLM: no.
# What it does: bundles everything a human reviewer needs into one packet: the claim,
#   fraud score, policy findings, the decision and why it was escalated.
# Output: EscalationPacket, which the API saves to the adjudicator queue.
# Claims that are approved or denied never reach this step.

from typing import Optional

from src.schemas import (
    AdjudicationDecision,
    DecisionStatus,
    EscalationPacket,
    EscalationQueue,
    EscalationReason,
    ExtractedClaim,
    FraudRiskResult,
    PolicyRetrievalResult,
)


def build_escalation_packet(
    submission_id: str,
    extracted_claim: Optional[ExtractedClaim],
    policy_result: Optional[PolicyRetrievalResult],
    fraud_result: Optional[FraudRiskResult],
    decision: AdjudicationDecision,
) -> EscalationPacket:
    """Assembles an audit-ready EscalationPacket for a human reviewer."""
    
    if decision.status not in (DecisionStatus.ESCALATED, DecisionStatus.PENDING_HUMAN_REVIEW):
        raise ValueError(f"build_escalation_packet called with non-escalated status: {decision.status}")
    
    queue = (
        EscalationQueue.FRAUD_REVIEW
        if EscalationReason.HIGH_FRAUD_RISK in decision.escalation_reasons
        else EscalationQueue.ADJUDICATOR
    )

    return EscalationPacket(
        submission_id=submission_id,
        extracted_claim=extracted_claim,
        policy_result=policy_result,
        fraud_result=fraud_result,
        planner_reasoning=decision.reasoning,
        escalation_reasons=decision.escalation_reasons,
        queue=queue,
        status=decision.status,
    )
