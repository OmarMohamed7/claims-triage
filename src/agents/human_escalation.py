# Human Escalation Agent — assembles an audit-ready EscalationPacket for a
# human reviewer. Assembly only, no LLM call.
# See PLAN.md, Phase 6.
#
# TODO: build_escalation_packet(claim, policy_result, fraud_result, decision) -> EscalationPacket
# - Populate every EscalationPacket field from the four inputs above —
#   nothing upstream should be silently dropped; the whole point of this
#   agent is a complete audit trail.
# - planner_reasoning comes from decision.reasoning.
# - status should mirror decision.status (typically ESCALATED or
#   PENDING_HUMAN_REVIEW — decide whether this function should assert
#   that, given it's only meant to be called in that case).

from src.schemas import AdjudicationDecision, DecisionStatus, EscalationPacket, FraudRiskResult, ExtractedClaim, PolicyRetrievalResult


def build_escalation_packet(
    submission_id: str,
    extracted_claim: ExtractedClaim,
    policy_result: PolicyRetrievalResult,
    fraud_result: FraudRiskResult,
    decision: AdjudicationDecision,
) -> EscalationPacket:
    """Assembles an audit-ready EscalationPacket for a human reviewer."""
    
    if decision.status not in (DecisionStatus.ESCALATED, DecisionStatus.PENDING_HUMAN_REVIEW):
        raise ValueError(f"build_escalation_packet called with non-escalated status: {decision.status}")
    
    return EscalationPacket(
        submission_id=submission_id,
        extracted_claim=extracted_claim,
        policy_result=policy_result,
        fraud_result=fraud_result,
        planner_reasoning=decision.reasoning,
        escalation_reasons=decision.escalation_reasons,
        status=decision.status,
    )