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
