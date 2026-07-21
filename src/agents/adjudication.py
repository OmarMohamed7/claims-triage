# Adjudication Agent (Manager/Planner) — combines Intake, Policy Retrieval,
# and Fraud Risk outputs into a final AdjudicationDecision.
# See PLAN.md, Phase 5.
#
# TODO: apply_rules(claim, policy_result, fraud_result) -> AdjudicationDecision | None
# - Fast-path rule layer, using settings.thresholds. Return a decision if
#   the rules are conclusive, otherwise None so the caller falls back to
#   LLM reasoning.
# - Auto-escalate (no LLM needed) when:
#   fraud_result.risk_score > thresholds.fraud_auto_escalate, OR
#   claim.claimed_amount > thresholds.high_value_claim_amount, OR
#   claim.extraction_confidence < thresholds.min_extraction_confidence, OR
#   policy_result.retrieval_confidence < thresholds.min_retrieval_confidence, OR
#   policy_result.is_covered in (None, False).
#   Populate escalation_reasons with every EscalationReason that applies
#   (a claim can hit more than one).
# - Auto-approve (no LLM needed) when:
#   fraud_result.risk_score < thresholds.fraud_auto_approve_ceiling AND
#   policy_result.is_covered is True AND both confidences clear their
#   minimums AND claimed_amount is under the high-value cutoff.
#   approved_amount should respect policy_result.deductible /
#   coverage_limit, not just claim.claimed_amount.
# - Anything else: return None (ambiguous — let the LLM reason about it).
#
# TODO: adjudicate(claim, policy_result, fraud_result) -> AdjudicationDecision
# - Try apply_rules() first; return its result if not None.
# - Otherwise, prompt the LLM (settings.models) with all three upstream
#   results and ask for a status + reasoning + escalation_reasons. Treat
#   the LLM here as a reasoning/justification layer, not a source of new
#   facts — it shouldn't invent coverage or fraud numbers.
# - Always populate `reasoning` with a human-readable justification, since
#   it's what an auditor or the Human Escalation Agent will read.
