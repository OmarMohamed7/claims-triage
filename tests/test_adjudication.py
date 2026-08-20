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
