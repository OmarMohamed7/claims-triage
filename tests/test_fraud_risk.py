# Tests for the Fraud Risk Agent (src/agents/fraud_risk.py). See PLAN.md, Phase 9.
#
# TODO:
# - score_fraud_risk() for a policy_number present in data/raw/claims.csv
#   returns a FraudRiskResult with risk_score in [0, 1] and a risk_tier
#   consistent with settings.thresholds boundaries.
# - lookup_policyholder_profile() for an unknown policy_number behaves as
#   decided in the TODO in fraud_risk.py (raises / returns a default) —
#   pin that behavior down with a test once implemented.
# - Mock joblib.load so the test doesn't depend on
#   models/fraud_classifier.joblib existing on disk (i.e. doesn't require
#   scripts/train_fraud_model.py to have been run first).
