# Fraud Risk Agent — classical ML (not an LLM call). Scores fraud risk for a
# claim using the classifier trained by scripts/train_fraud_model.py.
# See PLAN.md, Phase 4.
#
# TODO: load_model()
# - joblib.load(config.models.fraud_model_path); cache it so repeated
#   calls don't hit disk. Raise a clear error if the file is missing (i.e.
#   scripts/train_fraud_model.py hasn't been run yet).
#
# TODO: lookup_policyholder_profile(policy_number) -> dict
# - Mock policyholder/incident lookup, keyed by PolicyNumber, standing in
#   for a real policy system (see scripts/train_fraud_model.py docstring
#   for why data/raw/claims.csv is reused here).
# - Read data/raw/claims.csv (config.paths.raw_claims_dataset), find the
#   row matching policy_number, and return it as a dict of the same columns
#   the model was trained on (i.e. everything except PolicyNumber and
#   FraudFound_P — see DROP_COLS/TARGET_COL in scripts/train_fraud_model.py).
# - Decide what happens when policy_number isn't found in the CSV (no
#   history for a genuinely new policyholder) — this is a real case, not
#   just a test fixture gap.
#
# TODO: score_fraud_risk(claim) -> FraudRiskResult
# - load_model() + lookup_policyholder_profile(claim.policy_number) to
#   build the same feature row shape used in training.
# - model.predict_proba(...) -> risk_score in [0, 1].
# - Bucket risk_score into risk_tier ("low"/"medium"/"high") using
#   config.thresholds.fraud_auto_approve_ceiling / fraud_auto_escalate as
#   the boundaries.
# - model_version: read from models/fraud_classifier_metrics.json or
#   hardcode a version string bumped alongside retraining.
# - top_contributing_features can stay empty for the MVP (shap is commented
#   out of pyproject.toml) — wire it up later if needed.
