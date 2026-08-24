# Fraud Risk Agent — classical ML (not an LLM call). Scores fraud risk for a
# claim using the classifier trained by scripts/train_fraud_model.py.
# See PLAN.md, Phase 4.
#
# TODO: load_model()
# - joblib.load(config.models.fraud_model_path); cache it so repeated
#   calls don't hit disk. Raise a clear error if the file is missing (i.e.
#   scripts/train_fraud_model.py hasn't been run yet).
#

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.schemas import ExtractedClaim, FraudRiskResult
from src.config import config


_model_cache: Pipeline | None = None


def load_model() -> Optional[Pipeline]:

    global _model_cache
    if _model_cache is not None:
        return _model_cache

    model_path = Path(config.models.fraud_model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Fraud model not found at '{model_path.resolve()}' -- "
            "run scripts/train_fraud_model.py first."
        )

    _model_cache = joblib.load(model_path)  # type: ignore
    return _model_cache

_BASE_POLICY_PREFIX = {
    "Liability": "LIAB",
    "Collision": "COLL",
    "All Perils": "ALLP",
}


def _synthetic_policy_numbers(df: pd.DataFrame) -> pd.Series:
    prefix = df["BasePolicy"].map(_BASE_POLICY_PREFIX).fillna("GEN")
    return "POL-" + prefix + "-" + df["PolicyNumber"].astype(int).astype(str).str.zfill(6)


def lookup_policyholder_profile(policy_number: str) -> Dict[str, Any]:
    """
    Looks up a policyholder's historical feature profile from the raw claims dataset.

    Drops identification and target columns to return exactly the feature set
    expected by the fraud classification model.
    """

    csv_path = Path(config.paths.raw_claims_dataset)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Claims dataset not found at: {csv_path.resolve()}")

    df: pd.DataFrame = pd.read_csv(csv_path, dtype={"PolicyNumber": str})

    DROP_COLS = ["PolicyNumber", "FraudFound_P"]

    matched_row = df[_synthetic_policy_numbers(df) == policy_number]

    if matched_row.empty:
        # No history for a genuinely new policyholder is an expected,
        # common case here -- not an error.
        print(f"Policy number '{policy_number}' not found. Treating as a new policyholder.")
        return {}

    profile_df = matched_row.drop(columns=[col for col in DROP_COLS if col in matched_row.columns])
    profile_dict = profile_df.iloc[0].to_dict()

    return profile_dict # type: ignore

# metrics.json (models/fraud_classifier_metrics.json) is sklearn's
# classification_report output -- no version field in it, so this is
# hardcoded per the TODO's own fallback option. Bump alongside retraining.
MODEL_VERSION = "fraud_classifier_v1"


def _risk_tier(risk_score: float) -> str:
    if risk_score < config.thresholds.fraud_auto_approve_ceiling:
        return "low"
    if risk_score >= config.thresholds.fraud_auto_escalate:
        return "high"
    return "medium"


def score_fraud_risk(claim: ExtractedClaim) -> FraudRiskResult:

    model: Pipeline | None = load_model()

    if not model or not claim.policy_number:
        return FraudRiskResult(
            submission_id=claim.submission_id,
            risk_score=0,
            risk_tier="low",
            model_version="",
        )

    profile = lookup_policyholder_profile(claim.policy_number)

    if not profile:
        return FraudRiskResult(
            submission_id=claim.submission_id,
            risk_score=0,
            risk_tier="low",
            model_version="",
        )

    proba = model.predict_proba(pd.DataFrame([profile]))[0]
    fraud_class_index = list(model.classes_).index(1)
    risk_score = float(proba[fraud_class_index])

    return FraudRiskResult(
        submission_id=claim.submission_id,
        risk_score=risk_score,
        risk_tier=_risk_tier(risk_score),
        top_contributing_features=[],
        model_version=MODEL_VERSION,
    )
    