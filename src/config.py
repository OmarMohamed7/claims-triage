"""
Central configuration. Everything tunable lives here so thresholds aren't
buried inside agent logic where they're hard to find and harder to justify
in an audit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Thresholds:
    """Decision boundaries used by the Adjudication Agent's rule layer."""

    # Fraud risk score (0-1) above this triggers automatic escalation,
    # regardless of what the LLM reasoning layer thinks.
    fraud_auto_escalate: float = 0.80

    # Below this fraud score, and with clear policy coverage, auto-approve
    # is allowed without LLM deliberation.
    fraud_auto_approve_ceiling: float = 0.30

    # Claim amounts above this always route through the LLM reasoning step
    # (and likely escalation) rather than the fast-path rules.
    high_value_claim_amount: float = 25_000.0

    # Below this, Intake extraction is considered too unreliable to proceed
    # automatically; route straight to human review.
    min_extraction_confidence: float = 0.70

    # Below this, Policy Retrieval is considered inconclusive.
    min_retrieval_confidence: float = 0.65
    
    # Top-k results to retrieve from Qdrant and BM25 for policy retrieval
    top_k_policy_retrieval: int = 5


@dataclass
class ModelConfig:
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    llm_model: str = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    gcp_project: str = os.getenv("GCP_PROJECT", "")
    gcp_location: str = os.getenv("GCP_LOCATION", "global")
    fraud_model_path: str = os.getenv(
        "FRAUD_MODEL_PATH", "models/fraud_classifier.joblib"
    )
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    @property
    def default_llm(self) -> str:
        return f"{self.llm_provider}:{self.llm_model}"

    @property
    def llm_choices(self) -> list[str]:
        extra = [c.strip() for c in os.getenv("LLM_CHOICES", "").split(",") if c.strip()]
        return list(dict.fromkeys([self.default_llm, *extra]))


@dataclass
class PathConfig:
    data_dir: str = os.getenv("DATA_DIR", "data")
    model_dir: str = os.getenv("MODEL_DIR",  "models")
    policy_docs_dir: str = os.getenv("POLICY_DOCS_DIR", "data/policy_docs")
    qdrant_path: str = os.getenv("QDRANT_PATH", "data/qdrant")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "policy_clauses")
    reviews_database_url: str = os.getenv("REVIEWS_DATABASE_URL", "")
    reviews_db: str = os.getenv("REVIEWS_DB", "data/reviews.db")
    raw_claims_dataset: str = os.getenv("RAW_CLAIMS_DATASET", "data/raw/claims.csv")


@dataclass
class ApiConfig:
    rate_limit: str = os.getenv("RATE_LIMIT", "10/minute")


@dataclass
class Config:
    thresholds: Thresholds = field(default_factory=Thresholds)
    api: ApiConfig = field(default_factory=ApiConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    VERSION = "1.0.0"


config = Config()
