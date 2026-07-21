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
    fraud_auto_approve_ceiling: float = 0.20

    # Claim amounts above this always route through the LLM reasoning step
    # (and likely escalation) rather than the fast-path rules.
    high_value_claim_amount: float = 25_000.0

    # Below this, Intake extraction is considered too unreliable to proceed
    # automatically; route straight to human review.
    min_extraction_confidence: float = 0.70

    # Below this, Policy Retrieval is considered inconclusive.
    min_retrieval_confidence: float = 0.65


@dataclass
class ModelConfig:
    # Local by default: Ollama serves both chat and embeddings, so the whole
    # pipeline runs with no API key and no per-call cost. Swap LLM_PROVIDER
    # to point at a hosted model later without touching agent code.
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2:3b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    fraud_model_path: str = os.getenv("FRAUD_MODEL_PATH", "models/fraud_classifier.joblib")
    # all-minilm is small enough to bundle in the deployed container; override
    # to a larger Ollama embedding model (e.g. bge-m3) for local dev quality.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-minilm")


@dataclass
class PathConfig:
    data_dir: str = os.getenv("DATA_DIR", "data")
    policy_docs_dir: str = os.getenv("POLICY_DOCS_DIR", "data/policy_docs")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    raw_claims_dataset: str = os.getenv("RAW_CLAIMS_DATASET", "data/raw/claims.csv")


@dataclass
class Settings:
    thresholds: Thresholds = field(default_factory=Thresholds)
    models: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)


settings = Settings()
