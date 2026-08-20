# Policy Retrieval Agent — hybrid (Qdrant + BM25) search over data/policy_docs
# to determine coverage for an ExtractedClaim.
#
# Reads the index artifacts built by scripts/build_policy_index.py.
# See PLAN.md, Phase 3.
#
\

import json
from pathlib import Path
import pickle
import re
from typing import Any

from langchain_ollama import OllamaEmbeddings
from numpy import float64
from qdrant_client import QdrantClient

from src.config import config
from src.schemas import ExtractedClaim, PolicyClause, PolicyRetrievalResult
from src.text_utils import STOPWORDS
from rank_bm25 import BM25Okapi


# Anchored on the label so we grab the amount tied to it, not a later
# sub-limit in the same section (e.g. auto_theft.md's aftermarket-equipment
# cap after the real coverage limit).
_MONEY_RE = r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
_DEDUCTIBLE_RE = re.compile(rf"deductible:[^$]*{_MONEY_RE}", re.IGNORECASE)
_LIMIT_RE = re.compile(rf"limit:[^$]*{_MONEY_RE}", re.IGNORECASE)

_EXCLUSION_SECTION_KEYWORD = "exclusion"


def _parse_dollar_amount(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


_index_cache: tuple[QdrantClient, BM25Okapi, dict[str, Any], list[str]] | None = None


def load_index() -> tuple[QdrantClient, BM25Okapi, dict[str, Any], list[str]]:
    """Load the Qdrant + BM25 index artifacts for policy retrieval.

    Cached at module level: local-mode Qdrant holds an exclusive file lock
    while a client is open, and opening a second one before the first is
    closed raises "Storage folder ... already accessed by another
    instance". One client for the process's lifetime avoids that.

    Returns:
        tuple: (qdrant_client, bm25_index, docstore, chunk_ids). chunk_ids
        is docstore's keys in insertion order, which lines up 1:1 with the
        BM25 corpus order and the Qdrant point ids — all three were built
        from the same `chunks` list in scripts/build_policy_index.py.
    """
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    index_dir = Path(config.paths.qdrant_path)

    quadrant_client = QdrantClient(path=str(index_dir))

    with open(index_dir / "bm25.pkl", "rb") as f:
        bm25: BM25Okapi = pickle.load(f)

    with open(index_dir / "docstore.json") as f:
        docstore: dict = json.load(f)

    _index_cache = (quadrant_client, bm25, docstore, list(docstore.keys()))
    return _index_cache

def retrieve_policy(claim: ExtractedClaim) -> PolicyRetrievalResult:
    
    claim_type = claim.claim_type.value.replace("_", " ") if claim.claim_type is not None else ""

    query = f"${claim_type} {claim.incident_description}"
    embedded_query = embed_query(query = query)

    qdrant_client, bm25, docstore, _ = load_index()

    # embedding search
    dense_res = search_qdrant(embedded_query, qdrant_client)

    # Sparse Retrieval
    sparse_res: Any = search_bm25(query, bm25, docstore)
    
    # Fusing ( Merging the embedding search res+ sparse res)
    fused_res = fuse_result(dense_res, sparse_res)
    
    # Remove weak res
    relevant_res = apply_removal(fused_res)

    retrieval_confidence = relevant_res[0].relevance_score if relevant_res else 0.0

    if not relevant_res or retrieval_confidence < config.thresholds.min_retrieval_confidence:
        # Not enough signal -- inconclusive rather than a guess.
        return PolicyRetrievalResult(
            submission_id=claim.submission_id,
            policy_number=claim.policy_number or "",
            is_covered=None,
            relevant_clauses=relevant_res,
            retrieval_confidence=retrieval_confidence,
        )

    # Scope to the top-matched document only, in case a second document's
    # clause squeaked past apply_removal's floor.
    primary_document_id = relevant_res[0].document_id
    primary_doc_clauses = [c for c in relevant_res if c.document_id == primary_document_id]

    exclusion_clauses = [
        c for c in primary_doc_clauses if _EXCLUSION_SECTION_KEYWORD in c.section.lower()
    ]
    is_covered = _EXCLUSION_SECTION_KEYWORD not in primary_doc_clauses[0].section.lower()

    deductible: float | None = None
    coverage_limit: float | None = None
    for clause in primary_doc_clauses:
        if deductible is None:
            deductible = _parse_dollar_amount(_DEDUCTIBLE_RE, clause.text_excerpt)
        if coverage_limit is None:
            coverage_limit = _parse_dollar_amount(_LIMIT_RE, clause.text_excerpt)

    return PolicyRetrievalResult(
        submission_id=claim.submission_id,
        policy_number=claim.policy_number or "",
        is_covered=is_covered,
        deductible=deductible,
        coverage_limit=coverage_limit,
        relevant_clauses=relevant_res,
        exclusions_matched=[c.text_excerpt for c in exclusion_clauses],
        retrieval_confidence=retrieval_confidence,
    )

def embed_query(query: str) -> list[float]:
    return OllamaEmbeddings(model=config.models.embedding_model).embed_query(query)
    

def search_qdrant(embeddings: list[float], qdrant_client: QdrantClient, top_k: int = 5) -> list[PolicyClause]:
    '''
    For example:

    Claim:
    My car was stolen overnight.

    Policy:
    Coverage applies when a covered automobile is unlawfully taken without the owner's consent.

    There isn't necessarily strong word-for-word overlap, but semantically they're very similar.
    That's where dense retrieval is strong.
    '''
    res = qdrant_client.query_points(
        collection_name=config.paths.qdrant_collection_name,
        query=embeddings,
        limit=top_k,
        with_payload=True,
    )

    return [
        PolicyClause(
            text_excerpt=point.payload["text"],
            document_id=point.payload["document_id"],
            section=point.payload["section"],
            relevance_score=point.score,
        )
        for point in res.points
    ]
 
def search_bm25(
    query: str, bm25: BM25Okapi, docstore: dict[str, Any], top_k: int = 5
) -> list[PolicyClause]:
    
    '''
    You also need a traditional keyword search.

    BM25 asks:

    "Does this policy text contain important words from the claim?"

    For:

    auto theft Vehicle was stolen

    BM25 might strongly match:

    Theft of the insured automobile is covered...

    because:

    theft
    automobile

    appear explicitly.
    '''
    
    tokens = [t for t in query.split() if t.lower() not in STOPWORDS]
    scores = bm25.get_scores(tokens)
    max_score: float64 | float = max(scores) if len(scores) else 0.0

    records = list(docstore.values())
    ranked = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)[:top_k]

    clauses: list[PolicyClause] = []
    for i in ranked:
        record = records[i]
        clauses.append(
            PolicyClause(
                text_excerpt=record["text"],
                document_id=record["document_id"],
                section=record["section"],
                relevance_score=float(scores[i] / max_score) if max_score > 0 else 0.0,
            )
        )

    return clauses


def fuse_result(
    dense_res: list[PolicyClause], sparse_res: list[PolicyClause]
) -> list[PolicyClause]:
    '''
    This is the important architectural idea.

    Dense:

    semantic similarity

    BM25:

    lexical / keyword similarity

    Neither one is perfect.

    For example:

    Dense result
    Claim:
    Car was stolen.


    Policy:
    Loss caused by unlawful taking of an insured vehicle...

    Dense probably likes this.

    BM25 result
    Claim:
    Car was stolen.


    Policy:
    Theft coverage applies to automobiles...

    BM25 probably likes this.

    Combining both gives you a stronger retrieval system.

    This is called hybrid retrieval.
    '''
    
    DENSE_WEIGHT = 0.65
    SPARSE_WEIGHT = 0.35

    # keyed on full clause identity -- section alone isn't unique since a
    # section can be split into several chunks.
    clauses_by_key: dict[tuple[str, str, str], PolicyClause] = {}
    dense_scores: dict[tuple[str, str, str], float] = {}
    sparse_scores: dict[tuple[str, str, str], float] = {}

    for clause in dense_res:
        key = (clause.document_id, clause.section, clause.text_excerpt)
        clauses_by_key[key] = clause
        dense_scores[key] = clause.relevance_score

    for clause in sparse_res:
        key = (clause.document_id, clause.section, clause.text_excerpt)
        clauses_by_key.setdefault(key, clause)
        sparse_scores[key] = clause.relevance_score

    fused: list[PolicyClause] = []
    for key, clause in clauses_by_key.items():
        dense_score = dense_scores.get(key)
        sparse_score = sparse_scores.get(key)

        if dense_score is not None and sparse_score is not None:
            score = DENSE_WEIGHT * dense_score + SPARSE_WEIGHT * sparse_score
        elif dense_score is not None:
            score = DENSE_WEIGHT * dense_score  # uncorroborated -- floor it, don't trust in full
        else:
            assert sparse_score is not None  # dense_score is None, so this came from sparse_res
            score = SPARSE_WEIGHT * sparse_score

        fused.append(
            PolicyClause(
                text_excerpt=clause.text_excerpt,
                document_id=clause.document_id,
                section=clause.section,
                relevance_score=score,
            )
        )

    fused.sort(key=lambda c: c.relevance_score, reverse=True)
    return fused


def apply_removal(
    fused_res: list[PolicyClause], relative_floor: float = 0.5
) -> list[PolicyClause]:
    """
    Trim the fused list down to genuinely relevant clauses, using a floor
    relative to the top hit rather than a fixed top_k or absolute score --
    BM25/cosine scores aren't comparable across different queries in
    absolute terms.
    """
    if not fused_res:
        return []

    top_score = fused_res[0].relevance_score
    return [c for c in fused_res if c.relevance_score >= relative_floor * top_score]