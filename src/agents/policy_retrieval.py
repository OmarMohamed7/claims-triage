# Policy Retrieval Agent — hybrid (Qdrant + BM25) search over data/policy_docs
# to determine coverage for an ExtractedClaim.
#
# Reads the index artifacts built by scripts/build_policy_index.py.
# See PLAN.md, Phase 3.
#
# TODO: load_index()
# - Open the Qdrant collection (settings.paths.qdrant_path,
#   settings.paths.qdrant_collection_name) that
#   scripts/build_policy_index.py wrote to, plus whatever BM25 object/pickle
#   + chunk_id -> {text, document_id, section} docstore it persisted
#   alongside it. Keep the on-disk format in sync with Phase 1.
#
# TODO: retrieve_policy(claim) -> PolicyRetrievalResult
# - Embed a query built from claim.claim_type + incident_description.
# - Run dense (Qdrant) and sparse (BM25) search, combine rankings (e.g.
#   reciprocal rank fusion), keep the top-k chunks as PolicyClause.
# - Determine is_covered / deductible / coverage_limit by reading the top
#   clauses (LLM synthesis, or rule-based section matching against
#   "Exclusions" — pick one and document why). Use exclusions_matched to
#   record which specific exclusions apply.
# - Set retrieval_confidence from retrieval score strength / agreement
#   between dense and sparse results; below
#   settings.thresholds.min_retrieval_confidence should mean the
#   Adjudication Agent treats coverage as inconclusive.
