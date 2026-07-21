# Policy Retrieval Agent — hybrid (FAISS + BM25) search over data/policy_docs
# to determine coverage for an ExtractedClaim.
#
# Reads the index artifacts built by scripts/build_policy_index.py.
# See PLAN.md, Phase 3.
#
# TODO: load_index()
# - Load whatever artifacts scripts/build_policy_index.py wrote to
#   settings.paths.faiss_index_path (FAISS index + BM25 object/pickle +
#   chunk_id -> {text, document_id, section} docstore). Keep the on-disk
#   format in sync with Phase 1.
#
# TODO: retrieve_policy(claim) -> PolicyRetrievalResult
# - Embed a query built from claim.claim_type + incident_description.
# - Run dense (FAISS) and sparse (BM25) search, combine rankings (e.g.
#   reciprocal rank fusion), keep the top-k chunks as PolicyClause.
# - Determine is_covered / deductible / coverage_limit by reading the top
#   clauses (LLM synthesis, or rule-based section matching against
#   "Exclusions" — pick one and document why). Use exclusions_matched to
#   record which specific exclusions apply.
# - Set retrieval_confidence from retrieval score strength / agreement
#   between dense and sparse results; below
#   settings.thresholds.min_retrieval_confidence should mean the
#   Adjudication Agent treats coverage as inconclusive.
