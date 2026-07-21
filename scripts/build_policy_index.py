# Build the Policy Retrieval Agent's search index from data/policy_docs/*.md.
#
# Produces the FAISS + BM25 + docstore artifacts that
# src/agents/policy_retrieval.py loads at query time.
#
# Run with: uv run python scripts/build_policy_index.py
# See PLAN.md, Phase 1.
#
# TODO: load_and_chunk_docs()
# - Read every *.md file in data/policy_docs and split into chunks. Each
#   policy doc already has a "Document ID:" line and numbered
#   "## N. Section" headings (see data/policy_docs/auto_theft.md) — chunk
#   by section so each chunk maps to one clause, and carry document_id +
#   section through as metadata for PolicyClause provenance later.
#
# TODO: build_index(chunks)
# - Embed chunk texts with settings.models.embedding_model (via Ollama).
# - Build a FAISS index over the embeddings.
# - Build a BM25 corpus (rank-bm25) over the same chunk texts.
# - Persist both, plus a chunk_id -> {text, document_id, section} docstore,
#   to settings.paths.faiss_index_path. Keep this format in sync with what
#   src/agents/policy_retrieval.py.load_index() expects.
