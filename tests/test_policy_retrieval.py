# Tests for the Policy Retrieval Agent (src/agents/policy_retrieval.py).
# See PLAN.md, Phase 9.
#
# TODO:
# - retrieve_policy() for an auto_theft claim returns clauses whose
#   document_id matches POLICY-AUTO-THEFT-01, not an unrelated policy doc.
# - A claim narrative matching a documented exclusion (e.g. "keys were left
#   in the ignition") should surface that exclusion in exclusions_matched
#   and set is_covered to False.
# - retrieval_confidence drops below
#   config.thresholds.min_retrieval_confidence when no policy doc is a
#   good match, and is_covered should end up None, not a guessed True/False.
# - Build a small fixture index (few chunks) rather than depending on the
#   full Qdrant build_index() artifacts for unit tests. Keep the fixture
#   bigger than "one chunk per doc" (e.g. a handful of docs, several
#   sections each) — with only 3-4 total chunks, BM25's IDF weighting
#   over-rewards a single incidental shared word (like "from") enough
#   that it can outrank the actually-relevant doc. A handful of docs with
#   a few sections each is closer to how the real ~32-chunk corpus
#   behaves and avoids that false signal.
# - Stub out the embedding call (the one external-service dependency,
#   requires a live Ollama model) rather than mocking retrieval
#   internals — let real Qdrant + real BM25 search run against the
#   fixture index so the test exercises actual retrieval behavior.
# - If parsing deductible/coverage_limit with regex, add a case for a
#   dollar amount that wraps across a line break, and a section with more
#   than one "up to $X" mention (e.g. a sub-limit) to make sure the real
#   coverage limit is picked, not the sub-limit.
