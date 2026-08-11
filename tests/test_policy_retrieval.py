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
#   settings.thresholds.min_retrieval_confidence when no policy doc is a
#   good match, and is_covered should end up None, not a guessed True/False.
# - Build a small fixture index (few chunks) rather than depending on the
#   full Qdrant build_index() artifacts for unit tests.
