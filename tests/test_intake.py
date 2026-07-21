# Tests for the Intake Agent (src/agents/intake.py). See PLAN.md, Phase 9.
#
# TODO:
# - extract_claim() on a clean, unambiguous raw_text returns an
#   ExtractedClaim with high extraction_confidence and empty missing_fields.
# - extract_claim() on a raw_text missing an obvious required field (e.g.
#   no incident date) reports it in missing_fields with lower confidence,
#   rather than guessing.
# - extract_claim() retries and eventually succeeds (or fails predictably)
#   when the LLM's first response fails ExtractedClaim validation — mock
#   the LLM call for this rather than hitting a real model.
