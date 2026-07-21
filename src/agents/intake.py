# Intake Agent — turns a raw ClaimSubmission into a structured ExtractedClaim.
# See PLAN.md, Phase 2.
#
# TODO:
# - Build a prompt instructing the LLM (settings.models.llm_provider /
#   llm_model) to pull out every ExtractedClaim field from raw_text.
# - Ask the model to self-report `extraction_confidence` (0-1) and list any
#   fields it could not confidently fill in `missing_fields`.
# - Parse the model's response into ExtractedClaim; on a ValidationError,
#   retry with the error fed back into the prompt (cap retries, e.g. 2).
# - If extraction still fails after retries, decide how the caller finds out
#   (raise, or return a low-confidence ExtractedClaim with missing_fields
#   populated) — document the choice here once decided.
