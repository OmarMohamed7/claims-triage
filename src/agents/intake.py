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

from pydantic import ValidationError

from src.llm import get_llm
from src.schemas import ClaimSubmission, ExtractedClaim


MAX_EXTRACTION_RETRIES = 2


def extract_claim(submission: ClaimSubmission) -> ExtractedClaim:
    prompt = build_extraction_prompt(submission.raw_text)
    llm = get_llm()

    last_error: ValidationError | None = None
    for _ in range(MAX_EXTRACTION_RETRIES + 1):
        if last_error is not None:
            prompt = f"{prompt}\n\nThe previous response was invalid:\n{last_error}\n\nReturn corrected JSON only."

        response_text = llm.generate(prompt=prompt)
        try:
            return ExtractedClaim.model_validate_json(response_text)
        except ValidationError as e:
            last_error = e

    assert last_error is not None
    raise last_error


def build_extraction_prompt(raw_text: str) -> str:
    return f"""
You are a claim extraction system.

Extract every field required by the ExtractedClaim schema from the
provided raw text.

Rules:
1. Extract information only from the provided text.
2. Do not invent or assume values.
3. Every ExtractedClaim field must be considered.
4. If a field cannot be determined reliably, use the appropriate
   null/empty representation allowed by the schema.
5. Report every field that could not be confidently populated in
   `missing_fields`.
6. Report `extraction_confidence` as a number between 0 and 1.
7. The confidence should represent your confidence in the overall
   extraction, not whether the text itself is trustworthy.
8. Return ONLY valid JSON.
9. The JSON must contain exactly the fields required by ExtractedClaim.

Expected JSON structure:

{{
    ...ExtractedClaim fields...,
    "extraction_confidence": 0.0,
    "missing_fields": []
}}

Raw text:

{raw_text}
"""
