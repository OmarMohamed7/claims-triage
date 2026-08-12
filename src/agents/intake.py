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

import json

from pydantic import ValidationError

from src.llm import get_llm
from src.schemas import ClaimSubmission, ClaimType, ExtractedClaim

# Field names + a short description of what to put there. Hand-written
# (rather than derived from the schema's types) so the model gets plain
# instructions instead of raw Python type syntax. Keep this in sync with
# ExtractedClaim if fields change.
_FIELD_DESCRIPTIONS = {
    "policy_number": "the policy number referenced in the claim",
    "claimant_name": "full name of the person filing the claim",
    "claim_type": "one of: " + ", ".join(t.value for t in ClaimType),
    "incident_date": "date the incident occurred, as YYYY-MM-DD",
    "claimed_amount": "dollar amount being claimed, a number greater than 0",
    "incident_location": "where the incident occurred (optional)",
    "incident_description": "narrative description of what happened",
    "other_parties_involved": "true or false — were other parties involved",
}
# reported_date isn't in raw_text — it's when the submission arrived, which
# we already know (submission.submitted_at). Not asked of the LLM.
_EXTRACTABLE_FIELDS = list(_FIELD_DESCRIPTIONS)


MAX_EXTRACTION_RETRIES = 2


def extract_claim(submission: ClaimSubmission) -> ExtractedClaim:
    prompt = build_extraction_prompt(submission.raw_text)
    llm = get_llm()

    last_error: str | None = None
    for _ in range(MAX_EXTRACTION_RETRIES + 1):
        if last_error is not None:
            prompt = f"{prompt}\n\nThe previous response was invalid:\n{last_error}\n\nReturn corrected JSON only."

        response_text = llm.generate(prompt=prompt)
        try:
            data = json.loads(response_text)
            data["submission_id"] = submission.submission_id
            data["reported_date"] = submission.submitted_at.date().isoformat()
            if submission.policy_number is not None:
                data["policy_number"] = submission.policy_number
                if isinstance(data.get("missing_fields"), list):
                    data["missing_fields"] = [
                        f for f in data["missing_fields"] if f != "policy_number"
                    ]

            return ExtractedClaim(**data)
        except ValidationError as e:
            last_error = e.json()
        except json.JSONDecodeError as e:
            last_error = e.msg

    assert last_error is not None
    raise Exception(last_error)

def build_extraction_prompt(raw_text: str) -> str:
    field_lines = "\n".join(
        f"- {name}: {desc}" for name, desc in _FIELD_DESCRIPTIONS.items()
    )
    output_format_lines = ",\n    ".join(f'"{name}": ...' for name in _EXTRACTABLE_FIELDS)
    return f"""
You are a claim extraction system.

Your task is to extract exactly the following fields from the provided raw
text. Use these exact field names — do not invent field names that aren't
listed here, and do not omit any of them from your response:

{field_lines}

STRICT OUTPUT REQUIREMENTS:
- Your entire response MUST be a single valid JSON object.
- Start the response with '{{' and end the response with '}}'.
- Do NOT write any text before or after the JSON object.
- Do NOT use Markdown.
- Do NOT use ```json or ``` code fences.
- Do NOT include explanations, introductions, comments, or notes.
- Do NOT include comments anywhere inside the JSON (no // or /* */) — JSON
  does not support comments and they will break parsing.
- Do NOT say "Here is the extracted claim" or anything similar.
- The response must be directly parseable using json.loads().
- Use double quotes for JSON keys and string values.
- Do not invent or assume information.
- Remove any leading or trailing whitespace from string values.
- Remove any new lines.

EXTRACTION RULES:
1. Extract information only from the provided raw text.
2. Every field listed above must be considered.
3. If a field cannot be determined reliably, set it to null. Never use 0,
   an empty string "", or placeholder text to represent an unknown value —
   always use null.
4. Add every field that could not be confidently populated to
   `missing_fields`.
5. `extraction_confidence` must be a number between 0 and 1.
6. `extraction_confidence` represents your confidence in the extraction,
   not whether the claim itself is legitimate.
7. The JSON must contain exactly the fields listed above, plus
   `extraction_confidence` and `missing_fields` — no others.
8. Extract incident_location as a string, e.g.
   "123 Main St, Springfield, IL 62704" or "Springfield, IL".
9. If the text contains conflicting or ambiguous dates, do not guess.
   Set the affected date to null and add it to `missing_fields`.
10. Use ISO date format: YYYY-MM-DD.

MISSING FIELDS DEFINITION:

`missing_fields` must be a JSON array containing ONLY bare field-name
strings from the field list above — e.g. ["incident_date", "claimed_amount"].
Never attach a reason, comment, or explanation to an entry, and never write
anything other than a field name inside that array.

If a field's value cannot be determined, its value in the JSON must be
null AND its name must appear in `missing_fields` — never fill it with a
guess, description, or placeholder text instead.

A field belongs in `missing_fields` when it is:
1. Not marked "(optional)" above, AND
2. Not explicitly present in the raw text, OR
3. Present but too ambiguous to extract reliably.

Do NOT include incident_location merely because it's absent — it's optional.

Do NOT include fields that were successfully extracted.

For example:
- If the text does not state incident_location, which is optional,
  do NOT add `incident_location` to missing_fields.
- If the text does not clearly state incident_date, which is required,
  add `incident_date` to missing_fields.

CONFIDENCE:

Set extraction_confidence based on the completeness and reliability
of the extracted information.

Use approximately:

0.9 - 1.0:
All critical fields are present and unambiguous.

0.7 - 0.89:
Most critical fields are present, with minor missing information.

0.4 - 0.69:
Important information is missing or ambiguous.

0.0 - 0.39:
The submission contains insufficient information to reliably
understand the claim.

OUTPUT FORMAT:

{{
    {output_format_lines},
    "extraction_confidence": 0.0,
    "missing_fields": []
}}

IMPORTANT:
The `...` values above are placeholders — replace each with the extracted
value (or null) for that field. Do not add, rename, or omit any keys.

RAW TEXT:
{raw_text}
"""