# Tests for the Intake Agent (src/agents/intake.py). See PLAN.md, Phase 9.
#
from datetime import datetime, timezone

from src.agents.intake import extract_claim
from src.schemas import ClaimSubmission

mock_claim_submissions = [
    # 1. GOOD — complete auto accident claim
    ClaimSubmission(
        submission_id="CLM-2026-0001",
        raw_text=(
            "I was involved in a car accident on August 8, 2026, around 6:30 PM "
            "on Ring Road in Cairo. Another vehicle hit the rear of my car while "
            "I was stopped at a traffic light. The rear bumper and trunk were damaged. "
            "No one was injured. The other driver's insurance information was exchanged "
            "at the scene. I am requesting coverage for the vehicle repairs."
        ),
        policy_number="POL-AUTO-102938",
        submitted_at=datetime(2026, 8, 8, 18, 45, tzinfo=timezone.utc),
        attachment_paths=[
            "/claims/CLM-2026-0001/accident_photos.jpg",
            "/claims/CLM-2026-0001/police_report.pdf",
        ],
    ),

    # 2. DEFECTIVE — missing policy number and incomplete narrative
    ClaimSubmission(
        submission_id="CLM-2026-0002",
        raw_text=(
            "My car was damaged yesterday. I found it like this in the morning. "
            "I don't know who did it. Please process my claim."
        ),
        policy_number=None,
        submitted_at=datetime(2026, 8, 9, 9, 15, tzinfo=timezone.utc),
        attachment_paths=[],
    ),

    # 3. GOOD — property/water damage claim
    ClaimSubmission(
        submission_id="CLM-2026-0003",
        raw_text=(
            "On August 5, 2026, a water pipe burst in my apartment at approximately "
            "2:00 AM. Water leaked into the living room and bedroom and damaged the "
            "wooden flooring, sofa, and several electrical appliances. A plumber was "
            "called immediately and stopped the leak. Photos of the damaged property "
            "and the plumber's invoice are attached. No injuries occurred."
        ),
        policy_number="POL-HOME-554321",
        submitted_at=datetime(2026, 8, 5, 8, 20, tzinfo=timezone.utc),
        attachment_paths=[
            "/claims/CLM-2026-0003/damage_photos.zip",
            "/claims/CLM-2026-0003/plumber_invoice.pdf",
        ],
    ),

    # 4. DEFECTIVE — vague narrative, potentially missing critical information
    ClaimSubmission(
        submission_id="CLM-2026-0004",
        raw_text=(
            "I want to make a claim for something that happened recently. "
            "There was some damage to my property and I need compensation. "
            "Please check my account and let me know what I should do."
        ),
        policy_number="POL-HOME-778899",
        submitted_at=datetime(2026, 8, 10, 11, 5, tzinfo=timezone.utc),
        attachment_paths=[
            "/claims/CLM-2026-0004/document.pdf",
        ],
    ),

    # 5. DEFECTIVE — contradictory/incomplete information
    ClaimSubmission(
        submission_id="CLM-2026-0005",
        raw_text=(
            "I had an accident on August 7, 2026. It happened somewhere near Cairo, "
            "but I cannot remember the exact location. My car was damaged on the "
            "front and maybe the rear as well. I think another vehicle was involved, "
            "but I am not completely sure. I need the insurance company to handle it."
        ),
        policy_number="POL-AUTO-445566",
        submitted_at=datetime(2026, 8, 10, 14, 30, tzinfo=timezone.utc),
        attachment_paths=[
            "/claims/CLM-2026-0005/photos.zip",
        ],
    ),
]
# TODO:
# - extract_claim() on a clean, unambiguous raw_text returns an
#   ExtractedClaim with high extraction_confidence and empty missing_fields.

# - extract_claim() on a raw_text missing an obvious required field (e.g.
#   no incident date) reports it in missing_fields with lower confidence,
#   rather than guessing.


# - extract_claim() retries and eventually succeeds (or fails predictably)
#   when the LLM's first response fails ExtractedClaim validation — mock
#   the LLM call for this rather than hitting a real model.

