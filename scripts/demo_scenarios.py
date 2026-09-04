# Runs a representative claim submission through the compiled pipeline for
# each distinct adjudication outcome (auto-approve, every apply_rule()
# escalation reason, and the LLM fallback for an ambiguous case), and logs
# a structured summary of each run.
#
# Policy numbers below were chosen by sampling data/raw/claims.csv against
# the trained fraud model to land in a specific risk band (see git history
# for the lookup) -- they'll only stay valid as long as the dataset and
# models/fraud_classifier.joblib aren't retrained.

import logging
import sys
from dataclasses import dataclass

from src.graph import build_pipeline
from src.schemas import ClaimSubmission, DecisionStatus, PipelineState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("claims_triage.demo")

for _noisy in ("httpx", "httpcore", "qdrant_client"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_RULE = "-" * 72


@dataclass
class Scenario:
    name: str
    expected: str
    submission: ClaimSubmission


SCENARIOS = [
    Scenario(
        name="Clean auto-approve",
        expected="APPROVED (low fraud, covered, confident, low value)",
        submission=ClaimSubmission(
            submission_id="DEMO-APPROVE",
            raw_text=(
                "I was rear-ended at a red light on Main Street on "
                "2026-08-15 while driving my sedan. The other driver's "
                "insurance admitted fault. Repair estimate came to $4,200. "
                "No injuries, no other parties involved beyond the other "
                "driver."
            ),
            policy_number="POL-LIAB-000001",
        ),
    ),
    Scenario(
        name="High-value claim",
        expected="ESCALATED: HIGH_CLAIM_VALUE",
        submission=ClaimSubmission(
            submission_id="DEMO-HIGH-VALUE",
            raw_text=(
                "My vehicle was totaled in a collision on Interstate 5 on "
                "2026-07-02. The other driver ran a red light. Total loss "
                "estimate from the body shop is $48,500. No injuries."
            ),
            policy_number="POL-LIAB-000001",
        ),
    ),
    Scenario(
        name="High fraud risk",
        expected="ESCALATED: HIGH_FRAUD_RISK",
        submission=ClaimSubmission(
            submission_id="DEMO-HIGH-FRAUD",
            raw_text=(
                "My car was stolen from outside my apartment on 2026-06-10. "
                "It was recovered a week later with extensive interior "
                "damage. Claimed repair cost is $6,000."
            ),
            policy_number="POL-ALLP-006823",
        ),
    ),
    Scenario(
        name="Unrecognized policy number",
        expected="ESCALATED: INSUFFICIENT_FRAUD_DATA (no history to score)",
        submission=ClaimSubmission(
            submission_id="DEMO-UNKNOWN-POLICY",
            raw_text=(
                "My car was hit while parked outside my house on "
                "2026-08-01. Damage to the rear bumper, repair estimate "
                "$1,800."
            ),
            policy_number="POL-LIAB-999999",
        ),
    ),
    Scenario(
        name="Excluded loss (flood under water-damage policy)",
        expected="ESCALATED: AMBIGUOUS_POLICY_COVERAGE (is_covered=False)",
        submission=ClaimSubmission(
            submission_id="DEMO-EXCLUDED",
            raw_text=(
                "Heavy rain caused a nearby creek to overflow and flood my "
                "basement on 2026-08-20. Water damaged the flooring and "
                "furniture. Estimated loss is $9,000."
            ),
            policy_number="POL-LIAB-000001",
        ),
    ),
    Scenario(
        name="Vague, low-confidence submission",
        expected="ESCALATED: LOW_CONFIDENCE_EXTRACTION",
        submission=ClaimSubmission(
            submission_id="DEMO-VAGUE",
            raw_text="something happened to my car i think, need money for it",
            policy_number="POL-LIAB-000001",
        ),
    ),
    Scenario(
        name="Ambiguous fraud score (mid-band)",
        expected="apply_rule() abstains -> LLM adjudication decides",
        submission=ClaimSubmission(
            submission_id="DEMO-AMBIGUOUS",
            raw_text=(
                "My car was in a collision on 2026-08-05 while parked; "
                "another vehicle backed into it in a parking lot. Repair "
                "estimate is $3,100."
            ),
            policy_number="POL-COLL-007446",
        ),
    ),
]


def _log_result(scenario: Scenario, result: dict) -> str:
    """Logs a structured summary for one scenario run and returns the
    final status label used in the closing table."""

    logger.info(_RULE)
    logger.info("SCENARIO: %s", scenario.name)
    logger.info("Expected: %s", scenario.expected)
    logger.info(_RULE)

    errors = result.get("errors") or []
    if errors:
        for err in errors:
            logger.error("Pipeline error: %s", err)
        logger.info("Terminal state: FAILED (ended at first node error)")
        return "FAILED"

    claim = result.get("extracted_claim")
    if claim is not None:
        logger.info(
            "Intake        | confidence=%.2f  claim_type=%s  amount=%s",
            claim.extraction_confidence,
            claim.claim_type,
            claim.claimed_amount,
        )

    fraud = result.get("fraud_result")
    if fraud is not None:
        logger.info(
            "Fraud Risk    | tier=%-7s score=%.2f  model=%s",
            fraud.risk_tier,
            fraud.risk_score,
            fraud.model_version or "(not run)",
        )

    policy = result.get("policy_result")
    if policy is not None:
        logger.info(
            "Policy        | is_covered=%s  retrieval_confidence=%.2f  "
            "exclusions_matched=%s",
            policy.is_covered,
            policy.retrieval_confidence,
            policy.exclusions_matched or "none",
        )

    decision = result.get("decision")
    if decision is None:
        logger.info("Terminal state: NO DECISION REACHED")
        return "NO DECISION"

    logger.info("Decision      | status=%s", decision.status.value.upper())
    logger.info("Reasoning     | %s", decision.reasoning)
    if decision.escalation_reasons:
        logger.info(
            "Escalation reasons: %s",
            ", ".join(r.value for r in decision.escalation_reasons),
        )
    if decision.approved_amount is not None:
        logger.info("Approved amount: $%.2f", decision.approved_amount)

    packet = result.get("escalation_packet")
    if packet is not None:
        logger.info(
            "Escalation packet built for human review (submission_id=%s)",
            packet.submission_id,
        )

    return decision.status.value.upper()


def main() -> None:
    pipeline = build_pipeline()
    outcomes: list[tuple[str, str]] = []

    for scenario in SCENARIOS:
        result = pipeline.invoke(PipelineState(submission=scenario.submission))
        status = _log_result(scenario, result)
        outcomes.append((scenario.name, status))
        logger.info("")

    logger.info(_RULE)
    logger.info("SUMMARY")
    logger.info(_RULE)
    for name, status in outcomes:
        logger.info("%-45s -> %s", name, status)


if __name__ == "__main__":
    main()
