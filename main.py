from src.graph import build_pipeline
from src.schemas import ClaimSubmission, PipelineState


def main():
    submission = ClaimSubmission(
        submission_id="test_submission",
        raw_text=(
            "I was rear-ended at a red light on Main Street on 2026-08-15 "
            "while driving my sedan. The other driver's insurance admitted "
            "fault. Repair estimate came to $4,200. No injuries, no other "
            "parties involved beyond the other driver."
        ),
        policy_number="POL-LIAB-000001",
    )

    pipeline = build_pipeline()
    result = pipeline.invoke(PipelineState(submission=submission)) # type: ignore

    print("errors:", result.get("errors"))
    print()
    print("extracted_claim:", result.get("extracted_claim"))
    print()
    print("fraud_result:", result.get("fraud_result"))
    print()
    print("policy_result:", result.get("policy_result"))
    print()
    print("decision:", result.get("decision"))
    print()
    print("escalation_packet:", result.get("escalation_packet"))


if __name__ == "__main__":
    main()
