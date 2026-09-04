# Adjudication Agent (Manager/Planner) — combines Intake, Policy Retrieval,
# and Fraud Risk outputs into a final AdjudicationDecision.
# See PLAN.md, Phase 5.

from pydantic import ValidationError

from src.config import config
from src.llm import LLMProvider, get_llm
from src.schemas import (
    AdjudicationDecision,
    DecisionStatus,
    EscalationReason,
    ExtractedClaim,
    FraudRiskResult,
    LLMAdjudicationResult,
    PolicyRetrievalResult,
)

MAX_ADJUDICATION_RETRIES = 2


def adjudicate(claim: ExtractedClaim, policy_result: PolicyRetrievalResult, fraud_result: FraudRiskResult) -> AdjudicationDecision:
    
    decision: AdjudicationDecision | None = apply_rule(
        claim,
        policy_result,
        fraud_result
    )
    
    if decision is not None:
        return decision
    
    return adjudicate_with_llm(
        claim,
        policy_result,
        fraud_result
    )
    
    
def apply_rule(
    claim: ExtractedClaim, 
    policy_result: PolicyRetrievalResult, 
    fraud_result: FraudRiskResult
) -> AdjudicationDecision | None:
    
    # Define Auto Escalate
    thresholds = config.thresholds
    
    reasons: list[str] = []
    escalation_reasons: list[EscalationReason] = []

    fraud_unassessed = fraud_result.risk_tier == "unknown"

    fraud_escalate = (
        not fraud_unassessed
        and fraud_result.risk_score > thresholds.fraud_auto_escalate
    )

    claim_escalate = (
        claim.claimed_amount > thresholds.high_value_claim_amount
        if claim.claimed_amount is not None
        else False
    )

    claim_extraction_escalate = (
        claim.extraction_confidence
        < thresholds.min_extraction_confidence
    )

    policy_retrieval_escalate = (
        policy_result.retrieval_confidence
        < thresholds.min_retrieval_confidence
    )

    if fraud_unassessed:
        reasons.append(
            "Fraud risk could not be assessed (no policy number to look "
            "up or no feature profile available for it)."
        )
        escalation_reasons.append(EscalationReason.INSUFFICIENT_FRAUD_DATA)

    if fraud_escalate:
        reasons.append(
            f"Fraud risk score ({fraud_result.risk_score:.2f}) "
            f"exceeds the automatic escalation threshold "
            f"({thresholds.fraud_auto_escalate:.2f})."
        )
        escalation_reasons.append(EscalationReason.HIGH_FRAUD_RISK)

    if claim_escalate:
        reasons.append(
            f"Claimed amount ({claim.claimed_amount}) exceeds the "
            f"high-value claim threshold "
            f"({thresholds.high_value_claim_amount})."
        )
        escalation_reasons.append(EscalationReason.HIGH_CLAIM_VALUE)

    if claim_extraction_escalate:
        reasons.append(
            f"Claim extraction confidence "
            f"({claim.extraction_confidence:.2f}) is below the "
            f"minimum required confidence "
            f"({thresholds.min_extraction_confidence:.2f})."
        )
        escalation_reasons.append(EscalationReason.LOW_CONFIDENCE_EXTRACTION)

    if policy_retrieval_escalate:
        reasons.append(
            f"Policy retrieval confidence "
            f"({policy_result.retrieval_confidence:.2f}) is below the "
            f"minimum required confidence "
            f"({thresholds.min_retrieval_confidence:.2f})."
        )
        escalation_reasons.append(EscalationReason.AMBIGUOUS_POLICY_COVERAGE)

    if policy_result.is_covered is False:
        reasons.append(
            "The retrieved policy information indicates that the claim "
            "is not covered."
        )
        escalation_reasons.append(EscalationReason.AMBIGUOUS_POLICY_COVERAGE)

    elif policy_result.is_covered is None:
        reasons.append(
            "Policy coverage could not be determined with sufficient confidence."
        )
        escalation_reasons.append(EscalationReason.AMBIGUOUS_POLICY_COVERAGE)

    if reasons:
        return AdjudicationDecision(
            submission_id= claim.submission_id,
            status= DecisionStatus.ESCALATED,
            reasoning= "".join(reasons),
            # dict.fromkeys dedupes while preserving order -- multiple rules
            # (e.g. low retrieval confidence and is_covered=False/None) can
            # map to the same EscalationReason.
            escalation_reasons=list(dict.fromkeys(escalation_reasons)),
        )
        
    # Auto-approve rules
    fraud_approve = (
        fraud_result.risk_score
        < thresholds.fraud_auto_approve_ceiling
    )

    policy_approve = policy_result.is_covered is True

    extraction_confident = (
        claim.extraction_confidence
        >= thresholds.min_extraction_confidence
    )

    retrieval_confident = (
        policy_result.retrieval_confidence
        >= thresholds.min_retrieval_confidence
    )

    claim_not_high_value = (
        claim.claimed_amount is not None
        and claim.claimed_amount
        < thresholds.high_value_claim_amount
    )

    if (
        fraud_approve
        and policy_approve
        and extraction_confident
        and retrieval_confident
        and claim_not_high_value
    ):
        # Calculate approved amount
        approved_amount: None | float = claim.claimed_amount

        if policy_result.deductible is not None and approved_amount is not None:
            approved_amount = max(
                0,
                approved_amount - policy_result.deductible,
            )

        if policy_result.coverage_limit is not None and approved_amount is not None:
            approved_amount = min(
                approved_amount,
                policy_result.coverage_limit,
            )

        return AdjudicationDecision(
            submission_id=claim.submission_id,
            status=DecisionStatus.APPROVED,
            approved_amount=approved_amount,
            reasoning=(
                "The claim is covered under the retrieved policy. "
                "Fraud risk is below the automatic approval threshold, "
                "claim and policy retrieval confidences meet the required "
                "minimums, and the claimed amount is below the high-value "
                "claim threshold."
            ),
        )
    
    return None
    
def adjudicate_with_llm(
    claim: ExtractedClaim,
    policy_result: PolicyRetrievalResult,
    fraud_result: FraudRiskResult,
) -> AdjudicationDecision:

    prompt = f"""
        You are an insurance claim adjudication assistant.

        Make a decision based ONLY on the provided information.

        Do not invent or assume facts that are not present in the input.

        CLAIM:
        {claim.model_dump_json(indent=2)}

        POLICY RESULT:
        {policy_result.model_dump_json(indent=2)}

        FRAUD RESULT:
        {fraud_result.model_dump_json(indent=2)}

        Decision rules:

        - Use APPROVED only when the available evidence supports coverage.
        - Use ESCALATED when the evidence is ambiguous, conflicting,
        insufficient, or requires human review.
        - Do not invent fraud scores, coverage limits, deductibles,
        policy clauses, or claim amounts.
        - approved_amount must be based only on the provided claimed amount,
        deductible, and coverage limit.
        - Always provide clear human-readable reasoning.
        - When escalating, escalation_reasons must contain only values from
        this exact set (do not invent new ones):
        {", ".join(r.value for r in EscalationReason)}

        Respond with a single valid JSON object, matching exactly this
        shape, and nothing else (no Markdown, no code fences, no
        commentary before or after it):

        {{
            "status": "approved" | "denied" | "escalated" | "pending_human_review",
            "reasoning": "...",
            "escalation_reasons": [],
            "approved_amount": null
        }}
        """

    llm: LLMProvider = get_llm()

    last_error: str | None = None
    for _ in range(MAX_ADJUDICATION_RETRIES + 1):
        attempt_prompt = prompt
        if last_error is not None:
            attempt_prompt = (
                f"{prompt}\n\nThe previous response was invalid:\n{last_error}"
                "\n\nReturn corrected JSON only."
            )

        response_text = llm.generate(attempt_prompt)
        try:
            result = LLMAdjudicationResult.model_validate_json(response_text)
        except ValidationError as e:
            last_error = e.json()
            continue

        return AdjudicationDecision(
            submission_id=claim.submission_id,
            status=result.status,
            reasoning=result.reasoning,
            escalation_reasons=result.escalation_reasons,
            approved_amount=result.approved_amount,
        )

    assert last_error is not None
    raise ValueError(f"LLM adjudication failed after retries: {last_error}")