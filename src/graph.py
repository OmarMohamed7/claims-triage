import logging
import time
from functools import wraps
from typing import Callable, TypeVar

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.adjudication import adjudicate
from src.agents.fraud_risk import score_fraud_risk
from src.agents.human_escalation import build_escalation_packet
from src.agents.intake import extract_claim
from src.agents.policy_retrieval import retrieve_policy
from src.config import config
from src.llm import use_llm
from src.schemas import (
    AdjudicationDecision,
    DecisionStatus,
    EscalationReason,
    PipelineState,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def _need(value: T | None, name: str) -> T:
    if value is None:
        raise ValueError(f"missing {name}")
    return value


# Add try / catcch in every node to catch exceptions and log them, then return an error in the state. This way,
# the graph can continue processing other nodes even if one fails.
def _node(name: str):
    def wrap(fn: Callable[[PipelineState], dict]) -> Callable[[PipelineState], dict]:
        @wraps(fn)
        def run(state: PipelineState) -> dict:
            start = time.perf_counter()
            try:
                with use_llm(state.model, state.custom_llm):
                    result = fn(state)
                return {**result, "logs": [f"{name}: ok ({_ms(start)} ms)"]}
            except Exception as e:
                logger.exception(
                    "%s failed for submission %s", name, state.submission.submission_id
                )
                return {
                    "errors": [f"{name}: {e}"],
                    "logs": [f"{name}: failed after {_ms(start)} ms: {e}"],
                }

        return run

    return wrap


@_node("intake")
def _run_intake(state: PipelineState) -> dict:
    return {"extracted_claim": extract_claim(state.submission)}


@_node("fraud_risk")
def _run_fraud_risk(state: PipelineState) -> dict:
    claim = _need(state.extracted_claim, "extracted_claim")
    return {"fraud_result": score_fraud_risk(claim)}


@_node("policy_retrieval")
def _run_policy_retrieval(state: PipelineState) -> dict:
    claim = _need(state.extracted_claim, "extracted_claim")
    return {"policy_result": retrieve_policy(claim)}


def _run_join(state: PipelineState) -> dict:
    return {}


@_node("adjudication")
def _run_adjudication(state: PipelineState) -> dict:
    return {
        "decision": adjudicate(
            _need(state.extracted_claim, "extracted_claim"),
            _need(state.policy_result, "policy_result"),
            _need(state.fraud_result, "fraud_result"),
        )
    }


@_node("early_escalation")
def _run_early_escalation(state: PipelineState) -> dict:
    if state.errors:
        reasoning = "Processing failed: " + "; ".join(state.errors)
        reasons = [EscalationReason.PROCESSING_ERROR]
    else:
        claim = _need(state.extracted_claim, "extracted_claim")
        reasoning = (
            f"Claim extraction confidence "
            f"({claim.extraction_confidence:.2f}) is below the "
            f"minimum required confidence "
            f"({config.thresholds.min_extraction_confidence:.2f})."
        )
        reasons = [EscalationReason.LOW_CONFIDENCE_EXTRACTION]

    decision = AdjudicationDecision(
        submission_id=state.submission.submission_id,
        status=DecisionStatus.ESCALATED,
        reasoning=reasoning,
        escalation_reasons=reasons,
    )
    return {
        "decision": decision,
        "escalation_packet": build_escalation_packet(
            state.submission.submission_id,
            state.extracted_claim,
            state.policy_result,
            state.fraud_result,
            decision,
        ),
    }


@_node("human_escalation")
def _run_human_escalation(state: PipelineState) -> dict:
    return {
        "escalation_packet": build_escalation_packet(
            state.submission.submission_id,
            state.extracted_claim,
            state.policy_result,
            state.fraud_result,
            _need(state.decision, "decision"),
        )
    }


def _after_intake(state: PipelineState) -> str | list[str]:
    claim = state.extracted_claim
    if (
        state.errors
        or claim is None
        or claim.extraction_confidence < config.thresholds.min_extraction_confidence
    ):
        return "early_escalation"
    return ["fraud_risk", "policy_retrieval"]


def _after_join(state: PipelineState) -> str:
    return "early_escalation" if state.errors else "adjudication"


def _after_adjudication(state: PipelineState) -> str:
    if state.errors or state.decision is None:
        return "early_escalation"
    if state.decision.status in (
        DecisionStatus.ESCALATED,
        DecisionStatus.PENDING_HUMAN_REVIEW,
    ):
        return "human_escalation"
    return END


def build_pipeline() -> (
    CompiledStateGraph[PipelineState, None, PipelineState, PipelineState]
):
    graph: StateGraph[PipelineState] = StateGraph(state_schema=PipelineState)

    graph.add_node("intake", _run_intake)
    graph.add_node("fraud_risk", _run_fraud_risk)
    graph.add_node("policy_retrieval", _run_policy_retrieval)
    graph.add_node("join", _run_join)
    graph.add_node("adjudication", _run_adjudication)
    graph.add_node("early_escalation", _run_early_escalation)
    graph.add_node("human_escalation", _run_human_escalation)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake",
        _after_intake,
        ["early_escalation", "fraud_risk", "policy_retrieval"],
    )
    graph.add_edge(["fraud_risk", "policy_retrieval"], "join")
    graph.add_conditional_edges(
        "join", _after_join, ["early_escalation", "adjudication"]
    )
    graph.add_conditional_edges(
        "adjudication",
        _after_adjudication,
        ["early_escalation", "human_escalation", END],
    )
    graph.add_edge("early_escalation", END)
    graph.add_edge("human_escalation", END)

    return graph.compile()
