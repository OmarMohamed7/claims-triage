# Pipeline orchestration — wires the five agents into a LangGraph StateGraph
# over PipelineState.
#
# Depends on Phases 2-6 being implemented (src/agents/*.py).
# See PLAN.md, Phase 7.
#
# TODO: build_pipeline()
# - Node order: intake -> [conditional: extraction_confidence too low ->
#   jump straight to human_escalation] -> fraud_risk -> policy_retrieval ->
#   adjudication -> [conditional: decision.status in (ESCALATED,
#   PENDING_HUMAN_REVIEW) -> human_escalation] -> END. Sequential, not
#   parallel: fraud_risk is a fast local sklearn call, policy_retrieval
#   makes live embedding + Qdrant + BM25 calls, and neither needs the
#   other's output (adjudication needs both regardless) -- so fraud_risk
#   runs first just to surface the cheap signal sooner, not for
#   correctness. See PLAN.md's Pipeline Overview diagram.
# - Each node should catch its agent's exceptions and append to
#   PipelineState.errors rather than crashing the whole graph run, then
#   route to human_escalation as a safe fallback.
# - Return the compiled graph so the caller can do
#   build_pipeline().invoke(PipelineState(submission=...)).

from langgraph.graph import StateGraph
from src.schemas import PipelineState

from src.agents.intake import extract_claim
from src.agents.policy_retrieval import retrieve_policy
from src.agents.fraud_risk import score_fraud_risk
from src.agents.adjudication import adjudicate
from src.agents.human_escalation import build_escalation_packet

def build_pipeline() :
    
    graph: StateGraph[PipelineState] = StateGraph(state_schema=PipelineState)
    
    graph.add_node("intake", extract_claim) # type: ignore
    graph.add_node("extraction", retrieve_policy) # type: ignore
    graph.add_node("fraud_risk", score_fraud_risk) # type: ignore
    graph.add_node("policy_retrieval", retrieve_policy) # type: ignore
    graph.add_node("adjudication", adjudicate) # type: ignore
    graph.add_node("human_escalation", build_escalation_packet) # type: ignore
    
    graph.add_edge("intake", "extraction")
    graph.add_conditional_edges("extraction", route_after_extraction)
    graph.add_edge("fraud_risk", "policy_retrieval")
    graph.add_edge("policy_retrieval", "adjudication")
    graph.add_edge("adjudication", "human_escalation", condition=lambda state: state.decision is not None and state.decision.status in ("ESCALATED", "PENDING_HUMAN_REVIEW")) # type: ignore
    
    print(graph)
    
    return graph


def route_after_extraction(state: PipelineState) -> str:
    """Routes to fraud_risk or human_escalation based on extraction confidence."""
    if state.extracted_claim is None:
        raise ValueError("extracted_claim is None; cannot route.")
    if state.extracted_claim.extraction_confidence < 0.8:
        return "human_escalation"
    return "fraud_risk"