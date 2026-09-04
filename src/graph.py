# Pipeline orchestration — wires the five agents into a LangGraph StateGraph
# over PipelineState.



from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.adjudication import adjudicate
from src.agents.fraud_risk import score_fraud_risk
from src.agents.human_escalation import build_escalation_packet
from src.agents.intake import extract_claim
from src.agents.policy_retrieval import retrieve_policy
from src.schemas import DecisionStatus, PipelineState


def _run_intake(state: PipelineState) -> dict:
    try:
        return {"extracted_claim": extract_claim(state.submission)} # type: ignore
    except Exception as e:
        return {"errors": state.errors + [f"intake: {e}"]} # type: ignore


def _run_fraud_risk(state: PipelineState) -> dict:
    assert state.extracted_claim is not None
    try:
        return {"fraud_result": score_fraud_risk(state.extracted_claim)} # type: ignore
    except Exception as e:
        return {"errors": state.errors + [f"fraud_risk: {e}"]} # type: ignore


def _run_policy_retrieval(state: PipelineState) -> dict:
    assert state.extracted_claim is not None
    try:
        return {"policy_result": retrieve_policy(state.extracted_claim)} # type: ignore
    except Exception as e:
        return {"errors": state.errors + [f"policy_retrieval: {e}"]} # type: ignore


def _run_adjudication(state: PipelineState) -> dict:
    assert state.extracted_claim is not None
    assert state.policy_result is not None
    assert state.fraud_result is not None
    try:
        return {
            "decision": adjudicate(
                state.extracted_claim, state.policy_result, state.fraud_result
            )
        } # type: ignore
    except Exception as e:
        return {"errors": state.errors + [f"adjudication: {e}"]} # type: ignore


def _run_human_escalation(state: PipelineState) -> dict:
    assert state.extracted_claim is not None
    assert state.policy_result is not None
    assert state.fraud_result is not None
    assert state.decision is not None
    return {
        "escalation_packet": build_escalation_packet(
            state.submission.submission_id,
            state.extracted_claim,
            state.policy_result,
            state.fraud_result,
            state.decision,
        )
    } # type: ignore


def _after(next_node: str):
    """Routes to `next_node`, or to END if the node that just ran recorded
    an error onto state.errors."""

    def _route(state: PipelineState) -> str:
        return END if state.errors else next_node

    return _route


def _after_adjudication(state: PipelineState) -> str:
    if state.errors:
        return END
    assert state.decision is not None
    if state.decision.status in (
        DecisionStatus.ESCALATED,
        DecisionStatus.PENDING_HUMAN_REVIEW,
    ):
        return "human_escalation"
    return END


def build_pipeline() -> CompiledStateGraph[PipelineState, None, PipelineState, PipelineState]:
    """Builds and compiles the claims-triage pipeline graph.

    Returns a compiled graph: build_pipeline().invoke(PipelineState(submission=...)).
    """

    graph: StateGraph[PipelineState] = StateGraph(state_schema=PipelineState)

    graph.add_node("intake", _run_intake)   # type: ignore
    graph.add_node("fraud_risk", _run_fraud_risk)   # type: ignore
    graph.add_node("policy_retrieval", _run_policy_retrieval)   # type: ignore
    graph.add_node("adjudication", _run_adjudication)   # type: ignore
    graph.add_node("human_escalation", _run_human_escalation)   # type: ignore

    graph.add_edge(START, "intake")

    graph.add_conditional_edges(
        "intake", _after("fraud_risk"), {"fraud_risk": "fraud_risk", END: END}
    )
    graph.add_conditional_edges(
        "fraud_risk",
        _after("policy_retrieval"),
        {"policy_retrieval": "policy_retrieval", END: END},
    )
    graph.add_conditional_edges(
        "policy_retrieval",
        _after("adjudication"),
        {"adjudication": "adjudication", END: END},
    )
    graph.add_conditional_edges(
        "adjudication",
        _after_adjudication,
        {"human_escalation": "human_escalation", END: END},
    )
    graph.add_edge("human_escalation", END)
    
    return graph.compile() # type: ignore
