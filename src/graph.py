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
