# Pipeline orchestration — wires the five agents into a LangGraph StateGraph
# over PipelineState.
#
# Depends on Phases 2-6 being implemented (src/agents/*.py).
# See PLAN.md, Phase 7.
#
# TODO: build_pipeline()
# - Node order: intake -> [conditional: extraction_confidence too low ->
#   jump straight to human_escalation] -> policy_retrieval + fraud_risk
#   (independent of each other, could run as parallel branches) ->
#   adjudication -> [conditional: decision.status in (ESCALATED,
#   PENDING_HUMAN_REVIEW) -> human_escalation] -> END.
# - Each node should catch its agent's exceptions and append to
#   PipelineState.errors rather than crashing the whole graph run, then
#   route to human_escalation as a safe fallback.
# - Return the compiled graph so the caller can do
#   build_pipeline().invoke(PipelineState(submission=...)).
