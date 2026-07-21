# Tests for the Human Escalation Agent (src/agents/human_escalation.py).
# See PLAN.md, Phase 9.
#
# TODO:
# - build_escalation_packet() carries every field from the extracted
#   claim, policy result, fraud result, and decision into the
#   EscalationPacket — assert nothing upstream is dropped.
# - planner_reasoning in the packet matches decision.reasoning exactly.
