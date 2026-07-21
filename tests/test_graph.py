# End-to-end tests for the compiled pipeline (src/graph.py). See PLAN.md,
# Phase 9. Requires Phases 2-6 to be implemented first.
#
# TODO:
# - build_pipeline().invoke(...) on a sample ClaimSubmission runs through
#   to a final decision without raising, for at least one claim per
#   ClaimType that has a policy doc in data/policy_docs.
# - A submission with unparseable/garbage raw_text routes to
#   human_escalation rather than crashing the graph.
# - Errors raised by an individual agent are recorded in
#   PipelineState.errors and the graph still reaches a terminal state,
#   instead of propagating an unhandled exception.
