# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code style

Always keep changes as simple as possible and straight to the point. Do not add comments.

## Setup

```bash
uv sync
cp .env.example .env
ollama pull llama3.2:3b   # LLM_MODEL default
ollama pull all-minilm    # EMBEDDING_MODEL default
```

The LLM provider is local Ollama (`src/llm.py`), not a hosted API — no API key
needed. `ollama serve` must be running for anything that touches the Intake or
Adjudication agents (LLM calls) or Policy Retrieval (embeddings).

Two local artifacts must be built before the pipeline can run end to end:

```bash
uv run python -m scripts.train_fraud_model     # models/fraud_classifier.joblib
uv run python -m scripts.build_policy_index    # data/qdrant/ (Qdrant + BM25 + docstore)
```

These import from `src.*` and (for `build_policy_index`) `scripts.chunk.*`, so they
must run as `-m` modules from the project root — running the file directly
(`python scripts/train_fraud_model.py`, or a debugger config with `program`
set to the file path) fails with `ModuleNotFoundError: No module named 'src'`,
since only `-m` puts the project root on `sys.path`. Same applies to a VS
Code debug config — set `"module": "scripts.train_fraud_model"` instead of
`"program"`, or run it in the integrated terminal.

## Commands

```bash
uv run main.py                                        # run the pipeline on one sample submission
uv run python -m scripts.demo_scenarios               # run the pipeline over multiple scenarios
uv run python -m pytest                               # full test suite
uv run python -m pytest tests/test_adjudication.py    # one test file
uv run python -m pytest tests/test_adjudication.py -k test_name   # one test
```

Use `python -m pytest`, not the bare `pytest` command — tests import `src.*`,
and only `-m` puts the project root on `sys.path` (same reason the scripts
above need `-m`; the bare `pytest` console script doesn't add the cwd).

No lint/format/typecheck command is configured in this repo. `# type: ignore`
/ `# pyright: ignore[...]` comments throughout the codebase target Pylance in
the editor, not a CI-run type checker.

## Architecture

`PLAN.md` has the full phased build plan (what each agent does, in what
order, and why) — check it before assuming a gap is unimplemented rather than
intentional.

**Pipeline shape.** `src/graph.py` wires five agent functions into a fixed
LangGraph `StateGraph` over `PipelineState` (`src/schemas.py`):
`intake → fraud_risk → policy_retrieval → adjudication → (human_escalation) → END`.
Agents are plain functions (`extract_claim`, `score_fraud_risk`,
`retrieve_policy`, `adjudicate`, `build_escalation_packet`), not LangGraph
tool-calling agents — there's no dynamic tool selection anywhere in this
pipeline. Each node function catches its own exceptions into
`state.errors`; the `_after()` / `_after_adjudication()` routing functions
check `state.errors` first and short-circuit to `END` before checking
anything else. `human_escalation` is reached only through adjudication's
status check (`ESCALATED` / `PENDING_HUMAN_REVIEW`) — there is no separate
low-confidence shortcut route.

**Where LLM calls happen (and don't).** Only two of the five agents call an
LLM: `intake.extract_claim` (structured extraction from raw text) and
`adjudication.adjudicate_with_llm` (fallback for cases the rules layer can't
decide). `fraud_risk` is a plain XGBoost classifier (no LLM), and
`policy_retrieval` is hybrid Qdrant + BM25 search with regex-based
dollar-amount parsing (no LLM) — coverage/deductible/limit come from pattern
matching over retrieved clauses, not from an LLM reading them. Both LLM call
sites parse the raw string response into a Pydantic schema and retry on
`ValidationError` (capped retries) rather than trusting the string directly.

**Adjudication is rules-first, LLM-fallback.** `adjudicate()` calls
`apply_rule()` first; it returns a decision for clear-cut cases (auto-approve
or auto-escalate) and `None` for ambiguous ones, which then falls through to
`adjudicate_with_llm()`. All thresholds driving `apply_rule()` — fraud
auto-escalate, fraud auto-approve ceiling, high-value claim cutoff, minimum
extraction/retrieval confidence — live in the `Thresholds` dataclass in
`src/config.py`, not hardcoded in the agent. `fraud_risk._risk_tier()` buckets
`risk_score` into `low` / `medium` / `high` using those same two fraud
thresholds; `medium` cannot auto-approve (only `low` can) and cannot
auto-escalate on fraud alone (only `high` triggers that) — it always falls
through to the LLM.

**Schemas are the contract.** Every agent reads/writes shared Pydantic models
in `src/schemas.py`; `PipelineState` is what's threaded through the graph.
When changing a field, `src/schemas.py` is the one place to look — every
agent and the graph read from there.

**LLM provider abstraction.** Agents call `get_llm().generate(prompt)` from
`src/llm.py` and never import a provider SDK directly. Adding a provider
means subclassing `LLMProvider` and registering it in `_PROVIDER_FACTORIES`;
switching providers is an `.env` change (`LLM_PROVIDER` / `LLM_MODEL`), not a
code change.
