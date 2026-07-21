# Claims Triage

A multi-agent pipeline for triaging insurance claims: extracting structured
data from raw submissions, checking policy coverage, scoring fraud risk, and
routing each claim to an automated decision or a human reviewer.

## Architecture

```
                    ┌─────────────────────┐
                    │   Manager/Planner    │
                    │   (Adjudication)     │
                    └──────────┬───────────┘
                               │ orchestrates
        ┌──────────┬──────────┼──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼
    ┌───────┐  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
    │Intake │  │ Policy │ │ Fraud  │ │Decision│ │  Human   │
    │ Agent │  │Retrieval│ │  Risk  │ │ Logic  │ │Escalation│
    └───────┘  │ (RAG)  │ │(sklearn│ └────────┘ └──────────┘
               └────────┘ │/xgboost)│
                          └────────┘
```

- **Intake Agent** — parses a raw claim submission (form text, email body, PDF
  dump) into a structured `ExtractedClaim`.
- **Policy Retrieval Agent (RAG)** — retrieves relevant policy clauses (FAISS +
  BM25 hybrid search) to determine coverage, deductible, and exclusions.
- **Fraud Risk Agent** — a classical ML classifier (scikit-learn / XGBoost),
  not an LLM call, that scores fraud risk from claim and policy features.
- **Manager/Planner (Adjudication)** — orchestrates the pipeline and combines
  extraction confidence, policy coverage, and fraud risk into a decision:
  approve, deny, or escalate.
- **Human Escalation Agent** — assembles an audit-ready packet for a human
  reviewer when confidence is low, coverage is ambiguous, fraud risk is high,
  or the claim value is large.

All agents communicate through shared Pydantic schemas defined in
[`src/schemas.py`](src/schemas.py), threaded through a single `PipelineState`
object as the pipeline runs.

## Tech stack

- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) /
  LangChain, [Anthropic](https://www.anthropic.com/) (Claude) as the LLM
  provider
- **Policy retrieval**: FAISS, BM25 (`rank-bm25`), `sentence-transformers`
- **Fraud risk**: scikit-learn, XGBoost, pandas, numpy
- **Schemas / config**: Pydantic, `python-dotenv`
- **Package management**: [uv](https://docs.astral.sh/uv/)

## Project structure

```
src/
  config.py    # Central settings: decision thresholds, model/provider config, paths
  schemas.py   # Shared Pydantic data contracts for every agent in the pipeline
tests/
  test_schemas.py
data/
  raw/claims.csv     # Sample claims dataset
  policy_docs/        # Policy documents for the retrieval agent to index
notebooks/             # Exploration notebooks
main.py
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
# then fill in ANTHROPIC_API_KEY in .env
```

### Configuration

Runtime settings live in `src/config.py` and are overridable via environment
variables (see `.env.example`):

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | LLM used by Intake, Policy synthesis, and Adjudication agents |
| `ANTHROPIC_API_KEY` | API key for the LLM provider |
| `EMBEDDING_MODEL` | Embedding model for policy document retrieval |
| `DATA_DIR`, `POLICY_DOCS_DIR`, `FAISS_INDEX_PATH`, `RAW_CLAIMS_DATASET` | Data paths |
| `FRAUD_MODEL_PATH` | Path to the trained fraud classifier |

Decision boundaries (fraud auto-escalate threshold, high-value claim cutoff,
minimum extraction/retrieval confidence, etc.) are defined in the
`Thresholds` dataclass in `src/config.py`.

## Running

```bash
uv run main.py
```

## Testing

```bash
uv run pytest
```

## Status

Early stage. The shared schemas (`src/schemas.py`) and config layer
(`src/config.py`) that every agent will build on are in place; the individual
agents and the LangGraph pipeline that wires them together are not yet
implemented.
