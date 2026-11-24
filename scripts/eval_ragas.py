import argparse
import asyncio
import json
import statistics
import time
from datetime import date
from pathlib import Path

from openai import AsyncOpenAI
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from src.agents.adjudication import adjudicate_with_llm
from src.agents.policy_retrieval import load_index, retrieve_policy
from src.config import config
from src.embeddings import embed_documents
from src.schemas import ExtractedClaim, FraudRiskResult

CASES_PATH = Path("data/eval/retrieval_cases.json")
RESULTS_PATH = Path("data/eval/ragas_last_run.json")

OPENAI_COMPATIBLE = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", lambda: config.models.gemini_api_key),
    "groq": ("https://api.groq.com/openai/v1", lambda: config.models.groq_api_key),
    "ollama": (f"{config.models.ollama_base_url}/v1", lambda: "ollama"),
}

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class LocalEmbeddings(BaseRagasEmbedding):
    def embed_text(self, text: str, **kwargs) -> list[float]:
        return embed_documents([text])[0]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_text(text)


def throttle(client: AsyncOpenAI, requests_per_minute: int) -> None:
    interval = 60 / requests_per_minute
    lock = asyncio.Lock()
    state = {"next": 0.0}
    original = client.chat.completions.create

    async def paced(*args, **kwargs):
        async with lock:
            wait = state["next"] - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            state["next"] = time.monotonic() + interval
        return await original(*args, **kwargs)

    client.chat.completions.create = paced


def build_judge(requests_per_minute: int):
    base_url, get_key = OPENAI_COMPATIBLE[config.models.llm_provider]
    client = AsyncOpenAI(base_url=base_url, api_key=get_key(), max_retries=6)
    throttle(client, requests_per_minute)
    return llm_factory(config.models.llm_model, provider="openai", client=client)


def build_sample(case: dict, docstore: dict) -> dict:
    claim = ExtractedClaim(
        submission_id=case["id"],
        policy_number="POL-EVAL",
        claim_type=case["claim_type"],
        incident_description=case["description"],
        incident_date=date(2026, 8, 15),
        reported_date=date(2026, 8, 16),
        incident_location="Springfield",
        claimed_amount=3000.0,
        extraction_confidence=1.0,
    )
    fraud = FraudRiskResult(
        submission_id=case["id"], risk_score=0.1, risk_tier="low", model_version="eval"
    )
    policy = retrieve_policy(claim)
    decision = adjudicate_with_llm(claim, policy, fraud)
    reference = "\n".join(
        r["text"]
        for r in docstore.values()
        if (r["document_id"], r["section"])
        == (case["expected_document"], case["expected_section"])
    )
    return {
        "user_input": (
            f"Claim type: {case['claim_type']}. {case['description']} "
            "Is this claim covered under the policy, and what should the decision be?"
        ),
        "response": decision.reasoning,
        "retrieved_contexts": [c.text_excerpt for c in policy.relevant_clauses],
        "grounding": [
            *(c.text_excerpt for c in policy.relevant_clauses),
            claim.model_dump_json(),
            fraud.model_dump_json(),
        ],
        "reference": reference,
    }


async def score(sample: dict, metrics: dict) -> dict:
    results = await asyncio.gather(
        metrics["faithfulness"].ascore(
            user_input=sample["user_input"],
            response=sample["response"],
            retrieved_contexts=sample["grounding"],
        ),
        metrics["answer_relevancy"].ascore(
            user_input=sample["user_input"], response=sample["response"]
        ),
        metrics["context_precision"].ascore(
            user_input=sample["user_input"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
        ),
        metrics["context_recall"].ascore(
            user_input=sample["user_input"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
        ),
    )
    return {name: float(result.value) for name, result in zip(METRICS, results)}


async def run(cases: list[dict], requests_per_minute: int) -> list[dict]:
    judge = build_judge(requests_per_minute)
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=LocalEmbeddings()),
        "context_precision": ContextPrecision(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }
    _, _, docstore, _, _ = load_index()
    rows = []
    for case in cases:
        try:
            sample = build_sample(case, docstore)
            scores = await score(sample, metrics)
        except Exception as e:
            print(f"- {case['id']}: failed ({type(e).__name__}: {e})")
            continue
        rows.append({"id": case["id"], **scores, "response": sample["response"]})
        print(f"- {case['id']}: " + "  ".join(f"{m}={scores[m]:.2f}" for m in METRICS))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rpm", type=int, default=12)
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text())[: args.limit]
    print(f"judge: {config.models.llm_provider}/{config.models.llm_model}, cases: {len(cases)}")
    rows = asyncio.run(run(cases, args.rpm))
    if not rows:
        raise SystemExit("no cases scored")
    print(f"\nscored {len(rows)}/{len(cases)} cases")
    for metric in METRICS:
        print(f"{metric:20s} {statistics.mean(r[metric] for r in rows):.3f}")
    RESULTS_PATH.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
