import json
import sys
from pathlib import Path

from src.agents.policy_retrieval import load_index, retrieve_policy
from src.schemas import ExtractedClaim

CASES_PATH = Path("data/eval/retrieval_cases.json")


def evaluate(cases: list[dict]) -> dict:
    _, _, docstore, _, _ = load_index()
    rows = []
    for case in cases:
        claim = ExtractedClaim(
            submission_id=case["id"],
            claim_type=case["claim_type"],
            incident_description=case["description"],
            extraction_confidence=1.0,
        )
        result = retrieve_policy(claim)
        clauses = result.relevant_clauses
        target = (case["expected_document"], case["expected_section"])
        total_relevant = sum(
            1 for r in docstore.values() if (r["document_id"], r["section"]) == target
        )
        hits = [(c.document_id, c.section) == target for c in clauses]
        first_hit = hits.index(True) + 1 if True in hits else None
        rows.append(
            {
                "id": case["id"],
                "doc_hit_at_1": bool(clauses) and clauses[0].document_id == target[0],
                "clause_hit_at_1": bool(hits) and hits[0],
                "precision": sum(hits) / len(clauses) if clauses else 0.0,
                "recall": min(sum(hits), total_relevant) / total_relevant,
                "reciprocal_rank": 1 / first_hit if first_hit else 0.0,
                "coverage_correct": result.is_covered == case["expected_is_covered"],
                "returned": [f"{c.document_id}::{c.section}" for c in clauses],
                "expected": f"{target[0]}::{target[1]}",
            }
        )
    n = len(rows)
    return {
        "n": n,
        "doc_hit_at_1": sum(r["doc_hit_at_1"] for r in rows) / n,
        "clause_hit_at_1": sum(r["clause_hit_at_1"] for r in rows) / n,
        "context_precision": sum(r["precision"] for r in rows) / n,
        "context_recall": sum(r["recall"] for r in rows) / n,
        "mrr": sum(r["reciprocal_rank"] for r in rows) / n,
        "coverage_accuracy": sum(r["coverage_correct"] for r in rows) / n,
        "rows": rows,
    }


def main() -> None:
    summary = evaluate(json.loads(CASES_PATH.read_text()))
    print(f"cases: {summary['n']}")
    for name in (
        "doc_hit_at_1",
        "clause_hit_at_1",
        "context_precision",
        "context_recall",
        "mrr",
        "coverage_accuracy",
    ):
        print(f"{name:20s} {summary[name]:.3f}")
    failures = [
        r for r in summary["rows"] if not r["clause_hit_at_1"] or not r["coverage_correct"]
    ]
    if failures:
        print("\nmisses:")
        for r in failures:
            print(f"- {r['id']}: expected {r['expected']}, got {r['returned'][:3]}")
    if "--json" in sys.argv:
        Path("data/eval/last_run.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
