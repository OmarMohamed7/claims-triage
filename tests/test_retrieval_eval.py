import json

import pytest

from scripts.eval_retrieval import CASES_PATH, evaluate
from src.agents.policy_retrieval import load_index
from src.schemas import ClaimType


@pytest.fixture(scope="module")
def cases():
    return json.loads(CASES_PATH.read_text())


def test_eval_labels_exist_in_index(cases):
    _, _, docstore, _, _ = load_index()
    known = {(r["document_id"], r["section"]) for r in docstore.values()}
    for case in cases:
        assert (case["expected_document"], case["expected_section"]) in known
        ClaimType(case["claim_type"])


def test_retrieval_quality_floor(cases):
    summary = evaluate(cases)
    assert summary["doc_hit_at_1"] >= 0.7
    assert summary["context_recall"] >= 0.8
