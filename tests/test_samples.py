import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "frontend"))

from samples import SAMPLES


def test_samples_are_well_formed():
    names = [s["name"] for s in SAMPLES]
    assert len(names) == len(set(names))
    for sample in SAMPLES:
        assert set(sample) == {"name", "policy", "expect", "text"}
        assert sample["text"].strip()
        assert sample["expect"].strip()


def test_samples_include_non_claim_inputs():
    assert sum(s["name"].startswith("Not a claim") for s in SAMPLES) >= 8
