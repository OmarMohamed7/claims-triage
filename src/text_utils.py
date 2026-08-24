import json
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# sklearn's list is unusually aggressive and includes some words that are
# peril-discriminative in this corpus: "fire" is one of the 8 coverage
# types (property_fire.md), and "third" appears only in liability.md's
# "third party" language (verified against the actual corpus).
_PROTECTED = frozenset({"fire", "third"})
GENERIC_STOPWORDS = frozenset(ENGLISH_STOP_WORDS) - _PROTECTED

DOMAIN_STOPWORDS_FILENAME = "domain_stopwords.json"
DOMAIN_STOPWORD_DOC_FREQ_THRESHOLD = 0.6


def derive_domain_stopwords(texts: list[str]) -> frozenset[str]:
    """Words near-universal in this specific corpus (e.g. "coverage",
    appearing in most policy chunks) even though they're real English
    words elsewhere -- a generic stopword list can't know about these.
    """
    doc_freq: dict[str, int] = {}
    for text in texts:
        for token in {t.lower() for t in text.split() if t.lower() not in GENERIC_STOPWORDS}:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    n_docs = len(texts)
    if not n_docs:
        return frozenset()

    return frozenset(
        word for word, freq in doc_freq.items() if freq / n_docs >= DOMAIN_STOPWORD_DOC_FREQ_THRESHOLD
    )


def save_domain_stopwords(index_dir: Path, domain_stopwords: frozenset[str]) -> None:
    with open(index_dir / DOMAIN_STOPWORDS_FILENAME, "w") as f:
        json.dump(sorted(domain_stopwords), f)


def load_domain_stopwords(index_dir: Path) -> frozenset[str]:
    with open(index_dir / DOMAIN_STOPWORDS_FILENAME) as f:
        return frozenset(json.load(f))
