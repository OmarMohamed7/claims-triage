import itertools
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from scripts.build_policy_index import Document


@dataclass
class PolicyChunk:
    document_id: str
    section: str
    text: str
    chunk_id: str


class Chunk(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[PolicyChunk]:
        pass


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

DOCUMENT_ID_RE = re.compile(r"^Document ID:\s*(.+)$", re.MULTILINE)
SECTION_HEADING_RE = re.compile(r"^##\s*\d+\.\s*(.+)$", re.MULTILINE)


class StructureAwareChunker(Chunk):
    """Primary strategy: split by the doc's own 'Document ID:' line and
    numbered '## N. Section' headings, so each chunk is exactly one clause.

    Raises ValueError if the document doesn't have that structure, so the
    caller can fall back to RecursiveChunker instead.
    """

    def chunk(self, document: Document) -> list[PolicyChunk]:
        text = document.text

        doc_id_match = DOCUMENT_ID_RE.search(text)
        headings = list(SECTION_HEADING_RE.finditer(text))

        if not doc_id_match or not headings:
            raise ValueError(
                f"{document.source_ref} has no 'Document ID:' line and/or "
                "no '## N. Section' headings -- doesn't match the expected "
                "structure"
            )

        document_id = doc_id_match.group(1).strip()

        chunks = []
        for i, heading in enumerate(headings):
            section = heading.group(1).strip()
            start = heading.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            section_text = text[start:end].strip()

            chunks.append(
                PolicyChunk(
                    document_id=document_id,
                    section=section,
                    text=section_text,
                    chunk_id=f"{document_id}::{section}",
                )
            )

        return chunks


class RecursiveChunker(Chunk):
    """Fallback strategy for documents that don't follow the structured
    convention (e.g. a PDF, or a doc missing 'Document ID:'/'## N.'
    headings). Recursively splits on the largest available separator
    (paragraph, then line, then sentence, then word) that keeps each
    chunk under chunk_size, falling back to a hard character cut if a
    single "word" is still too long.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.sperators = ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, document: Document) -> list[PolicyChunk]:
        # No trustworthy "Document ID:" line, so fall back to the filename.
        document_id = document.source_ref
        pieces = self._split(document.text, self.sperators)

        return [
            PolicyChunk(
                document_id=document_id,
                section=f"chunk_{i}",
                text=piece,
                chunk_id=f"{document_id}::chunk_{i}",
            )
            for i, piece in enumerate(pieces)
        ]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        sep, *rest = separators
        parts = text.split(sep) if sep else list(text)

        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)

            if len(part) > self.chunk_size:
                if rest:
                    chunks.extend(self._split(part, rest))
                else:
                    # Last resort: no separators left, hard-cut by size.
                    chunks.extend(
                        part[j : j + self.chunk_size]
                        for j in range(0, len(part), self.chunk_size)
                    )
                current = ""
            else:
                current = part

        if current:
            chunks.append(current)

        if self.chunk_overlap and len(chunks) > 1:
            overlapped = [chunks[0]]
            for prev, curr in itertools.pairwise(chunks):
                overlapped.append(prev[-self.chunk_overlap :] + curr)
            return overlapped

        return [c.strip() for c in chunks if c.strip()]
