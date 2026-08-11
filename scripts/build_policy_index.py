# Build the Policy Retrieval Agent's search index from data/policy_docs/*.md (Mimic the Database).
#
# Produces the Qdrant + BM25 + docstore artifacts that
# src/agents/policy_retrieval.py loads at query time.
#
# Run with: uv run python scripts/build_policy_index.py
# See PLAN.md, Phase 1.
#
from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from scripts.chunk.chunk import PolicyChunk, RecursiveChunker, StructureAwareChunker
from src.config import settings

POLICIES_PATH = "data/policy_docs/"

SUPPORTED_EXTENSIONS = {".md", ".pdf"}


@dataclass
class Document:
    text: str
    source_ref: str
    source_type: str = "file"
    metadata: dict = field(default_factory=dict)


class Loader(ABC):
    @abstractmethod
    def load(self) -> list[Document]:
        """Load Docs from passed Path"""


class FileLoader(Loader):
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> list[Document]:
        docs = []
        actual_path = Path(self.path)

        if actual_path.is_dir() and actual_path.exists():
            for file in actual_path.iterdir():
                if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    if file.suffix.lower() == ".md":
                        doc = self._load_md_file(file)
                        docs.append(doc)

                    elif file.suffix.lower() == ".pdf":
                        doc = self._load_pdf_file(file)
                        docs.append(doc)

        return docs

    def _load_md_file(self, file: Path) -> Document:
        content = file.read_text(encoding="utf-8")

        return Document(text=content, source_ref=file.name, metadata=file.stat()) # pyright: ignore[reportArgumentType]

    def _load_pdf_file(self, file: Path) -> Document:
        reader = PdfReader(file)

        text = ""
        for page in reader.pages:
            text += page.extract_text()

        doc = Document(
            text=text,
            metadata=reader.metadata, # pyright: ignore[reportArgumentType]
            source_ref=file.name,
        )

        return doc


def chunk_document(document: Document) -> list[PolicyChunk]:
    """Structure-aware first; fall back to recursive chunking for any
    document that doesn't match the expected convention."""
    try:
        return StructureAwareChunker().chunk(document)
    except ValueError as e:
        print(f"[chunking] falling back to recursive for {document.source_ref}: {e}")
        return RecursiveChunker().chunk(document)


def load_and_chunk_docs() -> list[PolicyChunk]:
    docs = FileLoader(path=Path(POLICIES_PATH)).load()
    print(f"Found : {len(docs)} docs")

    chunks: list[PolicyChunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))

    print(f"Produced {len(chunks)} chunks")
    return chunks


def build_index(chunks: list[PolicyChunk]) -> None:
    embedding_model = OllamaEmbeddings(model=settings.models.embedding_model)

    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts)

    tokenized_corpus = [text.split() for text in texts]
    # BM25 (Best Matching 25) is a classic ranking algorithm used by search engines to score and rank documents based on exact keyword matches
    bm25 = BM25Okapi(tokenized_corpus)

    index_dir = Path(settings.paths.qdrant_path)
    index_dir.mkdir(parents=True, exist_ok=True)
    collection_name = settings.paths.qdrant_collection_name

    # Embedded/on-disk mode -- no separate Qdrant server needed.
    # We Can create a seprate server
    client = QdrantClient(path=str(index_dir))
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(embeddings[0]), distance=Distance.COSINE
            ),
        )

        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=i,
                    vector=embeddings[i],
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "section": chunk.section,
                        "text": chunk.text,
                    },
                )
                for i, chunk in enumerate(chunks)
            ],
        )
    finally:
        client.close()

    # rank-bm25 has no native save/load, so pickle the fitted corpus.
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # chunk_id -> {document_id, section, text}, keyed the same way the
    # Qdrant payload is, so a hit from either index resolves to the same
    # record.
    docstore = {
        chunk.chunk_id: {
            "document_id": chunk.document_id,
            "section": chunk.section,
            "text": chunk.text,
        }
        for chunk in chunks
    }
    with open(index_dir / "docstore.json", "w") as f:
        json.dump(docstore, f, indent=2)

    print(
        f"Indexed {len(chunks)} chunks into Qdrant collection "
        f"'{collection_name}' at {index_dir}"
    )


if __name__ == "__main__":
    build_index(load_and_chunk_docs())
