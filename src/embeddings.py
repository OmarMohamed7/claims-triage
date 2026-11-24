from functools import lru_cache

from fastembed import TextEmbedding

from src.config import config


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=config.models.embedding_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return [vector.tolist() for vector in _model().embed(texts)]


def embed_query(query: str) -> list[float]:
    return embed_documents([query])[0]
