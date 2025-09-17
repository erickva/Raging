"""Embedding client abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class EmbeddingResult:
    """Wrapper around embedding vectors for a batch request."""

    vectors: List[List[float]]
    model: str


class EmbeddingClient(ABC):
    """Interface for embedding backends."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str], model: str | None = None) -> EmbeddingResult:
        """Return embeddings for a batch of document chunks."""

    @abstractmethod
    def embed_query(self, text: str, model: str | None = None) -> List[float]:
        """Return a single embedding suitable for similarity search."""


class NullEmbeddingClient(EmbeddingClient):
    """Fallback client that produces zero vectors (useful for tests)."""

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str], model: str | None = None) -> EmbeddingResult:
        zeros = [[0.0] * self.dimension for _ in texts]
        return EmbeddingResult(vectors=zeros, model=model or "null")

    def embed_query(self, text: str, model: str | None = None) -> List[float]:
        return [0.0] * self.dimension
