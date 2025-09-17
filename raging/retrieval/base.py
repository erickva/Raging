"""Retrieval abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..embeddings.base import EmbeddingClient
from ..storage.base import VectorRecord, VectorStore


class Reranker(ABC):
    """Re-order raw vector results to improve relevance."""

    @abstractmethod
    def rerank(self, query: str, results: List[VectorRecord]) -> List[VectorRecord]:
        ...


class NullReranker(Reranker):
    """Default reranker that returns the original ordering."""

    def rerank(self, query: str, results: List[VectorRecord]) -> List[VectorRecord]:
        return results


class Retriever:
    """Glue together embedding client, vector store, and reranker."""

    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingClient,
        reranker: Reranker,
        *,
        top_k: int,
        score_threshold: float | None = None,
        rerank_top_n: int | None = None,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.reranker = reranker
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.rerank_top_n = rerank_top_n

    def query(self, text: str) -> List[VectorRecord]:
        vector = self.embeddings.embed_query(text)
        results = self.store.query(vector, top_k=self.top_k, score_threshold=self.score_threshold)
        if self.rerank_top_n:
            head = results[: self.rerank_top_n]
            tail = results[self.rerank_top_n :]
            reranked = self.reranker.rerank(text, head)
            return reranked + tail
        return self.reranker.rerank(text, results)
