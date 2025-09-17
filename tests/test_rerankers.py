from dataclasses import dataclass
from typing import List

import pytest
import requests

from raging.retrieval.base import NullReranker, Retriever
from raging.retrieval.rerankers import BM25Reranker, LLMReranker
from raging.embeddings.base import EmbeddingClient
from raging.storage.base import VectorRecord, VectorStore
from raging.config import ProjectConfig, EmbeddingConfig, StorageConfig, RetrievalConfig


@dataclass
class FakeStore(VectorStore):
    results: List[VectorRecord]

    def __init__(self, results: List[VectorRecord]) -> None:
        config = ProjectConfig(
            embedding=EmbeddingConfig(model="test", base_url="https://proxy"),
            storage=StorageConfig(
                connection_url="postgresql+psycopg://user:pass@localhost/db",
                schema="raging",
                collection="default",
            ),
            retrieval=RetrievalConfig(top_k=len(results)),
        )
        super().__init__(config)
        self.results = results

    def initialize(self) -> None:  # pragma: no cover - not used in tests
        raise NotImplementedError

    def upsert_chunks(self, chunks, embeddings):  # pragma: no cover - not used
        raise NotImplementedError

    def query(self, vector, top_k, score_threshold=None):
        return self.results[:top_k]

    def fetch_existing_checksums(self):  # pragma: no cover - not used here
        return {}

    def log_ingestion_run(self, run_id, totals, documents, *, recorded_at):  # pragma: no cover - not used
        return None


class FakeEmbeddings(EmbeddingClient):
    def embed_documents(self, texts, model=None):  # pragma: no cover - not used
        raise NotImplementedError

    def embed_query(self, text, model=None):
        return [0.0]


def make_record(content: str, score: float = 0.0) -> VectorRecord:
    return VectorRecord(
        chunk_id=content,
        source_id="src",
        content=content,
        metadata={},
        tags=(),
        score=score,
    )


def test_bm25_reranker_orders_by_term_match():
    reranker = BM25Reranker()
    results = [
        make_record("alpha beta gamma"),
        make_record("alpha alpha"),
        make_record("gamma delta"),
    ]
    ordered = reranker.rerank("alpha beta", results)
    assert ordered[0].content == "alpha beta gamma"
    assert ordered[1].content == "alpha alpha"


def test_retriever_honours_rerank_top_n():
    results = [
        make_record("alpha beta"),
        make_record("alpha"),
        make_record("beta"),
    ]
    store = FakeStore(results)
    embeddings = FakeEmbeddings()
    reranker = BM25Reranker()
    retriever = Retriever(
        store=store,
        embeddings=embeddings,
        reranker=reranker,
        top_k=3,
        score_threshold=None,
        rerank_top_n=2,
    )

    ordered = retriever.query("beta")
    # Only top two should be reranked, tail preserves original order
    assert ordered[0].content in {"alpha beta", "alpha"}
    assert ordered[2].content == "beta"


def test_retriever_with_null_reranker_returns_original_order():
    results = [make_record("alpha"), make_record("beta")]
    store = FakeStore(results)
    embeddings = FakeEmbeddings()
    retriever = Retriever(
        store=store,
        embeddings=embeddings,
        reranker=NullReranker(),
        top_k=2,
        score_threshold=None,
    )
    assert retriever.query("alpha") == results


def test_llm_reranker_scores(monkeypatch):
    results = [
        make_record("alpha beta"),
        make_record("gamma"),
    ]

    scores = ["0.9", "0.2"]

    class FakeResponse:
        status_code = 200

        def __init__(self, value: str) -> None:
            self._value = value

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": self._value}}]}

    def fake_post(self, url, json, timeout):
        return FakeResponse(scores.pop(0))

    monkeypatch.setattr(requests.Session, "post", fake_post)

    reranker = LLMReranker(base_url="https://proxy", model="rerank-model")
    ordered = reranker.rerank("alpha", results)
    assert ordered[0].content == "alpha beta"
    assert ordered[1].content == "gamma"
