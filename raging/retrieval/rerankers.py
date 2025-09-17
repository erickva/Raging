"""Built-in reranking strategies."""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from typing import List

import requests

from .base import Reranker
from ..storage.base import VectorRecord


class KeywordOverlapReranker(Reranker):
    """Re-rank results using a simple keyword overlap heuristic."""

    def rerank(self, query: str, results: List[VectorRecord]) -> List[VectorRecord]:
        tokens = _tokenize(query)
        query_counter = Counter(tokens)
        scored = []
        for record in results:
            doc_tokens = _tokenize(record.content)
            doc_counter = Counter(doc_tokens)
            overlap = sum(min(query_counter[token], doc_counter[token]) for token in query_counter)
            score = -record.score  # vector distance (lower is better)
            scored.append((overlap, score, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored]


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in text.split()] if text else []


class BM25Reranker(Reranker):
    """Re-rank using BM25 scoring over chunk content."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def rerank(self, query: str, results: List[VectorRecord]) -> List[VectorRecord]:
        if not results:
            return results

        query_tokens = _tokenize(query)
        if not query_tokens:
            return results

        doc_tokens = [_tokenize(record.content) for record in results]
        doc_lengths = [len(tokens) for tokens in doc_tokens]
        avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0 or 1

        df: Counter[str] = Counter()
        for tokens in doc_tokens:
            for token in set(tokens):
                df[token] += 1

        N = len(results)
        scores = []
        for record, tokens, dl in zip(results, doc_tokens, doc_lengths):
            counter = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                tf = counter.get(token)
                if not tf:
                    continue
                doc_freq = df.get(token, 0)
                idf = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                if denom == 0:
                    continue
                score += idf * (tf * (self.k1 + 1) / denom)
            scores.append((score, -record.score, record))

        scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scores]


class LLMReranker(Reranker):
    """Use an OpenAI-compatible LLM endpoint to score chunk relevance."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        headers = {"Content-Type": "application/json"}
        token = api_key
        if not token and api_key_env:
            token = os.getenv(api_key_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session.headers.update(headers)

    def rerank(self, query: str, results: List[VectorRecord]) -> List[VectorRecord]:
        if not results or not query.strip():
            return results

        scored: List[tuple[float, float, VectorRecord]] = []
        for record in results:
            score = self._score_candidate(query, record)
            scored.append((score, -record.score, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored]

    # ------------------------------------------------------------------
    def _score_candidate(self, query: str, record: VectorRecord) -> float:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You rate how relevant a document chunk is to a user query. "
                        "Respond with a single number between 0 and 1."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n"
                        f"Chunk: {record.content}\n"
                        "Relevance score (0-1):"
                    ),
                },
            ],
            "max_tokens": 8,
            "temperature": self.temperature,
        }

        delay = 1.0
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries):
            response = self.session.post(url, json=payload, timeout=self.timeout)
            if response.status_code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 10)
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError:
                time.sleep(delay)
                delay = min(delay * 2, 10)
                continue
            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            return _safe_parse_score(content)
        return 0.0


def _safe_parse_score(value: str) -> float:
    try:
        return max(0.0, min(1.0, float(value.split()[0]))) if value else 0.0
    except ValueError:
        return 0.0


__all__ = ["KeywordOverlapReranker", "BM25Reranker", "LLMReranker"]
