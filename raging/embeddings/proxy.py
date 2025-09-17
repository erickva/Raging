"""Embedding client that targets an OpenAI compatible proxy."""

from __future__ import annotations

import os
import time
from typing import Iterable, List, Sequence

import requests

from ..config import EmbeddingConfig
from .base import EmbeddingClient, EmbeddingResult


class OpenAIProxyEmbeddingClient(EmbeddingClient):
    """Call an OpenAI-compatible embeddings endpoint exposed via a proxy."""

    def __init__(self, config: EmbeddingConfig) -> None:
        if not config.base_url:
            raise ValueError("Embedding config requires base_url for proxy usage")
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        api_key = config.api_key
        if not api_key and config.api_key_env:
            api_key = os.getenv(config.api_key_env)
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    # ------------------------------------------------------------------
    def embed_documents(self, texts: Sequence[str], model: str | None = None) -> EmbeddingResult:
        chunks: List[List[float]] = []
        model_name = model or self.config.model
        for batch in _chunk_iterable(texts, self.config.batch_size):
            payload = {"input": list(batch), "model": model_name}
            vectors = self._post_embeddings(payload)
            chunks.extend(vectors)
        return EmbeddingResult(vectors=chunks, model=model_name)

    def embed_query(self, text: str, model: str | None = None) -> List[float]:
        payload = {"input": text, "model": model or self.config.model}
        vectors = self._post_embeddings(payload)
        if not vectors:
            raise RuntimeError("Embedding service returned no vectors")
        return vectors[0]

    # ------------------------------------------------------------------
    def _post_embeddings(self, payload: dict) -> List[List[float]]:
        url = f"{self.config.base_url.rstrip('/')}/embeddings"
        backoff = 1.0
        for attempt in range(5):
            response = self.session.post(url, json=payload, timeout=30)
            if response.status_code == 429:
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data.get("data", [])]
        raise RuntimeError("Embedding request failed after retries")


def _chunk_iterable(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    if size <= 0:
        yield items
        return
    for start in range(0, len(items), size):
        yield items[start : start + size]
