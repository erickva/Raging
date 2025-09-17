"""LLM-backed response generation for Raging."""

from __future__ import annotations

import os
import time
from typing import Iterable, List, Sequence

import requests

from ..config import GenerationConfig
from ..storage.base import VectorRecord


class ResponseGenerator:
    """Synthesize answers from retrieved chunks using an LLM."""

    def __init__(self, config: GenerationConfig) -> None:
        if config.provider != "openai-proxy":
            raise ValueError(f"Unsupported generation provider: {config.provider}")
        self.config = config
        self.session = requests.Session()
        headers = {"Content-Type": "application/json"}
        api_key = config.api_key
        if not api_key and config.api_key_env:
            api_key = os.getenv(config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers.update(headers)

    # ------------------------------------------------------------------
    def generate(self, question: str, chunks: Sequence[VectorRecord]) -> str:
        if not chunks:
            return "No supporting context was found to answer the question."
        prompt = self._build_prompt(question, chunks)
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions using only the provided context. "
                        "If the context is insufficient, say you do not know. Cite chunk numbers when relevant."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        delay = 1.0
        for attempt in range(max(1, self.config.max_retries + 1)):
            response = self.session.post(url, json=payload, timeout=self.config.timeout)
            if response.status_code == 429 and attempt < self.config.max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 10)
                continue
            response.raise_for_status()
            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            return content or "The model returned no content."
        raise RuntimeError("Generation request failed after retries")

    # ------------------------------------------------------------------
    def _build_prompt(self, question: str, chunks: Sequence[VectorRecord]) -> str:
        sections: List[str] = []
        for idx, record in enumerate(chunks, start=1):
            snippet = record.content.strip()
            metadata = record.metadata or {}
            source = metadata.get("source_path") or metadata.get("source_id") or record.source_id
            sections.append(
                f"Chunk {idx} (source: {source}):\n{snippet}"
            )
        context = "\n\n".join(sections)
        return (
            "Context:\n"
            f"{context}\n\n"
            f"Question: {question}\n"
            "Answer using only the provided context."
        )
