from typing import List

import requests

from raging.config import GenerationConfig
from raging.generation.generator import ResponseGenerator
from raging.storage.base import VectorRecord


class DummyResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if 400 <= self.status_code:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def make_record(content: str) -> VectorRecord:
    return VectorRecord(
        chunk_id="chunk",
        source_id="src",
        content=content,
        metadata={"source_path": "doc.md"},
        tags=(),
        score=0.1,
    )


def test_response_generator_uses_inline_api_key(monkeypatch):
    config = GenerationConfig(
        base_url="https://proxy",
        model="saga-chat",
        api_key="tenant-key",
    )
    generator = ResponseGenerator(config)

    def fake_post(url, json, timeout):
        assert json["messages"][1]["content"].startswith("Context:")
        assert timeout == config.timeout
        return DummyResponse(200, {"choices": [{"message": {"content": "Answer text"}}]})

    monkeypatch.setattr(generator.session, "post", fake_post)
    answer = generator.generate("Question?", [make_record("Some content")])
    assert answer == "Answer text"
    assert generator.session.headers["Authorization"] == "Bearer tenant-key"


def test_response_generator_handles_missing_chunks():
    config = GenerationConfig(
        base_url="https://proxy",
        model="saga-chat",
    )
    generator = ResponseGenerator(config)
    answer = generator.generate("Question?", [])
    assert "No supporting context" in answer
