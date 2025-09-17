"""Embedding helpers."""

from .base import EmbeddingClient, EmbeddingResult, NullEmbeddingClient
from .proxy import OpenAIProxyEmbeddingClient

__all__ = [
    "EmbeddingClient",
    "EmbeddingResult",
    "NullEmbeddingClient",
    "OpenAIProxyEmbeddingClient",
]
