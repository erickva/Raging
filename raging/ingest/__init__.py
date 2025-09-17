"""Ingestion package exports."""

from .base import DocumentChunk, SourceDocument
from .handlers import FAQHandler, MarkdownHandler, TextHandler
from .pipeline import IngestionPipeline
from .registry import HandlerRegistry, get_registry

__all__ = [
    "DocumentChunk",
    "SourceDocument",
    "FAQHandler",
    "MarkdownHandler",
    "TextHandler",
    "IngestionPipeline",
    "HandlerRegistry",
    "get_registry",
]
