"""Vector storage abstractions.""" 

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Mapping, Sequence

from ..config import ProjectConfig
from ..ingest.base import DocumentChunk, SourceDocument
from ..ingest.pipeline import IngestionStats


@dataclass
class VectorRecord:
    """Result row returned by vector search."""

    chunk_id: str
    source_id: str
    content: str
    metadata: Mapping[str, str]
    tags: Sequence[str]
    score: float


class VectorStore(ABC):
    """Interface for storing embeddings and metadata."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    @abstractmethod
    def initialize(self) -> None:
        """Create required tables/indexes."""

    @abstractmethod
    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Insert or update chunk rows with embeddings."""

    @abstractmethod
    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        score_threshold: float | None = None,
    ) -> List[VectorRecord]:
        """Return the top matching chunks for the query vector."""

    def fetch_existing_checksums(self) -> Mapping[str, str]:
        """Return mapping of chunk_id -> checksum for duplicate detection."""
        raise NotImplementedError

    def log_ingestion_run(
        self,
        run_id: str,
        totals: IngestionStats,
        documents: Sequence[tuple[SourceDocument, IngestionStats]],
        *,
        recorded_at: datetime,
    ) -> None:
        """Persist ingestion run statistics. Optional for stores."""
        raise NotImplementedError
