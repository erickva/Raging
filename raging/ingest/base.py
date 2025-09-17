"""Core ingestion data structures and interfaces."""

from __future__ import annotations

import hashlib
import itertools
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


CHUNK_ID_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000000")


def compute_checksum(content: str, algorithm: str = "sha256") -> str:
    """Return a hex digest for the given content using the desired algorithm."""

    algo = getattr(hashlib, algorithm, None)
    if algo is None:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    return algo(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DocumentChunk:
    """Normalized representation of a chunk ready for embedding."""

    chunk_id: str
    source_id: str
    content: str
    checksum: str
    chunk_index: int
    metadata: Dict[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SourceDocument:
    """Metadata describing a raw document before chunking."""

    source_id: str
    path: Path
    metadata: Dict[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = field(default_factory=tuple)


class IngestionHandler:
    """Interface that all ingestion handlers must follow."""

    name: str = "base"

    def iter_chunks(
        self,
        document: SourceDocument,
        checksum_algorithm: str,
    ) -> Iterator[DocumentChunk]:
        """Yield document chunks for a given source document."""

        raise NotImplementedError


def iter_chunk_indices() -> Iterator[int]:
    """Convenience generator that yields a sequence of chunk indexes."""

    return itertools.count(start=0, step=1)


def build_chunk_id(source_id: str, chunk_index: int) -> str:
    """Deterministically build a chunk identifier."""

    raw = f"{source_id}:{chunk_index}".encode("utf-8")
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, raw.decode("utf-8")))


def to_source_id(path: Path) -> str:
    """Create a stable identifier for a source document based on its path."""

    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, str(path.resolve())))
