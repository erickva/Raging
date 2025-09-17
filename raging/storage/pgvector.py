"""pgvector-backed storage implementation."""

from __future__ import annotations

from importlib import import_module
from datetime import datetime
from typing import List, Sequence

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.sql import select

from ..config import ProjectConfig
from ..ingest.base import DocumentChunk
from .base import VectorRecord, VectorStore


def _load_vector_type():
    try:
        module = import_module("pgvector.sqlalchemy")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "pgvector is required for PgVectorStore. Install raging with the pgvector backend enabled."
        ) from exc
    return getattr(module, "Vector")


class PgVectorStore(VectorStore):
    """Persist chunks and embeddings inside PostgreSQL using pgvector."""

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__(config)
        dimension = config.embedding_dimension or 1536
        self.collection_column = config.storage.collection_column
        collection_col = Column(self.collection_column, String, nullable=False)
        self.metadata = MetaData(schema=config.storage.schema)
        self.engine: Engine = create_engine(
            config.storage.connection_url,
            echo=config.storage.echo_sql,
            future=True,
        )
        Vector = _load_vector_type()
        self.chunks = Table(
            "chunks",
            self.metadata,
            Column("chunk_id", String, primary_key=True),
            Column("source_id", String, nullable=False),
            collection_col,
            Column("content", Text, nullable=False),
            Column("checksum", String, nullable=False),
            Column("metadata", JSON, nullable=False, default=dict),
            Column("tags", JSON, nullable=False, default=list),
            Column("embedding", Vector(dimension), nullable=False),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
            Column(
                "updated_at",
                DateTime(timezone=True),
                server_default=func.now(),
                onupdate=func.now(),
            ),
        )
        self.ingestion_runs = Table(
            "ingestion_runs",
            self.metadata,
            Column("run_id", String, primary_key=True),
            Column(self.collection_column, String, nullable=False),
            Column("recorded_at", DateTime(timezone=True), nullable=False),
            Column("processed", Integer, nullable=False),
            Column("emitted", Integer, nullable=False),
            Column("skipped", Integer, nullable=False),
        )
        self.ingestion_documents = Table(
            "ingestion_documents",
            self.metadata,
            Column("run_id", String, nullable=False),
            Column("source_id", String, nullable=False),
            Column("path", Text, nullable=False),
            Column(self.collection_column, String, nullable=True),
            Column("processed", Integer, nullable=False),
            Column("emitted", Integer, nullable=False),
            Column("skipped", Integer, nullable=False),
            PrimaryKeyConstraint("run_id", "source_id"),
        )

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        schema = self.config.storage.schema
        with self.engine.begin() as conn:
            conn.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        self.metadata.create_all(self.engine, checkfirst=True)

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        rows = []
        collection = self.config.storage.collection
        for chunk, vector in zip(chunks, embeddings):
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    self.collection_column: collection,
                    "content": chunk.content,
                    "checksum": chunk.checksum,
                    "metadata": dict(chunk.metadata),
                    "tags": list(chunk.tags),
                    "embedding": list(vector),
                }
            )

        if not rows:
            return

        stmt = insert(self.chunks).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[self.chunks.c.chunk_id],
            set_={
                "content": stmt.excluded.content,
                "checksum": stmt.excluded.checksum,
                "metadata": stmt.excluded.metadata,
                "tags": stmt.excluded.tags,
                "embedding": stmt.excluded.embedding,
                "updated_at": func.now(),
            },
        )

        with self.engine.begin() as conn:
            conn.execute(stmt)

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        score_threshold: float | None = None,
    ) -> List[VectorRecord]:
        embedding_col = self.chunks.c.embedding
        distance = embedding_col.l2_distance(vector).label("distance")

        stmt = (
            select(
                self.chunks.c.chunk_id,
                self.chunks.c.source_id,
                self.chunks.c.content,
                self.chunks.c.metadata,
                self.chunks.c.tags,
                distance,
            )
            .where(self.chunks.c[self.collection_column] == self.config.storage.collection)
            .order_by(distance)
            .limit(top_k)
        )

        results: List[VectorRecord] = []
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        for row in rows:
            score = float(row.distance)
            if score_threshold is not None and score > score_threshold:
                continue
            results.append(
                VectorRecord(
                    chunk_id=row.chunk_id,
                    source_id=row.source_id,
                    content=row.content,
                    metadata=row.metadata or {},
                    tags=row.tags or [],
                    score=score,
                )
            )
        return results

    def fetch_existing_checksums(self) -> dict[str, str]:
        stmt = select(self.chunks.c.chunk_id, self.chunks.c.checksum).where(
            self.chunks.c[self.collection_column] == self.config.storage.collection
        )
        checksums: dict[str, str] = {}
        with self.engine.connect() as conn:
            for chunk_id, checksum in conn.execute(stmt):
                checksums[str(chunk_id)] = str(checksum)
        return checksums

    def log_ingestion_run(
        self,
        run_id: str,
        totals,
        documents,
        *,
        recorded_at: datetime,
    ) -> None:
        run_row = {
            "run_id": run_id,
            self.collection_column: self.config.storage.collection,
            "recorded_at": recorded_at,
            "processed": totals.processed,
            "emitted": totals.emitted,
            "skipped": totals.skipped,
        }
        doc_rows = [
            {
                "run_id": run_id,
                "source_id": document.source_id,
                "path": str(document.path),
                self.collection_column: self.config.storage.collection,
                "processed": stats.processed,
                "emitted": stats.emitted,
                "skipped": stats.skipped,
            }
            for document, stats in documents
        ]

        with self.engine.begin() as conn:
            stmt = insert(self.ingestion_runs).values(run_row)
            stmt = stmt.on_conflict_do_nothing(index_elements=[self.ingestion_runs.c.run_id])
            conn.execute(stmt)
            if doc_rows:
                doc_stmt = insert(self.ingestion_documents).values(doc_rows)
                doc_stmt = doc_stmt.on_conflict_do_update(
                    index_elements=[
                        self.ingestion_documents.c.run_id,
                        self.ingestion_documents.c.source_id,
                    ],
                    set_={
                        "processed": doc_stmt.excluded.processed,
                        "emitted": doc_stmt.excluded.emitted,
                        "skipped": doc_stmt.excluded.skipped,
                        "path": doc_stmt.excluded.path,
                    },
                )
                conn.execute(doc_stmt)
