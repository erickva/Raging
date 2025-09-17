from pathlib import Path

from raging.config import (
    EmbeddingConfig,
    IngestConfig,
    ProjectConfig,
    RetrievalConfig,
    SourceConfig,
    StorageConfig,
)
from raging.ingest.base import DocumentChunk, IngestionHandler, compute_checksum
from raging.ingest.pipeline import IngestionPipeline
from raging.ingest.registry import HandlerRegistry


class DummyHandler(IngestionHandler):
    name = "dummy"

    def iter_chunks(self, document, checksum_algorithm):
        text = document.path.read_text(encoding="utf-8")
        checksum = compute_checksum(text, checksum_algorithm)
        yield DocumentChunk(
            chunk_id="dummy",
            source_id=document.source_id,
            content=text,
            checksum=checksum,
            chunk_index=0,
            metadata={},
            tags=(),
        )


def test_pipeline_uses_custom_registry(tmp_path: Path) -> None:
    data = tmp_path / "docs"
    data.mkdir()
    sample = data / "example.txt"
    sample.write_text("hello world", encoding="utf-8")

    config = ProjectConfig(
        embedding=EmbeddingConfig(model="test", base_url="https://proxy"),
        storage=StorageConfig(
            connection_url="postgresql+psycopg://user:pass@localhost/db",
            schema="raging",
            collection="default",
        ),
        ingest=IngestConfig(
            sources=[
                SourceConfig(
                    path=data,
                    handler="dummy",
                    include=["*.txt"],
                    exclude=[],
                )
            ]
        ),
        retrieval=RetrievalConfig(top_k=5),
    )

    registry = HandlerRegistry()
    registry.register("dummy", DummyHandler, patterns=["*.txt"])

    pipeline = IngestionPipeline(config, registry=registry)
    chunks = list(pipeline.run(force=True))
    assert len(chunks) == 1
    assert chunks[0].content == "hello world"
    assert pipeline.stats.emitted == 1
    assert pipeline.stats.processed == 1
    assert pipeline.stats.skipped == 0


def test_pipeline_skips_existing_checksums(tmp_path: Path) -> None:
    data = tmp_path / "docs"
    data.mkdir()
    sample = data / "example.txt"
    sample.write_text("hello world", encoding="utf-8")

    config = ProjectConfig(
        embedding=EmbeddingConfig(model="test", base_url="https://proxy"),
        storage=StorageConfig(
            connection_url="postgresql+psycopg://user:pass@localhost/db",
            schema="raging",
            collection="default",
        ),
        ingest=IngestConfig(
            sources=[
                SourceConfig(
                    path=data,
                    handler="dummy",
                    include=["*.txt"],
                    exclude=[],
                )
            ]
        ),
        retrieval=RetrievalConfig(top_k=5),
    )

    registry = HandlerRegistry()
    registry.register("dummy", DummyHandler, patterns=["*.txt"])
    pipeline = IngestionPipeline(config, registry=registry)

    initial_chunks = list(pipeline.run(force=True))
    existing = {chunk.chunk_id: chunk.checksum for chunk in initial_chunks}

    skipped_chunks = list(pipeline.run(existing_checksums=existing, force=False))
    assert skipped_chunks == []
    assert pipeline.stats.processed == 1
    assert pipeline.stats.skipped == 1
    assert pipeline.stats.emitted == 0
    doc_stats_list = list(pipeline.iter_document_stats())
    assert len(doc_stats_list) == 1
    _, doc_stats = doc_stats_list[0]
    assert doc_stats.skipped == 1

    forced_chunks = list(pipeline.run(existing_checksums=existing, force=True))
    assert forced_chunks
    assert pipeline.stats.processed == 1
    assert pipeline.stats.skipped == 0
    assert pipeline.stats.emitted == 1
    doc_stats_list = list(pipeline.iter_document_stats())
    assert doc_stats_list[0][1].emitted == 1
