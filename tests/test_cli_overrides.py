from pathlib import Path

from raging.cli.main import _override_sources_with_files
from raging.config import (
    EmbeddingConfig,
    IngestConfig,
    ProjectConfig,
    RetrievalConfig,
    SourceConfig,
    StorageConfig,
)


def test_override_sources_with_files(tmp_path: Path) -> None:
    base_cfg = ProjectConfig(
        embedding=EmbeddingConfig(model="test", base_url="https://proxy"),
        storage=StorageConfig(
            connection_url="postgresql+psycopg://user:pass@localhost/db",
            schema="raging",
            collection="default",
        ),
        ingest=IngestConfig(
            sources=[SourceConfig(path=tmp_path, include=["*.md"])]
        ),
        retrieval=RetrievalConfig(top_k=3),
    )

    file_path = (tmp_path / "fresh.txt").resolve()
    cfg = _override_sources_with_files(base_cfg, [file_path])
    assert cfg.ingest.sources[0].files == [file_path]
    assert cfg.ingest.sources[0].path is None
    assert cfg.ingest.sources[1:].pop().path == tmp_path
