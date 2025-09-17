from pathlib import Path

from raging.config import ConfigError, load_config


def test_load_config_roundtrip(tmp_path: Path) -> None:
    config_path = tmp_path / "raging.yaml"
    config_path.write_text(
        """
embedding:
  provider: openai-proxy
  model: text-embedding-3-small
  base_url: https://proxy.local/v1
storage:
  kind: pgvector
  connection_url: postgresql+psycopg://user:pass@localhost:5432/db
  schema: raging
  collection: default
ingest:
  sources:
    - path: .
retrieval:
  top_k: 4
rerank:
  strategy: none
generation:
  provider: openai-proxy
  base_url: https://proxy.local/v1
  model: saga-chat
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.embedding.model == "text-embedding-3-small"
    assert config.storage.schema == "raging"
    assert config.retrieval.top_k == 4
    assert config.generation is not None


def test_load_config_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    try:
        load_config(missing)
    except ConfigError as exc:
        assert "not found" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ConfigError for missing config")
