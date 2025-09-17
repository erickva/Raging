"""Configuration models and loader for Raging projects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding client."""

    provider: Literal["openai-proxy", "custom"] = "openai-proxy"
    model: str = Field(..., description="Embedding model name exposed by the proxy")
    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the embedding service (e.g. https://proxy/v1)",
    )
    api_key_env: Optional[str] = Field(
        default=None, description="Name of the environment variable that stores the API key"
    )
    api_key: Optional[str] = Field(
        default=None, description="Inline API key value (useful for multi-tenant injection)"
    )
    dimension: Optional[int] = Field(
        default=None, description="Embedding dimension; defaults to provider specific value"
    )
    batch_size: int = Field(default=32, ge=1, description="Number of texts per embed batch")


class StorageConfig(BaseModel):
    """Configuration for persistence of chunk metadata and vectors."""

    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["pgvector"] = "pgvector"
    connection_url: str = Field(
        ..., description="SQLAlchemy connection string for PostgreSQL"
    )
    db_schema: str = Field(
        default="raging", alias="schema", description="Database schema used for tables"
    )
    collection: str = Field(
        default="default",
        description="Logical name for grouping documents inside the same database",
    )
    embedding_dimension: Optional[int] = Field(
        default=None, description="Vector dimension; overrides embedding config when provided"
    )
    echo_sql: bool = Field(default=False, description="Enable SQL echo for debugging")

    @property
    def schema(self) -> str:
        return self.db_schema


class SourceConfig(BaseModel):
    """Configuration describing a single ingestion source."""

    path: Optional[Path] = Field(
        default=None, description="Base path for the source; optional when specifying explicit files"
    )
    handler: Optional[str] = Field(
        default=None,
        description="Explicit handler name; falls back to registry pattern matching when omitted",
    )
    include: List[str] = Field(
        default_factory=lambda: ["**/*"],
        description="Filename glob patterns to include",
    )
    exclude: List[str] = Field(
        default_factory=list,
        description="Filename glob patterns to exclude",
    )
    recursive: bool = Field(default=True, description="Recurse into subdirectories when scanning")
    tags: List[str] = Field(default_factory=list, description="Extra metadata tags for chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Static metadata to attach")
    files: List[Path] = Field(
        default_factory=list,
        description="Explicit list of files to ingest (bypasses globbing when provided)",
    )

    @field_validator("path")
    @classmethod
    def _expand_path(cls, value: Path | None) -> Path | None:
        return value.expanduser().resolve() if value else None

    @field_validator("files")
    @classmethod
    def _expand_files(cls, values: List[Path]) -> List[Path]:
        return [value.expanduser().resolve() for value in values]


class IngestConfig(BaseModel):
    """Configuration for running ingestion."""

    sources: List[SourceConfig] = Field(default_factory=list)
    checksum_algorithm: Literal["sha1", "sha256", "md5"] = "sha256"
    default_handler: Optional[str] = Field(
        default=None, description="Fallback handler name when registry cannot match"
    )


class RetrievalConfig(BaseModel):
    """Configuration for querying chunks."""

    top_k: int = Field(default=8, ge=1, description="Number of chunks to return per query")
    score_threshold: Optional[float] = Field(
        default=None,
        description="Optional minimum similarity score for results (0..1 depending on metric)",
    )


class RerankConfig(BaseModel):
    """Optional reranking stage after vector similarity search."""

    strategy: Literal["none", "bm25", "llm", "custom"] = "none"
    top_n: int = Field(
        default=20, ge=1, description="Number of initial hits to pass into the reranker"
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)


class GenerationConfig(BaseModel):
    """Configuration for answer synthesis with an LLM."""

    provider: Literal["openai-proxy"] = "openai-proxy"
    base_url: str = Field(..., description="Base URL for chat/completions endpoint")
    model: str = Field(..., description="LLM model name to generate answers")
    api_key_env: Optional[str] = Field(
        default=None, description="Environment variable with API key"
    )
    api_key: Optional[str] = Field(default=None, description="Inline API key value")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=512, ge=1)
    timeout: int = Field(default=30, description="HTTP timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Retries on rate limit")


class ProjectConfig(BaseModel):
    """Top-level configuration wrapper."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage: StorageConfig
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    generation: Optional[GenerationConfig] = Field(default=None)

    @property
    def embedding_dimension(self) -> Optional[int]:
        """Return explicit embedding dimension when configured."""

        return self.storage.embedding_dimension or self.embedding.dimension


class ConfigError(RuntimeError):
    """Raised when configuration parsing fails."""


def load_config(path: str | Path) -> ProjectConfig:
    """Parse a YAML config file into a :class:`ProjectConfig` instance."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    try:
        return ProjectConfig.model_validate(raw)
    except ValidationError as exc:  # pragma: no cover - informative error path
        raise ConfigError(str(exc)) from exc


def default_config_path() -> Path:
    """Return the default config file path (`raging.yaml`)."""

    return Path("raging.yaml").resolve()


def resolve_config_path(path: Optional[str | Path]) -> Path:
    """Resolve the config path, falling back to :func:`default_config_path`."""

    if path:
        return Path(path).expanduser().resolve()
    return default_config_path()
