"""Command-line interface for Raging."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from ..config import ProjectConfig, load_config, resolve_config_path
from ..embeddings.base import EmbeddingClient
from ..embeddings.proxy import OpenAIProxyEmbeddingClient
from ..ingest.pipeline import IngestionPipeline
from ..retrieval.base import NullReranker, Retriever
from ..retrieval.rerankers import BM25Reranker, KeywordOverlapReranker, LLMReranker
from ..storage.pgvector import PgVectorStore

app = typer.Typer(help="Raging CLI")


@app.command()
def init_db(config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file")) -> None:
    """Create pgvector tables and indexes."""

    cfg = _load_config(config)
    store = PgVectorStore(cfg)
    store.initialize()
    typer.echo("Database schema initialized.")


@app.command()
def ingest(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-embed all chunks even when checksums match existing records",
    ),
    embedding_api_key: Optional[str] = typer.Option(
        None,
        "--embedding-api-key",
        help="Override embedding provider API key for this run",
    ),
    rerank_api_key: Optional[str] = typer.Option(
        None,
        "--rerank-api-key",
        help="Override reranker LLM API key (llm strategy only)",
    ),
) -> None:
    """Ingest documents configured in raging.yaml."""

    cfg = _load_config(config)
    pipeline = IngestionPipeline(cfg)
    store = PgVectorStore(cfg)
    embeddings = _build_embedding(cfg, api_key=embedding_api_key)

    try:
        existing = store.fetch_existing_checksums()
    except Exception:  # pragma: no cover - store not initialized yet
        existing = {}

    chunks = list(pipeline.run(existing_checksums=existing, force=force))
    if not chunks:
        typer.echo("No documents matched the configured sources.")
        raise typer.Exit(code=0)

    vectors = embeddings.embed_documents([chunk.content for chunk in chunks])
    store.upsert_chunks(chunks, vectors.vectors)

    run_id = uuid.uuid4().hex
    try:
        store.log_ingestion_run(
            run_id,
            pipeline.stats,
            list(pipeline.iter_document_stats()),
            recorded_at=datetime.utcnow(),
        )
    except NotImplementedError:  # pragma: no cover - optional for other stores
        run_id = "n/a"

    stats = pipeline.stats
    typer.echo(
        "Ingest run {run_id}: {emitted} chunks stored (processed={processed}, skipped={skipped}) in '{collection}'.".format(
            run_id=run_id,
            emitted=stats.emitted,
            processed=stats.processed,
            skipped=stats.skipped,
            collection=cfg.storage.collection,
        )
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Question or search query"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
    json_output: bool = typer.Option(False, "--json", help="Return machine-readable output"),
    embedding_api_key: Optional[str] = typer.Option(
        None,
        "--embedding-api-key",
        help="Override embedding provider API key for this query",
    ),
    rerank_api_key: Optional[str] = typer.Option(
        None,
        "--rerank-api-key",
        help="Override reranker LLM API key (llm strategy only)",
    ),
) -> None:
    """Run an ad-hoc query against the vector store."""

    cfg = _load_config(config)
    store = PgVectorStore(cfg)
    embeddings = _build_embedding(cfg, api_key=embedding_api_key)
    reranker, rerank_top_n = _build_reranker(cfg, llm_api_key=rerank_api_key)
    retriever = Retriever(
        store=store,
        embeddings=embeddings,
        reranker=reranker,
        top_k=cfg.retrieval.top_k,
        score_threshold=cfg.retrieval.score_threshold,
        rerank_top_n=rerank_top_n,
    )

    results = retriever.query(question)
    if json_output:
        import json

        typer.echo(
            json.dumps(
                [
                    {
                        "chunk_id": record.chunk_id,
                        "source_id": record.source_id,
                        "score": record.score,
                        "metadata": dict(record.metadata),
                        "tags": list(record.tags),
                        "content": record.content,
                    }
                    for record in results
                ],
                indent=2,
            )
        )
        return

    for idx, record in enumerate(results, start=1):
        typer.echo(f"[{idx}] score={record.score:.4f} source={record.source_id}")
        typer.echo(record.content)
        typer.echo("-")


@app.command()
def sources(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
) -> None:
    """List source documents discovered by the pipeline."""

    cfg = _load_config(config)
    pipeline = IngestionPipeline(cfg)
    documents = pipeline.plan_documents()
    if not documents:
        typer.echo("No documents matched.")
        raise typer.Exit(code=0)

    for doc in documents:
        typer.echo(f"{doc.source_id} -> {doc.path}")


# ---------------------------------------------------------------------------

def _load_config(path: Optional[Path]) -> ProjectConfig:
    resolved = resolve_config_path(path)
    return load_config(resolved)


def _build_embedding(config: ProjectConfig, api_key: Optional[str] = None) -> EmbeddingClient:
    if config.embedding.provider == "openai-proxy":
        if api_key:
            updated = config.embedding.model_copy(update={"api_key": api_key, "api_key_env": None})
        else:
            updated = config.embedding
        return OpenAIProxyEmbeddingClient(updated)
    raise ValueError(f"Unsupported embedding provider: {config.embedding.provider}")


def _build_reranker(config: ProjectConfig, llm_api_key: Optional[str] = None):
    strategy = config.rerank.strategy
    params = config.rerank.parameters
    top_n = max(config.rerank.top_n, 0)
    top_n_value = top_n or None
    if strategy == "none":
        return NullReranker(), None
    if strategy == "bm25":
        k1 = float(params.get("k1", 1.5)) if params else 1.5
        b = float(params.get("b", 0.75)) if params else 0.75
        return BM25Reranker(k1=k1, b=b), top_n_value
    if strategy == "llm":
        if not params:
            raise ValueError("LLM reranker requires parameters 'base_url' and 'model'")
        base_url = params.get("base_url") or config.embedding.base_url
        model = params.get("model")
        if not base_url or not model:
            raise ValueError("LLM reranker requires both base_url and model settings")
        api_key_env = params.get("api_key_env") or config.embedding.api_key_env
        api_key = llm_api_key or params.get("api_key")
        reranker = LLMReranker(
            base_url=base_url,
            model=model,
            api_key_env=(None if api_key else api_key_env),
            api_key=api_key,
            temperature=float(params.get("temperature", 0.0)),
            timeout=int(params.get("timeout", 30)),
            max_retries=int(params.get("max_retries", 3)),
        )
        return reranker, top_n_value
    if strategy == "custom":
        raise ValueError("Custom reranker strategy requires project-defined implementation")
    return KeywordOverlapReranker(), top_n_value


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
