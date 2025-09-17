# Raging

Raging is a composable toolkit that helps small teams stand up Retrieval Augmented Generation (RAG) workflows quickly. It packages ingestion, embedding, storage, and retrieval primitives so each project can stitch together a tailored flow while relying on maintained connectors from LangChain and LlamaIndex.

## Highlights

- **Pluggable ingestion registry** – map file patterns to handlers that know how to load and chunk documents.
- **Embeddings via proxy** – call your OpenAI-compatible proxy with retries and caching hooks.
- **pgvector integration** – store chunk metadata, text, and embeddings in PostgreSQL with IVFFLAT indexing.
- **Optional reranking** – re-order results with lightweight heuristics or custom scorers.
- **CLI ready** – run `raging ingest` and `raging query` commands from project configs.
- **Checksum aware** – skip re-embedding unchanged chunks unless you explicitly force it, with metrics logged per run and per source.
- **Flexible reranking** – choose between keyword, BM25, or LLM-based reordering via the proxy you already run, supplying per-tenant API keys at runtime.

## Package Layout

```
raging/
  cli/            # Typer-based CLI entry points
  config.py       # Project configuration models and loader
  embeddings/     # Embedding client abstractions + proxy client
  ingest/         # Base handler interfaces and default registry
  retrieval/      # Retriever and reranker interfaces
  storage/        # Vector store abstractions and pgvector implementation
  plugins/        # Future optional loaders (S3, Confluence, etc.)
```

## Quick Start

1. Install the package (extras enable specific ingestion handlers):

   ```bash
   pip install -e .[ingest,pdf,excel,docx]
   ```

2. Create `raging.yaml` in your project:

   ```yaml
   embedding:
     provider: openai-proxy
     model: text-embedding-3-small
     base_url: https://your-proxy/v1
   storage:
     kind: pgvector
     connection_url: postgresql+psycopg://user:pass@host:5432/db
     schema: raging
     collection: default
   ingest:
     checksum_algorithm: sha256
     sources:
       - path: ./docs
         include: ["**/*.md", "**/*.txt"]
         handler: markdown
         tags: [knowledge-base]
   retrieval:
     top_k: 8
   rerank:
     strategy: bm25
     top_n: 10
     parameters:
       k1: 1.2
       b: 0.7
   ```

3. Initialize the database schema:

   ```bash
   raging init-db --config raging.yaml
   ```

4. Ingest documents:

   ```bash
   raging ingest --config raging.yaml
   ```

   Add `--force` to re-embed everything regardless of cached checksums.
   Each run stores summary stats in Postgres so you can audit what changed.

   Inject different tenant credentials on demand:

   ```bash
   raging ingest --config raging.yaml --embedding-api-key tenant-token --rerank-api-key tenant-rerank
   ```

5. Run an ad-hoc query:

   ```bash
   raging query --config raging.yaml "How do I deploy the service?"
   ```

## Reranking Options

Set `rerank.strategy` to `none`, `bm25`, `llm`, or leave it blank to use the keyword overlap heuristic. Example LLM configuration (reuses your proxy):

```yaml
rerank:
  strategy: llm
  top_n: 6
  parameters:
    base_url: https://your-proxy/v1
    model: gpt-4o-mini
    api_key_env: PROXY_API_KEY
    temperature: 0.1
```

Only the top `top_n` hits are reranked; the remainder keep the original vector order.

## Multi-Tenant Usage

- Treat `raging.yaml` as a template. In code, load it once (`load_config`) and `model_copy` it per tenant to override `storage.collection`, `embedding.api_key`, etc.
- Collections are just string labels stored with each chunk; every ingest/query filters on that value. Use `collection = f"tenant_{tenant_id}"` to isolate tenants inside the same tables.
- If you prefer CLI-only workflows, generate a temporary config per tenant or add `--collection` and API key flags when invoking the commands.
- Schema separation (`storage.schema`) is orthogonal: set it if you want Raging tables in their own Postgres schema, but you can still join across schemas as needed.

## Philosophy

Raging favors clarity over magic: components are plain Python classes with explicit interfaces. Adding new ingestion sources or rerankers means subclassing the relevant base class and registering it with the dispatcher. The CLI is a convenience layer that uses the same building blocks to keep parity between automation and development workflows.
