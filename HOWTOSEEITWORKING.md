# Try It Out

1. **Install dependencies** (inside your virtualenv):
   ```bash
   pip install -e .[ingest,pdf,docx,excel]
   pip install psycopg[binary] pgvector
   ```

2. **Enable `pgvector` extension** (once per database, run as superuser):
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **Create `raging.yaml`** (template that you can tweak programmatically per tenant):
   ```yaml
   embedding:
     provider: openai-proxy
     base_url: https://your-proxy/v1
     model: text-embedding-3-small
   storage:
     kind: pgvector
     connection_url: postgresql+psycopg://user:pass@host:5432/db
     schema: raging
     collection: default
   ingest:
     checksum_algorithm: sha256
     sources:
       - path: ./docs
         include: ["**/*.md", "**/*.pdf"]
   retrieval:
     top_k: 6
   rerank:
     strategy: bm25
     top_n: 3
   ```

4. **Initialize database schema**:
   ```bash
   raging init-db --config raging.yaml
   ```

5. **Ingest documents** (optionally injecting per-tenant keys):
   ```bash
   raging ingest --config raging.yaml \
     --embedding-api-key tenant-embed-token \
     --rerank-api-key tenant-rerank-token
   ```

6. **Query**:
   ```bash
   raging query --config raging.yaml "How do I deploy the service?" --rerank-api-key tenant-rerank-token
   ```

7. **Programmatic multi-tenant example**:
   ```python
   from raging.config import load_config
   from raging.ingest.pipeline import IngestionPipeline
   from raging.storage.pgvector import PgVectorStore
   from raging.embeddings.proxy import OpenAIProxyEmbeddingClient

   base_cfg = load_config("raging.yaml")

   def ingest_for_tenant(tenant_id: str, api_key: str):
       cfg = base_cfg.model_copy(
           update={
               "storage": {"collection": f"tenant_{tenant_id}"},
               "embedding": {"api_key": api_key, "api_key_env": None},
           }
       )
       pipeline = IngestionPipeline(cfg)
       store = PgVectorStore(cfg)
       embeddings = OpenAIProxyEmbeddingClient(cfg.embedding)
       # ... supply tenant-specific documents to pipeline.run()
   ```

The CLI prints processed/skipped/emitted counts, and queries show top matches (use `--json` for machine-readable output).
